import torch

from arguments import ModelArguments, TrainingArguments, DataArguments
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer

tqdm.pandas()


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    match model_args.model_type:
        case "chatglm":
            data_args.train_format = "input-output"
            from chatglm.chatglm_dataset import get_dataset
            train_dataset = get_dataset(data_args, tokenizer)
            data_module = {"train_dataset": train_dataset}
        case "qwen":
            tokenizer.pad_token_id = tokenizer.eod_id
            from qwen.qwen_dataset import make_supervised_data_module
            data_module = make_supervised_data_module(
                tokenizer=tokenizer, data_args=data_args, max_len=data_args.max_seq_length
            )
        case "baichuan":
            from baichuan.supervised_dataset import SupervisedDataset
            dataset = SupervisedDataset(
                data_args.train_data_path, tokenizer, data_args.max_seq_length
            )
            data_module = {"train_dataset": dataset}
        case "yi":
            from yi.yi_dataset import create_prompt_dataset
            train_dataset, eval_dataset = create_prompt_dataset(
                0,
                [data_args.train_data_path],
                '10, 0, 0',
                data_args.output_path,
                0,
                training_args.seed,
                tokenizer,
                data_args.max_seq_length,
                end_of_conversation_token="<|endoftext|>"
            )
            data_module = {"train_dataset": train_dataset, "eval_dataset": eval_dataset}

    if model_args.use_4_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=model_args.use_4_bit,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    print(model)

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=model_args.lora_target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=['output_layer']
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        packing=True,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_model(data_args.output_path)


if __name__ == '__main__':
    main()
