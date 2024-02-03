from dataclasses import dataclass, field
from typing import List

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bf16: bool = field(default=False)
    fp16: bool = field(default=False)
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size (per device) for the training dataloader."}
    )
    per_device_eval_batch_size: int = field(
        default=16, metadata={"help": "Batch size (per device) for the evaluation dataloader."}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay to use."}
    )
    num_train_epochs: int = field(
        default=5, metadata={"help": "Number of training epochs to perform."}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "The scheduler type to use."}
    )
    num_warmup_steps: int = field(
        default=0, metadata={"help": "Number of steps for the warmup in the lr scheduler."}
    )
    seed: int = field(
        default=1234, metadata={"help": "A seed for reproducible training."}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Enable HF gradient checkpointing for model."}
    )
    report_to: str = field(
        default="tensorboard", metadata={"help": "report the results and logs to."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(metadata={"help": "chatglm,qwen,yi,baichuan"})
    ptuning_checkpoint: str = field(
        default=None, metadata={"help": "Path to p-tuning v2 checkpoints"}
    )
    config_name: str = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_4_bit: bool = field(default=False)
    lora: bool = field(default=False)
    prefix_tuning: bool = field(default=False)
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: bool = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    pre_seq_len: int = field(
        default=None
    )
    prefix_projection: bool = field(
        default=False
    )
    lora_r: int = field(
        default=64, metadata={"help": ""}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": ""}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": ""}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [""]
    )

@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None, metadata={"help": "The input training data file."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    output_path: str = field(
        default="output", metadata={"help": ""}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    max_target_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    train_format: str = field(
        default=None, metadata={"help": "The format of the training data file (mulit-turn or input-output)"},
    )
    preprocessing_num_workers: int = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
