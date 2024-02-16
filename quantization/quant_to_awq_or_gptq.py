from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AwqConfig, GenerationConfig, GPTQConfig


# awq量化
def quantize():
    model_path = "/home/chuan/models/qwen/Qwen1___5-7B-Chat"
    quant_path = "/home/chuan/models/qwen/Qwen1___5-7B-Chat-awq"
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path,
                                               trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    print(quant_config)
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    # the pretrained transformers model is stored in the model attribute + we need to pass a dict
    model.model.config.quantization_config = quantization_config
    # a second solution would be to use Autoconfig and push to hub (what we do at llm-awq)

    # save model weights
    model.save_quantized(quant_path, safetensors=True)
    tokenizer.save_pretrained(quant_path)


# gptq量化
def quantize2():
    model_path = "/home/chuan/models/qwen/Qwen1___5-7B-Chat"
    quant_path = "/home/chuan/models/qwen/Qwen1___5-7B-Chat-gptq"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    dataset = [
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

    gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=gptq_config, device_map="auto",
                                                 trust_remote_code=True)

    model.save_pretrained(quant_path)
    tokenizer.save_pretrained(quant_path)


# 量化及测试量化后模型
def main():
    quantize2()
    tokenizer = AutoTokenizer.from_pretrained("/home/chuan/models/qwen/Qwen1___5-7B-Chat-gptq",
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("/home/chuan/models/qwen/Qwen1___5-7B-Chat-gptq",
                                                 device_map="auto",
                                                 bfloat16=True,
                                                 trust_remote_code=True).eval()

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    device = "cuda"
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == '__main__':
    main()
