git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j LLAMA_CUBLAS=1

python convert-hf-to-gguf.py /home/chuan/models/qwen/Qwen1___5-7B-Chat
# quantize the model to 4-bits (using Q4_K_M method)
./quantize /home/chuan/models/qwen/Qwen1___5-7B-Chat/ggml-model-f16.gguf /home/chuan/models/qwen/Qwen1___5-7B-Chat/ggml-model-Q4_K_M.gguf Q4_K_M

# start inference on a gguf model
./main -m /home/chuan/models/qwen/Qwen1___5-7B-Chat/ggml-model-Q4_K_M.gguf -n 128