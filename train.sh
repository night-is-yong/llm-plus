DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6002
"

torchrun $DISTRIBUTED_ARGS train.py \
    --model_name_or_path /home/chuan/models/baichuan-inc/Baichuan2-13B-Chat \
    --train_data_path baichuan/baichuan_example_dataset/belle_chat_ramdon_10k.json \
    --model_type baichuan \
    --lora_target_modules W_pack \
    --use_4_bit \
    --bf16 \
    --max_seq_length 512 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --report_to "none" \
    --seed 1234 \
    --gradient_checkpointing \
    --deepspeed baichuan/deepspeed.json \
    --output_dir output \
