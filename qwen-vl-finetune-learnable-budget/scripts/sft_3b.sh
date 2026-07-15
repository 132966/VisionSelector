#!/bin/bash
export http_proxy=http://10.229.18.27:8412 export https_proxy=http://10.229.18.27:8412 export HTTP_PROXY=http://10.229.18.27:8412 export HTTPS_PROXY=http://10.229.18.27:8412

# Wandb configuration
export WANDB_PROJECT="VisionSelector"
export WANDB_MODE=online

# Distributed training configuration
export TORCH_DISTRIBUTED_RUN_TIMEOUT=3600
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs
NPROC_PER_NODE=4

# DeepSpeed configuration
deepspeed=./scripts/zero3.json
# Model configuration
llm=../pretrained/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=5e-5
batch_size=8
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen_selector.py

# Dataset configuration (replace with public dataset names)
datasets=chartqa,coco%10,ocr_vqa
# Output configuration
run_name="VisionSelector-Qwen2.5-VL-3B-LearnableBudget-3epoch-20260714"
output_dir=../output_ckpt/VisionSelector-Qwen2.5-VL-3B-LearnableBudget-3epoch-20260714

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_compressor True \
    --budget 0.5 \
    --compression_weight_start 0.05 \
    --compression_weight_end 0.5 \
    --budget_lr 0.01 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --reg_weight_start 0.1 \
    --reg_weight_end 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb"

# Launch training
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
