#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=1 #机器数量
ARG_NPROC_PER_NODE=1 #使用GPU数量
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
if [ -z "$WORLD_SIZE" ] || [ -z "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=16
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE)))

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_PROJECT=videollama2qwen2_downstream_sft
RUN_NAME=siglip_tcv35_7b_16f_lora
DATA_DIR=datasets
OUTP_DIR=VideoLLaMA2-tuned-highway-epoch5

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK  \
    videollama2/train.py \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --mm_projector_lr 2e-5 \
    --deepspeed "scripts/zero333.json" \
    --model_type videollama2_qwen2 \
    --model_path "../VideoLLaMA2.1-7B-16F" \
    --vision_tower ../siglip-so400m-patch14-384 \
    --mm_projector_type stc_connector_v35 \
    --pretrain_mm_mlp_adapter "../VideoLLaMA2.1-7B-16F-Base/mm_projector.bin" \
    --data_path "${DATA_DIR}/custom_sft/Videochat.json" \
    --data_folder "${DATA_DIR}/custom_sft" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 False \
    --tf32 False \
    --fp16 True \
    --output_dir "${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME}" \
    --num_train_epochs 5 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0.03 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --report_to tensorboard \
    --run_name "$RUN_NAME" \
 #   2025-03-23 15:28:35,584] [WARNING] [stage3.py:2008:step] 1 pytorch allocator cache flushes since last step. this happens when there is high memory pressure and is detrimental to performance. if this is happening frequently consider adjusting settings to reduce memory consumption. If you are unable to make the cache flushes go away consider adding get_accelerator().empty_cache() calls in your training loop to ensure that all ranks flush their caches at the same time