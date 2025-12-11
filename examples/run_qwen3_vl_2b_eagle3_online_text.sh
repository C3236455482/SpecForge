#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# support tp1 train eagle3 for Qwen3-VL-2B-Instruct
NUM_GPUS=8


export WANDB_API_KEY="42a32cbe01f63106f00434f65f9375800959b897"
export WANDB_PROJECT="Eagle3-Qwen3-VL-2B-sharegpt"
export WANDB_NAME="run-$(date +%Y%m%d-%H%M%S)"

wandb login --relogin $WANDB_API_KEY

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /ch/pretrained_models/Qwen3-VL-2B-Instruct \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-2b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --output-dir $ROOT_DIR/outputs/Qwen3-VL-2B-eagle3 \
    --build-dataset-num-proc 0 \
    --num-epochs 10 \
    --batch-size 3 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template qwen3-vl \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --vlm-text-only \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --report-to wandb \
    --wandb-name $WANDB_NAME \
    --wandb-project $WANDB_PROJECT

