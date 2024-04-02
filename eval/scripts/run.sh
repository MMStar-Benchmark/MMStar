#!/bin/bash
set -x

export MASTER_PORT=29512
export NUM_GPUS=8
# --model  LLaMA2-7B Nous_Yi_34B llava_next_yi_34b

torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
    --verbose \
    --data MMStar \
    --model llava_next_yi_34b \
    --max-new-tokens 32 \
    --gen-mode mm