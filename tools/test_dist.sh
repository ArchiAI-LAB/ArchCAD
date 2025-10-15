#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=8
# 可选：调试信息
# export NCCL_DEBUG=INFO


# 测试
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
    --dist \
    configs/svg/svg_pointT.yaml \
    model.pth 