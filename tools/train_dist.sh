# !/usr/bin/env bash

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS=8


# 调试模式
# export NCCL_DEBUG=INFO  

OMP_NUM_THREADS=$GPUS torchrun 
    --nproc_per_node=$GPUS 
    --master_port=$((RANDOM + 10000)) tools/train.py \
    configs/svg/svg_pointT.yaml \
    --dist \
    --exp_name ArchCAD_XXX \
    --sync_bn \
    
#--resume  path to resume from
    
