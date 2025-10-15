#!/usr/bin/env bash
#SBATCH -p vanke
#SBATCH -N 1
#SBATCH -J 7g
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem 128GB

export PYTHONPATH=./
GPUS=1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train2.py \
	--dist configs/svg/svg_pointT_O.yaml  \
	--exp_name debug \
	--sync_bn
