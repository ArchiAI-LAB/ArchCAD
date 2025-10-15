#!/usr/bin/env bash
#SBATCH -p vanke
#SBATCH -N 1
#SBATCH -J cdn-exp-l6
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:hgx:8
#SBATCH --mem 128GB

export PYTHONPATH=./


datadir=dataset
out=./
python tools/inference.py \
	 configs/svg/svg_pointT_O.yaml  \
	 model.pth  \
	 --datadir $datadir \
	 --out $out 
