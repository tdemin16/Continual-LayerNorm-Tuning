#!/bin/bash

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate cln

# Train in single stage using classifier rectification, alpha and shuffling the order of classes

SEED=0
TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_LAUNCH_BLOCKING=1 \
CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
         --master_port=29407 \
        main.py imr \
        --name INR-Single-Stage \
        --method 'single_stage' \
        --unbiased \
        --use_att \
        --seed $SEED \
        --shuffle
