#!/bin/bash

declare -a beta_lrs=(10 1 0.1 0.0001 0.00000001)
declare -a gamma_lrs=(10 1 0.1 0.0001 0.00000001)
declare -a lambda_lrs=(10 1 0.1 0.0001 0.00000001)
i=0

for LR_beta in "${beta_lrs[@]}" ; do
    for LR_gamma in "${gamma_lrs[@]}" ; do
        for LR_lambda in "${lambda_lrs[@]}" ; do
            CMD="CUDA_VISIBLE_DEVICES=2 python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-beta.yaml --device 0 --dataset lob --model gru-rnn --preprocessing-method identity --num-cross-validation-folds 1 --experiment-name bin-lob-tuning-${i} --override='bin:beta_lr:${LR_beta} bin:gamma_lr:${LR_gamma} bin:lambda_lr:${LR_lambda}' --adaptive-layer bin"
            i=$((i+1))
            eval "$CMD"
        done 
    done 
done
