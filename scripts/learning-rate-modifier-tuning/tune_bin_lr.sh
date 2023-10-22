#!/bin/bash

declare -a beta_lrs=(10 1 0.1 0.01 0.000001)
declare -a gamma_lrs=(10 1 0.1 0.01 0.000001)
declare -a lambda_lrs=(10 1 0.1 0.01 0.000001)
i=0

for LR_B in "${beta_lrs[@]}" ; do
    for LR_G in "${gamma_lrs[@]}" ; do
        for LR_L in "${lambda_lrs[@]}" ; do
            CMD="CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 0 --dataset amex --model gru-rnn --preprocessing-method standard-scaler --num-cross-validation-folds 1 --adaptive-layer bin --experiment-name bin-tuning-${i} --override='bin:beta_lr:${LR_B} bin:gamma_lr:${LR_G} bin:lambda_lr:${LR_L} fit:num_epochs:10'"
            eval "$CMD"
            i=$((i+1))
        done
    done 
done
