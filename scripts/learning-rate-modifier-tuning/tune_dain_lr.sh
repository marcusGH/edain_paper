#!/bin/bash

declare -a mean_lrs=(10 1 0.1 0.01 0.001 0.0001)
declare -a scale_lrs=(10 1 0.1 0.01 0.001 0.0001)
i=0

for LR_M in "${mean_lrs[@]}" ; do
    for LR_S in "${scale_lrs[@]}" ; do
        CMD="CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 0 --dataset amex --model gru-rnn --preprocessing-method standard-scaler --num-cross-validation-folds 1 --adaptive-layer dain --experiment-name dain-tuning-${i} --override='dain:mean_lr:${LR_M} dain:scale_lr:${LR_S} fit:num_epochs:10'"
        i=$((i+1))
        eval "$CMD"
    done 
done
