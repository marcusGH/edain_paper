#!/bin/bash

declare -a scale_lrs=(0.1 0.0001 0.00000001)
declare -a shift_lrs=(0.1 0.01)
declare -a outlier_lrs=(10 1 0.1 0.01 0.001 0.00001 0.0000001)
declare -a power_lrs=(10 1 0.1 0.01 0.001 0.00001 0.0000001)
i=0

for LR_scale in "${scale_lrs[@]}" ; do
    for LR_shift in "${shift_lrs[@]}" ; do
        for LR_out in "${outlier_lrs[@]}" ; do
            for LR_pow in "${power_lrs[@]}" ; do
                CMD="CUDA_VISIBLE_DEVICES=2 python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-beta.yaml --device 0 --dataset lob --model gru-rnn --preprocessing-method identity --num-cross-validation-folds 1 --experiment-name edain-ba-lob-tuning-${i} --override='edain_bijector_fit:scale_lr:${LR_scale} edain_bijector_fit:shift_lr:${LR_shift} edain_bijector_fit:outlier_lr:${LR_out} edain_bijector_fit:power_lr:${LR_pow}' --adaptive-layer edain"
                i=$((i+1))
                eval "$CMD"
            done
        done 
    done 
done
