#!/bin/bash

declare -a num_clusts=(2 4 6 8 10 20 30)
i=0

for K in "${num_clusts[@]}" ; do
    CMD="CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 0 --dataset amex --model gru-rnn --preprocessing-method mixed --num-cross-validation-folds 5 --experiment-name mixture-clustering-tuning-${i} --mixture-device-ids 2 3 4 5 --override='mixture:number_of_clusters:${K}'"
    i=$((i+1))
    eval "$CMD"
done
