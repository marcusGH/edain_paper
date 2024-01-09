#!/bin/sh
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/config-hpc.yaml --device 1 --dataset hpc --model gru-rnn --preprocessing-method mccarter-0.1 --ignore-time --experiment-name="HPC-KDIT-0.1" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/config-hpc.yaml --device 1 --dataset hpc --model gru-rnn --preprocessing-method mccarter-1 --ignore-time --experiment-name="HPC-KDIT-1" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/config-hpc.yaml --device 1 --dataset hpc --model gru-rnn --preprocessing-method mccarter-10 --ignore-time --experiment-name="HPC-KDIT-10" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/config-hpc.yaml --device 1 --dataset hpc --model gru-rnn --preprocessing-method mccarter-100 --ignore-time --experiment-name="HPC-KDIT-100" --num-cross-validation-folds 5 --random-state 42

