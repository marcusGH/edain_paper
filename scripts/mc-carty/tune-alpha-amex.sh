#!/bin/sh
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 7 --dataset amex --model gru-rnn --preprocessing-method mccarter-0.1 --ignore-time --experiment-name="mcCarter-amex-0.1" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 7 --dataset amex --model gru-rnn --preprocessing-method mccarter-1 --ignore-time --experiment-name="mcCarter-amex-1" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 7 --dataset amex --model gru-rnn --preprocessing-method mccarter-10 --ignore-time --experiment-name="mcCarter-amex-10" --num-cross-validation-folds 5 --random-state 42
python3 src/experiments/run_experiment.py --experiment-config src/experiments/configs/experiment-config-alpha.yaml --device 7 --dataset amex --model gru-rnn --preprocessing-method mccarter-100 --ignore-time --experiment-name="mcCarter-amex-100" --num-cross-validation-folds 5 --random-state 42

