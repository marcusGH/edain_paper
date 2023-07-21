import yaml
import os
import numpy as np
import argparse


preprocessing_methods = {
    'identity' : IdentityTransform,
    'standard-scaler' : StandardScalerTimeSeries,
    # TODO: continue
}


parser = argparse.ArgumentParser(
    prog="Experiment runner script",
    description="Runs a cross-validation experiment using specified model, preprocessing method and dataset",
)


################################################################
###                     Argument setup                       ###
################################################################
parser.add_argument("--experiment-config",
    help="The path to the yaml experiment configuration file",
    required=True,
)
parser.add_argument("--device",
    help="The device to run the experiment on",
    choices=list(range(8)) + ['cpu'],
    default=0,
    required=True,
)
parser.add_argument("--dataset",
    help="The dataset to use for the experiment",
    choices=['amex'],
    required=True,
)
parser.add_argument("--model",
    help="The model to train for the experiment."
    choices=['gru-rnn'],
    required=True,
)
parser.add_argument("--adaptive-layer",
        help="If set, the adaptive preprocessing layer to prepend to the model and train with backpropagation",
        choices=['dain', 'bin', 'edain'],
        default=None,
        required=False,
)
parser.add_argument("--preprocessing-method",
    help="The preprocessing method to use",
    choices=list(preprocessing_methods.keys()),
    required=True,
)
# TODO: arguments to add:
# - winsorize for if apply winsorization
# - ignore time decorator boolean


if __name__ == '__main__':
    args = parser.parse_args()
    # TODO: ...
