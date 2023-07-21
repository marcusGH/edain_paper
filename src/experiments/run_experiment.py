import yaml
import torch
import time
import torch.nn.functional as F
import os
import numpy as np
import argparse
from src.preprocessing.static_transformations import (
    IdentityTransform,
    StandardScalerTimeSeries,
    LogMinMaxTimeSeries,
    LogStandardScalerTimeSeries,
    MinMaxTimeSeries,
    TanhStandardScalerTimeSeries,
    WinsorizeDecorator,
    IgnoreTimeDecorator,
    MixedTransformsTimeSeries,
)
from src.preprocessing.normalizing_flows import (
    EDAIN_Layer,
    EDAINScalerTimeSeries,
    EDAINScalerTimeSeriesDecorator,
)
from src.models.basic_grunet import GRUNetBasic
from src.models.adaptive_grunet import AdaptiveGRUNet
from src.lib.experimentation import (
    EarlyStopper,
    cross_validate_experiment,
    load_numpy_amex_data,
)

static_preprocessing_methods = {
    'identity' : IdentityTransform,
    'standard-scaler' : StandardScalerTimeSeries,
    'log-min-max' : LogMinMaxTimeSeries,
    'log-standard-scaler' : LogStandardScalerTimeSeries,
    'min-max' : MinMaxTimeSeries,
    'tanh-standard-scaler' : TanhStandardScalerTimeSeries,
    # 'edain-kl' : EDAINScalerTimeSeries,
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
    help="The model to train for the experiment.",
    choices=['gru-rnn'],
    required=True,
)
parser.add_argument("--adaptive-layer",
    help="If set, the adaptive preprocessing layer to prepend to the model and train with backpropagation",
    dest='adaptive_layer',
    choices=['dain', 'bin', 'edain'],
    default=None,
    required=False,
)
parser.add_argument("--preprocessing-method",
    help="The preprocessing method to use",
    dest='preprocessing_method',
    choices=list(static_preprocessing_methods.keys()),
    required=True,
)
parser.add_argument("--edain-kl",
    help="If set, use the EDAIN-KL method for preprocessing. The '--preprocessing-method' argument will then be applied before fitting EDAIN with KL divergence loss",
    dest='edain_kl',
    action='store_true',
    required=False,
)
parser.add_argument("--winsorize",
    help="If set, winsorize the data before applying the preprocessing method",
    action='store_true',
    required=False,
)
parser.add_argument("--ignore-time",
    help="If set, ignore the time dimension of the data",
    dest='ignore_time',
    action='store_true',
    required=False,
)
parser.add_argument("--num-cross-validation-folds",
    help="The number of cross-validation folds to use",
    dest='num_cross_validation_folds',
    type=int,
    default=5,
    required=False,
)
parser.add_argument("--random-state",
    help="The random state to use for the experiment",
    dest='random_state',
    type=int,
    default=42,
    required=False,
)
parser.add_argument("--experiment-name",
    help="The name of the experiment",
    dest='experiment_name',
    type=str,
    required=True,
)

if __name__ == '__main__':
    args = parser.parse_args()
    # read main config
    with open("config.yaml", 'r') as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # read experiment config
    with open(args.experiment_config, 'r') as f:
        exp_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # parse device
    if args.device == 'cpu':
        dev = torch.device('cpu')
    else:
        dev = torch.device('cuda', args.device)

    start_time = time.time()
    optimizer_init_fn = None

    # set random seed
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    ################################################################
    ###            Part 1: Loading dataset and model             ###
    ################################################################

    # load dataset
    if args.dataset == 'amex':
        # TODO: refactor load_numpy_data to work with this interface
        # TODO: refactor out dataset-specific config into amex_dataset.yaml file
        X, y = None, None
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")

    # load model
    if args.model == 'gru-rnn' and args.adaptive_layer is None:
        model_init_fn = lambda : GRUNetBasic(
            num_cat_columns=exp_cfg['num_categorical_columns'],
            **exp_cfg['gru_model']
        )
    elif args.model == 'gru-rnn' and args.adaptive_layer is not None:
        # TODO: Do the following:
        #       1. refactor adaptive grunet to not take time_series_length argument
        #       2. refactor adaptive grunet to take adaptive layer instance instead of init function
        if args.adaptive_layer == 'dain':
            adaptiver_layer_init_fn = None
        elif args.adaptive_layer == 'bin':
            adaptive_layer = None
        elif args.adaptive_layer == 'edain':
            adaptive_layer = None

            # set up optimizer
            base_lr = exp_cfg['edain_bijector_fit']['lr']
            optimizer_init_fn = lambda mod: torch.optim.Adam(
                # the learning rate for each layer in EDAIN
                mod.preprocess.get_optimizer_param_list(
                    **{k : exp_cfg['edain_bijector_fit'][k] for k in exp_cfg['edain_bijector_fit'] if 'lr' in k}
                ) +
                # the rest of the parameters of the daptive gru model
                [
                    {'params' : mod.gru.parameters(), 'lr' : base_lr},
                    {'params': mod.emb_layers.parameters(), 'lr': base_lr},
                    {'params': mod.feed_forward.parameters(), 'lr': base_lr}
                ], lr=0.001)
        else:
            raise ValueError(f"Adaptive layer not supported: {args.adaptive_layer}")

        # TODO: refactor adaptive grunet as above (1) and (2)
        # model_init_fn = lambda : AdaptiveGRUNet(
        #     adaptive_layer=adaptive_layer_init_fn(),
        #     num_cat_columns=exp_cfg['num_categorical_columns'],
        #     **exp_cfg['gru_model'],
        # )
    else:
        raise ValueError(f"Model not supported: {args.model}")

    ################################################################
    ###            Part 2: Setup preprocessing methods           ###
    ################################################################

    # TODO: include other parameters such as a and b for min-max
    scaler_init_fn = lambda : static_preprocessing_methods[args.preprocessing_method](exp_cfg['time_series_length'])
    # TODO: add optional winsorization
    # TODO: add optinal time dimension removal (requires setting exp cfg time_series_length to 1)

    # use the above scaler function to decorate our EDAIN-KL scaler
    if args.edain_kl:
        fit_kwargs = exp_cfg['edain_bijector_fit']
        fit_kwargs['device'] = dev
        scaler_init_fn = lambda : EDAINScalerTimeSeriesDecorator(
            scaler=scaler_init_fn(),
            time_series_length=exp_cfg['time_series_length'],
            input_dim=exp_cfg['gru_model']['num_features'] - exp_cfg['num_categorical_columns'],
            bijector_kwargs=exp_cfg['edain_bijector'],
            bijector_fit_kwargs=fit_kwargs,
        )

    ################################################################
    ###          Part 3: Run cross-validation experiment         ###
    ################################################################

    # set up loss function
    if exp_cfg['loss'] == 'bce':
        loss_fn = F.binary_cross_entropy
    else:
        raise ValueError(f"Loss not supported: {exp_cfg['loss']}")

    # set up optimizer function
    if optimizer_init_fn is None:
        if exp_cfg['fit']['optimizer'] == 'adam':
            optimizer_init_fn = lambda mod : torch.optim.Adam(
                mod.parameters(),
                lr=exp_cfg['fit']['lr'],
            )
        else:
            raise ValueError(f"Optimizer not supported: {args.optimizer}")

    # set up scheduler function
    scheduler_init_fn = lambda opt : torch.optim.lr_scheduler.MultiStepLR(
        optimizer=opt,
        milestones=exp_cfg['fit']['scheduler_milestones'],
        gamma=0.1,
    )

    # set up early stopper
    early_stopper_init_fn = lambda : EarlyStopper(
        patience=exp_cfg['fit']['early_stopper_patience'],
        min_delta=exp_cfg['fit']['early_stopper_min_delta'],
    )

    history = cross_validate_experiment(
        model_init_fn=model_init_fn,
        preprocess_init_fn=scaler_init_fn,
        optimizer_init_fn=optimizer_init_fn,
        scheduler_init_fn=scheduler_init_fn,
        early_stopper_init_fn=early_stopper_init_fn,
        loss_fn=loss_fn,
        X=X,
        y=y,
        num_epochs=exp_cfg['fit']['num_epochs'],
        dataloader_kwargs=exp_cfg['fit']['data_loader'],
        num_folds=args.num_cross_validation_folds,
        device=dev,
        random_state=args.random_state,
    )

    ################################################################
    ###            Part 4: Save experiment results               ###
    ################################################################

    # augment history with experiment metadata
    history['experiment_name'] = args.experiment_name
    history['experiment_config'] = exp_cfg
    history['cli_arguments'] = vars(args)

    save_path = os.path.join(main_cfg['experiment_directory'], f"{args.experiment_name}.npy")
    np.save(save_path, history)
    # print time in minutes and seconds
    print(f"Completed experiment: {args.experiment_name}.")
    print(f"Experiment took {(time.time() - start_time) / 60:.0f} minutes and {(time.time() - start_time) % 60:.0f} seconds")