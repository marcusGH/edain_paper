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
from src.preprocessing.adaptive_transformations import (
    DAIN_Layer,
    BiN_Layer,
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
    load_amex_numpy_data,
    undo_min_max_corrupt_func,
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
    choices=[str(i) for i in range(8)] + ['cpu'],
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
        dev = torch.device('cuda', int(args.device))

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
        # minor TODO: refactor out dataset-specific config into amex_dataset.yaml file
        X, y = load_amex_numpy_data(
            split_data_dir=os.path.join(main_cfg['dataset_directory'], 'derived', 'processed-splits'),
            fill_dict=exp_cfg['fill'],
            corrupt_func=lambda X, y: undo_min_max_corrupt_func(X, y, args.random_state),
            num_categorical_features=exp_cfg['num_categorical_features']
        )
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")
    print(f"Finished loading dataset with covariates of shape {X.shape} and responses of shape {y.shape}")

    # load model
    if args.model == 'gru-rnn' and args.adaptive_layer is None:
        model_init_fn = lambda : GRUNetBasic(
            num_cat_columns=exp_cfg['num_categorical_features'],
            **exp_cfg['gru_model']
        )
    elif args.model == 'gru-rnn' and args.adaptive_layer is not None:
        ####################### DAIN layer #########################
        if args.adaptive_layer == 'dain':
            dain_layer_kwargs = exp_cfg['dain']
            dain_layer_kwargs['input_dim'] = exp_cfg['gru_model']['num_features'] - exp_cfg['num_categorical_features']
            adaptive_layer_init_fn = lambda : DAIN_Layer(**dain_layer_kwargs)
            adaptive_layer_optim_args = {
                'base_lr' : exp_cfg['fit']['base_lr']
            }
            # TODO: the averaging happens over time dimension now (???)
            dim_first = True
        ######################## BiN layer #########################
        elif args.adaptive_layer == 'bin':
            adaptive_layer_init_fn = lambda : BiN_Layer(
                input_shape=(exp_cfg['gru_model']['num_features'] - exp_cfg['num_categorical_features'], exp_cfg['time_series_length']),
            )
            dim_first = True
            adaptive_layer_optim_args = exp_cfg['bin']
            adaptive_layer_optim_args['base_lr'] = exp_cfg['fit']['base_lr']
        ######################## EDAIN layer #######################
        elif args.adaptive_layer == 'edain':
            # TODO: currently, the time axis is broadcasted, so only learn for each dim, like in DAIN and BiN
            #       add option to learn for each of the T * D dimensions?
            adaptive_layer_init_fn = lambda : EDAIN_Layer(
                # TODO: when added flatten_time_dim, make this multiply with time_series_length
                input_dim=exp_cfg['gru_model']['num_features'] - exp_cfg['num_categorical_features'],
                # when using it as adaptive layer, we do a forward pass, not inverse
                invert_bijector=False,
                **exp_cfg['edain_bijector'],
            )
            adaptive_layer_optim_args = {
                k: exp_cfg['edain_bijector_fit'][k] for k in exp_cfg['edain_bijector_fit'] if 'lr' in k
            }
            dim_first = False
        else:
            raise ValueError(f"Adaptive layer not supported: {args.adaptive_layer}")

        # set up model
        print(f"Setting up adaptive model using layer: {args.adaptive_layer}")
        model_init_fn = lambda : AdaptiveGRUNet(
            adaptive_layer=adaptive_layer_init_fn(),
            num_cat_columns=exp_cfg['num_categorical_columns'],
            time_series_length=exp_cfg['time_series_length'],
            dim_first=dim_first,
            **exp_cfg['gru_model'],
        )

        # set up optimizer
        base_lr = exp_cfg['fit']['base_lr']
        optimizer_init_fn = lambda mod: torch.optim.Adam(
            # the learning rate for each layer in EDAIN, DAIN or BiN
            mod.preprocess.get_optimizer_param_list(
                **adaptive_layer_optim_args
            ) +
            # the rest of the parameters of the GRU RNN model
            [
                {'params': mod.gru.parameters(), 'lr' : base_lr},
                {'params': mod.emb_layers.parameters(), 'lr': base_lr},
                {'params': mod.feed_forward.parameters(), 'lr': base_lr}
            ], lr=exp_cfg['fit']['base_lr'])
    else:
        raise ValueError(f"Model not supported: {args.model}")
    print(f"Finished loading model")

    ################################################################
    ###            Part 2: Setup preprocessing methods           ###
    ################################################################

    if args.ignore_time:
        original_time_length = exp_cfg['time_series_length']
        exp_cfg['time_series_length'] = 1

    # minor TODO: include other parameters such as a and b for min-max
    msg = args.preprocessing_method
    preprocess_init_fn = lambda : static_preprocessing_methods[args.preprocessing_method](exp_cfg['time_series_length'])
    # TODO: add optional winsorization

    # use the above scaler function to decorate our EDAIN-KL scaler
    if args.edain_kl:
        fit_kwargs = exp_cfg['edain_bijector_fit']
        fit_kwargs['device'] = dev
        _scaler_init_fn = lambda : EDAINScalerTimeSeriesDecorator(
            scaler=preprocess_init_fn(),
            time_series_length=exp_cfg['time_series_length'],
            input_dim=exp_cfg['gru_model']['num_features'] - exp_cfg['num_categorical_features'],
            bijector_kwargs=exp_cfg['edain_bijector'],
            bijector_fit_kwargs=fit_kwargs,
        )
        msg = f"EDAIN-KL({msg})"
    else:
        _scaler_init_fn = preprocess_init_fn

    # check if scaler should be decorated with time-flatten scaler
    if args.ignore_time:
        scaler_init_fn = lambda : IgnoreTimeDecorator(
            scaler=_scaler_init_fn(),
            time_series_length=original_time_length,
        )
        msg = f"IgnoreTime({msg})"
    else:
        scaler_init_fn = _scaler_init_fn
    print(f"Finished setting up preprocessing technique: {msg}")

    ################################################################
    ###          Part 3: Run cross-validation experiment         ###
    ################################################################

    # set up loss function
    if exp_cfg['fit']['loss'] == 'bce':
        loss_fn = F.binary_cross_entropy
    else:
        raise ValueError(f"Loss not supported: {exp_cfg['loss']}")

    # set up optimizer function
    if optimizer_init_fn is None:
        if exp_cfg['fit']['optimizer'] == 'adam':
            optimizer_init_fn = lambda mod : torch.optim.Adam(
                mod.parameters(),
                lr=exp_cfg['fit']['base_lr'],
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

    print(f"Starting cross-validation experiment")
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
        dataloader_kwargs=exp_cfg['data_loader'],
        num_folds=args.num_cross_validation_folds,
        device=dev,
        random_state=args.random_state,
        num_categorical_features=exp_cfg['num_categorical_features'],
    )

    ################################################################
    ###            Part 4: Save experiment results               ###
    ################################################################

    # print time in minutes and seconds
    elapsed_time = time.time() - start_time
    print(f"Completed experiment: {args.experiment_name}.")
    print(f"Experiment took {elapsed_time // 60} minutes and {elapsed_time % 60:.0f} seconds")

    # augment history with experiment metadata
    history['experiment_name'] = args.experiment_name
    history['experiment_config'] = exp_cfg
    history['cli_arguments'] = vars(args)
    history['elapsed_time'] = elapsed_time

    save_path = os.path.join(main_cfg['experiment_directory'], f"{args.experiment_name}.npy")
    np.save(save_path, history)
