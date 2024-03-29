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
    BaselineTransform,
    McCarterTimeSeries,
    LogMinMaxTimeSeries,
    LogStandardScalerTimeSeries,
    MinMaxTimeSeries,
    TanhStandardScalerTimeSeries,
    WinsorizeDecorator,
    IgnoreTimeDecorator,
    InvertCDFTimeSeries,
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
from src.preprocessing.mixture import (
    MixedTransformsTimeSeries,
    find_optimal_preprocessing_mixture_with_brute_force,
)
from src.models.basic_grunet import GRUNetBasic, GRUNetLOB
from src.models.adaptive_grunet import AdaptiveGRUNet, AdaptiveGRUNetLOB
from src.lib.experimentation import (
    EarlyStopper,
    cross_validate_experiment,
    train_evaluate_lob_anchored,
    load_amex_numpy_data,
    load_power_numpy_data,
    undo_min_max_corrupt_func,
)

static_preprocessing_methods = {
    'identity' : IdentityTransform,
    'standard-scaler' : StandardScalerTimeSeries,
    'mccarter-0.1' : lambda time_series_length : McCarterTimeSeries(time_series_length, alpha=0.1),
    'mccarter-1' : lambda time_series_length : McCarterTimeSeries(time_series_length, alpha=1),
    'mccarter-10' : lambda time_series_length : McCarterTimeSeries(time_series_length, alpha=10),
    'mccarter-100' : lambda time_series_length : McCarterTimeSeries(time_series_length, alpha=100),
    # the numbers represent binary flags for the three layers: winsorize, z_score, yeo-johnson
    'baseline-111' : lambda time_series_length : BaselineTransform(time_series_length, True, True, True),
    'baseline-010' : lambda time_series_length : BaselineTransform(time_series_length, False, True, False),
    'baseline-011' : lambda time_series_length : BaselineTransform(time_series_length, False, True, True),
    'baseline-110' : lambda time_series_length : BaselineTransform(time_series_length, True, True, False),
    'log-min-max' : LogMinMaxTimeSeries,
    'log-standard-scaler' : LogStandardScalerTimeSeries,
    'min-max' : MinMaxTimeSeries,
    'tanh-standard-scaler' : TanhStandardScalerTimeSeries,
    'cdf-invert' : InvertCDFTimeSeries,
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
parser.add_argument("--mixture-device-ids",
    help="The device ids to use for the brute force mixture preprocessing search routine. Required if the '--preprocessing-method' argument is set to 'mixed'",
    nargs='+',
    type=int,
    required=False,
    dest='mixture_device_ids',
)
parser.add_argument("--dataset",
    help="The dataset to use for the experiment",
    choices=['amex', 'lob', 'hpc'],
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
    choices=list(static_preprocessing_methods.keys()) + ["mixed"],
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
    help="The number of cross-validation folds to use. If using the LOB dataset, this decides the number of splits to use for the LOB data.",
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
parser.add_argument("--override",
    help="String with commands for overriding the experiment config",
    type=str,
    required=False,
    default=None,
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
    print(f"Using random seed = {args.random_state}")
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    ################## Parse override commands #####################

    if args.override is not None:
        # commands are separated by spaces
        commands = args.override.split(" ")

        for cmd_str in commands:
            # key(s) and values are separated by colons
            keys = cmd_str.split(":")[:-1]
            val = cmd_str.split(":")[-1]

            # find the value in the dictionary recursively
            d = exp_cfg
            for k in keys[:-1]:
                d = d[k]
            # then update the value after casting it to the appropriate type
            d[keys[-1]] = type(d[keys[-1]])(val)

    print(f"Start experiment '{args.experiment_name}'.") 

    ################################################################
    ###            Part 1: Load the dataset                      ###
    ################################################################

    if args.dataset == 'amex':
        X, y = load_amex_numpy_data(
            split_data_dir=os.path.join(main_cfg['dataset_directory'], 'derived', 'processed-splits'),
            fill_dict=exp_cfg['amex_dataset']['fill'],
            corrupt_func=lambda X, y: undo_min_max_corrupt_func(X, y, args.random_state),
            num_categorical_features=exp_cfg['amex_dataset']['num_categorical_features']
        )
        print(f"Finished loading dataset '{args.dataset}' with covariates of shape {X.shape} and responses of shape {y.shape}")
    elif args.dataset == 'hpc':
        X, y = load_power_numpy_data()
        print(f"Finished loading dataset '{args.dataset}' with covariates of shape {X.shape} and responses of shape {y.shape}")
    elif args.dataset == 'lob':
        X, y = None, None
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")

    ################################################################
    ###            Part 2: Setup the model initialisation        ###
    ################################################################

    input_dim = exp_cfg[f"{args.dataset}_dataset"]['numeric_input_dim']
    time_series_length = exp_cfg[f"{args.dataset}_dataset"]['time_series_length']

    if args.model == 'gru-rnn' and args.adaptive_layer is None:
        if args.dataset == 'amex' or args.dataset == 'hpc':
            model_init_fn = lambda : GRUNetBasic(
                num_cat_columns=exp_cfg['amex_dataset']['num_categorical_features'],
                **exp_cfg['gru_model_amex']
            )
        elif args.dataset == 'lob':
            model_init_fn = lambda : GRUNetLOB(
                **exp_cfg['gru_model_lob']
            )
        else:
            raise ValueError(f"Dataset not supported: {args.dataset}")
    elif args.model == 'gru-rnn' and args.adaptive_layer is not None:

        ####################### DAIN layer #########################
        if args.adaptive_layer == 'dain':
            dain_layer_kwargs = exp_cfg['dain']
            dain_layer_kwargs['input_dim'] = input_dim
            adaptive_layer_init_fn = lambda : DAIN_Layer(**dain_layer_kwargs)
            adaptive_layer_optim_args = {
                'base_lr' : exp_cfg['fit']['base_lr']
            }
            dim_first = True
        ######################## BiN layer #########################
        elif args.adaptive_layer == 'bin':
            adaptive_layer_init_fn = lambda : BiN_Layer(
                input_shape=(input_dim, time_series_length),
            )
            dim_first = True
            adaptive_layer_optim_args = exp_cfg['bin']
            adaptive_layer_optim_args['base_lr'] = exp_cfg['fit']['base_lr']
        ######################## EDAIN layer #######################
        elif args.adaptive_layer == 'edain':
            adaptive_layer_init_fn = lambda : EDAIN_Layer(
                input_dim=input_dim,
                # when using it as adaptive layer, we do a forward pass, not inverse
                invert_bijector=False,
                outlier_removal_residual_connection=True,
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
        if args.dataset == 'amex' or args.dataset == 'hpc':
            model_init_fn = lambda : AdaptiveGRUNet(
                adaptive_layer=adaptive_layer_init_fn(),
                num_cat_columns=exp_cfg['amex_dataset']['num_categorical_features'],
                time_series_length=time_series_length,
                dim_first=dim_first,
                **exp_cfg['gru_model_amex'],
            )
        elif args.dataset == 'lob':
            model_init_fn = lambda : AdaptiveGRUNetLOB(
                adaptive_layer=adaptive_layer_init_fn(),
                time_series_length=time_series_length,
                dim_first=dim_first,
                **exp_cfg['gru_model_lob'],
            )
        else:
            raise ValueError(f"Dataset not supported: {args.dataset}")

        # set up optimizer
        base_lr = exp_cfg['fit']['base_lr']
        if exp_cfg['fit']['optimizer'] == 'adam':
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
        elif exp_cfg['fit']['optimizer'] == 'rmsprop':
            optimizer_init_fn = lambda mod: torch.optim.RMSprop(
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
            raise ValueError(f"Optimizer not supported: {exp_cfg['fit']['optimizer']}")
    else:
        raise ValueError(f"Model not supported: {args.model}")
    print(f"Finished loading model")


    ################################################################
    ###        Part 3: Setup optimizer related utilities         ###
    ################################################################

    # set up optimizer function (some models have their own optimizers)
    if optimizer_init_fn is None:
        if exp_cfg['fit']['optimizer'] == 'adam':
            optimizer_init_fn = lambda mod : torch.optim.Adam(
                mod.parameters(),
                lr=exp_cfg['fit']['base_lr'],
            )
        elif exp_cfg['fit']['optimizer'] == 'rmsprop':
            optimizer_init_fn = lambda mod : torch.optim.RMSprop(
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

    ################################################################
    ###            Part 4: Setup preprocessing methods           ###
    ################################################################

    if args.ignore_time:
        original_time_length = time_series_length
        time_series_length = 1

    msg = args.preprocessing_method
    if args.preprocessing_method == 'mixed':
        assert args.mixture_device_ids is not None and "Device IDs for mixture not provided"
        assert args.dataset == 'amex' and "Mixture only supported for AMEX dataset"
        assert not args.ignore_time and "Mixture not supported for time-agnostic models"
        # start the brute force search
        transform_list = find_optimal_preprocessing_mixture_with_brute_force(
            args.experiment_name,
            args.mixture_device_ids,
            # the following are all the job kwargs required by `run_parallel_mixture_jobs`
            model_init_fn=model_init_fn,
            optimizer_init_fn=optimizer_init_fn,
            scheduler_init_fn=scheduler_init_fn,
            early_stopper_init_fn=early_stopper_init_fn,
            X=X,
            y=y,
            exp_cfg=exp_cfg,
            random_state=args.random_state,
        )

        # setup the mixed scaler based on result
        _scaler_init_fn_1 = lambda : MixedTransformsTimeSeries(
            transform_list,
            time_series_length=time_series_length,
        )
    else:
        _scaler_init_fn_1 = lambda : static_preprocessing_methods[args.preprocessing_method](time_series_length=time_series_length)
    # TODO: add optional winsorization

    # use the above scaler function to decorate our EDAIN-KL scaler
    if args.edain_kl:
        fit_kwargs = exp_cfg['edain_bijector_fit']
        fit_kwargs['device'] = dev
        _scaler_init_fn_2 = lambda : EDAINScalerTimeSeriesDecorator(
            scaler=_scaler_init_fn_1(),
            time_series_length=time_series_length,
            input_dim=input_dim,
            bijector_kwargs=exp_cfg['edain_bijector'],
            bijector_fit_kwargs=fit_kwargs,
        )
        msg = f"EDAIN-KL({msg})"
    else:
        _scaler_init_fn_2 = _scaler_init_fn_1

    # check if scaler should be decorated with time-flatten scaler
    if args.ignore_time:
        scaler_init_fn = lambda : IgnoreTimeDecorator(
            scaler=_scaler_init_fn_2(),
            time_series_length=original_time_length,
        )
        msg = f"IgnoreTime({msg})"
    else:
        scaler_init_fn = _scaler_init_fn_2
    print(f"Finished setting up preprocessing technique: {msg}")


    ################################################################
    ###            Part 5: Run the experiment                    ###
    ################################################################

    print(f"Starting cross-validation experiment on dataset {args.dataset}")
    if args.dataset == "amex" or args.dataset == "hpc":
        history = cross_validate_experiment(
            model_init_fn=model_init_fn,
            preprocess_init_fn=scaler_init_fn,
            optimizer_init_fn=optimizer_init_fn,
            scheduler_init_fn=scheduler_init_fn,
            early_stopper_init_fn=early_stopper_init_fn,
            loss_fn=F.binary_cross_entropy,
            X=X,
            y=y,
            num_epochs=exp_cfg['fit']['num_epochs'],
            dataloader_kwargs=exp_cfg['amex_dataset']['data_loader'],
            num_folds=args.num_cross_validation_folds,
            device=dev,
            random_state=args.random_state,
            num_categorical_features=exp_cfg['amex_dataset']['num_categorical_features'],
        )
    elif args.dataset == "lob":
        history = train_evaluate_lob_anchored(
            h5_file_path=exp_cfg['lob_dataset']['preprocessed_lob_path'],
            model_init_fn=model_init_fn,
            preprocess_init_fn=scaler_init_fn,
            optimizer_init_fn=optimizer_init_fn,
            scheduler_init_fn=scheduler_init_fn,
            early_stopper_init_fn=early_stopper_init_fn,
            num_epochs=exp_cfg['fit']['num_epochs'],
            device=dev,
            random_state=args.random_state,
            horizon=exp_cfg['lob_dataset']['horizon'],
            windows=exp_cfg['lob_dataset']['time_series_length'],
            batch_size=exp_cfg['lob_dataset']['batch_size'],
            use_resampling=exp_cfg['lob_dataset']['use_resampling'],
            splits=exp_cfg['lob_dataset']['splits'][:min(args.num_cross_validation_folds, 9)],
        )
    else:
        raise NotImplementedError(f"No experiment utilties for dataset: {args.dataset}")

    ################################################################
    ###            Part 6: Save experiment results               ###
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
