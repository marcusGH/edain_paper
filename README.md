# Extended Deep Adaptive Input Normalization for Preprocessing Time Series Data for Neural Networks

Data preprocessing is a crucial part of any machine learning pipeline, and it
can have a significant impact on both performance and training efficiency.
This is especially evident when using deep neural networks for time series
prediction and classification: real-world time series data often exhibit
irregularities such as multi-modality, skewness and outliers, and the model
performance can degrade rapidly if these characteristics are not adequately
addressed.  In our work, we propose the EDAIN (Extended Deep Adaptive Input
Normalization) layer, a novel adaptive neural layer that learns how to
appropriately normalize irregular time series data for a given task in an
end-to-end fashion, instead of using a fixed normalization scheme.  This is
achieved by optimizing its unknown parameters simultaneously with the deep
neural network using back-propagation.  Our experiments, conducted using
synthetic data, a credit default prediction dataset, and a large-scale limit
order book benchmark dataset, demonstrate the superior performance of the EDAIN
layer when compared to conventional normalization methods and existing adaptive
time series preprocessing layers.

In this repository we provide an implementation of the Extended Deep Adaptive
Input Normalization (EDAIN) layer using PyTorch. We also provide all the
necessary code and instructions for reproducing the results in our paper.
The associated thesis can be found [here](reports/thesis/build/main.pdf) and
our paper is available at TODO.

<!-- vim-markdown-toc GFM -->

* [Main contributions](#main-contributions)
  * [Extended Deep Adaptive Input Normalization (EDAIN) preprocessing layer](#extended-deep-adaptive-input-normalization-edain-preprocessing-layer)
* [Repository overview](#repository-overview)
* [Reproducing the results](#reproducing-the-results)
  * [Setting up the Amex dataset](#setting-up-the-amex-dataset)
  * [Setting up the FI-2010 LOB dataset](#setting-up-the-fi-2010-lob-dataset)
  * [Running the experiments](#running-the-experiments)
    * [Default prediction dataset experiments](#default-prediction-dataset-experiments)
    * [Financial forecasting dataset (FI-2010) experiments](#financial-forecasting-dataset-fi-2010-experiments)
    * [Synthetic data experiments](#synthetic-data-experiments)
  * [Reproducing the tables and plots](#reproducing-the-tables-and-plots)
* [Examples](#examples)
  * [Using the EDAIN preprocessing layer](#using-the-edain-preprocessing-layer)
  * [Generating synthetic multivariate time-series data](#generating-synthetic-multivariate-time-series-data)
* [Known issues](#known-issues)
* [License information](#license-information)

<!-- vim-markdown-toc -->

# Main contributions

## Extended Deep Adaptive Input Normalization (EDAIN) preprocessing layer

![EDAIN-architecture-diagram](https://github.com/marcusGH/automated-preprocessing-for-deep-neural-networks/assets/29378769/2646f250-6b96-4b77-8fa2-91a87f0e67c7)

This adaptive preprocessing layer can be attached in front of any deep sequence model
and its unknown parameters can be trained in an end-to-end fashion while the sequence
model is being trained with backpropagation. It has two modes, _local-aware_ and
_global-aware_, where the former is most suitable for highly multimodal data and the
latter is most suitable for data all originating from the same data generation mechanism.

# Repository overview

Coming soon!

# Reproducing the results

First, make sure all the python libraries listed in `requirements.txt` are installed.
Note that the `cudf` and `cupy` are only needed to setup the American Express default
prediction dataset.

## Setting up the Amex dataset

1. Download raddar's deanonymized dataset as well as the raw dataset:

   ```
   kaggle datasets download -d raddar/amex-data-integer-dtypes-parquet-format
   kaggle competitions download -c amex-default-prediction
   ```
2. The dataset directory should look like:

   ```
   data_dir/
     * raw/
       * train_labels.csv
       * train_data.csv
       * test_data.csv
       * sample_submission.csv
     * derived/
       * train.parquet
       * test.parquet
       * processed-splits/
    ```

3. Run the following to split the datasets into suitable partitions

   ```python3
   from src.lib import initial_preprocessing

   initial_preprocessing.compress_csvs_to_feather(DATA_DIR)

   test_path = os.path.join(DATA_DIR, 'derived', 'test.parquet')
   train_path = os.path.join(DATA_DIR, 'derived', 'train.parquet')
   save_path = os.path.join(DATA_DIR, 'derived', 'processed-splits')

   initial_preprocessing.split_raddars_parquet(test_path, save_path, num_splits=20)
   initial_preprocessing.split_raddars_parquet(train_path, save_path, num_splits=10)
   ```
4. Change the `train_split_data_dir` key in the experiment config (`src/experiments/configs/`) to point to the directory with the processed splits generated in step 3

## Setting up the FI-2010 LOB dataset

1. Download the preprocessed data from [here](https://www.dropbox.com/s/vvvqwfejyertr4q/lob.tar.xz?dl=0) (courtesy of [passalis](https://github.com/passalis)'s [DAIN repo](https://github.com/passalis/dain))
2. Change  the `preprocessed_lob_path` in the experiment config (`src/experiments/configs/`) to point to the `lob.h5` file downloaded

## Running the experiments

Make sure `config.yaml` has been updated to match your filesystem, and add the project root to
your python path with `export PYTHONPATH=$(pwd)`.

### Default prediction dataset experiments

To generate the history files for the different preprocessing methods on the American Express default prediction, run the following commands from the project root:
* **No preprocessing**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --preprocessing_method=identity --num_cross_validation_folds=5 --random_state=42 --experiment_name=no-preprocess-amex-RECENT`
* **$z$-score scaling**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --preprocessing_method=standard-scaler --ignore_time=True --num_cross_validation_folds=5 --random_state=42 --experiment_name=standard-scaling-no-time-1`
* **CDF inversion**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --preprocessing_method=cdf-invert --num_cross_validation_folds=5 --random_state=42 --experiment_name=cdf-inversion-amex`
* **EDAIN-KL**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=2 --dataset=amex --model=gru-rnn --preprocessing_method=min-max --edain_kl=True --num_cross_validation_folds=5 --random_state=42 --experiment_name=amex-edain-kl-preprocessing-1 --override='edain_bijector_fit:scale_lr:10.0 edain_bijector_fit:shift_lr:10.0 edain_bijector_fit:outlier_lr:100.0 edain_bijector_fit:power_lr:0.0000001'`
* **EDAIN (global-aware)**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --adaptive_layer=edain --preprocessing_method=standard-scaler --num_cross_validation_folds=5 --random_state=42 --experiment_name=edain-preprocessing-1`
* **EDAIN (local-aware)**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --adaptive_layer=edain --preprocessing_method=standard-scaler --num_cross_validation_folds=5 --random_state=42 --experiment_name=edain-local-aware-amex-RECENT --override='edain_bijector_fit:scale_lr:1.0 edain_bijector_fit:shift_lr:1.0 edain_bijector_fit:outlier_lr:10.0 edain_bijector_fit:power_lr:10.0'`
* **DAIN**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=1 --dataset=amex --model=gru-rnn --adaptive_layer=dain --preprocessing_method=standard-scaler --num_cross_validation_folds=5 --random_state=42 --experiment_name=amex-dain-preprocessing-1`
* **BIN**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-alpha.yaml --device=0 --dataset=amex --model=gru-rnn --adaptive_layer=bin --preprocessing_method=standard-scaler --num_cross_validation_folds=5 --random_state=42 --experiment_name=amex-bin-preprocessing-1`

### Financial forecasting dataset (FI-2010) experiments

To generate the history files for the different preprocessing methods on the American Express default prediction, run the following commands from the project root:
* **No preprocessing**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=5 --dataset=lob --model=gru-rnn --preprocessing_method=identity --num_cross_validation_folds=99 --random_state=42 --experiment_name=no-preprocess-lob-RECENT`
* **$z$-score scaling**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --preprocessing_method=standard-scaler --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-standard-scaling-experiment-final`
* **CDF inversion**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=6 --dataset=lob --model=gru-rnn --preprocessing_method=cdf-invert --num_cross_validation_folds=99 --random_state=42 --experiment_name=cdf-inversion-lob-v2`
* **BIN**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --adaptive_layer=bin --preprocessing_method=identity --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-BIN-experiment-final`
* **DAIN**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --adaptive_layer=dain --preprocessing_method=identity --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-DAIN-experiment-final`
* **EDAIN (local-aware)**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --adaptive_layer=edain --preprocessing_method=identity --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-EDAIN-experiment-final-v1` (after changing the $\beta$ config to `batch_aware=True` and using the learning rate modifiers for local-aware)
* **EDAIN (global-aware)**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --adaptive_layer=edain --preprocessing_method=standard-scaler --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-EDAIN-global-experiment-final-v1`
* **EDAIN-KL**: `python3 src/experiments/run_experiments.py --experiment_config=src/experiments/configs/experiment-config-beta.yaml --device=0 --dataset=lob --model=gru-rnn --preprocessing_method=standard-scaler --edain_kl=True --num_cross_validation_folds=100 --random_state=42 --experiment_name=LOB-EDAIN-KL-experiment-final-v1`

### Synthetic data experiments

1. Run the script `python3 src/experiments/misc/synthetic_data_performance_compare.py`

## Reproducing the tables and plots

* To reproduce the plot in Figure 4, run `python3 scripts/plots/amex_performance_convergence.py`.
* To reproduce the latex code for Table 2, run `python3 scripts/plots/amex_performance_convergence.py`.
* To reproduce the latex code for Table 1, run `python3 scripts/plots/synthetic_data_table.py`
* To reproduce the latex code for Table 3, run `python3 scripts/plots/lob_performance_table.py`.

# Examples

## Using the EDAIN preprocessing layer

Coming soon!

## Generating synthetic multivariate time-series data

Coming soon!

# Known issues

Gradients in power transformation layer might becomes `NaN`s. TODO: elaborate

This might also happen if not using standard scaler as preprocessing method
before applying EDAIN, as gradients might explode for this case as well!

# License information

In our work, we used the following assets:
* The [`dain`](https://github.com/passalis/dain/) preprocessing layer developed by [Passalis et al. (2019)](https://arxiv.org/pdf/1902.07892.pdf). No license information is specified in their repository
* The [`bin`](https://github.com/viebboy/mlproject/blob/main/mlproject/models/tabl.py#L106) preprocessing layer developed by [Tran et al. (2020)](https://ieeexplore.ieee.org/document/9412547). Their implementation is publicly released under the Apache License 2.0.
* The benchmark dataset for mid-price forecasting of limit order books (FI-2010), available [here](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649). This dataset is released under the creative commons attribution 4.0.
* The American Express default prediction dataset used as part of [this Kaggle competition](https://www.kaggle.com/competitions/amex-default-prediction). We have received permission to use this dataset in our research.
