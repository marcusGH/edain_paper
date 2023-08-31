# Automated data preprocessing for deep neural networks

Introduction coming soon!

The report/thesis can be found [here](reports/thesis/build/main.pdf).

<!-- vim-markdown-toc GFM -->

* [Main contributions](#main-contributions)
  * [Extended Deep Adaptive Input Normalization (EDAIN) preprocessing layer](#extended-deep-adaptive-input-normalization-edain-preprocessing-layer)
* [Repository overview](#repository-overview)
* [Reproducing the results](#reproducing-the-results)
  * [Setting up the Amex dataset](#setting-up-the-amex-dataset)
  * [Setting up the FI-2010 LOB dataset](#setting-up-the-fi-2010-lob-dataset)
* [Examples](#examples)
  * [Using the EDAIN preprocessing layer](#using-the-edain-preprocessing-layer)
  * [Generating synthetic multivariate time-series data](#generating-synthetic-multivariate-time-series-data)
* [Known issues](#known-issues)

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

Instructions coming soon!

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
    ```

3. Run the following to split the datasets into suitable partitions
   
   ```python3
   from src.lib import initial_preprocessing
   
   test_path = os.path.join(DATA_DIR, 'derived', 'test.parquet')
   # run the other utility function to compress csv to feather first
   labels_path = os.path.join(DATA_DIR, 'derived', 'train_labels.feather')
   train_path = os.path.join(DATA_DIR, 'derived', 'train.parquet')
   save_path = os.path.join(DATA_DIR, 'derived', 'processed-splits')

   initial_preprocessing.split_raddars_parquet(test_path, save_path, num_splits=20)   
   initial_preprocessing.split_raddars_parquet(train_path, save_path, num_splits=10)   
   ```

## Setting up the FI-2010 LOB dataset

Instructions coming soon!

# Examples

## Using the EDAIN preprocessing layer

Coming soon!

## Generating synthetic multivariate time-series data

Coming soon!

# Known issues

Gradients in power transformation layer might becomes `NaN`s. TODO: elaborate

This might also happen if not using standard scaler as preprocessing method
before applying EDAIN, as gradients might explode for this case as well!
