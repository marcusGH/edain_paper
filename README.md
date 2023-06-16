# Automating data preprocessing for sequential neural network models (tentative)

## Setting up the data

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

