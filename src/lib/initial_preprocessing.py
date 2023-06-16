import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import gc
import cupy
import cudf

def compress_csvs_to_feather(dataset_dir):
    for f in ['train_data', 'test_data', 'train_labels']:
        print(f"Converting {f}")
        df = pd.read_csv(os.path.join(dataset_dir, "raw", f"{f}.csv"))
        df.to_feather(os.path.join(dataset_dir, "derived", f"{f}.feather"))
        del df

def split_raddars_parquet(data_path, save_path, label_path=None, pad_customers_to_13_rows=True,
                          num_splits=10):
    # df = cudf.read_parquet(os.path.join(DATA_DIR, "derived", "train.parquet"))
    df = cudf.read_parquet(data_path)
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')

    # LOAD TARGETS
    if label_path is not None:
        # targets = cudf.read_feather(os.path.join(DATA_DIR, 'derived', 'train_labels.feather'))
        targets = cudf.read_feather(label_path)
        targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
        print(f'There are {targets.shape[0]} train targets')
    else:
        targets = None

    # GET TRAIN COLUMN NAMES
    # train = cudf.read_csv(os.path.join(DATA_DIR, 'raw', 'train_data.csv'), nrows=1)
    # T_COLS = train.columns
    # print(f'There are {len(T_COLS)} train dataframe columns')

    customers = df.customer_ID.unique().values.flatten()
    print(f'There are {len(customers)} unique customers in train.')

    # extract the Y, M and D from the date column, then sort by time (after customer_ID)
    df.S_2 = cudf.to_datetime(df.S_2)
    df['year'] = (df.S_2.dt.year-2000).astype('int8')
    df['month'] = (df.S_2.dt.month).astype('int8')
    df['day'] = (df.S_2.dt.day).astype('int8')
    del df['S_2']


    CATS = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68'] + ['D_63','D_64']

    if pad_customers_to_13_rows:
        tmp = df[['customer_ID']].groupby('customer_ID').customer_ID.agg('count')
        more = cupy.array([], dtype='int64') 
        for j in range(1, 13):
            i = tmp.loc[tmp == j].index.values
            more = cupy.concatenate([more, cupy.repeat(i, 13-j)])
        df_pad = df.iloc[:len(more)].copy().fillna(0)
        df_pad = df_pad * 0 - 3 #pad numerical columns with -3
        df_pad[CATS] = (df_pad[CATS] * 0 - 2).astype('int8') #pad categorical columns with -2
        df_pad['customer_ID'] = more
        df = cudf.concat([df, df_pad], axis = 0, ignore_index=True)

        del tmp, df_pad
        gc.collect()

    if targets is not None:
        df = df.merge(targets, on='customer_ID', how='left')
        df.target = df.target.astype('int8')

    df = df.sort_values(['customer_ID', 'year', 'month', 'day']).reset_index(drop = True)
    df = df.drop(['year', 'month', 'day'], axis=1)

    COLS = list(df.columns[1:])
    COLS = ['customer_ID'] + CATS + [c for c in COLS if c not in CATS]
    df = df[COLS]

    df = df.fillna(-0.5)

    for i in range(num_splits):
        lower = (len(customers) // num_splits) * i
        upper = (len(customers) // num_splits) * (i + 1)
        if i == num_splits - 1:
            upper = len(customers)

        sub_df = df[df.customer_ID.isin(customers[lower:upper])]
        print(f"[{i + 1} / {num_splits}] Saving data with {len(sub_df)} rows...")

        if targets is not None:
            sub_targets = sub_df[['customer_ID', 'target']].drop_duplicates().sort_index()
            # sub_targets.to_parquet(os.path.join(DATA_DIR, 'derived', 'processed-splits', f"train-targets_{i}.parquet"))
            sub_targets.to_parquet(os.path.join(save_path, f"train-targets_{i}.parquet"))

        if targets is not None:
            # remove the customer ID and the target column 190 -> 188
            sub_data = sub_df.iloc[:, 1:-1].values.reshape((-1, 13, 188))
            # cupy.save(os.path.join(DATA_DIR, "derived", "processed-splits", f"train-data_{i}.npy"), sub_data.astype('float32'))
            cupy.save(os.path.join(save_path, f"train-data_{i}.npy"), sub_data.astype('float32'))
        else:
            # remove the customer ID column 189 -> 188
            sub_data = sub_df.iloc[:, 1:].values.reshape((-1, 13, 188))
            # cupy.save(os.path.join(DATA_DIR, "derived", "processed-splits", f"train-data_{i}.npy"), sub_data.astype('float32'))
            cupy.save(os.path.join(save_path, f"test-data_{i}.npy"), sub_data.astype('float32'))

        del sub_df, sub_data
        if targets is not None:
            del sub_targets
        gc.collect()

    # clean up
    del df
    gc.collect()
