# Please run 0-1-2-3 files before running this one + put raw data to ../data/raw

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def label_encode_target(df, _inplace = True):
    df.replace('unrelated', 0, inplace=_inplace)
    df.replace('agreed'   , 1, inplace=_inplace)
    df.replace('disagreed', 2, inplace=_inplace)

data_folder = os.path.dirname(os.getcwd())+'/data'
processed_folder = f'{data_folder}/processed'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
    
train = pd.read_csv(data_folder+'/raw/train.csv')
test = pd.read_csv(data_folder+'/raw/test.csv')
label_encode_target(train)

# Process train embeddings
enc_train1 = pd.read_csv('encoded_train1.csv', index_col=0)
enc_train2 = pd.read_csv('encoded_train2.csv', index_col=0)
enc_train3 = pd.read_csv('encoded_ch_train1.csv', index_col=0)
enc_train4 = pd.read_csv('encoded_ch_train2.csv', index_col=0)
X_train = enc_train1.merge(enc_train2, how='left', on='id')
X_train = X_train.merge(enc_train3, how='left', on='id')
X_train = X_train.merge(enc_train4, how='left', on='id')
X_train = X_train.merge(train[['id', 'label']], how='left', on='id')

float_cols = [c for c in X_train.columns if X_train.loc[:, c].dtype=='float64']
for col in tqdm(float_cols):
    X_train.loc[:, col] = X_train.loc[:, col].astype(np.float32)
int_cols = [c for c in X_train.columns if X_train.loc[:, c].dtype=='int64']
for col in int_cols:
    X_train.loc[:, col] = X_train.loc[:, col].astype(np.int32)
X_train.to_pickle(processed_folder+'/train')
X_train.to_csv(processed_folder+'/train.csv')

# Process test embeddings
enc_test1 = pd.read_csv('encoded_test1.csv', index_col=0)
enc_test2 = pd.read_csv('encoded_test2.csv', index_col=0)
enc_test3 = pd.read_csv('encoded_ch_test1.csv', index_col=0)
enc_test4 = pd.read_csv('encoded_ch_test2.csv', index_col=0)
X_test = enc_test1.merge(enc_test2, how='left', on='id')
X_test = X_test.merge(enc_test3, how='left', on='id')
X_test = X_test.merge(enc_test4, how='left', on='id')

float_cols = [c for c in X_test.columns if X_test.loc[:, c].dtype=='float64']
for col in tqdm(float_cols):
    X_test.loc[:, col] = X_test.loc[:, col].astype(np.float32)
X_test.to_pickle(processed_folder+'/test')
X_test.to_csv(processed_folder+'/test.csv')