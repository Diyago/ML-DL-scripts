# Please run bert-serving-start before running this notebook
# Setup: https://github.com/hanxiao/bert-as-service
# Examples (change folders to your locals)
# english cased: bert-serving-start -model_dir /bert-as-service/cased_L-24_H-1024_A-16/ -num_worker=4
# multi cased: bert-serving-start -model_dir /bert-as-service/multi_cased_L-12_H-768_A-12/ -num_worker=4
# chinese: bert-serving-start -model_dir /bert-as-service/chinese_L-12_H-768_A-12/ -num_worker=4

# launch bert (valilenk):
# english cased: bert-serving-start -model_dir /media/airvetra/1tb/valilenk/nlp/bert-as-service/cased_L-24_H-1024_A-16/ -num_worker=2
# multi cased: bert-serving-start -model_dir /media/airvetra/1tb/valilenk/nlp/bert-as-service/multi_cased_L-12_H-768_A-12/ -num_worker=2
# chinese: bert-serving-start -model_dir /media/airvetra/1tb/valilenk/nlp/bert-as-service/chinese_L-12_H-768_A-12/ -num_worker=2

import pandas as pd
import torch
import os
from time import time
from tqdm import tqdm
from bert_serving.client import BertClient

data_folder = os.path.dirname(os.getcwd()) + "/data"
test = pd.read_csv(data_folder + "/raw/test.csv")

bc = BertClient()


def gen_encodings(df, column):
    t0 = time()
    _list = list(df.loc[:, column])
    for i, text in enumerate(_list):
        if not isinstance(_list[i], str):
            _list[i] = str(text)
        if not _list[i].strip():
            _list[i] = _list[i].strip()
        if len(_list[i]) == 0:
            _list[i] = "temp"
    arr = bc.encode(_list)
    temp = pd.DataFrame(arr)
    temp.columns = [f"{column}_{c}" for c in range(len(arr[0]))]
    temp = temp.join(df.id)
    print(f"time: {time() - t0}")
    return temp


encoded_test = gen_encodings(test, "title1_zh")
encoded_test.to_csv("encoded_ch_test1.csv")
encoded_test = gen_encodings(test, "title2_zh")
encoded_test.to_csv("encoded_ch_test2.csv")
