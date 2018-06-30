#!/usr/bin/python2
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import six
from chainer import cuda
from chainer import serializers
from sklearn.feature_extraction.image import extract_patches
from tqdm import tqdm

from deepIQA.fr_model import FRModel
from deepIQA.nr_model import Model

top='models/nr_live_weighted.model'

model = Model(top=top)

cuda.cudnn_enabled = True
cuda.check_cuda_available()
xp = cuda.cupy
serializers.load_hdf5(top, model)
model.to_gpu()

images_path = '../../test_jpg/'
# images_path_test = '../input/test_jpg/'
names = []
extracted_features = []

file_path = '../input/deepIQA_features_test.csv'
os.mknod(file_path)

train_ids = next(os.walk(images_path))[2]
f = True
for name in tqdm(train_ids):
    try:
        img = cv2.imread(images_path + name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = extract_patches(img, (32, 32, 3), 32)
        X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

        y = []
        weights = []
        batchsize = min(2000, X.shape[0])
        t = xp.zeros((1, 1), np.float32)
        for i in six.moves.range(0, X.shape[0], batchsize):
            X_batch = X[i:i + batchsize]
            X_batch = xp.array(X_batch.astype(np.float32))

            model.forward(X_batch, t, False, X_batch.shape[0])

            y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
            weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))

        y = np.concatenate(y)
        weights = np.concatenate(weights)
        names.append(name[:-4])

        v = np.sum(y * weights) / np.sum(weights)
        extracted_features.append(v)

        if len(names) >= 10000:
            df = pd.DataFrame(extracted_features)
            se = pd.Series(names)
            df['ids'] = se.values  # df.set_index('id', inplace=True)
            if f:
                df.to_csv(file_path, mode='a', index_label=False, index=False, chunksize=len(names))
                f = False
            else:
                df.to_csv(file_path, mode='a', index_label=False, index=False, chunksize=len(names), header=False)
            names = []
            extracted_features = []
    except:
        print(name)

if len(names) > 0:
    df = pd.DataFrame(extracted_features)
    se = pd.Series(names)
    df['ids'] = se.values  # df.set_index('id', inplace=True)
    if f:
        df.to_csv(file_path, mode='a', index_label=False, index=False, chunksize=len(names))
        f = False
    else:
        df.to_csv(file_path, mode='a', index_label=False, index=False, chunksize=len(names), header=False)

    '''
    --model
    models/nr_tid_patchwise.model
    --top
    patchwise
    /home/alex/work/py/avito/input/train_jpg/0a0a5a3f22320e0508139273d23f390ca837aef252036034ed640fb939529bd9.jpg
    '''
