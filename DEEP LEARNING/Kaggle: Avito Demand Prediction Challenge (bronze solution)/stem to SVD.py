#Thanks for the approach https://github.com/ML-Person/My-solution-to-Avito-Challenge-2018 (@nikita)
import pandas as pd
import numpy as np
import gc
import os
import re
import pickle
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb

# for text data
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
pd.set_option('max_columns', 84)
import warnings
warnings.filterwarnings('ignore')

PATH_TO_DATA = '/Avito'



traintrain  ==  pdpd..read_csvread_cs (os.path.join(PATH_TO_DATA, 'train.csv'))
test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv'))

'''
item_id - Ad id.
user_id - User id.
region - Ad region.
city - Ad city.
parent_category_name - Top level ad category as classified by Avito's ad model.
category_name - Fine grain ad category as classified by Avito's ad model.
param_1 - Optional parameter from Avito's ad model.
param_2 - Optional parameter from Avito's ad model.
param_3 - Optional parameter from Avito's ad model.
title - Ad title.
description - Ad description.
price - Ad price.
item_seq_number - Ad sequential number for user.
activation_date - Date ad was placed.
user_type - User type.
image - Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.
image_top_1 - Avito's classification code for the image.
deal_probability - The target variable. This is the likelihood that an ad actually sold something. It's not possible to verify every transaction with certainty, so this column's value can be any float from zero to one.
'''
categorical = [
    'image_top_1', 'param_1', 'param_2', 'param_3', 
    'city', 'region', 'category_name', 'parent_category_name', 'user_type'
]

# easy preprocessing
text_cols = [
    'title', 'description', 'param_1', 'param_2', 'param_3',
    'city', 'region', 'category_name', 'parent_category_name'
]
for col in text_cols:
    for df in [train, test]:
        df[col] = df[col].str.replace(r"[^А-Яа-яA-Za-z0-9,!?@\'\`\"\_\n]", ' ')
        df[col].fillna("NA", inplace=True)
        df[col] = df[col].str.lower()


forfor  dfdf  inin  [[traintrain,,  testtest]:]:
         dfdf[['len_description''len_de ] = df['description'].apply(lambda x: len(str(x)))
    df['num_desc_punct'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation])) / df['len_description']
    
    for col in ['description', 'title']:
        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))

    # percentage of unique words
    df['words_vs_unique_title'] = df['num_unique_words_title'] / df['num_words_title'] * 100
    df['words_vs_unique_description'] = df['num_unique_words_description'] / df['num_words_description'] * 100
 

# [DUMP] TRAIN + TEST# [DUMP] 
train.to_csv(os.path.join(PATH_TO_DATA, 'train_all_features.csv'), index=False, encoding='utf-8')
test.to_csv(os.path.join(PATH_TO_DATA, 'test_all_features.csv'), index=False, encoding='utf-8')

del train, test
gc.collect()

train = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv'))
test = pd.read_csv(os.path.join(PATH_TO_DATA, 'test.csv


stemmer = SnowballStemmer("russian",  ignore_stopwords=False)

train['title_stemm'] = train['title'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))
test['title_stemm'] = test['title'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))
train['description_stemm'] = train['description'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))
test['description_stemm'] = test['description'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))


train['text'] = train['param_1'] + " " + train['param_2'] + " " + train['param_3'] + " " + \
                train['city'] + " " + train['category_name'] + " " + train['parent_category_name']
test['text'] =  test['param_1'] + " " + test['param_2'] + " " + test['param_3'] + " " + \
                test['city'] + " " + test['category_name'] + " " + test['parent_category_name']

train['text_stemm'] = train['text'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))
test['text_stemm'] = test['text'].apply(lambda string: ' '.join([stemmer.stem(w) for w in string.split()]))


for df in [train, test]:
    df.drop(['title', 'description', 'text'], axis=1, inplace=True)


#TF-IDF + SVD 

# CountVectorizer for 'title'
title_tfidf = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True,
                              token_pattern=r'\w{1,}', ngram_range=(1, 1))

full_tfidf = title_tfidf.fit_transform(train['title_stemm'].values.tolist() + test['title_stemm'].values.tolist())
train_title_tfidf = title_tfidf.transform(train['title_stemm'].values.tolist())
test_title_tfidf = title_tfidf.transform(test['title_stemm'].values.tolist())

### SVD Components ###
n_comp = 10
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_title_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_title_tfidf))
train_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_title_'+str(i+1) for i in range(n_comp)]
train_svd['item_id'] = train['item_id']
test_svd['item_id'] = test['item_id']

# Merge and delete
train = train.merge(train_svd, on='item_id', how='left')
test = test.merge(test_svd, on='item_id', how='left')
del full_tfidf, train_svd, test_svd
gc.collect()


# TF-IDF for 'description'
desc_tfidf = TfidfVectorizer(stop_words=stopwords.words('russian'), token_pattern=r'\w{1,}',
                             lowercase=True, ngram_range=(1, 2),  norm='l2', smooth_idf=False,
                             max_features=17000)
full_tfidf = desc_tfidf.fit_transform(train['description_stemm'].values.tolist() + test['description_stemm'].values.tolist())
train_desc_tfidf = desc_tfidf.transform(train['description_stemm'].values.tolist())
test_desc_tfidf = desc_tfidf.transform(test['description_stemm'].values.tolist())

### SVD Components ###
n_comp = 10
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)
train_svd = pd.DataFrame(svd_obj.transform(train_desc_tfidf))
test_svd = pd.DataFrame(svd_obj.transform(test_desc_tfidf))
train_svd.columns = ['svd_description_'+str(i+1) for i in range(n_comp)]
test_svd.columns = ['svd_description_'+str(i+1) for i in range(n_comp)]
train_svd['item_id'] = train['item_id']
test_svd['item_id'] = test['item_id']

# Merge and delete
train = train.merge(train_svd, on='item_id', how='left')
test = test.merge(test_svd, on='item_id', how='left')
del full_tfidf, train_svd, test_svd
gc.collect()


# [STACKING]# [STACK 
train_tfidf = csr_matrix(hstack([train_title_tfidf, train_desc_tfidf, train_text_tfidf])) 
test_tfidf = csr_matrix(hstack([test_title_tfidf, test_desc_tfidf, test_text_tfidf]))

del train_title_tfidf, train_desc_tfidf, train_text_tfidf
del test_title_tfidf, test_desc_tfidf, test_text_tfidf
gc.collect()

vocab = np.hstack([
    title_tfidf.get_feature_names(),
    desc_tfidf.get_feature_names(),
    text_tfidf.get_feature_names()
])

 [DUMP] TF-IDF pickle files + vocabulary
with open(os.path.join(PATH_TO_DATA, 'train_tfidf.pkl'), 'wb') as train_tfidf_pkl:
    pickle.dump(train_tfidf, train_tfidf_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'test_tfidf.pkl'), 'wb') as test_tfidf_pkl:
    pickle.dump(test_tfidf, test_tfidf_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'vocab.pkl'), 'wb') as vocab_pkl:
    pickle.dump(vocab, vocab_pkl, protocol=2)

del train, train_tfidf, test, test_tfidf, vocab
gc.collect()

