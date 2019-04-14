import cv2
import os
import time
import gc
import glob
import json
import pprint
import joblib
import warnings
import random

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf

from collections import Counter
from functools import partial
from math import sqrt
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF

from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
from pandas.io.json import json_normalize

from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D, CuDNNLSTM, CuDNNGRU
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling2D, GlobalAveragePooling1D, GlobalAvgPool2D, \
    GlobalMaxPool2D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, SpatialDropout2D
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers


preload = False


class PetFinderParser(object):

    def __init__(self, debug=False):

        self.debug = debug
        self.sentence_sep = ' '

        # Does not have to be extracted because main DF already contains description
        self.extract_sentiment_text = False

    def open_metadata_file(self, filename):
        """
        Load metadata file.
        """
        with open(filename, 'r') as f:
            metadata_file = json.load(f)
        return metadata_file

    def open_sentiment_file(self, filename):
        """
        Load sentiment file.
        """
        with open(filename, 'r') as f:
            sentiment_file = json.load(f)
        return sentiment_file

    def open_image_file(self, filename):
        """
        Load image file.
        """
        image = np.asarray(Image.open(filename))
        return image

    def parse_sentiment_file(self, file):
        """
        Parse sentiment file. Output DF with sentiment features.
        """

        file_sentiment = file['documentSentiment']
        file_entities = [x['name'] for x in file['entities']]
        file_entities = self.sentence_sep.join(file_entities)

        # file_entities_new =[x['type'] for x in file['entities']]
        # file_entities_new = self.sentence_sep.join(file_entities_new)

        if self.extract_sentiment_text:
            file_sentences_text = [x['text']['content'] for x in file['sentences']]
            file_sentences_text = self.sentence_sep.join(file_sentences_text)
        file_sentences_sentiment = [x['sentiment'] for x in file['sentences']]

        file_sentences_sentiment = pd.DataFrame.from_dict(
            file_sentences_sentiment, orient='columns').sum()
        file_sentences_sentiment = file_sentences_sentiment.add_prefix('document_').to_dict()

        file_sentiment.update(file_sentences_sentiment)

        df_sentiment = pd.DataFrame.from_dict(file_sentiment, orient='index').T
        if self.extract_sentiment_text:
            df_sentiment['text'] = file_sentences_text

        df_sentiment['entities'] = file_entities
        # df_sentiment['entities_type'] = file_entities_new
        df_sentiment = df_sentiment.add_prefix('sentiment_')

        return df_sentiment

    def parse_metadata_file(self, file):
        """
        Parse metadata file. Output DF with metadata features.
        """

        file_keys = list(file.keys())

        if 'labelAnnotations' in file_keys:
            file_annots = file['labelAnnotations'][:int(len(file['labelAnnotations']))]
            file_top_score = np.asarray([x['score'] for x in file_annots]).mean()
            file_top_desc = [x['description'] for x in file_annots]
        else:
            file_top_score = np.nan
            file_top_desc = ['']

        file_colors = file['imagePropertiesAnnotation']['dominantColors']['colors']
        file_crops = file['cropHintsAnnotation']['cropHints']

        file_color_score = np.asarray([x['score'] for x in file_colors]).mean()
        file_color_pixelfrac = np.asarray([x['pixelFraction'] for x in file_colors]).mean()

        file_crop_conf = np.asarray([x['confidence'] for x in file_crops]).mean()

        if 'importanceFraction' in file_crops[0].keys():
            file_crop_importance = np.asarray([x['importanceFraction'] for x in file_crops]).mean()
        else:
            file_crop_importance = np.nan

        df_metadata = {
            'annots_score': file_top_score,
            'color_score': file_color_score,
            'color_pixelfrac': file_color_pixelfrac,
            'crop_conf': file_crop_conf,
            'crop_importance': file_crop_importance,
            'annots_top_desc': self.sentence_sep.join(file_top_desc)
        }

        df_metadata = pd.DataFrame.from_dict(df_metadata, orient='index').T
        df_metadata = df_metadata.add_prefix('metadata_')

        return df_metadata


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


def rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# Helper function for parallel data processing:
def extract_additional_features(pet_id, mode='train'):
    pet_parser = PetFinderParser()
    sentiment_filename = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(mode, pet_id)
    try:
        sentiment_file = pet_parser.open_sentiment_file(sentiment_filename)
        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)
        df_sentiment['PetID'] = pet_id
    except FileNotFoundError:
        df_sentiment = []

    dfs_metadata = []
    metadata_filenames = sorted(
        glob.glob('../input/petfinder-adoption-prediction/{}_metadata/{}*.json'.format(mode, pet_id)))
    if len(metadata_filenames) > 0:
        for f in metadata_filenames:
            metadata_file = pet_parser.open_metadata_file(f)
            df_metadata = pet_parser.parse_metadata_file(metadata_file)
            df_metadata['PetID'] = pet_id
            dfs_metadata.append(df_metadata)
        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)
    dfs = [df_sentiment, dfs_metadata]

    return dfs


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


# Helper Functions
# ---------------------
@contextmanager
def faith(title):
    start_time = time.time()
    yield
    print(">> {} - done in {:.0f}s".format(title, time.time() - start_time))


def reduce_mem_usage(df, verbose=True):
    numerics = ['uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
                                                                              100 * (start_mem - end_mem) / start_mem))
    return df


def clean_name(x):
    x = str(x)
    no_names = ["No Name Yet", "Nameless", "no_Name_Yet", "No Name Yet God Bless", "-no Name-", "[No Name]",
                "(No Name)", "No Names", "Not Yet Named"]
    for n in no_names:
        x.replace(n, "No Name")
    return x


def relative_age(cols):
    pet_type = cols[0]
    age = cols[1]
    if pet_type == 1:
        relage = age / 144  # Dog Avergae Life Span - 12 years
    else:
        relage = age / 180  # Cat Average Span - 15 years
    return relage


def VerifibalePhotoAmy(number):
    if number > 1:
        vfp = 1
    else:
        vfp = 0
    return vfp


def seo_value(cols):
    photos = cols[0]
    videos = cols[1]
    seo = .7 * videos + .3 * photos
    return seo


def genuine_name(cols):
    name = cols[0]
    quantity = cols[1]
    try:
        is_gen = int(len(name.split()) == 1)
    except:
        is_gen = np.nan
    if int(quantity) > 1:
        is_gen = 1
    return is_gen


def rankbyG(alldata, group):
    rank_telemetry = pd.DataFrame()
    for unit in (alldata[group].unique()):
        tf = alldata[alldata[group] == unit][['PetID', 'InstaFeature', group]]
        col_name = "Insta" + str(group).title() + "Rank"
        tf[col_name] = tf['InstaFeature'].rank(method='max')
        rank_telemetry = pd.concat([rank_telemetry, tf[['PetID', col_name]]])
        del tf
    alldata = pd.merge(alldata, rank_telemetry, on=['PetID'], how='left')
    return alldata


def get_new_columns(name, aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def agg_features(df, groupby, agg, prefix):
    agg_df = df.groupby(groupby).agg(agg)
    agg_df.columns = get_new_columns(prefix, agg)
    return agg_df


def bounding_features(df, meta_path="../input/petfinder-adoption-prediction/train_metadata/"):
    
    df_id = df['PetID']
    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []
    nf_count = 0
    nl_count = 0
    for pet in (df_id):
        try:
            with open(str(meta_path) + pet + '-1.json', 'r') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            nf_count += 1
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)
    df.loc[:, 'vertex_x'] = vertex_xs
    df.loc[:, 'vertex_y'] = vertex_ys
    df.loc[:, 'bounding_confidence'] = bounding_confidences
    df.loc[:, 'bounding_importance'] = bounding_importance_fracs
    df.loc[:, 'dominant_blue'] = dominant_blues
    df.loc[:, 'dominant_green'] = dominant_greens
    df.loc[:, 'dominant_red'] = dominant_reds
    df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    df.loc[:, 'dominant_score'] = dominant_scores
    # df.loc[:, 'label_description'] = label_descriptions
    df.loc[:, 'label_score'] = label_scores
    return df


def open_breeds_info_file(filename):
    with open(filename, 'r') as f:
        breedsdata_file = json.load(f)
    return breedsdata_file


def parse_sentiment_file(file):
    df = pd.DataFrame()
    breeds_file = open_breeds_info_file(file)
    cat_data, dog_data = breeds_file['cat_breeds'], breeds_file['dog_breeds']
    ### Cats
    for idx, cat_breed in enumerate((cat_data.keys())):
        temp = pd.DataFrame.from_dict(json_normalize(cat_data[cat_breed]), orient='columns')
        temp.insert(0, 'Breed', cat_breed)
        for col in temp.columns:
            if col not in ['Breed']:
                df.loc[idx, f'cat_{col}'] = temp[col].values[0]
            else:
                df.loc[idx, f'{col}'] = temp[col].values[0]
    return df


def resize_to_square(im, img_size):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im


def load_image(path):
    image = cv2.imread(path).astype(np.float32)
    new_image = resize_to_square(image)
    new_image = preprocess_input(new_image)
    return new_image


def load_image2(path, image_size):
    image = cv2.imread(path).astype(np.float32)
    new_image = resize_to_square(image, image_size)
    new_image = preprocess_input(new_image)
    return new_image


def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def getDimensions(filename):
    img_size = Image.open(filename).size
    return img_size


def meta_nlp_feats(df,col):
    
    df[col] = df[col].fillna("None")
    df['length'] = df[col].apply(lambda x : len(x))
    df['capitals'] = df[col].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df[col].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df[col].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
    df['num_words'] = df[col].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
    df['num_sad'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(')))
    
    return df
# ============================== PROCESS IN ORDER ===========================


def load_tabular_data():
    train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
    test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

    label_metadata = {}
    labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
    labels_color = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
    labels_state = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

    # print("Mapping Breed Labels...")
    # breed_label_map = {}
    # for idx, row in (enumerate(labels_breed[['BreedID', 'BreedName']].values)):
    #     breed_label_map[row[1]] = int(row[0])
    # temp = parse_sentiment_file('../input/cat-and-dog-breeds-parameters/rating.json')
    # temp['Breed'] = temp['Breed'].map(breed_label_map)
    # train = train.merge(temp, how='left', left_on='Breed1', right_on='Breed')
    # train[temp.columns.tolist()[1:]] = train[temp.columns.tolist()[1:]].fillna(2)
    # train.drop('Breed', axis=1, inplace=True)
    # test = test.merge(temp, how='left', left_on='Breed1', right_on='Breed')
    # test[temp.columns.tolist()[1:]] = test[temp.columns.tolist()[1:]].fillna(2)
    # test.drop('Breed', axis=1, inplace=True)

    return train, test, labels_state, labels_breed, labels_color


def load_image_data():
    
    train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    return train_df_imgs, test_df_imgs


def load_metadata():
    
    train_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_metadata/*.json'))
    test_metadata_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_metadata/*.json'))

    train_df_metadata = pd.DataFrame(train_metadata_files)
    train_df_metadata.columns = ['metadata_filename']
    train_metadata_pets = train_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_metadata = train_df_metadata.assign(PetID=train_metadata_pets)

    test_df_metadata = pd.DataFrame(test_metadata_files)
    test_df_metadata.columns = ['metadata_filename']
    test_metadata_pets = test_df_metadata['metadata_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_metadata = test_df_metadata.assign(PetID=test_metadata_pets)

    return train_df_metadata, test_df_metadata


def load_sentiment_data():
    train_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))
    test_sentiment_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))

    train_df_sentiment = pd.DataFrame(train_sentiment_files)
    train_df_sentiment.columns = ['sentiment_filename']
    train_sentiment_pets = train_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    train_df_sentiment = train_df_sentiment.assign(PetID=train_sentiment_pets)

    test_df_sentiment = pd.DataFrame(test_sentiment_files)
    test_df_sentiment.columns = ['sentiment_filename']
    test_sentiment_pets = test_df_sentiment['sentiment_filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    test_df_sentiment = test_df_sentiment.assign(PetID=test_sentiment_pets)

    return train_df_sentiment, test_df_sentiment


def build_model(shape=(256, 256, 3), weights_path="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5"):
    inp = Input(shape)
    backbone = DenseNet121(input_tensor=inp,
                           weights=weights_path,
                           include_top=False)
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = AveragePooling1D(4)(x)
    out = Lambda(lambda x: x[:, :, 0])(x)
    model = Model(inp, out)
    return model


def train_model(model, train, test, nn_params={"batch_size": 64, "img_size": 256}):
    batch_size = nn_params['batch_size']
    img_size = nn_params['img_size']
    pet_ids = train['PetID'].values
    train_df_ids = train[['PetID']]

    # Train images
    features = {}
    train_image = glob.glob("../input/petfinder-adoption-prediction/train_images/*.jpg")
    n_batches = len(train_image) // batch_size + (len(train_image) % batch_size != 0)
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = train_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    train_feats = pd.DataFrame.from_dict(features, orient='index')
    train_feats.columns = ['pic_' + str(i) for i in range(train_feats.shape[1])]

    train_feats = train_feats.reset_index()
    train_feats['PetID'] = train_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
    train_feats = train_feats.drop("index", axis=1)
    train_feats = train_feats.groupby('PetID').agg("mean")
    train_feats = train_feats.reset_index()

    # Test images
    features = {}

    test_image = glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg")
    n_batches = len(test_image) // batch_size + (len(test_image) % batch_size != 0)
    for b in (range(n_batches)):
        start = b * batch_size
        end = (b + 1) * batch_size
        batch_pets = test_image[start:end]
        batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
        for i, pet_id in enumerate(batch_pets):
            try:
                batch_images[i] = load_image(pet_id)
            except:
                pass
        batch_preds = model.predict(batch_images)
        for i, pet_id in enumerate(batch_pets):
            features[pet_id] = batch_preds[i]

    test_feats = pd.DataFrame.from_dict(features, orient='index')
    test_feats.columns = ['pic_' + str(i) for i in range(test_feats.shape[1])]

    test_feats = test_feats.reset_index()
    test_feats['PetID'] = test_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
    test_feats = test_feats.drop("index", axis=1)
    test_feats = test_feats.groupby('PetID').agg("mean")
    test_feats = test_feats.reset_index()
    pretrained_feats = pd.concat([train_feats, test_feats], axis=0)

    return pretrained_feats


def image_feature(model, train, test, nn_params={"batch_size": 64, "img_size": 256}):
    if not preload:
        batch_size = nn_params['batch_size']
        img_size = nn_params['img_size']
        train_df_ids = train[['PetID']]

        # Train images
        features = {}
        train_image = glob.glob("../input/petfinder-adoption-prediction/train_images/*.jpg")
        n_batches = len(train_image) // batch_size + 1
        for b in (range(n_batches)):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = train_image[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image2(pet_id, img_size)
                except:
                    print(pet_id)
                    pass
            batch_preds = model.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]

        train_feats = pd.DataFrame.from_dict(features, orient='index')
        train_feats.columns = ['pic_' + str(i) for i in range(train_feats.shape[1])]

        train_feats = train_feats.reset_index()
        train_feats['PetID'] = train_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
        train_feats = train_feats.drop("index", axis=1)
        train_feats = train_feats.groupby('PetID').agg("mean")
        train_feats = train_feats.reset_index()

        # Test images
        features = {}

        test_image = glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg")
        n_batches = len(test_image) // batch_size + 1
        for b in (range(n_batches)):
            start = b * batch_size
            end = (b + 1) * batch_size
            batch_pets = test_image[start:end]
            batch_images = np.zeros((len(batch_pets), img_size, img_size, 3))
            for i, pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = load_image2(pet_id, img_size)
                except:
                    print(pet_id)
                    pass
            batch_preds = model.predict(batch_images)
            for i, pet_id in enumerate(batch_pets):
                features[pet_id] = batch_preds[i]

        test_feats = pd.DataFrame.from_dict(features, orient='index')
        test_feats.columns = ['pic_' + str(i) for i in range(test_feats.shape[1])]

        test_feats = test_feats.reset_index()
        test_feats['PetID'] = test_feats['index'].apply(lambda x: x.split("/")[-1].split("-")[0])
        test_feats = test_feats.drop("index", axis=1)
        test_feats = test_feats.groupby('PetID').agg("mean")
        test_feats = test_feats.reset_index()
        pretrained_feats = pd.concat([train_feats, test_feats], axis=0)
    else:
        train_feats = pd.read_csv("./processed_data/train_img.csv")
        test_feats = pd.read_csv("./processed_data/test_img.csv")
        pretrained_feats = pd.concat([train_feats, test_feats], axis=0)

    return pretrained_feats


def basic_features(train, test):
    
    alldata = pd.concat([train, test], sort=False)
    print(train.shape, test.shape, alldata.shape)
    #########################################################################################################
    # Breed create columns
    alldata['weeks'] = alldata['Age']*31//7
    alldata['L_Breed1_Siamese'] =(alldata['Breed1']== 292).astype(int)
    alldata['L_Breed1_Persian']=(alldata['Breed1']== 285).astype(int)
    alldata['L_Breed1_Labrador_Retriever']=(alldata['Breed1']== 141).astype(int)
    alldata['L_Breed1_Terrier']=(alldata['Breed1']==218).astype(int)
    alldata['L_Breed1_Golden_Retriever ']=(alldata['Breed1']==109).astype(int)
    alldata['shorthair_hairless_domestic_hair'] = 0
    alldata.loc[alldata['Breed1'].isin([9 ,104 ,106 ,236 ,237 ,238 ,243 ,244 ,251 ,255 ,264 ,265 ,266 ,268 ,282 ,283 ,298]) == True, 'shorthair_hairless_domestic_hair'] = 1
    
    alldata['#Feature_avg_age_breed1_fee'] = alldata[['Age', 'Breed1', 'Fee']].groupby(['Age', 'Breed1'])['Fee'].transform('mean')
    alldata['#Feature_avg_age_breed2_fee'] = alldata[['Age', 'Breed2', 'Fee']].groupby(['Age', 'Breed2'])['Fee'].transform('mean')
    alldata['#Feature_age_breed1_maturity_sz'] = alldata[[ 'Age', 'Breed1', 'MaturitySize']].groupby([ 'Age', 'Breed1'])['MaturitySize'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_maturity_sz'] = alldata[[ 'Age', 'Breed2', 'MaturitySize']].groupby([ 'Age', 'Breed2'])['MaturitySize'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_age_breed1_fur'] = alldata[[ 'Age', 'Breed1', 'FurLength']].groupby([ 'Age', 'Breed1'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_fur'] = alldata[[ 'Age', 'Breed2', 'FurLength']].groupby([ 'Age', 'Breed2'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed1_fee'] = alldata[[ 'Age', 'Breed1', 'Fee']].groupby([ 'Age', 'Breed1'])['Fee'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_breed2_fee'] = alldata[[ 'Age', 'Breed2', 'Fee']].groupby([ 'Age', 'Breed2'])['Fee'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_state_breed1_age_freq']     = alldata[[ 'State', 'Breed1', 'Age']].groupby([ 'State', 'Breed1'])['Age'].transform('mean')
    alldata['#Feature_state_breed1_age_fee_freq'] = alldata[[ 'State', 'Breed1', 'Age', 'Fee']].groupby([ 'State', 'Breed1', 'Age'])['Fee'].transform('mean')
    alldata['#Feature_state_breed2_age_freq']     = alldata[[ 'State', 'Breed2', 'Age']].groupby([ 'State', 'Breed2'])['Age'].transform('mean')
    alldata['#Feature_state_breed2_age_fee_freq'] = alldata[[ 'State', 'Breed2', 'Age', 'Fee']].groupby([ 'State', 'Breed2', 'Age'])['Fee'].transform('mean')
    
    alldata['#Feature_avg_type_age_breed1_fee'] = alldata[['Type','Age', 'Breed1', 'Fee']].groupby(['Type','Age', 'Breed1'])['Fee'].transform('mean')
    alldata['#Feature_avg_type_age_breed2_fee'] = alldata[['Type','Age', 'Breed2', 'Fee']].groupby(['Type','Age', 'Breed2'])['Fee'].transform('mean')
    alldata['#Feature_age_type_breed1_maturity_sz'] = alldata[['Type', 'Age', 'Breed1', 'MaturitySize']].groupby(['Type', 'Age', 'Breed1'])['MaturitySize'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_maturity_sz'] = alldata[['Type', 'Age', 'Breed2', 'MaturitySize']].groupby(['Type', 'Age', 'Breed2'])['MaturitySize'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_age_type_breed1_fur'] = alldata[['Type', 'Age', 'Breed1', 'FurLength']].groupby(['Type', 'Age', 'Breed1'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_fur'] = alldata[['Type', 'Age', 'Breed2', 'FurLength']].groupby(['Type', 'Age', 'Breed2'])['FurLength'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed1_fee'] = alldata[['Type', 'Age', 'Breed1', 'Fee']].groupby(['Type', 'Age', 'Breed1'])['Fee'].transform('count') / alldata.shape[0]
    alldata['#Feature_age_type_breed2_fee'] = alldata[['Type', 'Age', 'Breed2', 'Fee']].groupby(['Type', 'Age', 'Breed2'])['Fee'].transform('count') / alldata.shape[0]
    
    alldata['#Feature_state_type_breed1_age_freq']     = alldata[['Type', 'State', 'Breed1', 'Age']].groupby(['Type', 'State', 'Breed1'])['Age'].transform('mean')
    alldata['#Feature_state_type_breed1_age_fee_freq'] = alldata[['Type', 'State', 'Breed1', 'Age', 'Fee']].groupby(['Type', 'State', 'Breed1', 'Age'])['Fee'].transform('mean')
    alldata['#Feature_state_type_breed2_age_freq']     = alldata[['Type', 'State', 'Breed2', 'Age']].groupby(['Type', 'State', 'Breed2'])['Age'].transform('mean')
    alldata['#Feature_state_type_breed2_age_fee_freq'] = alldata[['Type', 'State', 'Breed2', 'Age', 'Fee']].groupby(['Type', 'State', 'Breed2', 'Age'])['Fee'].transform('mean')
    
    ###########################################################################################################
    
    alldata['RelAge'] = alldata[['Type', 'Age']].apply(relative_age, axis=1)
    alldata['IsNameGenuine'] = alldata[['Name', 'Quantity']].apply(genuine_name, axis=1)
    alldata['InstaFeature'] = alldata[['PhotoAmt', 'VideoAmt']].apply(seo_value, axis=1)
    alldata['ShowsMore'] = alldata['PhotoAmt'].apply(VerifibalePhotoAmy)
    alldata["Vaccinated_Deworked_Mutation"] = alldata['Vaccinated'].apply(str) + "_" + alldata['Dewormed'].apply(str)
    alldata["Vaccinated_Deworked_Mutation"] = alldata['Vaccinated'].apply(str) + "_" + alldata['Dewormed'].apply(str)
    alldata = pd.get_dummies(alldata, columns=['Vaccinated_Deworked_Mutation'], prefix="Vaccinated_Dewormed")
    alldata['GlobalInstaRank'] = alldata['InstaFeature'].rank(method='max')
    print(">> Ranking Features By State")
    alldata = rankbyG(alldata, "State")
    print(">> Ranking Features By Animal")
    alldata = rankbyG(alldata, "Type")
    print(">> Ranking Features By Breed1")
    alldata = rankbyG(alldata, "Breed1")
    print(">> Ranking Features By Gender")
    alldata = rankbyG(alldata, "Gender")

    top_dogs = [179, 205, 195, 178, 206, 109, 189, 103]
    top_cats = [276, 268, 285, 252, 243, 251, 288, 247, 280, 290]

    alldata['#Feature_SecondaryColors'] = alldata['Color2'] + alldata['Color3']
    alldata['#Feature_MonoColor'] = np.where(alldata['#Feature_SecondaryColors'], 1, 0)
    alldata['top_breeds'] = 0
    alldata.loc[alldata['Breed1'].isin(top_dogs + top_cats) == True, 'top_breeds'] = 1
    alldata['top_breed_free'] = 0
    alldata.loc[alldata[(alldata['Fee'] == 0) & (alldata['top_breeds'] == 1)].index, 'top_breed_free'] = 1
    alldata['free_pet'] = 0
    alldata.loc[alldata[alldata['Fee'] == 0].index, 'free_pet'] = 1
    alldata['free_pet_age_1'] = 0
    alldata.loc[alldata[(alldata['Fee'] == 0) & (alldata['Age'] == 1)].index, 'free_pet_age_1'] = 1
    alldata['year'] = alldata['Age'] / 12.
    alldata['#Feature_less_a_year'] = np.where(alldata['Age'] < 12, 1, 0)
    alldata['#Feature_top_2_states'] = 0
    alldata.loc[alldata['State'].isin([41326, 41401]) == True, '#Feature_top_2_states'] = 1
    alldata['#Feature_age_exact'] = 0
    alldata.loc[alldata['Age'].isin([12, 24, 36, 48, 60, 72, 84, 96, 108]) == True, '#Feature_age_exact'] = 1
    alldata['#Feature_isLonely'] = np.where(alldata['Quantity'] > 1, 1, 0)
    alldata['total_img_video'] = alldata['PhotoAmt'] + alldata['VideoAmt']
    
    # alldata['#Feature_avg_age_breed1_fee'] = alldata[['Age', 'Breed1', 'Fee']].groupby(['Age', 'Breed1'])[
    #     'Fee'].transform('mean')
    # alldata['#Feature_avg_age_breed2_fee'] = alldata[['Age', 'Breed2', 'Fee']].groupby(['Age', 'Breed2'])[
    #     'Fee'].transform('mean')
    # alldata['#Feature_age_breed1_maturity_sz'] = alldata[['Age', 'Breed1', 'MaturitySize']].groupby(['Age', 'Breed1'])[
    #                                                  'MaturitySize'].transform('count') / alldata.shape[0]
    # alldata['#Feature_age_breed2_maturity_sz'] = alldata[['Age', 'Breed2', 'MaturitySize']].groupby(['Age', 'Breed2'])[
    #                                                  'MaturitySize'].transform('count') / alldata.shape[0]
    # alldata['#Feature_age_breed1_fur'] = alldata[['Age', 'Breed1', 'FurLength']].groupby(['Age', 'Breed1'])[
    #                                          'FurLength'].transform('count') / alldata.shape[0]
    # alldata['#Feature_age_breed2_fur'] = alldata[['Age', 'Breed2', 'FurLength']].groupby(['Age', 'Breed2'])[
    #                                          'FurLength'].transform('count') / alldata.shape[0]
    # alldata['#Feature_age_breed1_fee'] = alldata[['Age', 'Breed1', 'Fee']].groupby(['Age', 'Breed1'])['Fee'].transform(
    #     'count') / alldata.shape[0]
    # alldata['#Feature_age_breed2_fee'] = alldata[['Age', 'Breed2', 'Fee']].groupby(['Age', 'Breed2'])['Fee'].transform(
    #     'count') / alldata.shape[0]
    # alldata['#Feature_state_breed1_age_freq'] = alldata[['State', 'Breed1', 'Age']].groupby(['State', 'Breed1'])[
    #     'Age'].transform('mean')
    # alldata['#Feature_state_breed1_age_fee_freq'] = \
    # alldata[['State', 'Breed1', 'Age', 'Fee']].groupby(['State', 'Breed1', 'Age'])['Fee'].transform('mean')
    # alldata['#Feature_state_breed2_age_freq'] = alldata[['State', 'Breed2', 'Age']].groupby(['State', 'Breed2'])[
    #     'Age'].transform('mean')
    # alldata['#Feature_state_breed2_age_fee_freq'] = \
    # alldata[['State', 'Breed2', 'Age', 'Fee']].groupby(['State', 'Breed2', 'Age'])['Fee'].transform('mean')

    # Clean the name
    # alldata['Name'] = alldata['Name'].apply(lambda x: clean_name(x))
    # alldata['Name'] = alldata['Name'].fillna("No Name")

    rescuer_count = alldata.groupby(['RescuerID'])['PetID'].count().reset_index()
    rescuer_count.columns = ['RescuerID', 'RescuerID_COUNT']
    alldata = alldata.merge(rescuer_count, how='left', on='RescuerID')

    Description_count = alldata.groupby(['Description'])['PetID'].count().reset_index()
    Description_count.columns = ['Description', 'Description_COUNT']
    alldata = alldata.merge(Description_count, how='left', on='Description')

    Name_count = alldata.groupby(['Name'])['PetID'].count().reset_index()
    Name_count.columns = ['Name', 'Name_COUNT']
    alldata = alldata.merge(Name_count, how='left', on='Name')

    agg = {}
    agg['Quantity'] = ['mean', 'var', 'max', 'min', 'skew', 'median']
    agg['Fee'] = ['mean', 'var', 'max', 'min', 'skew', 'median']
    agg['Age'] = ['mean', 'sum', 'var', 'max', 'min', 'skew', 'median']
    agg['Breed1'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Breed2'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Type'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Gender'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Color1'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Color2'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Color3'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['MaturitySize'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['FurLength'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Vaccinated'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Sterilized'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg['Health'] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg["PhotoAmt"] = ['nunique', 'var', 'max', 'min', 'skew', 'median']
    agg["RelAge"] = ['nunique', 'var', 'max', 'min', 'skew', 'median']

    # RescuerID
    grouby = 'RescuerID'
    agg_df = agg_features(alldata, grouby, agg, grouby)
    alldata = alldata.merge(agg_df, on=grouby, how='left')
    
    agg_kurt_df = alldata.groupby(grouby)[list(agg.keys())].apply(pd.DataFrame.kurt)
    agg_kurt_df.columns = [f"{key}_kurt" for key in list(agg.keys())]
    alldata = alldata.merge(agg_kurt_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.25)
    agg_perc_df.columns = [f"{key}_perc_25" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.75)
    agg_perc_df.columns = [f"{key}_perc_75" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')
    
    
    # State
    
    ################################################CREATING MULTIPLE COLUMNS WITH_X NEED TO BE FIXED
    grouby = 'State'
    agg_df = agg_features(alldata, grouby, agg, grouby)
    alldata = alldata.merge(agg_df, on=grouby, how='left')
    
    agg_kurt_df = alldata.groupby(grouby)[list(agg.keys())].apply(pd.DataFrame.kurt)
    agg_kurt_df.columns = [f"{key}_kurt" for key in list(agg.keys())]
    alldata = alldata.merge(agg_kurt_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.25)
    agg_perc_df.columns = [f"{key}_perc_25" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')
    
    agg_perc_df = alldata.groupby(grouby)[list(agg.keys())].quantile(.75)
    agg_perc_df.columns = [f"{key}_perc_75" for key in list(agg.keys())]
    alldata = alldata.merge(agg_perc_df, on=grouby, how='left')

    train = alldata[:len(train)]
    test  = alldata[len(train):]

    return train, test


def image_dim_features(train, test):
    # Load IDs and Image data
    # ===========================================
    split_char = "/"
    train_df_ids = train[['PetID']]
    test_df_ids = test[['PetID']]

    train_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/train_images/*.jpg'))
    test_image_files = sorted(glob.glob('../input/petfinder-adoption-prediction/test_images/*.jpg'))

    train_df_imgs = pd.DataFrame(train_image_files)
    train_df_imgs.columns = ['image_filename']
    train_imgs_pets = train_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)

    test_df_imgs = pd.DataFrame(test_image_files)
    test_df_imgs.columns = ['image_filename']
    test_imgs_pets = test_df_imgs['image_filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

    # ===========================================

    train_df_imgs['image_size'] = train_df_imgs['image_filename'].apply(getSize)
    train_df_imgs['temp_size'] = train_df_imgs['image_filename'].apply(getDimensions)
    train_df_imgs['width'] = train_df_imgs['temp_size'].apply(lambda x: x[0])
    train_df_imgs['height'] = train_df_imgs['temp_size'].apply(lambda x: x[1])
    train_df_imgs = train_df_imgs.drop(['temp_size'], axis=1)

    test_df_imgs['image_size'] = test_df_imgs['image_filename'].apply(getSize)
    test_df_imgs['temp_size'] = test_df_imgs['image_filename'].apply(getDimensions)
    test_df_imgs['width'] = test_df_imgs['temp_size'].apply(lambda x: x[0])
    test_df_imgs['height'] = test_df_imgs['temp_size'].apply(lambda x: x[1])
    test_df_imgs = test_df_imgs.drop(['temp_size'], axis=1)

    aggs = {
        'image_size': ['sum', 'mean', 'var'],
        'width': ['sum', 'mean', 'var'],
        'height': ['sum', 'mean', 'var'],
    }

    agg_train_imgs = train_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_train_imgs.columns = new_columns
    agg_train_imgs = agg_train_imgs.reset_index()

    agg_test_imgs = test_df_imgs.groupby('PetID').agg(aggs)
    new_columns = [
        k + '_' + agg for k in aggs.keys() for agg in aggs[k]
    ]
    agg_test_imgs.columns = new_columns
    agg_test_imgs = agg_test_imgs.reset_index()

    agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
    return agg_imgs


def metadata_features(train, test):

    if not preload:
        train_pet_ids = train.PetID.unique()
        test_pet_ids = test.PetID.unique()

        # Train Feature Extractions
        # ===============================

        dfs_train = Parallel(n_jobs=12, verbose=1)(
            delayed(extract_additional_features)(i, mode='train') for i in train_pet_ids)
        train_dfs_sentiment = [x[0] for x in dfs_train if isinstance(x[0], pd.DataFrame)]
        train_dfs_metadata = [x[1] for x in dfs_train if isinstance(x[1], pd.DataFrame)]
        train_dfs_sentiment = pd.concat(train_dfs_sentiment, ignore_index=True, sort=False)
        train_dfs_metadata = pd.concat(train_dfs_metadata, ignore_index=True, sort=False)

        # Test Feature Extractions
        # ===============================
        dfs_test = Parallel(n_jobs=6, verbose=1)(delayed(extract_additional_features)(i, mode='test') for i in test_pet_ids)
        test_dfs_sentiment = [x[0] for x in dfs_test if isinstance(x[0], pd.DataFrame)]
        test_dfs_metadata = [x[1] for x in dfs_test if isinstance(x[1], pd.DataFrame)]
        test_dfs_sentiment = pd.concat(test_dfs_sentiment, ignore_index=True, sort=False)
        test_dfs_metadata = pd.concat(test_dfs_metadata, ignore_index=True, sort=False)

    else:
        train_dfs_sentiment = pd.read_csv("./processed_data/train_dfs_sentiment.csv")
        train_dfs_metadata = pd.read_csv("./processed_data/train_dfs_metadata.csv")
        test_dfs_sentiment = pd.read_csv("./processed_data/test_dfs_sentiment.csv")
        test_dfs_metadata = pd.read_csv("./processed_data/test_dfs_metadata.csv")

        train_dfs_sentiment['sentiment_entities'].fillna('', inplace=True)
        train_dfs_metadata['metadata_annots_top_desc'].fillna('', inplace=True)
        test_dfs_sentiment['sentiment_entities'].fillna('', inplace=True)
        test_dfs_metadata['metadata_annots_top_desc'].fillna('', inplace=True)


    # Meta data Aggregates
    # ===============================
    aggregates = ['mean', 'sum', 'var']

    # Train Aggregates
    # ---------------------------
    train_metadata_desc = train_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    train_metadata_desc = train_metadata_desc.reset_index()
    train_metadata_desc['metadata_annots_top_desc'] = train_metadata_desc['metadata_annots_top_desc'].apply(
        lambda x: ' '.join(x.tolist()))

    prefix = 'metadata'
    train_metadata_gr = train_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
    for i in train_metadata_gr.columns:
        if 'PetID' not in i:
            train_metadata_gr[i] = train_metadata_gr[i].astype(float)
    train_metadata_gr = train_metadata_gr.groupby(['PetID']).agg(aggregates)
    train_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in train_metadata_gr.columns.tolist()])
    train_metadata_gr = train_metadata_gr.reset_index()

    train_sentiment_desc = train_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    train_sentiment_desc = train_sentiment_desc.reset_index()
    train_sentiment_desc['sentiment_entities'] = train_sentiment_desc['sentiment_entities'].apply(
        lambda x: ' '.join(x.tolist()))

    prefix = 'sentiment'
    train_sentiment_gr = train_dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in train_sentiment_gr.columns:
        if 'PetID' not in i:
            train_sentiment_gr[i] = train_sentiment_gr[i].astype(float)
    train_sentiment_gr = train_sentiment_gr.groupby(['PetID']).agg(aggregates)
    train_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in train_sentiment_gr.columns.tolist()])
    train_sentiment_gr = train_sentiment_gr.reset_index()

    # Test data Aggregates
    # ---------------------------
    test_metadata_desc = test_dfs_metadata.groupby(['PetID'])['metadata_annots_top_desc'].unique()
    test_metadata_desc = test_metadata_desc.reset_index()
    test_metadata_desc[
        'metadata_annots_top_desc'] = test_metadata_desc[
        'metadata_annots_top_desc'].apply(lambda x: ' '.join(x.tolist()))

    prefix = 'metadata'
    test_metadata_gr = test_dfs_metadata.drop(['metadata_annots_top_desc'], axis=1)
    for i in test_metadata_gr.columns:
        if 'PetID' not in i:
            test_metadata_gr[i] = test_metadata_gr[i].astype(float)
    test_metadata_gr = test_metadata_gr.groupby(['PetID']).agg(aggregates)
    test_metadata_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in test_metadata_gr.columns.tolist()])
    test_metadata_gr = test_metadata_gr.reset_index()

    test_sentiment_desc = test_dfs_sentiment.groupby(['PetID'])['sentiment_entities'].unique()
    test_sentiment_desc = test_sentiment_desc.reset_index()
    test_sentiment_desc[
        'sentiment_entities'] = test_sentiment_desc[
        'sentiment_entities'].apply(lambda x: ' '.join(x.tolist()))

    prefix = 'sentiment'
    test_sentiment_gr = test_dfs_sentiment.drop(['sentiment_entities'], axis=1)
    for i in test_sentiment_gr.columns:
        if 'PetID' not in i:
            test_sentiment_gr[i] = test_sentiment_gr[i].astype(float)
    test_sentiment_gr = test_sentiment_gr.groupby(['PetID']).agg(aggregates)
    test_sentiment_gr.columns = pd.Index(['{}_{}_{}'.format(
        prefix, c[0], c[1].upper()) for c in test_sentiment_gr.columns.tolist()])
    test_sentiment_gr = test_sentiment_gr.reset_index()


    # Mergining Features with Train/Test
    # =======================================
    train_proc = train.copy()
    train_proc = train_proc.merge(train_sentiment_gr, how='left', on='PetID')
    train_proc = train_proc.merge(train_metadata_gr, how='left', on='PetID')
    train_proc = train_proc.merge(train_metadata_desc, how='left', on='PetID')
    train_proc = train_proc.merge(train_sentiment_desc, how='left', on='PetID')

    test_proc = test.copy()
    test_proc = test_proc.merge(test_sentiment_gr, how='left', on='PetID')
    test_proc = test_proc.merge(test_metadata_gr, how='left', on='PetID')
    test_proc = test_proc.merge(test_metadata_desc, how='left', on='PetID')
    test_proc = test_proc.merge(test_sentiment_desc, how='left', on='PetID')

    return train_proc, test_proc


def breed_maps(train_proc, test_proc, labels_breed):
    train_breed_main = train_proc[['Breed1']].merge(labels_breed, how='left', left_on='Breed1', right_on='BreedID',
                                                    suffixes=('', '_main_breed'))
    train_breed_main = train_breed_main.iloc[:, 2:]
    train_breed_main = train_breed_main.add_prefix('main_breed_')
    train_breed_second = train_proc[['Breed2']].merge(labels_breed, how='left', left_on='Breed2', right_on='BreedID',
                                                      suffixes=('', '_second_breed'))
    train_breed_second = train_breed_second.iloc[:, 2:]
    train_breed_second = train_breed_second.add_prefix('second_breed_')
    train_proc = pd.concat([train_proc, train_breed_main, train_breed_second], axis=1)
    test_breed_main = test_proc[['Breed1']].merge(labels_breed, how='left', left_on='Breed1', right_on='BreedID',
                                                  suffixes=('', '_main_breed'))
    test_breed_main = test_breed_main.iloc[:, 2:]
    test_breed_main = test_breed_main.add_prefix('main_breed_')
    test_breed_second = test_proc[['Breed2']].merge(labels_breed, how='left', left_on='Breed2', right_on='BreedID',
                                                    suffixes=('', '_second_breed'))
    test_breed_second = test_breed_second.iloc[:, 2:]
    test_breed_second = test_breed_second.add_prefix('second_breed_')
    test_proc = pd.concat([test_proc, test_breed_main, test_breed_second], axis=1)
    X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
    categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']

    for i in categorical_columns:
        X.loc[:, i] = pd.factorize(X.loc[:, i])[0]
    return X


def nlp_features(X_temp):
    text_columns = ['Description', 'metadata_annots_top_desc', 'sentiment_entities']
    X_text = X_temp[text_columns]

    for i in X_text.columns:
        X_text.loc[:, i] = X_text.loc[:, i].fillna('<MISSING>')

    n_components = 50
    text_features = []

    # Generate text features:
    for i in X_text.columns:
        # Initialize decomposition methods:
        print('Generating features from: {}'.format(i))
        svd_ = TruncatedSVD(n_components=n_components, random_state=1337)
        nmf_ = NMF(n_components=n_components, random_state=1337)
        tfidf_col = TfidfVectorizer().fit_transform(X_text.loc[:, i].values)
        svd_col = svd_.fit_transform(tfidf_col)
        svd_col = pd.DataFrame(svd_col)
        svd_col = svd_col.add_prefix('SVD_{}_'.format(i))
        nmf_col = nmf_.fit_transform(tfidf_col)
        nmf_col = pd.DataFrame(nmf_col)
        nmf_col = nmf_col.add_prefix('NMF_{}_'.format(i))
        text_features.append(svd_col)
        text_features.append(nmf_col)

    # Combine all extracted features:
    text_features = pd.concat(text_features, axis=1)
    # Concatenate with main DF:
    X_temp = pd.concat([X_temp, text_features], axis=1)
    # Remove raw text columns:
    for i in X_text.columns:
        X_temp = X_temp.drop(i, axis=1)
    # Remove unnecessary columns:
    to_drop_columns = ['PetID', 'Name']
    X_temp = X_temp.drop(to_drop_columns, axis=1)

    return X_temp


def run_lgbm(X_temp, test):
    params = {'application': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'num_leaves': 70,
              'max_depth': 9,
              'learning_rate': 0.01,
              'bagging_fraction': 0.6,  # .85 previously
              'feature_fraction': 0.6,  # .8 previously
              'min_split_gain': 0.02,
              'min_child_samples': 150,
              'min_child_weight': 0.02,
              'lambda_l2': 0.0475,
              'verbosity': -1,
              'data_random_seed': 17,
              }
    # Additional parameters:
    early_stop = 500
    verbose_eval = 500
    num_rounds = 10000
    n_splits = 10

    # Split into train and test again:
    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed'], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    train_cols.remove('RescuerID')

    test_cols = X_test.columns.tolist()

    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    rescuer_gb_mean = X_train.groupby('RescuerID')['AdoptionSpeed'].agg("mean").reset_index()
    rescuer_gb_mean.columns = ['RescuerID', 'AdoptionSpeed_mean']

    rescuer_ids = rescuer_gb_mean['RescuerID'].values
    rescuer_as_mean = rescuer_gb_mean['AdoptionSpeed_mean'].values

    i = 0

    for train_index, valid_index in kfold.split(rescuer_ids, rescuer_as_mean.astype(np.int)):
        rescuser_train_ids = rescuer_ids[train_index]
        rescuser_valid_ids = rescuer_ids[valid_index]

        X_tr = X_train[X_train["RescuerID"].isin(rescuser_train_ids)]
        X_val = X_train[X_train["RescuerID"].isin(rescuser_valid_ids)]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        print('\ny_tr distribution: {}'.format(Counter(y_tr)))

        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_val, label=y_val)
        watchlist = [d_train, d_valid]

        print('training LGB:')
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop
                          )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        test_pred = model.predict(X_test.drop(["RescuerID"], axis=1), num_iteration=model.best_iteration)

        oof_train[X_val.index] = val_pred
        oof_test[:, i] = test_pred

        i += 1

    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_cols)
    imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
    imp_df["importance_split"] = model.feature_importance(importance_type='split')
    imp_df.to_csv('imps.csv', index=False)

    # Compute QWK based on OOF train predictions:
    optR = OptimizedRounder()
    optR.fit(oof_train, X_train['AdoptionSpeed'].values)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(oof_train, coefficients)
    print("\nValid Counts = ", Counter(X_train['AdoptionSpeed'].values))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, pred_test_y_k)
    print("QWK = ", qwk)

    coefficients_ = coefficients.copy()
    print(f'coefficients returned From optim for LGBM are {coefficients_}')

    coefficients_[0] = 1.645
    #coefficients_[1] = 2.115
    #coefficients_[3] = 2.84

    print(f'coefficients actually used are {coefficients_}')

    train_predictions = optR.predict(oof_train, coefficients_).astype(int)
    print('train pred distribution: {}'.format(Counter(train_predictions)))

    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_)
    print('test pred distribution: {}'.format(Counter(test_predictions)))

    # Distribution inspection of original target and predicted train and test:
    print("True Distribution:")
    print(pd.value_counts(X_train['AdoptionSpeed'], normalize=True).sort_index())
    print("\nTrain Predicted Distribution:")
    print(pd.value_counts(train_predictions, normalize=True).sort_index())
    print("\nTest Predicted Distribution:")
    print(pd.value_counts(test_predictions, normalize=True).sort_index())
    submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions.astype(np.uint16)})
    return submission, oof_train, oof_test


def run_xgb(X_temp, test):
 
    params = {'eval_metric': 'rmse',
              'seed': 1337,
              'eta': 0.0123,
              'subsample': 0.7,
              'colsample_bytree': 0.75,
              'tree_method': 'gpu_hist',
              'device': 'gpu',
              'silent': 1,
              'gamma' : 8,
              'max_depth' : 7
              }
    n_splits = 10
    verbose_eval = 1000
    num_rounds = 10000
    early_stop = 500
    
    # Split into train and test again:
    X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
    X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]

    # Remove missing target column from test:
    X_test = X_test.drop(['AdoptionSpeed', "RescuerID"], axis=1)

    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))

    # Check if columns between the two DFs are the same:
    train_cols = X_train.columns.tolist()
    train_cols.remove('AdoptionSpeed')
    train_cols.remove('RescuerID')

    test_cols = X_test.columns.tolist()

    kfold = StratifiedKFold(n_splits=n_splits, random_state=1337)
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    rescuer_gb_mean = X_train.groupby('RescuerID')['AdoptionSpeed'].agg("mean").reset_index()
    rescuer_gb_mean.columns = ['RescuerID', 'AdoptionSpeed_mean']

    rescuer_ids = rescuer_gb_mean['RescuerID'].values
    rescuer_as_mean = rescuer_gb_mean['AdoptionSpeed_mean'].values

    i = 0

    for train_index, valid_index in kfold.split(X_train, X_train['AdoptionSpeed'].astype(np.int)):
        print(f'Fold {i+1}')

        X_tr = X_train.iloc[train_index]
        X_val = X_train.iloc[valid_index]

        y_tr = X_tr['AdoptionSpeed'].values
        X_tr = X_tr.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        y_val = X_val['AdoptionSpeed'].values
        X_val = X_val.drop(['AdoptionSpeed', 'RescuerID'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,
                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[X_val.index] = valid_pred
        oof_test[:, i] = test_pred

        i += 1

    optR = OptimizedRounder()
    optR.fit(oof_train, X_train['AdoptionSpeed'].values)
    coefficients = optR.coefficients()
    valid_pred = optR.predict(oof_train, coefficients)
    qwk = quadratic_weighted_kappa(X_train['AdoptionSpeed'].values, valid_pred)
    print("QWK = ", qwk)
    coefficients_ = coefficients.copy()
    print(f'coefficients returned From optim for XGB are {coefficients_}')
    coefficients_[0] = 1.645
    # coefficients_[1] = 2.115
    # coefficients_[3] = 2.84
    print(f'coefficients used for XGB are {coefficients_}')
    train_predictions = optR.predict(oof_train, coefficients_).astype(np.int8)
    print(f'train pred distribution: {Counter(train_predictions)}')
    test_predictions = optR.predict(oof_test.mean(axis=1), coefficients_).astype(np.int8)
    print(f'test pred distribution: {Counter(test_predictions)}')

    submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})
    return submission, oof_train, oof_test


set_seed(2411)
train, test, labels_state, labels_breed, labels_color = load_tabular_data()

train, test = basic_features(train, test)

train =  meta_nlp_feats(train, 'Description')
test  =  meta_nlp_feats(test,  'Description')

train = bounding_features(train, meta_path="../input/petfinder-adoption-prediction/train_metadata/")
test  = bounding_features(test, meta_path="../input/petfinder-adoption-prediction/test_metadata/")

train, test = metadata_features(train, test)

###########################################################################

###### https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,tst_series=None,target=None,min_samples_leaf=1,smoothing=1,noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
###########################################################################
#######################################################################
trn, sub = target_encode(train["Breed1"], 
                         test["Breed1"], 
                         target=train.AdoptionSpeed, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
train['tencode_breed1'] = trn
test['tencode_breed1'] = sub

trn, sub = target_encode(train["Breed2"], 
                         test["Breed2"], 
                         target=train.AdoptionSpeed, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
train['tencode_breed2'] = trn
test['tencode_breed2'] = sub

trn, sub = target_encode(train["Age"], 
                         test["Age"], 
                         target=train.AdoptionSpeed, 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
train['tencode_Age'] = trn
test['tencode_Age'] = sub

del trn, sub
gc.collect()
######################################################
################################################################################################################
sentimental_analysis = sorted(glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json'))
#Define Empty lists
score=[]
magnitude=[]
petid=[]

for filename in sentimental_analysis:
    with open(filename, 'r') as f:
        sentiment_file = json.load(f)
    file_sentiment = sentiment_file['documentSentiment']
    file_score =  np.asarray(sentiment_file['documentSentiment']['score'])
    file_magnitude =np.asarray(sentiment_file['documentSentiment']['magnitude'])
    score.append(file_score)
    magnitude.append(file_magnitude)
    petid.append(filename.replace('.json','').replace('../input/petfinder-adoption-prediction/train_sentiment/', ''))

# Output with sentiment data for each pet
# Output with sentiment data for each pet
sentimental_analysis = pd.concat([ pd.DataFrame(petid, columns =['PetID']) ,pd.DataFrame(score, columns =['sentiment_document_score']),
                                                pd.DataFrame(magnitude, columns =['sentiment_document_magnitude'])],axis =1)

train = pd.merge(train, sentimental_analysis, how='left', on='PetID')

sentimental_analysis = sorted(glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json'))
#Define Empty lists
score=[]
magnitude=[]
petid=[]

for filename in sentimental_analysis:
    with open(filename, 'r') as f:
        sentiment_file = json.load(f)
    file_sentiment = sentiment_file['documentSentiment']
    file_score =  np.asarray(sentiment_file['documentSentiment']['score'])
    file_magnitude =np.asarray(sentiment_file['documentSentiment']['magnitude'])
    score.append(file_score)
    magnitude.append(file_magnitude)
    petid.append(filename.replace('.json','').replace('../input/petfinder-adoption-prediction/test_sentiment/', ''))

# Output with sentiment data for each pet
# Output with sentiment data for each pet
sentimental_analysis = pd.concat([ pd.DataFrame(petid, columns =['PetID']) ,pd.DataFrame(score, columns =['sentiment_document_score']),
                                                pd.DataFrame(magnitude, columns =['sentiment_document_magnitude'])],axis =1)

test = pd.merge(test, sentimental_analysis, how='left', on='PetID')
del sentimental_analysis, score, magnitude, petid
gc.collect()

train['neg_sentiment'] = 0
train.loc[train[(train['sentiment_document_score'] < 0)].index, 'neg_sentiment'] = 1

test['neg_sentiment'] = 0
test.loc[test[(test['sentiment_document_score'] < 0)].index, 'neg_sentiment'] = 1
###########################################################################################
X_temp = breed_maps(train, test, labels_breed)

keywords = ['urgent', 'lost', 'fast', 'left', 'immediate', 'critical', 'rescued', 'free', 'trained']
val = []
for idx in range(X_temp.shape[0]):
    i = ''
    i = X_temp.loc[idx,'Description']
    if not isinstance(i,float):
        if 'trained' in i: 
            val.append(1)
        elif 'urgent' in i:
            val.append(1)
        elif 'lost' in i:
            val.append(1)
        elif 'fast' in i:
            val.append(1)
        elif 'left' in i:
            val.append(1)
        elif 'immediate' in i:
            val.append(1)
        elif 'rescued' in i:
            val.append(1)
        else:
            val.append(0)
    else:
#         print(i)
        val.append(0)
print(len(val), X_temp.shape)
X_temp['keywords'] = val
del val
gc.collect()

denseNet121 = build_model()

pretrained_feats = image_feature(denseNet121, train, test)

X_temp = X_temp.merge(pretrained_feats, how='left', on='PetID')

X_feat = nlp_features(X_temp)

lgb_submission, lgb_oof_train, lgb_oof_test = run_lgbm(X_feat, test)
lgb_submission.to_csv("submission_lgb.csv", index=None)

from numba import cuda
cuda.close()

xgb_submission, xgb_oof_train, xgb_oof_test = run_xgb(X_feat, test)
xgb_submission.to_csv("submission_xgb.csv", index=None)

# Blend
submission = lgb_submission[['PetID']]
submission['AdoptionSpeed'] = (lgb_submission.AdoptionSpeed + xgb_submission.AdoptionSpeed)/2
submission['AdoptionSpeed'] = submission.AdoptionSpeed.astype(int)

submission.to_csv('submission.csv', index=False)