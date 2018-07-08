# @kmike `s code

# image feature extractions

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import traceback
from functools import partial


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 3 CPU
# 2 1060
# 0 1080Ti
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.preprocessing import image
from keras.applications import vgg16, vgg19, resnet50, inception_v3, inception_resnet_v2, xception, mobilenet, densenet

import config as cfg
import utils


@hdd_memory.cache
def get_names_paths(images_dir):
    img_names = sorted(os.listdir(images_dir))
    img_paths = [os.path.join(images_dir, name) for name in img_names]
    img_names = [name.split('.', 1)[0] for name in img_names]

    return img_paths, img_names


def img_generator(img_paths, img_names, target_size, preprocess_func=None, batch_size=128):
    pos = 0
    preprocess_func = preprocess_func or (lambda arg: arg)

    while True:
        try:
            names_batch = img_names[pos:pos + batch_size]
            path_batch = img_paths[pos:pos + batch_size]
            x = [image.img_to_array(image.load_img(path, target_size=target_size)) for path in path_batch]
            x = np.array(x)
            x = preprocess_func(x)

            yield pos, x, names_batch

        except Exception:
            traceback.print_exc()

        finally:
            pos += batch_size
            if pos >= len(img_paths):
                break


def get_model_and_data(mode, model_name):
    if model_name == 'vgg16':
        model = vgg16.VGG16(weights='imagenet', include_top=True)
        preprocess_func = partial(vgg16.preprocess_input, mode='tf')
        target_size = (224, 224)

    elif model_name == 'vgg19':
        model = vgg19.VGG19(weights='imagenet', include_top=True)
        preprocess_func = partial(vgg19.preprocess_input, mode='tf')
        target_size = (224, 224)

    elif model_name == 'resnet50':
        model = resnet50.ResNet50(weights='imagenet', include_top=True)
        preprocess_func = partial(resnet50.preprocess_input, mode='tf')
        target_size = (224, 224)

    elif model_name == 'inception_v3':
        model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
        preprocess_func = inception_v3.preprocess_input
        target_size = (299, 299)

    elif model_name == 'inception_resnet_v2':
        model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=True)
        preprocess_func = inception_resnet_v2.preprocess_input
        target_size = (299, 299)

    elif model_name == 'xception':
        model = xception.Xception(weights='imagenet', include_top=True)
        preprocess_func = xception.preprocess_input
        target_size = (299, 299)

    elif model_name == 'mobilenet':
        model = mobilenet.MobileNet(weights='imagenet', include_top=True)
        preprocess_func = mobilenet.preprocess_input
        target_size = (224, 224)

    elif model_name.startswith('densenet'):
        model_type = int(model_name[len('densenet'):])

        if model_type == 121:
            model = densenet.DenseNet121(weights='imagenet', include_top=True)
        elif model_type == 169:
            model = densenet.DenseNet169(weights='imagenet', include_top=True)
        elif model_type == 201:
            model = densenet.DenseNet201(weights='imagenet', include_top=True)
        else:
            raise ValueError(f'Got incorrect DenseNet model type ({model_type}).')

        preprocess_func = densenet.preprocess_input
        target_size = (224, 224)

    else:
        raise ValueError(f'Got unknown NN model ({model_name}).')

    if mode == 'train':
        input_dir = cfg.TRAIN_IMAGES_DIR

    elif mode == 'test':
        input_dir = cfg.TEST_IMAGES_DIR
    else:
        raise ValueError(f'Got unknown proc mode ({mode}).')

    output_file = cfg.NN_IMAGE_FEATURES[model_name][mode]['memmap']

    return model, input_dir, output_file, preprocess_func, target_size


def extract_features(model_name='vgg16', batch_size=64):

    def proc_dataset(mode):
        # create appropriate model and in/out data paths
        model, input_dir, output_file, preprocess_func, target_size = get_model_and_data(mode, model_name)

        # run image features extraction
        img_paths, img_names = get_names_paths(input_dir)

        gen = img_generator(img_paths, img_names, target_size, preprocess_func=preprocess_func, batch_size=batch_size)

        features = np.memmap(output_file, dtype='float32', mode='w+', shape=(len(img_paths), 1000))

        for pos, x_test, names in tqdm(gen, total=len(img_paths) // batch_size + 1):
            features[pos:pos + batch_size, :] = model.predict(x_test)
            features.flush()

        del features

    proc_dataset('train')
    proc_dataset('test')


def create_features_df(model_name='vgg16', mode='train'):
    if mode == 'train':
        images_dir_path = cfg.TRAIN_IMAGES_DIR
    elif mode == 'test':
        images_dir_path = cfg.TEST_IMAGES_DIR
    else:
        raise ValueError(f'Got unknown proc mode ({mode}).')

    features_file = cfg.NN_IMAGE_FEATURES[model_name][mode]['memmap']
    df_file = cfg.NN_IMAGE_FEATURES[model_name][mode]['df']

    df = pd.DataFrame()
    img_paths, img_names = get_names_paths(images_dir_path)
    df['image'] = img_names

    features = np.memmap(features_file, dtype='float32', mode='r', shape=(df.shape[0], 1000))
    for idx in tqdm(range(1000)):
        df[f'{model_name}_{idx}'] = features[:, idx]

    df.to_pickle(df_file, compression='gzip')
