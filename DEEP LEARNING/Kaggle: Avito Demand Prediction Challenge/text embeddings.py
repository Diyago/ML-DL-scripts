# @Kmike `s code
# https://github.com/deepmipt/DeepPavlov/blob/a59703de60deda349fc39918a1fc1b242638b7f7/pretrained-vectors.md

from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd

#read embeding
def embeding_reading(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split(' ')[:-1]
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    return embeddings_index

def text2features(embeddings_index, text):
    vec_stack = []
    for w in nltk.word_tokenize(text.lower()):
        v = embeddings_index.get(w, None)
        if v is not None:
            vec_stack.append(v)

    if len(vec_stack) != 0:
        v_mean = np.mean(vec_stack, axis=0)
    else:
        v_mean = np.zeros(300)

    return v_mean
    
def df_to_embed_features(df, column, embeddings_index):
    embed_size = 300
    X = np.zeros((df.shape[0], embed_size), dtype='float32')

    for i, text in tqdm(enumerate(df[column])):
        X[i] = text2features(embeddings_index, text)
    
    return X
       
path = '/mnt/nvme/jupyter/avito/embeding/ft_native_300_ru_wiki_lenta_lower_case.vec'
embeddings_index = embeding_reading(path)
X = df_to_embed_features(test_df, column='title', embeddings_index=embeddings_index)


# 2nd aproach  @artgor aproach
def load_emb(embedding_path, tokenizer, max_features, default=False, embed_size=300):
    """Load embeddings."""

    fasttext_model = FastText.load(embedding_path)
    word_index = tokenizer.word_index
    
    # my pretrained embeddings have different index, so need to add offset.
    if default:
        nb_words = min(max_features, len(word_index))
    else:
        nb_words = min(max_features, len(word_index)) + 2
        
    embedding_matrix = np.zeros((nb_words, embed_size))
    
    for word, i in word_index.items():
        if i >= max_features: continue
        try:
            embedding_vector = fasttext_model[word]
        except KeyError:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

embedding_matrix = nn_functions.load_emb('f:/Avito/embeddings/avito_big_150m_sg1.w2v', tokenizer, max_features, embed_size)
