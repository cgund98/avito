# -*- coding: utf-8 -*-

# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Merge, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, GaussianDropout
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from string import punctuation
punct = set(punctuation)
import os
import gc
import re
os.environ['OMP_NUM_THREADS'] = '8'

pd.options.mode.chained_assignment = None

# %% Load data
kaggle_path = '~/hdd/data/avito/'
output_file = 'submits/nn_avito.csv'
embeddings_file = '/home/callum/Coding/deeplearning/data/cc.ru.300.vec'
save_text = False
load_text = True
save_path = 'data/'

print('\nLoading data...\n')
train = pd.read_csv(kaggle_path + 'train.csv', parse_dates=['activation_date'])
y_tr = train.deal_probability.values
test = pd.read_csv(kaggle_path + 'test.csv', parse_dates=['activation_date'])
submit = pd.read_csv(kaggle_path + 'sample_submission.csv')
train_len = len(train)
data = pd.concat([train, test], axis=0)

# Add aggregated features
print('Adding aggregated features...')
agg = pd.read_csv(kaggle_path + 'aggregated_features.csv')
data = data.merge(agg, how='left', on=['user_id'])
del agg; gc.collect()

print('Train Length: {} \nTest Length: {} \n'.format(train_len, len(test)))

# %% Columns
#print('Columns:\n', data.columns.values)
data.head()

# %% Cleaning
data[['param_1', 'param_2', 'param_3']].fillna('missing', inplace=True)
data[['param_1', 'param_2', 'param_3']] = data[['param_1', 'param_2', 'param_3']].astype(str)

for s in data.description.astype(str):
    for c in s:
        if not c.isalpha() and not c.isdigit():
            punct.add(c)

for s in data.title.astype(str):
    for c in s:
        if not c.isalpha() and not c.isdigit():
            punct.add(c)

def clean_text(s):
    s = re.sub('м²|\d+\\/\d|\d+-к|\d+к', ' ', s.lower())
    s = ''.join([' ' if c in punct or c.isdigit() else c for c in s])
    s = re.sub('\\s+', ' ', s)
    s = s.strip()
    return s

print('Cleaning text...')
data['title'] = data.title.fillna('').astype(str).apply(clean_text)
data['description'] = data.description.fillna('').astype(str).apply(clean_text)

print('Creating more text features...')
for col in ['description', 'title']:
  data[col + '_len'] = data[col].map(lambda x: len(str(x))).astype(np.float16) #Lenth
  data[col + '_wc'] = data[col].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
  data[col + '_num_unique_words'] = data[col].apply(lambda comment: len(set(w for w in comment.split())))
  data[col + '_words_vs_unique'] = data[col + '_num_unique_words'] / data[col + '_wc'] * 100 # Count Unique Words

print('Concatting params...')
data['params'] = data.apply(lambda row: ' '.join([
    str(row['param_1']),
    str(row['param_2']),
    str(row['param_3'])]),axis=1)

# %% Process words
word_vec_size = 300
max_word_features = 100000
desc_tokenizer = text.Tokenizer(num_words=max_word_features)
title_tokenizer = text.Tokenizer(num_words=max_word_features)

def transformText(text_df, tokenizer, maxlen=100):
    max_features = max_word_features
    embed_size = word_vec_size
    X_text = text_df.astype(str).fillna('NA')
    tokenizer.fit_on_texts(list(X_text))
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embeddings_file))

    word_index = tokenizer.word_index
    print('Word index len:', len(word_index))
    nb_words = min(max_features, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix

if not load_text:
    print('\nCreating word embeddings...')
    print('Description embeddings...')
    desc_embs = transformText(data['description'], desc_tokenizer, maxlen=data['desc_wc'].max())
    print('Title embeddings...')
    title_embs = transformText(data['title'], title_tokenizer, maxlen=data['title_wc'].max())

    print('Encoding desc...')
    data_desc = desc_tokenizer.texts_to_sequences(data['description'])
    data_desc = sequence.pad_sequences(data_desc, maxlen=100)
    print('Encoding title...')
    data_title = title_tokenizer.texts_to_sequences(data['title'])
    data_title = sequence.pad_sequences(data_title, maxlen=30)

    print(data_title[1:3])

# %% Normalize data
eps = .00001
data['image_top_1'].fillna(-999, inplace=True)
data['image'].loc[data.image.notnull()] = 1; data['image'].loc[data.image.isnull()] = 0

features = [
    'region', 'city', 'parent_category_name', 'category_name', 'user_type', 'image_top_1', 'params', 'param_1',
    'price', 'title_len', 'title_wc', 'description_len', 'description_wc', 'item_seq_number',
    'description_num_unique_words', 'description_words_vs_unique', 'title_num_unique_words', 'title_words_vs_unique',
    'avg_days_up_user', 'avg_times_up_user', 'n_user_items'
]
cat_cols = ["region", "city", "parent_category_name", "category_name", "user_type", "image_top_1", "params", "param_1" ]
cont_cols = [col for col in features if col not in cat_cols]

for col in cont_cols:
  data[col] = np.log(data[col] + eps); data[col].fillna(data[col].mean(), inplace=True)

# %% Encoding
print('\nEncoding cat vars...')
data[cat_cols] = data[cat_cols].apply(LabelEncoder().fit_transform).astype(np.int32)

# Delete unused cols
for col in [col for col in data.columns.values if col not in features]:
    data.drop([col], inplace=True, axis=1)

# Assign max values for embedding

maxes = {}
for col in cat_cols:
    maxes[col] = data[col].max() + 1

def emb_depth(max_size): return min(16, int(max_size**.33))

cat_szs = {}
for col in cat_cols:
    cat_szs[col] = ( maxes[col], emb_depth(maxes[col]) )

# %% Split datasets
def getKerasData(dataset, desc=None, title=None):
    X = {
        'desc': desc,
        'title': title,
    }
    for col in cat_cols + cont_cols:
        X[col] = np.array(dataset[col])
    return X

test = data.iloc[train_len:].copy()
train = data.iloc[:train_len].copy()
del data; gc.collect()

if not load_text: # Splitting/loading text data
    desc_te = data_desc[train_len:]
    title_te = data_title[train_len:]
    desc_tr = data_desc[:train_len]
    title_tr = data_title[:train_len]
    del data_desc; del data_title; gc.collect()
else:
    print('Loading text...')
    desc_te =np.load(save_path + 'fasttext_desc_te.npy')
    title_te = np.load(save_path + 'fasttext_title_te.npy')
    desc_tr = np.load(save_path + 'fasttext_desc_tr.npy')
    title_tr = np.load(save_path + 'fasttext_title_tr.npy')
    desc_embs = np.load(save_path + 'fasttext_desc_embs.npy')
    title_embs = np.load(save_path + 'fasttext_title_embs.npy')

if save_text: # Save text data
    print('Saving text...')
    np.save(save_path + 'fasttext_desc_tr', desc_tr)
    np.save(save_path + 'fasttext_title_tr', title_tr)
    np.save(save_path + 'fasttext_desc_te', desc_te)
    np.save(save_path + 'fasttext_title_te', title_te)
    np.save(save_path + 'fasttext_desc_embs', desc_embs)
    np.save(save_path + 'fasttext_title_embs', title_embs)

# %% Define model
def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

def getModel():
    cont_size = 16

    in_desc = Input(shape=(100,), name='desc')
    emb_desc = SpatialDropout1D(.2)( Embedding(max_word_features+1, word_vec_size, weights=[desc_embs], trainable=False)(in_desc) )
    in_title = Input(shape=(30,), name='title')
    emb_title = SpatialDropout1D(.2)( Embedding(max_word_features+1, word_vec_size, weights=[title_embs], trainable=False)(in_title) )

    inps = [in_desc, in_title]

    cat_embs = []
    for idx, col in enumerate(cat_cols):
        #x = Lambda(lambda x: x[:, idx, None])(cat_inp)
        inp = Input(shape=[1], name=col)
        x = Embedding(cat_szs[col][0], cat_szs[col][1], input_length=1)(inp)
        cat_embs.append((x))
        inps.append(inp)
    cat_embs = concatenate(cat_embs)

    cont_embs = []
    for idx, col in enumerate(cont_cols):
        #x = Lambda(lambda x: x[:, idx, None])(cont_inp)
        inp = Input(shape=[1], name=col)
        x = Dense(cont_size, activation='tanh')(inp)
        cont_embs.append((x))
        inps.append(inp)
    cont_embs = concatenate(cont_embs)

    cat_dout = Flatten()(SpatialDropout1D(.4)(cat_embs))
    cont_dout = Dropout(.2)(cont_embs)

    descConv = Conv1D(100, kernel_size=3, strides=1, padding="same")(emb_desc)
    descGAP = GlobalAveragePooling1D()( descConv )
    descGMP = GlobalMaxPooling1D()( descConv )

    titleConv = Conv1D(32, kernel_size=3, strides=1, padding="same")(emb_title)
    titleGAP = GlobalAveragePooling1D()( titleConv )
    titleGMP = GlobalMaxPooling1D()( titleConv )
    convs = ( concatenate([ (descGAP), (descGMP), (titleGAP), (titleGMP) ]) )

    x = concatenate([(cat_dout), (cont_dout)])
    x = Dropout(.4)(Dense(512, activation='relu')(x))
    #x = BatchNormalization()(x)
    x = Dropout(.4)(Dense(64, activation='relu')(x))
    x = concatenate([x, (convs)])
    #x = BatchNormalization()(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inps, outputs=out)

    from keras import backend as K

    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    steps = int(len(y_tr)/bs) * epochs
    lr_init, lr_fin = 5e-3, 1e-3
    lr_decay = exp_decay(lr_init, lr_fin, steps)

    opt = Adam(lr=lr_init, decay=lr_decay)
    model.compile(optimizer=opt, loss=root_mean_squared_error)
    return model

# %% Train model
print('\nTraining...')

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=218)
models = []
cv_tr = np.zeros((len(y_tr), 1))

bs=512*3
epochs = 6

for i, (train_idx, valid_idx) in enumerate(kfold.split(train[cat_cols], np.round(y_tr))):
    print('\nTraining model #{}'.format(i+1))
    X_valid = getKerasData(train.iloc[valid_idx], desc_tr[valid_idx], title_tr[valid_idx])
    X_train = getKerasData(train.iloc[train_idx], desc_tr[train_idx], title_tr[train_idx])
    y_valid = y_tr[valid_idx]
    y_train = y_tr[train_idx]
    model = getModel()
    model.fit(X_train, y_train, batch_size=bs, validation_data=(X_valid, y_valid), epochs=epochs, verbose=1)
    cv_tr[valid_idx] = model.predict(X_valid, batch_size=bs)
    models.append(model)

print('\nFold RMSE: {}'.format(rmse(y_tr, cv_tr)))
#Fold RMSE: 0.22594231705501946 < With 3-wide conv
#Fold RMSE: 0.226058889143819 < With 3-wide conv and all-1 params feature
#Fold RMSE: 0.22571788218558694 < Added GMP
#Fold RMSE: 0.22560138317572037 < desc conv up to 100 filters
#Fold RMSE: 0.22466267000045032 < batch size to 512*3
#Fold RMSE: 0.2245272222547689 < Added new text features and param_1
#Fold RMSE: 0.2244001706665085 < Added another epoch
#Fold RMSE: 0.2243065138497809 < Upped 1st Dense layer from 256 to 512
#Fold RMSE: 0.22408247760573333 < fill cont. cols NA with mean
#Fold RMSE: 0.22288079052919876 < Added aggregated features
#Fold RMSE: 0.222707960 < Added exponential decay LR

# %% Predict
preds = np.zeros((len(test), 1))
for model in models:
    preds += model.predict(getKerasData(test, desc_te, title_te), batch_size=bs)

submit['deal_probability'] = preds / len(models)
print(submit.head())

submit.to_csv('nn/' + output_file, index=False)
print('\nSaved: ' + output_file + '!')
