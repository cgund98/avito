# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Merge, Conv1D, GlobalAveragePooling1D, GaussianDropout
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from string import punctuation
punct = set(punctuation)
import os
import gc
import re
os.environ['OMP_NUM_THREADS'] = '3'

pd.options.mode.chained_assignment = None

# %% Load data
kaggle_path = '/home/callum/.kaggle/competitions/avito-demand-prediction/'
output_file = 'submits/nn_avito.csv'
embeddings_file = '/home/callum/.kaggle/competitions/avito-demand-prediction/cc.ru.300.vec'
save_text = False
load_text = True
save_path = 'data/'

print('\nLoading data...\n')
train = pd.read_csv(kaggle_path + 'train.csv', parse_dates=['activation_date'])
test = pd.read_csv(kaggle_path + 'test.csv', parse_dates=['activation_date'])
submit = pd.read_csv(kaggle_path + 'sample_submission.csv')
train_len = len(train)
data = pd.concat([train, test], axis=0)

print('Train Length: {} \nTest Length: {} \n'.format(train_len, len(test)))
# %% Info
print('Columns:\n', data.columns.values)

# %% Preprocess
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

#print('clean text')
data['title'] = data.title.fillna('').astype(str).apply(clean_text)
data['description'] = data.description.fillna('').astype(str).apply(clean_text)

data['desc_len'] = data['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
data['desc_wc'] = data['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
data['title_len'] = data['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
data['title_wc'] = data['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count

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
    #X_text = tokenizer.texts_to_sequences(X_text)
    #X_text = sequence.pad_sequences(X_text, maxlen=maxlen)
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
    #print(sequence.pad_sequences(data['description'], maxlen=max_word_len).shape)
    print('Encoding title...')
    data_title = title_tokenizer.texts_to_sequences(data['title'])
    data_title = sequence.pad_sequences(data_title, maxlen=30)

    print(data_title[1:3])

#print(data['proc_description'].head())

# %% Normalize data
eps = .00001
data['price'] = np.log(data['price'] + eps); data['price'].fillna(-999, inplace=True)
data['image_top_1'].fillna(-999, inplace=True)
data['desc_len'] = np.log(data['desc_len'] + eps); data['desc_len'].fillna(-999, inplace=True)
data['desc_wc'] = np.log(data['desc_wc'] + eps); data['desc_wc'].fillna(-999, inplace=True)
data['title_len'] = np.log(data['title_len'] + eps); data['title_len'].fillna(-999, inplace=True)
data['title_wc'] = np.log(data['title_wc'] + eps); data['title_wc'].fillna(-999, inplace=True)
data['item_seq_number'] = np.log(data['item_seq_number'] + eps); data['item_seq_number'].fillna(-999, inplace=True)
data['image'].loc[data.image.notnull()] = 1; data['image'].loc[data.image.isnull()] = 0

# %% Encoding
print('\nEncoding cat vars...')
cat_cols_old = ["region", "city", "parent_category_name", "category_name", "user_type", "image_top_1", "param_1", "param_2", "param_3", ]#"item_seq_number"]
data[cat_cols_old] = data[cat_cols_old].apply(LabelEncoder().fit_transform).astype(np.int32)

# Assign max values for embedding
max_region = data['region'].max() + 1
max_city = data['city'].max() + 1
max_pcat = data['parent_category_name'].max() + 1
max_cat = data['category_name'].max() + 1
max_seq = data['item_seq_number'].max() + 1
max_utype = data['user_type'].max() + 1
max_itop1 = data['image_top_1'].max() + 1
max_param_1 = data['param_1'].max() + 1
max_param_2 = data['param_2'].max() + 1
max_param_3 = data['param_3'].max() + 1

features = ['category_name', 'city', 'desc_len', 'desc_wc', 'title_len', 'title_wc',
            #'image',
            'image_top_1', 'item_seq_number', 'param_1', 'param_2',
            'param_3', 'parent_category_name', 'price', 'region','user_type', ]
cont_cols = [col for col in features if col not in cat_cols]

def emb_depth(max_size): return min(16, int(max_size**.33))

cat_szs = {
    "region": (max_region, emb_depth(max_region)),
    "city": (max_city, emb_depth(max_city)),
    "parent_category_name": (max_pcat, emb_depth(max_pcat)),
    "category_name": (max_cat, emb_depth(max_cat)),
    "user_type": (max_utype, emb_depth(max_utype)),
    "image_top_1": (max_itop1, emb_depth(max_itop1)),
    "param_1": (max_param_1, emb_depth(max_param_1)),
    "param_2": (max_param_2, emb_depth(max_param_2)),
    "param_3": (max_param_3, emb_depth(max_param_3)),
    "image": (2, 2)
}

# %% Split datasets
# def getKerasData(dataset, desc=None, title=None):
#     X = {
#         'cat': np.array(dataset[cat_cols]),
#         'cont': np.array(dataset[cont_cols]),
#         'desc': desc,
#         'title': title,
#     }; return X

def getKerasData(dataset, desc=None, title=None):
    X = {
        'cat': np.array(dataset[cat_cols]),
        'cont': np.array(dataset[cont_cols]),
        'desc': desc,
        'title': title,
    }; return X

cat_cols = ['region', 'city', 'pcat', 'cat', 'utype', 'itop1', 'param_1', 'param_2', 'param_3']
cont_cols = ['price', 'title_len', 'title_wc', 'desc_len', 'desc_wc', 'seq']

test = data.iloc[train_len:].copy()
train = data.iloc[:train_len].copy()
y_tr = train.deal_probability.values
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

# %% Create model
print('Creating model...')

def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

def getModel(): # Model for making the NN
    cont_size = 16

    cat_inp = Input(shape=(len(cat_cols),), name='cat')
    cat_embs = []
    for idx, col in enumerate(cat_cols):
        x = Lambda(lambda x: x[:, idx, None])(cat_inp)
        x = Embedding(cat_szs[col][0], cat_szs[col][1], input_length=1)(x)
        cat_embs.append(x)
    cat_embs = concatenate(cat_embs)
    cat_embs = SpatialDropout1D(.4)(cat_embs)

    cont_inp = Input(shape=(len(cont_cols),), name='cont')
    cont_embs = []
    for idx, col in enumerate(cont_cols):
        x = Lambda(lambda x: x[:, idx, None])(cont_inp)
        x = Dense(cont_size, activation='tanh')(x)
        cont_embs.append(x)
    cont_embs = concatenate(cont_embs)
    cont_embs = Dropout(.4)(cat_embs)

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
    cont_dout = Dropout(.4)(cont_embs)

    descConv = GlobalAveragePooling1D()( Conv1D(64, kernel_size=7, strides=1, padding="same")(emb_desc) )
    titleConv = GlobalAveragePooling1D()( Conv1D(32, kernel_size=7, strides=1, padding="same")(emb_title) )
    convs = ( concatenate([ (descConv), (titleConv) ]) )

    x = concatenate([(cat_dout), (cont_dout)])
    x = Dropout(.4)(Dense(256, activation='relu')(x))
    #x = BatchNormalization()(x)
    x = Dropout(.4)(Dense(64, activation='relu')(x))
    x = Flatten()(x)
    x = concatenate([x, (convs)])
    #x = BatchNormalization()(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=out)

    from keras import backend as K

    opt = Adam(lr=2e-3,)
    model.compile(optimizer=opt, loss=root_mean_squared_error)
    return model

# %% Train model
print('\nTraining...')

print(cat_cols)
print(cont_cols)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=218)
models = []
cv_tr = np.zeros((len(y_tr), 1))

bs=4000

for i, (train_idx, valid_idx) in enumerate(kfold.split(train[cat_cols_old], np.round(y_tr))):
    print('\nTraining model #{}'.format(i+1))
    X_valid = getKerasData(train.iloc[valid_idx], desc_tr[valid_idx], title_tr[valid_idx])
    X_train = getKerasData(train.iloc[train_idx], desc_tr[train_idx], title_tr[train_idx])
    y_valid = train.iloc[valid_idx].deal_probability
    y_train = train.iloc[train_idx].deal_probability
    model = getModel()
    model.fit(X_train, y_train, batch_size=bs, validation_data=(X_valid, y_valid), epochs=3, verbose=1)
    for layer in model.layers[:len(cat_cols)*2]: # Freeze cat embedding layers
        layer.trainable = False
    model.fit(X_train, y_train, batch_size=bs, validation_data=(X_valid, y_valid), epochs=7, verbose=1)
    cv_tr[valid_idx] = model.predict(X_valid, batch_size=bs)
    models.append(model)

print('\nFold RMSE: {}'.format(rmse(y_tr, cv_tr)))

# %% Predict
preds = np.zeros((len(test), 1))
for model in models: # Make predictions for each model
    preds += model.predict(getKerasData(test, desc_te, title_te), batch_size=bs)

submit['deal_probability'] = preds / len(models) # Average predictions of each model
print(submit.head())

submit.to_csv('nn/'+output_file, index=False)
print('\nSaved: ' + output_file + '!')
