# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Input, Embedding, SpatialDropout1D, concatenate, Merge, Conv1D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
import os
import gc
os.environ['OMP_NUM_THREADS'] = '3'

pd.options.mode.chained_assignment = None

# %% Load data
kaggle_path = '~/hdd/data/avito/'
output_file = 'submits/nn_avito.csv'
embeddings_file = '/home/callum/Coding/deeplearning/data/cc.ru.300.vec'

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
eps = .001
data['price'] = np.log(data['price'] + eps)
data['price'].fillna(-999, inplace=True)
data['image_top_1'].fillna(-999, inplace=True)
data[['param_1', 'param_2', 'param_3']].fillna('missing', inplace=True)
data[['param_1', 'param_2', 'param_3']] = data[['param_1', 'param_2', 'param_3']].astype(str)
data['desc_len'] = data['description'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
data['desc_wc'] = data['description'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
data['title_len'] = data['title'].map(lambda x: len(str(x))).astype(np.float16) #Lenth
data['title_wc'] = data['title'].map(lambda x: len(str(x).split(' '))).astype(np.float16) #Word Count
data[['description', 'title']] = data[['description', 'title']].astype(str)
data['desc_len'] = np.log(data['desc_len'] + eps)
data['desc_len'].fillna(-999, inplace=True)
data['desc_wc'] = np.log(data['desc_wc'] + eps)
data['desc_wc'].fillna(-999, inplace=True)
data['title_len'] = np.log(data['title_len'] + eps)
data['title_len'].fillna(-999, inplace=True)
data['title_wc'] = np.log(data['title_wc'] + eps)
data['title_wc'].fillna(-999, inplace=True)
data['item_seq_number'] = np.log(data['item_seq_number'] + eps)
data['item_seq_number'].fillna(-999, inplace=True)

# %% Process words
word_vec_size = 100
max_word_len = 100
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

print('\nCreating word embeddings...')
print('Description embeddings...')
desc_embs = transformText(data['description'], desc_tokenizer, maxlen=data['desc_wc'].max())
print('Title embeddings...')
title_embs = transformText(data['title'], title_tokenizer, maxlen=data['title_wc'].max())

print('Encoding desc...')
data_desc = desc_tokenizer.texts_to_sequences(data['description'])
data_desc = sequence.pad_sequences(data_desc, maxlen=data['desc_wc'].max())
#print(sequence.pad_sequences(data['description'], maxlen=max_word_len).shape)
print('Encoding title...')
data_title = title_tokenizer.texts_to_sequences(data['title'])
data_title = sequence.pad_sequences(data_title, maxlen=data['title_wc'].max())

print(data_desc[1:3])
#print(data['proc_description'].head())

# %% Encoding
print('\nEncoding cat vars...')
#cat_cols = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type", "image_top_1"]
cat_cols = ["region", "city", "parent_category_name", "category_name", "user_type", "image_top_1", "item_seq_number", "param_1", "param_2", "param_3"]
data[cat_cols] = data[cat_cols].apply(LabelEncoder().fit_transform).astype(np.int32)

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

# %% Split datasets
def getKerasData(dataset, desc=None, title=None):
    X = {
        'region': np.array(dataset.region),
        'city': np.array(dataset.city),
        'pcat': np.array(dataset.parent_category_name),
        'cat': np.array(dataset.category_name),
        'seq': np.array(dataset.item_seq_number),
        'utype': np.array(dataset.user_type),
        'price': np.array(dataset.price),
        'itop1': np.array(dataset.image_top_1),
        'param_1': np.array(dataset.param_1),
        'param_2': np.array(dataset.param_2),
        'param_3': np.array(dataset.param_3),
        'title_len': np.array(dataset.title_len),
        'title_wc': np.array(dataset.title_wc),
        'desc_len': np.array(dataset.desc_len),
        'desc_wc': np.array(dataset.desc_wc),
        'desc': desc,
        'title': title,
    }; return X

test = data.iloc[train_len:].copy()
desc_te = data_desc[train_len:]
title_te = data_title[train_len:]
train = data.iloc[:train_len].copy()
desc_tr = data_desc[:train_len]
title_tr = data_title[:train_len]
y_tr = train.deal_probability.values
del data; del data_desc; del data_title; gc.collect()

# %% Create model
print('Creating model...')

def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

def getModel():
    def emb_depth(max_size): return min(16, int(max_size**.25))
    cont_size = 16

    in_region = Input(shape=[1], name='region')
    emb_region = Embedding(max_region, emb_depth(max_region))(in_region)
    in_city = Input(shape=[1], name='city')
    emb_city = Embedding(max_city, emb_depth(max_city))(in_city)
    in_pcat = Input(shape=[1], name='pcat')
    emb_pcat = Embedding(max_pcat, emb_depth(max_pcat))(in_pcat)
    in_cat = Input(shape=[1], name='cat')
    emb_cat = Embedding(max_cat, emb_depth(max_cat))(in_cat)
    in_utype = Input(shape=[1], name='utype')
    emb_utype = Embedding(max_utype, emb_depth(max_utype))(in_utype)
    in_itop1 = Input(shape=[1], name='itop1')
    emb_itop1 = Embedding(max_itop1, emb_depth(max_itop1))(in_itop1)
    # in_desc = Input(shape=(max_word_len,), name='desc')
    # emb_desc = Embedding(max_word_features+1, word_vec_size, weights=[desc_embs], trainable=False)(in_desc)
    # in_title = Input(shape=(max_word_len,), name='title')
    # emb_title = Embedding(max_word_features+1, word_vec_size, weights=[title_embs], trainable=False)(in_title)
    in_param_1 = Input(shape=[1], name='param_1')
    emb_param_1 = Embedding(max_param_1, emb_depth(max_param_1))(in_param_1)
    in_param_2 = Input(shape=[1], name='param_2')
    emb_param_2 = Embedding(max_param_2, emb_depth(max_param_2))(in_param_2)
    in_param_3 = Input(shape=[1], name='param_3')
    emb_param_3 = Embedding(max_param_3, emb_depth(max_param_3))(in_param_3)

    in_price = Input(shape=[1], name='price')
    emb_price = Dense(cont_size, activation='tanh')(in_price)
    in_title_len = Input(shape=[1], name='title_len')
    emb_title_len = Dense(cont_size, activation='tanh')(in_title_len)
    in_title_wc = Input(shape=[1], name='title_wc')
    emb_title_wc = Dense(cont_size, activation='tanh')(in_title_wc)
    in_desc_len = Input(shape=[1], name='desc_len')
    emb_desc_len = Dense(cont_size, activation='tanh')(in_desc_len)
    in_desc_wc = Input(shape=[1], name='desc_wc')
    emb_desc_wc = Dense(cont_size, activation='tanh')(in_desc_wc)
    in_seq = Input(shape=[1], name='seq')
    emb_seq = Dense(cont_size, activation='tanh')(in_seq)

    inps = [in_region, in_city, in_pcat, in_cat, in_utype, in_itop1, in_seq, in_param_1, in_param_2, in_param_3,
            in_price, in_title_len, in_title_wc, in_desc_len, in_desc_wc,
            #in_desc, in_title
            ]
    cat_embs = concatenate([ (emb_region), (emb_city), (emb_pcat), (emb_cat), (emb_utype),
                             (emb_itop1),  (emb_param_1), (emb_param_2), (emb_param_3)
                            ])
    cont_embs = concatenate([ (emb_price), (emb_title_len), (emb_title_wc), (emb_desc_len),
                              (emb_desc_wc), (emb_seq)
                             ])
    #nums = [(in_price), (in_title_len), (in_title_wc), (in_desc_len), (in_desc_wc)]
    cat_dout = Flatten()(SpatialDropout1D(.4)(cat_embs))
    cont_dout = Dropout(.4)(cont_embs)

    #descConv = Conv1D(100, kernel_size=10, strides=1, padding="same")(emb_desc)
    #titleConv = Conv1D(100, kernel_size=10, strides=1, padding="same")(emb_title)
    #convs = Flatten()( concatenate([ (descConv), (titleConv) ]) )

    x = concatenate([(cat_dout), (cont_dout)])
    # x = concatenate([(s_dout), (convs), *nums])
    x = Dropout(.4)(Dense(256, activation='relu')(x))
    #x = BatchNormalization()(x)
    x = Dropout(.4)(Dense(64, activation='relu')(x))
    #x = BatchNormalization()(x)
    out = Dense(1, activation='sigmoid')(x)


    model = Model(inputs=inps, outputs=out)

    from keras import backend as K

    opt = Adam(lr=3e-3, decay=5e-5)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=[root_mean_squared_error])
    return model

# %% Train model
print('\nTraining...')

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=218)
models = []
cv_tr = np.zeros((len(y_tr), 1))

for i, (train_idx, valid_idx) in enumerate(kfold.split(train[cat_cols], np.round(y_tr))):
    print('\nTraining model #{}'.format(i+1))
    X_valid = getKerasData(train.iloc[valid_idx])#, desc_tr[valid_idx], title_tr[valid_idx])
    X_train = getKerasData(train.iloc[train_idx])#, desc_tr[train_idx], title_tr[train_idx])
    y_valid = train.iloc[valid_idx].deal_probability
    y_train = train.iloc[train_idx].deal_probability
    model = getModel()
    model.fit(X_train, y_train, batch_size=4000, validation_data=(X_valid, y_valid), epochs=10, verbose=1)
    cv_tr[valid_idx] = model.predict(X_valid, batch_size=4000)
    models.append(model)

# Fold RMSE: 0.23286290473878124

print('\nFold RMSE: {}'.format(rmse(y_tr, cv_tr)))

# %% Predict
preds = np.zeros((len(test), 1))
for model in models:
    preds += model.predict(getKerasData(test, desc_te, title_te), batch_size=4000)

submit['deal_probability'] = preds / len(models)
print(submit.head())

submit.to_csv(output_file, index=False)
print('\nDone!')
