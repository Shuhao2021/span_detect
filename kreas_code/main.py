import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense
import os
import codecs
import pickle
from keras import backend as K

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
train = train.fillna(" ")
test = test.fillna(" ")
dev_size = int(train.shape[0]*0.2)
dev = train[:dev_size]
train = train[dev_size:]

X_train = train['subject'] + ' ' + train['email']
y_train = train['spam']
X_dev = dev['subject'] + ' ' + dev['email']
y_dev =dev['spam']
X_test = test['subject'] + ' ' + test['email']

def get_word_embed(word2ix, max_words): # get embedding matrix
    row = 0
    file = 'glove.6B.100d.txt'
    whole = os.path.join('../data', 'glove', file)
    words_embed = {}
    with open(whole, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1
    num_words = min(max_words,len(word2ix) + 1)
    embedding_matrix = np.zeros((num_words,100))
    for word,i in word2ix.items():
        if i >= max_words:
            continue
        embedding_vector = words_embed.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


max_words = 300
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(list(X_train)+list(X_test)) # tokenizer
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_dev_tokens = tokenizer.texts_to_sequences(X_dev)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# pad
maxlen = 100
from keras.preprocessing import sequence
X_train_tokens_pad = sequence.pad_sequences(X_train_tokens, maxlen=maxlen,padding='post')
X_dev_tokens_pad = sequence.pad_sequences(X_dev_tokens, maxlen=maxlen,padding='post')
X_test_tokens_pad = sequence.pad_sequences(X_test_tokens, maxlen=maxlen,padding='post')

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r+K.epsilon()))

class LossHistory(keras.callbacks.Callback): # metrics
    def on_train_begin(self, logs={}):
        self.record = {'train_loss': [], 'valid_loss': [], 'f1':[], 'recall':[], 'precision':[]}
 
    def on_epoch_end(self, batch, logs={}):
        self.record['train_loss'].append(logs.get('loss'))
        self.record['valid_loss'].append(logs.get('val_loss'))
        self.record['f1'].append(logs.get('f1'))
        self.record['precision'].append(logs.get('precision'))
        self.record['recall'].append(logs.get('recall'))
    

history = LossHistory()
weights = get_word_embed(tokenizer.word_index, max_words)
embeddings_dim = 100 
model = Sequential()
model.add(Embedding(input_dim=max_words, # Size of the vocabulary
                    output_dim=embeddings_dim,
                    weights = [weights],
                    input_length=maxlen))
model.add(LSTM(units=64)) 
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[f1, precision, recall]) 
model.fit(X_train_tokens_pad, y_train, validation_data=(X_dev_tokens_pad, y_dev),
                    batch_size=128, epochs=10, callbacks =[history])
model.save("email_cat_lstm.h5")

dir_name = os.path.join('../output', 'result', 'result_lstm.txt')
print(history.record)
with open(dir_name,'wb') as f:
    pickle.dump(history.record, f)

pred_prob = model.predict(X_test_tokens_pad).squeeze()
pred_class = np.asarray(pred_prob > 0.5).astype(np.int32)
id = test['id']
output = pd.DataFrame({'id':id, 'Class': pred_class})
output.to_csv("submission_gru.csv",  index=False)