import os

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.engine.base_layer import Layer
from keras.preprocessing.text import Tokenizer
from keras.saving.save import load_model
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WordAttention(Layer):
    def __init__(self, **kwargs):
        super(WordAttention, self).__init__(**kwargs)
        self.b_w = None
        self.w_h = None

    def build(self, input_shape):
        self.w_h = self.add_weight(name='w_h',
                                   shape=(input_shape[-1],),
                                   initializer='glorot_uniform',
                                   trainable=True)
        self.b_w = self.add_weight(name='b_w',
                                   shape=(input_shape[-1],),
                                   initializer='zeros',
                                   trainable=True)
        super(WordAttention, self).build(input_shape)

    def call(self, h, **kwargs):
        u_it = K.tanh(K.dot(h, K.expand_dims(self.w_h)) + self.b_w)

        alpha_it = K.softmax(u_it, axis=1)

        s_i = K.sum(alpha_it * h, axis=1)

        return s_i

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


input_file = '../Datasets/recombined_data_cleaned_eng.csv'
df = pd.read_csv(input_file, header=None)

X = df.iloc[:, 0]
y = df.iloc[:, 1]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

max_len = 70

sequences_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences_test, maxlen=max_len)

label_encoder = LabelEncoder()
y_test_labels = label_encoder.fit_transform(y_test)
# y_test_labels = 1 - y_test_labels
num_classes = len(np.unique(y_test_labels))
y_test = to_categorical(y_test_labels, num_classes=num_classes)

gru = load_model('../GRU_final/gru_rec_100.h5', custom_objects={'WordAttention': WordAttention})
lstm = load_model('../LSTM_final/lstm_rec_200.h5')
cnn_lstm = load_model('../CNN_LSTM_final/cnn_lstm_rec_100.h5')
cnn = load_model('../CNN_final/cnn_rec_100.h5')
cnn_gru = load_model('../CNN_GRU_final/cnn_gru_rec_100.h5')
ensemble = load_model('../ENSEMBLE_final/ensemble_model_rec_100.h5', custom_objects={'WordAttention': WordAttention})

models = [cnn, gru, lstm, cnn_gru, cnn_lstm, ensemble]

# Analyse performance of all individual models
for model in models:
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}\n")
