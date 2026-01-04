import os

import numpy as np
import tensorflow.keras.backend as K
from keras.engine.base_layer import Layer
from keras.preprocessing.text import tokenizer_from_json
from keras.saving.save import load_model
from keras.utils import pad_sequences

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


with open('tokenizer.json') as json_file:
    tokenizer_json = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

max_len = 70

ensemble = load_model('../ENSEMBLE_final/ensemble_model_rec_100.h5', custom_objects={'WordAttention': WordAttention})

while True:
    user_input = input('please enter your text: ')

    sequences_train = tokenizer.texts_to_sequences([user_input])
    X_train = pad_sequences(sequences_train, maxlen=max_len)

    predictions = ensemble.predict(X_train)

    predicted_class = np.argmax(predictions, axis=1)

    if predicted_class == 0:
        print('Not Hate')
    else:
        print('Hate')

    retry = input(f'Re-run test? (y/n)\n')

    if retry == 'n':
        break
