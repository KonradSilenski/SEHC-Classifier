import os
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.engine.base_layer import Layer
from keras.saving.save import load_model
from keras_preprocessing.text import tokenizer_from_json

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

hate_count = 0
nothate_count = 0

input_file = '../Datasets/cleaned_dataset_binary.csv'
df = pd.read_csv(input_file, header=None, skiprows=1)

output_file = 'predictions.csv'
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write("Text,Prediction\n")

    for index, row in df.iterrows():
        text = row[0]

        tokenized_text = tokenizer.texts_to_sequences([text])
        tokenized_text = tokenized_text[0]

        if len(tokenized_text) > max_len:
            tokenized_text = tokenized_text[:max_len]

        padded_text = np.pad(tokenized_text, (0, max_len - len(tokenized_text)), mode='constant')

        padded_text = np.expand_dims(padded_text, axis=0)

        prediction = ensemble.predict(np.array(padded_text))
        predicted_label = np.argmax(prediction, axis=1)[0]

        if predicted_label == 1:
            hate_count += 1
        else:
            nothate_count += 1

        if predicted_label == 0:
            predicted_label = "Not Hate"
        else:
            predicted_label = "Offensive or Hate Speech"

        f_out.write(f"{text},{predicted_label}\n")

print(f"Hate predictions: {hate_count}")
print(f"Not Hate predictions: {nothate_count}")
