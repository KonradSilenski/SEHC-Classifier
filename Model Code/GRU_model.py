import pickle

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import zeros
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Layer

# Load the dataset and split it into training, validation and test sets
input_file = '../Datasets/recombined_data_cleaned_eng.csv'
df = pd.read_csv(input_file, header=None)

X = df.iloc[:, 0]
y = df.iloc[:, 1]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit the tokenizer on the training set and tokenize the sentences for all sets
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)

# Set the word index length
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_valid = tokenizer.texts_to_sequences(X_val)
sequences_test = tokenizer.texts_to_sequences(X_test)

# Set the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocab size: ", vocab_size)

max_len = 70  # Maximum sentence length
out_dim = 100  # Output dimension size

# Pad the sentences
X_train = pad_sequences(sequences_train, maxlen=max_len)
X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1])
X_test = pad_sequences(sequences_test, maxlen=max_len)

# Two-hot encode the labels
label_encoder = LabelEncoder()
y_train_labels = label_encoder.fit_transform(y_train)
y_val_labels = label_encoder.transform(y_val)
y_test_labels = label_encoder.transform(y_test)
num_classes = len(np.unique(y_train_labels))
y_train = to_categorical(y_train_labels, num_classes=num_classes)
y_val = to_categorical(y_val_labels, num_classes=num_classes)
y_test = to_categorical(y_test_labels, num_classes=num_classes)

for index, label in enumerate(label_encoder.classes_):
    print(f"Class {index}: {label}")

print('Shape of X train and X validation tensor:', X_train.shape, X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape, y_val.shape)

# Load the GloVe dictionary and transform to matrix
embeddings_dictionary = dict()
glove_file = open('../GloVe/glove.twitter.27B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, out_dim))
print("matrix: ", embedding_matrix.shape)
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


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

    def call(self, h):
        u_it = K.tanh(K.dot(h, K.expand_dims(self.w_h)) + self.b_w)

        alpha_it = K.softmax(u_it, axis=1)

        s_i = K.sum(alpha_it * h, axis=1)

        return s_i

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


input_layer = Input(shape=(max_len,))

embedding_layer = Embedding(vocab_size, out_dim, weights=[embedding_matrix], trainable=False)(input_layer)

bi_gru = Bidirectional(GRU(256, return_sequences=True))(embedding_layer)

gru = GRU(128, return_sequences=True)(bi_gru)

attention_output = WordAttention()(gru)

dense_1 = Dense(100, activation='relu')(attention_output)
dense_2 = Dense(100, activation='relu')(dense_1)

output = Dense(2, activation='softmax')(dense_2)

gru = Model(inputs=input_layer, outputs=output, name='gru')

gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

gru.summary()

gru_model_history = gru.fit(X_train, y_train, batch_size=64, epochs=6, verbose=2,
                            validation_data=(X_val, y_val))

# Test the model on the test set
score = gru.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

gru.save('../GRU_final/gru_rec_100.h5')

with open('../GRU_final/gru_rec_100_history.pkl', 'wb') as file:
    pickle.dump(gru_model_history.history, file)

# Construct and display evaluation metrics
y_pred_prob = gru.predict(X_test)

y_pred_labels = np.argmax(y_pred_prob, axis=1)

y_test_labels = np.argmax(y_test, axis=1)

target_names = list(map(str, label_encoder.classes_))
print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))

n_classes = y_test.shape[1]
fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
