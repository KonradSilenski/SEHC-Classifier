import numpy as np
import pandas as pd
from keras.layers import Conv1D, Add, Dropout, GRU
from keras.layers import Embedding, MaxPooling1D
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
from tensorflow.keras.layers import Input, Bidirectional, Dense, Attention, GlobalAveragePooling1D, Layer
import tensorflow.keras.backend as K

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

max_len = 70    # Maximum sentence length

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

embedding_matrix = zeros((vocab_size, 100))
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

        s_i = K.expand_dims(K.sum(alpha_it * h, axis=1), axis=1)

        return s_i

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


input_layer = Input(shape=(max_len,))

embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, trainable=False, weights=[embedding_matrix])(
    input_layer)

conv1_2 = Conv1D(150, 2, activation='relu', padding='same')(embedding_layer)
conv1_3 = Conv1D(150, 3, activation='relu', padding='same')(embedding_layer)
conv1_4 = Conv1D(150, 4, activation='relu', padding='same')(embedding_layer)
merged_conv1 = Add()([conv1_2, conv1_3, conv1_4])

dropout1 = Dropout(0.25)(merged_conv1)
pool1 = MaxPooling1D(pool_size=2)(dropout1)

conv2_2 = Conv1D(150, 2, activation='relu', padding='same')(pool1)
conv2_3 = Conv1D(150, 3, activation='relu', padding='same')(pool1)
conv2_4 = Conv1D(150, 4, activation='relu', padding='same')(pool1)
merged_conv2 = Add()([conv2_2, conv2_3, conv2_4])

dropout2 = Dropout(0.25)(merged_conv2)
pool2 = MaxPooling1D(pool_size=2)(dropout2)

bi_gru = Bidirectional(GRU(256, return_sequences=True))(pool2)
gru = GRU(128, return_sequences=True)(bi_gru)

attention = Attention()([gru, gru])

pooled_output = GlobalAveragePooling1D()(attention)

dense1 = Dense(100, activation='relu')(pooled_output)
dense2 = Dense(100, activation='relu')(dense1)

output_layer = Dense(2, activation='softmax')(dense2)

cnn_gru_model = Model(inputs=input_layer, outputs=output_layer, name='cnn_gru')

cnn_gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_gru_model.summary()

cnn_gru_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=8, batch_size=32)

cnn_gru_model.save('../CNN_GRU_final/cnn_gru_rec_100.h5')

# Construct and display evaluation metrics
y_pred_prob = cnn_gru_model.predict(X_test)

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
