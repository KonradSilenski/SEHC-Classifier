import tensorflow.keras.backend as K
from keras.layers import Average
from keras.models import Model
from keras.models import load_model
from tensorflow.keras.layers import Input, Layer


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


# Load individual models and import custom attention objects
cnn = load_model('../CNN_final/cnn_rec_100.h5')
gru = load_model('../GRU_final/gru_rec_100.h5', custom_objects={'WordAttention': WordAttention})
lstm = load_model('../LSTM_final/lstm_rec_200.h5')
cnn_gru = load_model('../CNN_GRU_final/cnn_gru_rec_100.h5', custom_objects={'WordAttention': WordAttention})
cnn_lstm = load_model('../CNN_LSTM_final/cnn_lstm_rec_100.h5')

input_layer = Input(shape=(70,))

cnn_output = cnn(input_layer)
cnn_output._name = 'cnn_output'

gru_output = gru(input_layer)
gru_output._name = 'gru_output'

lstm_output = lstm(input_layer)
lstm_output._name = 'lstm_output'

cnn_gru_output = cnn_gru(input_layer)
cnn_gru_output._name = 'cnn_gru_output'

cnn_lstm_output = cnn_lstm(input_layer)
cnn_lstm_output._name = 'cnn_lstm_output'

# Create the Average layer for predictions
averaged_output = Average()([cnn_output, gru_output, lstm_output, cnn_gru_output, cnn_lstm_output])

ensemble_model = Model(inputs=input_layer, outputs=averaged_output, name='ensemble_model')

ensemble_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ensemble_model.save('../ENSEMBLE_final/ensemble_model_rec_100.h5')
