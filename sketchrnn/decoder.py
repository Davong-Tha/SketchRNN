import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sketchrnn.customlstm import customLSTM

class Decoder(keras.layers.Layer):
    def __init__(self, dec_hidden, output_dim, dec_cell_type='customlstm', recurrent_dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        # output dim is 6M+3 where M is number of Gaussian mixtures
        if dec_cell_type == 'customlstm':
            self.cell = customLSTM(dec_hidden, recurrent_dropout=recurrent_dropout)
            self.unidirectional = layers.RNN(self.cell, return_sequences=True, return_state=True)
        elif dec_cell_type == 'lstm':
            self.cell = layers.LSTM(dec_hidden, recurrent_dropout=False, return_sequences=True, return_state=True)
            self.unidirectional = self.cell
        self.w_out = layers.Dense(output_dim)
        self.normalize = layers.LayerNormalization()
        self.dropout = layers.Dropout(0.1, noise_shape=(None, 1, None))
        self.masking = layers.Masking(mask_value=0.0)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, inputs, initial_state=None, training=None, mask=None):
        x = inputs
        x = self.dropout(x, training=training)
        # x = self.masking(x)
        # x = self.normalize(x)
        seq_out, h, c = self.unidirectional(x, initial_state=initial_state, training=training, mask=mask)
        out = self.w_out(seq_out)
        return out, h, c