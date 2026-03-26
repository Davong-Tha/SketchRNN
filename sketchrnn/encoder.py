import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sketchrnn.customlstm import customLSTM

class Encoder(keras.layers.Layer):
    def __init__(self, hidden, latent_dim, dec_hidden, enc_cell_type='customlstm', recurrent_dropout=False, **kwargs):
        super().__init__(**kwargs)
        
        self.supports_masking = True
        if enc_cell_type == 'customlstm':
            self.cell = customLSTM(hidden, recurrent_dropout=recurrent_dropout)
            self.bidirectional = layers.Bidirectional(layers.RNN(self.cell, return_state=True, return_sequences=True))
        elif enc_cell_type == 'lstm':
            self.cell = layers.LSTM(hidden, recurrent_dropout=False,  return_state=True, return_sequences=True, dropout=0.0)
            self.bidirectional = layers.Bidirectional(self.cell)
        self.w_mean = layers.Dense(latent_dim)
        self.w_logvar = layers.Dense(latent_dim)
        self.reparam = layers.Lambda(self.reparameterize)
        self.w_out = layers.Dense(dec_hidden, activation='tanh')
        self.c_out = layers.Dense(dec_hidden, activation='tanh')
        self.normalize = layers.LayerNormalization()
        self.masking = layers.Masking(mask_value=0.0)
        self.dropout = layers.Dropout(0, noise_shape=(None, 1, None))
    def build(self, input_shape):
        super().build(input_shape)
    def reparameterize(self, args):
        z_mean, z_logvar = args
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_logvar * 0.5) * eps
    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.dropout(x, training=training)
        # x = self.masking(x)
        seq_out, fh, fc, bh, bc = self.bidirectional(x, training=training, mask=mask)
        h_enc = tf.concat([fh, bh], axis=-1) # concatenate forward and backward hidden states leaving batching dimension intact
        c_enc = tf.concat([fc, bc], axis=-1)
        # h_enc = self.normalize(h_enc)
        # c_enc = self.normalize(c_enc)
        z_mean = self.w_mean(tf.concat([h_enc, c_enc], axis=-1))
        z_logvar = self.w_logvar(tf.concat([h_enc, c_enc], axis=-1))
        # z_logvar = tf.clip_by_value(z_logvar, -6.0, 2.0)
        z = self.reparam((z_mean, z_logvar))
        h0 = self.w_out(z)
        c0 = self.c_out(z)
        return h0, z, z_mean, z_logvar, c0