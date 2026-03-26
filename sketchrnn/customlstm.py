import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class customLSTM(keras.layers.Layer):
    def __init__(self, units, recurrent_dropout=False, **kwargs):
        super().__init__(**kwargs)
        self.recurrent_dropout = recurrent_dropout
        self.dropout_prob = 0.2
        self.units = units
        self.state_size = [units, units]
        self.Wx = layers.Dense(4 * units, use_bias=True)
        self.Wh = layers.Dense(4 * units, use_bias=False)
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, inputs, states, training=None):
        c0 = states[1]
        h0 = states[0]
        Wxt = self.Wx(inputs)
        Wht = self.Wh(h0)
        f, i, o, c = tf.split(Wxt + Wht, num_or_size_splits=4, axis=1)
        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        o = tf.sigmoid(o)
        c = tf.tanh(c)
        if self.recurrent_dropout and training:
            c = tf.nn.dropout(c, rate=self.dropout_prob)
        c_new = f * c0 + i * c
        h_new = o * tf.tanh(c_new)
        return h_new, [h_new, c_new]
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


        