import tensorflow as tf
from tensorflow.keras import layers

class ProjectionHead(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=768, proj_dim=768):
        super().__init__()
        self.d1 = layers.Dense(input_dim, activation='relu')
        self.d2 = layers.Dense(hidden_dim, activation='relu')
        self.out = layers.Dense(proj_dim)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        x = tf.nn.l2_normalize(x, axis=-1)  # unit sphere (important!)
        return x