import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow_core.python.keras.regularizers import l2


def cca_loss(outdim, use_all_singular_values, r1=1e-4, r2=1e-4, eps=1e-12):
    def call(_, y_pred):
        # Retrieve x and y from concat layer
        x, y = tf.split(y_pred, num_or_size_splits=2, axis=1)
        x = tf.transpose(x)
        y = tf.transpose(y)

        # Get the number of inputs
        m = tf.shape(x)[1]

        m1_div = tf.cast(tf.divide(1, m - 1), tf.float32)
        m_div = tf.cast(tf.divide(1, m), tf.float32)

        # Center x and y
        x_centered = x - m_div * tf.matmul(x, tf.ones([m, m]))
        y_centered = y - m_div * tf.matmul(y, tf.ones([m, m]))

        # Compute cross covariance 1/(m-1) * XY
        c_xy = m1_div * tf.matmul(x_centered, y_centered, transpose_b=True)
        # Compute regularised X covariance 1/(m-1) * XX' + r1I
        c_xx = m1_div * tf.matmul(x_centered, x_centered, transpose_b=True) + r1 * tf.eye(x.shape[0])
        # Compute regularised Y covariance 1/(m-1) * YY' + r2I
        c_yy = m1_div * tf.matmul(y_centered, y_centered, transpose_b=True) + r2 * tf.eye(y.shape[0])

        # Compute the square root inverse of covariance matrices, c_xx^-1/2 and c_yy^-1/2
        c_xx_sqrt_inv = tf.linalg.sqrtm(tf.linalg.inv(c_xx))
        c_yy_sqrt_inv = tf.linalg.sqrtm(tf.linalg.inv(c_yy))

        # Compute T = c_xx^-1/2 * c_xy * c_yy^-1/2
        t = tf.matmul(tf.matmul(c_xx_sqrt_inv, c_xy), c_yy_sqrt_inv)

        if use_all_singular_values:
            # Total correlation is the trace of (TT')^1/2
            corr = tf.sqrt(tf.linalg.trace(tf.matmul(t, t, transpose_a=True)))
        else:
            # Use only the top outdim correlation
            s = tf.linalg.svd(t, compute_uv=False)
            tf.print(s)
            corr = tf.reduce_sum(s[:, outdim])

        return -corr

    return call


def build_mlp(input_layer, layers_dim, reg):
    last_idx = len(layers_dim) - 1
    x = input_layer
    for i, l in enumerate(layers_dim):
        activation = "sigmoid" if i < last_idx else "linear"
        x = keras.layers.Dense(l, activation=activation, kernel_regularizer=l2(reg))(x)
    return x


class DeepCCA:
    def __init__(self, x_input_dim, y_input_dim, layer_dims, l2_reg, lr, output_dim, use_all_singular_values,
                 model_folder):
        self.outdim = output_dim
        self.model_folder = model_folder

        x_input_layer = keras.Input(shape=(x_input_dim,))
        y_input_layer = keras.Input(shape=(y_input_dim,))
        x_net = build_mlp(x_input_layer, layer_dims, l2_reg)
        y_net = build_mlp(y_input_layer, layer_dims, l2_reg)
        concat = keras.layers.Concatenate()([x_net, y_net])

        self.model = keras.models.Model(inputs=[x_input_layer, y_input_layer], outputs=concat)
        self.model.compile(loss=cca_loss(output_dim, use_all_singular_values), optimizer=RMSprop(lr=lr))

    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs):
        callbacks = [keras.callbacks.ModelCheckpoint(filepath=self.model_folder + "/dcca_{epoch}", verbose=1,
                                                     save_best_only=True, save_weights_only=True)]
        self.model.fit([x_train, y_train], np.zeros(len(x_train)), batch_size=batch_size, epochs=epochs, shuffle=True,
                       callbacks=callbacks, validation_data=([x_val, y_val], np.zeros(len(x_val))))
        return self.model

    def test(self, x_test, y_test, batch_size, to_load=None):
        if to_load:
            self.model.load_weights(to_load)
        results = self.model.evaluate([x_test, y_test], np.zeros(len(x_test)), batch_size=batch_size, verbose=1)
        print("loss on test data: {}".format(results))
        return results

    def predict(self, x, y, batch_size, to_load=None):
        if to_load:
            self.model.load_weights(to_load)
        results = self.model.predict([x, y], batch_size)
        print("Predicted values: {}".format(results))
        return results
