from tensorflow import keras
from tensorflow.keras.layers import Dense, InputLayer, Conv2DTranspose
from tensorflow_core.python.keras.layers import Reshape


def deconvolutional_decoder(output_dim, latent_dim=10):
    """ Decoder proposed by [1] for image content is a transpose of the encoder architecture:
    4 convolutional layers with 32 channels, 4*4 kernels, strides of 2 with relu activation
    2 fully connected layers of 256 units with relu activation
    1 fully connected layer of 20 units for mean and variance estimation (2 * latent_dim)

    :param output_dim: the output size
    :param latent_dim: the size of the estimated latent variable. By default value proposed by [2].
    :return: The initialised encoder

    [1] Burgess, C.P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G. and Lerchner, A., 2018.
    Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599.
    """
    dense_units = 4 * 4 * 32  # units of the last dense layer to match deconvolutional layer dimensions after reshape
    decoder = keras.Sequential()
    decoder.add(InputLayer(input_shape=(latent_dim,)))
    decoder.add(Dense(256, activation="relu"))
    decoder.add(Dense(dense_units, activation="relu"))
    decoder.add(Reshape(target_shape=(4, 4, 32)))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="SAME"))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="SAME"))
    decoder.add(Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation="relu", padding="SAME"))
    decoder.add(Conv2DTranspose(filters=output_dim[2], kernel_size=4, strides=2, padding="SAME"))

    return decoder
