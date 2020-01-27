from tensorflow import keras
from tensorflow.keras.layers import Dense, InputLayer, Conv2D, Flatten


def convolutional_encoder(input_dim, latent_dim=10):
    """ Encoder proposed by [2] for image content:
    4 convolutional layers with 32 channels, 4*4 kernels, strides of 2 with relu activation
    2 fully connected layers of 256 units with relu activation
    1 fully connected layer of 20 units for mean and variance estimation (2 * latent_dim)

    :param input_dim: the input size
    :param latent_dim: the size of the estimated latent variable. By default value proposed by [2].
    :return: The initialised encoder

    [2] Burgess, C.P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G. and Lerchner, A., 2018.
    Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599.
    """
    encoder = keras.Sequential()
    encoder.add(InputLayer(input_shape=input_dim))
    encoder.add(Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
    encoder.add(Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
    encoder.add(Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
    encoder.add(Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
    encoder.add(Flatten())
    encoder.add(Dense(256, activation="relu"))
    encoder.add(Dense(256, activation="relu"))
    encoder.add(Dense(2 * latent_dim))  # here mu and log_var are on the same layer so we need twice the size of Z

    return encoder
