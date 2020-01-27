import tensorflow as tf
from tensorflow import keras


def _reparametrize(mu, log_var):
    """ Reparametrisation trick used by [1].
    Compute z as a mapping from g(x, eps) = mu + sigma ⊙ eps where eps ~ N(0, 1)

    :param mu: the estimated mean value returned by the encoder
    :param log_var: the estimated variance returned by the encoder
    :return: z

    [1] Kingma, D.P. and Welling, M., 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
    """
    eps = tf.random.normal(shape=mu.shape)
    log_var_sqrt = tf.exp(log_var * 0.5)
    z = mu + eps * log_var_sqrt
    return z


class VAE(keras.Model):
    def __init__(self, encoder, decoder, loss_func, input_dim, latent_dim=300, optimizer=None):
        """ Initialise the VAE model with a given encoder, decoder and loss function.

        :param encoder: Encoder model to use
        :param decoder: Decoder model to use
        :param loss_func: Loss function to call
        :param input_dim: Input shape
        :param latent_dim: Latent variable shape
        :param optimizer: optimizer to use. (Default: Adam(5e-4), similarly to [1])

        [1] Burgess, C.P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G. and Lerchner, A., 2018.
        Understanding disentangling in $\beta $-VAE. arXiv preprint arXiv:1804.03599.
        """
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self.encoder = encoder(input_dim, latent_dim)
        self.decoder = decoder(input_dim, latent_dim)
        self.loss_func = loss_func
        self.optimizer = tf.keras.optimizers.Adam(5e-4) if optimizer is None else optimizer

    def encode(self, X):
        """ Encode X and retrieve mean and variance by splitting the last layer in two

        :param X: input to encode
        :return: mean and variance returned by the encoder
        """
        mu, log_var = tf.split(self.encoder(X), num_or_size_split=2, axis=1)
        return mu, log_var

    def decode(self, Z, sigmoid=False):
        """ Reconstruct X from its latent representation Z

        :param Z: latent representation obtained after reparametrisation of the encoder output
        :param sigmoid: if True returns the sigmoid of the reconstructed X, otherwise returns X
        :return: the reconstructed X (or its sigmoid if sigmoid = True)
        """
        X_hat = self.decode(Z)
        return tf.sigmoid(X_hat) if sigmoid else X_hat

    @tf.function
    def forward(self, X):
        """ A forward pass during training where X is encoded, the decoder estimations used to retrieve z and
        pass it to the decoder to reconstruct X.

        :param X: the input value
        :return: mean, variance and reconstructed X value
        """
        mu, log_var = self.encode(X)
        Z = _reparametrize(mu, log_var)
        X_hat = self.decode(Z)
        return mu, log_var, Z, X_hat

    @tf.function
    def backward(self, X):
        """ A backward pass which computes and apply the gradients during training

        :param X: the input value
        :return: None
        """
        with tf.GradientTape() as gtape:
            loss = self.compute_loss(X)
        gradients = gtape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def compute_loss(self, X):
        """ Compute the loss

        :param X: the input
        :return: the computed loss
        """
        mu, log_var, Z, X_hat = self.forward(X)
        return self.loss_func(X_hat, X, mu, log_var)

    @tf.function
    def sample(self, eps=None):
        """ Sample X from decoder during test time

        :param eps: epsilon value from which z will be approximated, if None eps ~ N(0,1)
        :return: the sigmoid value of the generated X
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self._latent_dim))
        return self.decode(eps, sigmoid=True)
