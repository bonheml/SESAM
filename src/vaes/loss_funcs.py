import tensorflow as tf
import numpy as np


def gaussian_log_pdf(X, mu, log_var):
    inv_var = tf.exp(-log_var)
    log2pi = tf.math.log(2 * np.pi)
    return -0.5 * (log2pi + log_var + inv_var * tf.square(X - mu))


def reconstruction_loss(X_pred, X_true):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X_true)
    return tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))


def gaussian_kld(mu, log_var):
    to_reduce = tf.square(mu) + tf.exp(log_var) - log_var - 1
    return tf.reduce_mean(0.5 * tf.reduce_sum(to_reduce, axis=1))


def beta_vae_loss(X_pred, X_true, mu, log_var, beta=1):
    kld = gaussian_kld(mu, log_var)
    recon_loss = reconstruction_loss(X_pred, X_true)
    return recon_loss + beta * kld
