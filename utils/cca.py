import numpy as np
from scipy import linalg


def cross_covariance(x, y):
    """ Compute the cross covariance matrix

    :param x: the first matrix
    :param y: the second matrix
    :return: variance of x, covariance of (x,y), variance of y
    """
    n = x.shape[0]

    # Compute variance of x and y and covariance of xy
    c_xx = np.dot(x.T, x) / (n - 1)
    c_yy = np.dot(y.T, y) / (n - 1)
    c_xy = np.dot(x.T, y) / (n - 1)

    return c_xx, c_xy, c_yy


def matrix_sqrt_inv(x):
    """ Compute the inverse square root of a matrix

    :param x: the matrix to use
    :return: the inverse square root of the matrix (x^-1/2)
    """
    # Add small value for a more stable computation as done in https://github.com/google/svcca/blob/master/cca_core.py
    x += np.finfo(x.dtype).eps * np.eye(x.shape[0])
    x_inv = np.linalg.pinv(x)
    x_sqrt_inv = linalg.sqrtm(x_inv)
    return x_sqrt_inv


def corr_coef(x, y):
    """ Correlate each n with each m.

    :param x: matrix of size T x N
    :type x: np.array
    :param y: matrix of size T x M
    :type y: np .array
    :return: N x M matrix containing the correlation coefficients
    :rtype: np.array
    """
    n = x.shape[0]
    assert n == y.shape[0]

    std_x = x.std(0, ddof=n - 1)
    std_y = y.std(0, ddof=n - 1)
    corr = np.dot(x.T, y) / np.dot(std_x[:, None], std_y[None, :])
    return corr


def compute_cca(x, y):
    """ Compute canonical correlation analysis of matrices x and y using SVD

    :param x: the matrix X
    :param y: the matrix Y
    :return: a dictionary containing svd values, the coefficients w_x and w_y, the canonical loadings and cross loadings,
    the canonical variates of x and y (z_x and z_y) and the diagonal matrix of correlations
    """
    res = {}

    # Center x and y
    x = (x - x.mean(axis=0))
    y = (y - y.mean(axis=0))

    c_xx, c_xy, c_yy = cross_covariance(x, y)  # Compute cross covariance matrix

    c_xx_sqrt_inv = matrix_sqrt_inv(c_xx)  # Compute c_xx^-1/2
    c_yy_sqrt_inv = matrix_sqrt_inv(c_yy)  # Compute c_yy^-1/2

    # Doing SVD decomposition for c_xx^-1/2 * c_xy * c_yy^-1/2
    to_decompose = np.matmul(c_xx_sqrt_inv, np.matmul(c_xy, c_yy_sqrt_inv))
    u, s, v = np.linalg.svd(to_decompose)
    v = v.T
    res["svd"] = (u, s, v)
    res["sigma"] = s

    dim_to_keep = range(0, len(s))
    u = u[:, dim_to_keep]
    v = v[:, dim_to_keep]

    # Compute w_x = c_xx^-1/2 * U and w_y = c_yy^-1/2 * V
    res["w_x"] = np.matmul(c_xx_sqrt_inv, u)
    res["w_y"] = np.matmul(c_yy_sqrt_inv, v)

    # Compute the projections (canonical variates) of x, z_x = x * c_xx^-1/2 * U and y, z_y = y * c_yy^-1/2 * V
    res["z_x"] = np.matmul(x, res["w_x"])
    res["z_y"] = np.matmul(y, res["w_y"])

    # Compute the canonical loadings
    res["loadings_x"] = corr_coef(x, res["z_x"])
    res["loadings_y"] = corr_coef(y, res["z_y"])

    # Compute the canonical cross-loadings
    res["loadings_xy"] = corr_coef(x, res["z_y"])
    res["loadings_yx"] = corr_coef(y, res["z_x"])
    return res
