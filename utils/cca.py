import numpy as np


def matrix_sqrt(x):
    """ Compute the square root of the matrix
    :param x: the matrix
    :return: the square root of matrix x
    """
    w, v = np.linalg.eigh(x)
    d = np.diag(w)
    d_sqrt = np.sqrt(d)

    # Compute the square root with eigendecomposition such that X = V * D^1/2 * VT
    return np.dot(v, np.dot(d_sqrt, v.T))


def cross_covariance(x, y):
    """ Compute the cross covariance matrix
    :param x: the first matrix
    :param y: the second matrix
    :return: variance of x, covariance of (x,y), variance of y
    """
    n = x.shape[0]
    # Center x and y
    xm = (x - x.mean())
    ym = (y - y.mean())

    # Compute variance of xm and ym and covariance of xy
    c_xx = np.dot(xm.T, xm) / n
    c_yy = np.dot(ym.T, ym) / n
    c_xy = np.dot(xm.T, ym) / n

    return c_xx, c_xy, c_yy


def matrix_sqrt_inv(x):
    """ Compute the inverse square root of a matrix
    :param x: the matrix to use
    :return: the inverse square root of the matrix (x^-1/2)
    """
    # Add small value for a more stable computation as done in https://github.com/google/svcca/blob/master/cca_core.py
    x += np.finfo(x.dtype).eps * np.eye(x.shape[0])
    x_inv = np.linalg.pinv(x)
    x_sqrt_inv = matrix_sqrt(x_inv)
    return x_sqrt_inv


def compute_cca(x, y):
    """ Compute canonical correlation analysis of matrices x and y using SVD

    :param x: the matrix X
    :param y: the matrix Y
    :return: a dictionary containing svd values, the weights w_x and w_y, the projected values of x and y (z_x and z_y)
    amd the diagonal matrix of correlations
    """
    res = {}

    c_xx, c_xy, c_yy = cross_covariance(x, y)  # Compute cross covariance matrix
    c_xx_sqrt_inv = matrix_sqrt_inv(c_xx)  # Compute c_xx^-1/2
    c_yy_sqrt_inv = matrix_sqrt_inv(c_yy)  # Compute c_yy^-1/2

    # Doing SVD decomposition for C_xx^-1/2 * c_xy * c_yy^-1/2
    to_decompose = np.dot(c_xx_sqrt_inv, np.dot(c_xy, c_yy_sqrt_inv))
    U, S, V = np.linalg.svd(to_decompose)
    res["svd"] = (U, S, V)

    # Compute w_x = c_xx^-1/2 * U and w_y = c_yy^-1/2 * V
    res["w_x"] = np.dot(c_xx_sqrt_inv, U)
    res["w_y"] = np.dot(c_yy_sqrt_inv, V)

    # Compute the projections of x, z_x = c_xx^-1/2 * U and y, z_y = c_yy^-1/2 * V
    res["z_x"] = np.dot(x, res["w_x"])
    res["z_y"] = np.dot(y, res["w_y"])
    res["sigma"] = np.diag(S)
    return res
