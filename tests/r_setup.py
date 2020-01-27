import rpy2.robjects.packages as rpackages
from rpy2.rinterface import NULL
from rpy2.robjects import default_converter, numpy2ri, pandas2ri
from rpy2.robjects.vectors import DataFrame, FloatVector, IntVector, StrVector, ListVector
import numpy as np

converter = default_converter + numpy2ri.converter + pandas2ri.converter


def install_r_packages(to_install):
    """ Install R packages listed in to_install if they are not yet installed.

    :param to_install: list of string containing the names of the R packages to install
    :type to_install: list
    :return: None
    """
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list
    not_yet_installed = [package for package in to_install if not rpackages.isinstalled(package)]
    if len(not_yet_installed) > 0:
        utils.install_packages(StrVector(not_yet_installed))


def rpy2dict(x):
    return dict(zip(x.names, [converter.rpy2py(e) for e in x]))


def rpy2array(x):
    return np.array(x)


def rpy2list(x):
    return [converter.rpy2py(e) for e in x]


def rpy2none(x):
    return None


def rtp_list(data):
    """Map R nested list to python equivalent recursively

    :param data: the nested R list to convert
    :return: the python equivalent
    """
    converter.rpy2py.register(FloatVector, rpy2array)
    converter.rpy2py.register(IntVector, rpy2array)
    converter.rpy2py.register(DataFrame, rpy2dict)
    converter.rpy2py.register(ListVector, rpy2dict)
    converter.rpy2py.register(StrVector, rpy2list)
    converter.rpy2py.register(type(NULL), rpy2none)
    return converter.rpy2py(data)

