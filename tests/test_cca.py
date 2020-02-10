import pandas as pd
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from tests.r_setup import install_r_packages, rtp_list
from src.models.cca import compute_cca


def r_cca(X, Y):
    # Install CCA and CCP if they are not yet installed
    install_r_packages(["CCA"])
    # Enable conversion of numpy to R objects
    cca = importr("CCA").cc
    numpy2ri.activate()
    r_res = cca(X, Y)
    numpy2ri.deactivate()
    res = rtp_list(r_res)
    return res


if __name__ == "__main__":
    df = pd.read_csv("tests/data/toy_example.csv")
    x = df[["locus_of_control", "self_concept", "motivation"]].to_numpy()
    y = df[["read", "write", "math", "science", "female"]].to_numpy()

    res = r_cca(x, y)
    res2 = compute_cca(x, y)
    print("--------------Correlation--------------------------")
    print(res["cor"])
    print(res2["sigma"])

    print("--------------Canonical coefficients----------------")
    print("X:")
    print(res["xcoef"])
    print(res2["w_x"])
    print("Y:")
    print(res["ycoef"])
    print(res2["w_y"])

    print("---------------Canonical variates------------------")
    print("X:")
    print(res["scores"]["xscores"])
    print(res2["z_x"])
    print("Y:")
    print(res["scores"]["yscores"])
    print(res2["z_y"])

    print("---------------Canonical loadings------------------")
    print("X:")
    print(res["scores"]["corr.X.xscores"])
    print(res2["loadings_x"])
    print("Y:")
    print(res["scores"]["corr.Y.yscores"])
    print(res2["loadings_y"])

    print("---------------Canonical cross-loadings------------------")
    print("XY:")
    print(res["scores"]["corr.X.yscores"])
    print(res2["loadings_xy"])
    print("YX:")
    print(res["scores"]["corr.Y.xscores"])
    print(res2["loadings_yx"])
