import numpy as np
from src import ModelEstimation as me


if __name__ == '__main__':
    x = np.loadtxt("data/x.txt", delimiter=",")
    y = np.loadtxt("data/y.txt", delimiter=",")
    index = np.loadtxt("data/index.txt", delimiter=",", dtype=bool)
    names = np.loadtxt("data/names.txt", delimiter=",", dtype=str)
    osr = me.ModelEstimation(x, y, index, names)
    res = osr.osr()
    print(res)