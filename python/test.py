import sys
from pathlib import Path
import numpy as np
import time
import ctypes

_current_root = str(Path(__file__).resolve().parents[1])
sys.path.append(_current_root + '/cmake-build-debug/lib')

import pyelnet


def fit_model(X, y):
    y = y.reshape(-1, 1)  # make it Eigen acceptable format

    m = pyelnet.PyElNet()
    start = time.time()
    m.fit(X, y, alpha=0.5)
    end = time.time()
    print("took ", end - start)
    return m


if __name__ == '__main__':
    n = 100000  # Number of observations
    p = 1000  # Number of predictors included in model
    real_p = 15  # Number of true predictors
    X = np.random.rand(n, p)
    y = np.add.reduce(X[:, 0:real_p], 1)
    X = X.astype(dtype=np.float64, order='F', copy=False)

    m = fit_model(X, y)
    pred = m.predict(X)

    mse = np.mean((y - pred)**2)
    pcor = np.corrcoef(y, pred)
    print('mse ', mse, "pcor", pcor[0][1])
    print('end')
