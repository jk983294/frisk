import numpy as np
import time
import ctypes
import copy


class ElNet(object):
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("/opt/version/latest/frisk/lib/libelnet.so")
        self.lib.get_model.restype = ctypes.c_void_p
        self.lib.free_model.argtypes = [ctypes.c_void_p]
        self.lib.free_predicted_y.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.lib.fit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_long, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                 ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
        self.lib.predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                                     ctypes.c_long, ctypes.c_int, ctypes.c_double]
        self.lib.predict.restype = ctypes.c_void_p
        self.model = self.lib.get_model()

    def __del__(self):
        self.lib.free_model(self.model)
        print('Destructor called, Employee deleted.')

    def fit_model(self, X, y, alpha: float = 0.5, n_lambda: int = 100,
                  lambda_path: np.array = np.empty(0, dtype=np.float64),
                  standardize: bool = True, fit_intercept: bool = True,
                  max_iter: int = 100000, max_features: int = -1):
        start = time.time()
        X = X.astype(dtype=np.float64, order='F', copy=False)
        pX = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        py = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_lambda_path = lambda_path.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n_lambda_path = lambda_path.shape[0]
        self.lib.fit(self.model, pX, py, X.shape[0], X.shape[1], alpha, n_lambda, p_lambda_path, n_lambda_path,
                     standardize, fit_intercept, max_iter, max_features)
        end = time.time()
        print("took ", end - start)
        return m

    def model_predict(self, X, s: float = np.nan):
        X = X.astype(dtype=np.float64, order='F', copy=False)
        pX = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pointer = self.lib.predict(self.model, pX, X.shape[0], X.shape[1], s)
        pred = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_double))
        ret = copy.deepcopy(np.ctypeslib.as_array((ctypes.c_double * X.shape[0]).from_address(pointer)))
        self.lib.free_predicted_y(pred)
        return ret


if __name__ == '__main__':
    n = 100000  # Number of observations
    p = 20  # Number of predictors included in model
    real_p = 5  # Number of true predictors
    X = np.random.rand(n, p)
    y = np.add.reduce(X[:, 0:real_p], 1)
    # X = X.astype(dtype=np.float64, order='F', copy=False)

    m = ElNet()
    m.fit_model(X, y)
    pred = m.model_predict(X)

    mse = np.mean((y - pred) ** 2)
    pcor = np.corrcoef(y, pred)
    print('mse ', mse, "pcor", pcor[0][1])
    print('end')
