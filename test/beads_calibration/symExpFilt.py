import numpy as np
from numba import jit

@jit(nopython=True)
def symExpFilt(X,XLen,C0,Zi,K0,KVec):
    Y = np.zeros(X.size).astype(np.clongdouble)
    for k in np.arange(0, K0):
        Y[0] = Y[0] + np.power(Zi, k) * X[np.int32(KVec[k])]
    for k in np.arange(1, XLen):
        Y[k] = X[k] + Zi * Y[k - 1]
    Y[XLen - 1] = (2 * Y[XLen - 1] - X[XLen - 1]) * C0
    for k in np.arange(XLen - 2, -1, -1):
        Y[k] = (Y[k + 1] - Y[k]) * Zi
    return Y
