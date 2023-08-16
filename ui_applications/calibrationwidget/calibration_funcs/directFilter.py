import numpy as np
from numba import jit
import time
import copy

from .getBSplineFiltCoeffs import *
from .symExpFilt import *
from .interp3_0 import *


def directFilter(A,D,tensorOrder,lambda_array):
    if(np.all(D<2) and np.all(lambda_array==0)):
        padNum = np.floor(D/2)
        idx = []
        for k in range(tensorOrder):
            M = np.size(A,k)
            dimNums = np.concatenate((np.arange(0,M),np.arange(M-2,0,-1)))
            p = padNum[k]
            idx.append(dimNums[np.mod(np.arange(-p,M+p),2*M-2)])

        CPad = A[idx]

        return CPad

    F = []
    F0 = []
    for i in range(tensorOrder):
        F_i, F0_i = getBSplineFiltCoeffs((D[i], True, 'direct',lambda_array[i]))
        F.append(F_i)
        F0.append(F0_i)

    C = copy.copy(A).astype(np.longdouble)

    K0Tol = 2.2204e-16
    CRows = np.size(C,0)
    CCols = np.size(C,1)
    CSlices = np.size(C,2)

    if tensorOrder > 3:
        raise('multidimensional is not implemented yet')
    C = C.astype(np.clongdouble)
    for t in range(tensorOrder):
        if (np.size(F[t]) == 1):
            F[t] = [F[t]]
        for i in range(len(F[t])):
            calFilt(t, CRows, CCols, CSlices, F[t][i], C, K0Tol)
        C = np.real(C)*F0[t]
        C = C.astype(np.clongdouble)

    C = np.real(C).astype(np.longdouble)
    padNum = np.floor(D/2)

    idx = []
    for k in range(tensorOrder):
        M = np.size(C,k)
        dimNums = np.concatenate((np.arange(0,M),np.arange(M-2,0,-1)))
        p = int(padNum[k])
        idx.append(dimNums[np.mod(np.arange(-p,M+p+1),2*M-2)])

    C1 = C[idx[0],:,:]
    C2 = C1[:,idx[1],:]
    CPad = C2[:,:,idx[2]]
    return CPad


@jit(nopython=True)
def calFilt(type, CRows, CCols, CSlices, F, C, K0Tol):
    K0 = int(np.ceil(np.log(K0Tol) / np.log(np.abs(F))))
    if type == 0:
        indReflect = np.concatenate((np.arange(0, CRows), np.arange(CRows - 2, 0, -1)))
        numReps = np.ceil(K0 / (2 * CRows - 2))
    elif type == 1:
        indReflect = np.concatenate((np.arange(0, CCols), np.arange(CCols - 2, 0, -1)))
        numReps = np.ceil(K0 / (2 * CCols - 2))
    else:
        indReflect = np.concatenate((np.arange(0, CSlices), np.arange(CSlices - 2, 0, -1)))
        numReps = np.ceil(K0 / (2 * CSlices - 2))
    KVec = indReflect * np.ones((1, np.int32(numReps)))[0]
    KVec = KVec[0:K0]
    C0 = -F / (1 - F ** 2)
    if type == 0:
        for k in range(CSlices):
            for j in range(CCols):
                C[:, j, k] = symExpFilt(C[:, j, k], CRows, C0, F, K0, KVec)
    elif type == 1:
        for k in range(CSlices):
            for j in range(CRows):
                C[j, :, k] = symExpFilt(C[j, :, k], CCols, C0, F, K0, KVec)
    else:
        for k in range(CCols):
            for j in range(CRows):
                C[j, k, :] = symExpFilt(C[j, k, :], CSlices, C0, F, K0, KVec)

if __name__ == '__main__':
    dataFile = 'directfilter.mat'
    import scipy.io as scio
    data = scio.loadmat(dataFile)
    A = data['A']
    D = data['D'][0]
    tensorOrder = data['tensorOrder'][0][0]
    lambda_array = data['lambda'][0]
    coeffs = data['coeffs']
    coeffs2 = directFilter(A,D,tensorOrder,lambda_array)
    diff = coeffs2 - coeffs