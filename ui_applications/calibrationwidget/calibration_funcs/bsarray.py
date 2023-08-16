import numpy as np
import scipy.io as scio
import copy

from .ext_round import *
from .directFilter import *


def bsarray(a=[],lambda_array=[]):
    if(a==[]):
        b=dict()
        b['coeffs']=[]
        b['tensorOrder'] = 0
        b['dataSize'] = []
        b['coeffsSize'] = []
        b['degree'] = []
        b['centred'] = []
        b['elementSpacing'] = []
        b['lambda'] = []

        return b
    # ignore the copy construction case in MATLAB
    else:
        b = constructBsarray(a,lambda_array = lambda_array)

        return b

def constructBsarray(a,lambda_array=0,deg=3,eSp=1):
    n = np.ndim(a)
    if((n==2) and (np.size(a,0)==1)):
        n = 1
        a = a[0,:]
    if((n==2) and (np.size(a,1)==1)):
        n = 1
        a = a[:,1]

    if(np.size(deg)==1):
        deg = (deg*np.ones((1,n)))[0]
    if(np.size(deg)!=n):
        raise('Degree must be scalar or vector of same length as the number of nonsingleton dimensions of a.')
    if(np.any(deg<0) or np.any(deg>7) or np.any(deg!=ext_round(deg))):
        raise('Each element of BSpline degree must be an integer in the interval [0,7]')

    cflag = np.full(np.shape(deg), True, dtype=bool)

    if(np.size(eSp)==1):
        eSp = (eSp*np.ones((1,n)))[0]
    if(np.size(eSp)!=n):
        raise('ElementSpacing must be scalar or vector of same length as the number of nonsingleton dimensions of a.')
    if(np.any(eSp<=0)):
        raise('Each element of ElementSpacing must be a positive integer')

    if(lambda_array == []):
        lambda_array = 0
    if(np.size(lambda_array)==1):
        lambda_array = (lambda_array*np.ones((1,n)))[0]
    if(np.size(lambda_array)!=n):
        raise('lambda must be scalar or vector of same length as the number of nonsingleton dimensions of a.')
    if(np.any(lambda_array<0)):
        raise('Each element of lambda must be a positive integer')

    if(np.any(lambda_array[np.mod(deg,2)==0])):
        raise('Nonzero lambda values not supported for even degree BSplines.')

    coeffs = directFilter(a,deg,n,lambda_array)
    b = dict()
    b['coeffs']=coeffs
    b['tensorOrder'] = n
    b['dataSize'] = np.shape(a)
    b['coeffsSize'] = np.shape(coeffs)
    b['degree'] = deg
    b['centred'] = cflag
    b['elementSpacing'] = eSp
    b['lambda'] = lambda_array

    return b
