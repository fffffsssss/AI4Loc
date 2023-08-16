# interp3_0 function
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import numpy as np
import numba
from numba import jit

from .evalBSpline import *
from .logger import *


def parseInputs(varargin: list):
    b = varargin[0]
    if b["tensorOrder"] != 3:
        error_log("bsarray/interp3 can only be used with bsarray objects having tensor order 3.")
    xi = varargin[1]
    if False in np.isreal(xi):
        error_log("The interpolation points XI should be real")
    yi = varargin[2]
    if False in np.isreal(yi):
        error_log("The interpolation points YI should be real")
    if yi.shape != xi.shape:
        error_log("YI must be the same size as XI")
    zi = varargin[3]
    if False in np.isreal(zi):
        error_log("The interpolation points ZI should be real")
    if zi.shape != xi.shape:
        error_log("ZI must be the same size as XI")
    if len(varargin) > 4:
        extrapval = varargin[4]
    else:
        extrapval = []
    if isinstance(extrapval, list) and not extrapval:
        extrapval = np.NaN
    if not np.isscalar(extrapval):
        error_log("EXTRAP option must be a scalar")
    return b, xi, yi, zi, extrapval


def interp3_0(b, xi, yi, zi, extrapval = 0):
    m = b['centred']
    mx = m[1]
    my = m[0]
    mz = m[2]

    nData = np.array(b['dataSize'])
    nDatax = nData[1]
    nDatay = nData[0]
    nDataz = nData[2]

    n = np.array(b['coeffsSize'])
    nx = n[1]
    ny = n[0]
    nz = n[2]

    padNum = (n - nData - 1) / 2
    padNumx = padNum[1]
    padNumy = padNum[0]
    padNumz = padNum[2]

    h = b['elementSpacing']
    hx = h[1]
    hy = h[0]
    hz = h[2]

    xCol = hx * np.arange((1 - padNumx), (nDatax + padNumx + 1) + 1)
    xDataCol = xCol[int(padNumx):int(nDatax + 1) + 1]
    yCol = hy* np.arange((1 - padNumy), (nDatay + padNumy + 1) + 1)
    yDataCol = yCol[int(padNumy):int(nDatay + 1) + 1]
    zCol = hz * np.arange((1 - padNumz), (nDataz + padNumz + 1) + 1)
    zDataCol = zCol[int(padNumz):int(nDataz + 1) + 1]

    siz_xi = xi.shape
    siz_vi = siz_xi

    cMat = b['coeffs']
    numelXi = xi.size

    p = np.arange(0, numelXi)

    kx = np.minimum(np.maximum(1 + np.floor((xi.reshape(numelXi, order='F') - xCol[0]) / hx), 1 + padNumx), nx - padNumx - 1) + 1 - mx
    ky = np.minimum(np.maximum(1 + np.floor((yi.reshape(numelXi, order='F') - yCol[0]) / hy), 1 + padNumy), ny - padNumy - 1) + 1 - my
    kz = np.minimum(np.maximum(1 + np.floor((zi.reshape(numelXi, order='F') - zCol[0]) / hz), 1 + padNumz), nz - padNumz - 1) + 1 - mz

    sx = (xi.reshape(numelXi, order='F') - xCol[kx.astype(int) - 1]) / hx
    sy = (yi.reshape(numelXi, order='F') - yCol[ky.astype(int) - 1]) / hy
    sz = (zi.reshape(numelXi, order='F') - zCol[kz.astype(int) - 1]) / hz

    d = b['degree']
    dx = d[1]
    dy = d[0]
    dz = d[2]

    xflag = (mx == 0 and np.mod(dx, 2))
    yflag = (my == 0 and np.mod(dy, 2))
    zflag = (mz == 0 and np.mod(dz, 2))

    viMat = interp3_0_cal(mx, my, mz, dx, dy, dz, sx, sy, sz, xflag, yflag, zflag, kx, ky, kz, numelXi, cMat)
    outOfBounds = np.where((xi.reshape(numelXi, order='F') < xDataCol[0]) | (xi.reshape(numelXi, order='F') > xDataCol[nDatax - 1]) |
                           (yi.reshape(numelXi, order='F') < yDataCol[0]) | (yi.reshape(numelXi, order='F') > yDataCol[nDatay - 1]) |
                           (zi.reshape(numelXi, order='F') < zDataCol[0]) | (zi.reshape(numelXi, order='F') > zDataCol[nDataz - 1]))
    if len(outOfBounds[0]) > 0:
        viMat[p[outOfBounds]] = extrapval

    vi = np.reshape(viMat, siz_vi, order="F")
    return vi

@numba.jit(nopython=True)
def interp3_0_cal(mx, my, mz, dx, dy, dz, sx, sy, sz, xflag, yflag, zflag, kx, ky, kz, numelXi, cMat):
    viMat = np.zeros(numelXi).astype(np.longdouble)
    for q in np.arange(1, np.ceil((dz + 1) / 2) + 1):
        Bz1 = evalBSpline(sz + q - (1 + mz) / 2, dz)
        Bz2 = evalBSpline(sz - q + (1 - mz) / 2, dz)
        for j in np.arange(1, np.ceil((dx + 1) / 2) + 1):
            Bx1 = evalBSpline(sx + j - (1 + mx) / 2, dx)
            Bx2 = evalBSpline(sx - j + (1 - mx) / 2, dx)
            for i in np.arange(1, np.ceil((dy + 1) / 2) + 1):
                By1 = evalBSpline(sy + i - (1 + my) / 2, dy)
                By2 = evalBSpline(sy - i + (1 - my) / 2, dy)
                for ind in np.arange(0, numelXi):
                    viMat[ind] = viMat[ind] + \
                                 (cMat[int(ky[ind] - i + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] - q + mz - 1)] * By1[ind] * Bx1[ind] * Bz1[ind] +
                                  cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] - q + mz - 1)] * By2[ind] * Bx1[ind] * Bz1[ind] +
                                  cMat[int(ky[ind] - i + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] - q + mz - 1)] * By1[ind] * Bx2[ind] * Bz1[ind] +
                                  cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] - q + mz - 1)] * By2[ind] * Bx2[ind] * Bz1[ind] +
                                  cMat[int(ky[ind] - i + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * By1[ind] * Bx1[ind] * Bz2[ind] +
                                  cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * By2[ind] * Bx1[ind] * Bz2[ind] +
                                  cMat[int(ky[ind] - i + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * By1[ind] * Bx2[ind] * Bz2[ind] +
                                  cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * By2[ind] * Bx2[ind] * Bz2[ind])
            if yflag:
                By1 = evalBSpline(sy + i + 1 / 2, dy)
                for ind in np.arange(0, numelXi):
                    viMat[ind] = viMat[ind] + By1[ind] * \
                        (cMat[int(ky[ind] - i - 1 - 1), int(kx[ind] - j + mx - 1), int(kz[ind] - q + mz - 1)] * Bx1[ind] * Bz1[ind] +
                         cMat[int(ky[ind] - i - 1 - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] - q + mz - 1)] * Bx2[ind] * Bz1[ind] +
                         cMat[int(ky[ind] - i - 1 - 1), int(kx[ind] - j + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * Bx1[ind] * Bz2[ind] +
                         cMat[int(ky[ind] - i - 1 - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] + q - 1 + mz - 1)] * Bx2[ind] * Bz2[ind])
        if xflag:
            Bx1 = evalBSpline(sx + j + 1 / 2, dx)
            for i in np.arange(1, np.ceil((dy + 1) / 2) + 1):
                By1 = evalBSpline(sy + i - (1 + my) / 2, dy)
                By2 = evalBSpline(sy - i + (1 - my) / 2, dy)
                for ind in np.arange(0, numelXi):
                    viMat[ind] = viMat[ind] + Bx1[ind] * \
                        (cMat[int(ky[ind] - i + my - 1), int(kx[ind] - j - 1 - 1), int(kz[ind] - q + mz - 1)] * By1[ind] * Bz1[ind] +
                         cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] - j - 1 - 1), int(kz[ind] - q + mz - 1)] * By2[ind] * Bz1[ind] +
                         cMat[int(ky[ind] - i + my - 1), int(kx[ind] - j - 1 - 1), int(kz[ind] + q - 1 + mz - 1)] * By1[ind] * Bz2[ind] +
                         cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] - j - 1 - 1), int(kz[ind] + q - 1 + mz - 1)] * By2[ind] * Bz2[ind])
    if zflag:
        Bz1 = evalBSpline(sz + q + 1 / 2, dz)
        for j in np.arange(1, np.ceil((dx + 1) / 2) + 1):
            Bx1 = evalBSpline(sx + j - (1 + mx) / 2, dx)
            Bx2 = evalBSpline(sx - j + (1 - mx) / 2, dx)
            for i in np.arange(1, np.ceil((dy + 1) / 2) + 1):
                By1 = evalBSpline(sy + i - (1 + my) / 2, dy)
                By2 = evalBSpline(sy - i + (1 - my) / 2, dy)
                for ind in np.arange(0, numelXi):
                    viMat[ind] = viMat[ind] + Bz1[ind] * \
                        (cMat[int(ky[ind] - i + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] - q - 1 - 1)] * By1[ind] * Bx1[ind] +
                         cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] - j + mx - 1), int(kz[ind] - q - 1 - 1)] * By2[ind] * Bx1[ind] +
                         cMat[int(ky[ind] - i + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] - q - 1 - 1)] * By1[ind] * Bx2[ind] +
                         cMat[int(ky[ind] + i - 1 + my - 1), int(kx[ind] + j - 1 + mx - 1), int(kz[ind] - q - 1 - 1)] * By2[ind] * Bx2[ind])
    return viMat


if __name__ == '__main__':
    dataFile = 'interp3_0.mat'
    import scipy.io as scio
    data = scio.loadmat(dataFile)
    b = dict()
    b['coeffs'] = data['b3']['coeffs'][0][0]
    b['tensorOrder'] = data['b3']['tensorOrder'][0][0][0][0]
    b['dataSize'] =data['b3']['dataSize'][0][0][0]
    b['coeffsSize'] = data['b3']['coeffsSize'][0][0][0]
    b['degree'] = data['b3']['degree'][0][0][0]
    b['centred'] = data['b3']['centred'][0][0][0]
    b['elementSpacing'] = data['b3']['elementSpacing'][0][0][0]
    b['lambda'] = data['b3']['lambda'][0][0][0]

    Xq = data['Xq']
    Yq = data['Yq']
    Zq = data['Zq']
    shift = data['shift']
    zshiftf0 = data['zshiftf0']
    shiftedh = data['shiftedh']
    rsu = interp3_0(b,Xq-shift[0,1],Yq-shift[0,0],Zq-shift[0,2]-np.float64(zshiftf0[0]),0)
    c = rsu - shiftedh