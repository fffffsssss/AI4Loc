# stackas2z function
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

from .ext_round import *


def stackas2z(sxa, sya, za, na, p):
    """
    stackas2z
    :param data: Parameter, contains sxa, sya, za, na, p
    :return: zas
    """
    dz = p['dz']
    fminmax = p['fminmax']

    wind = 4 / dz * 50
    windn = 10 / dz * 50
    midp = np.mean(fminmax)
    numf = max(500 / dz, len(sxa) / 8) #index: should -1
    range = np.where(abs(za - midp) < numf)

    sx = sxa[range]
    sy = sya[range]
    z = za[range]
    n = na[range]

    indm = np.argmax(np.true_divide(np.true_divide(n, np.maximum(0.5, sx)), np.maximum(0.5, sy)))

    winstd = ext_round(500 / dz)
    range = np.arange(max(0, indm - winstd),min(indm + winstd, len(n))).astype(np.int32)

    nindx = np.argmin(sx[range])
    nindy = np.argmin(sy[range])

    nxind = ext_round((nindx + nindy) / 2) + range[1] - 1

    fstartn = int(max(nxind - windn, 0))
    fstopn = int(min(nxind + windn, len(n)))

    indx1 = np.where(sx[fstartn:fstopn] > sy[fstartn:fstopn])[0][0] + fstartn
    indx2 = np.where(sy[fstartn:fstopn] > sx[fstartn:fstopn])[0][0] + fstartn
    sxind = max(indx1, indx2) - 1
    fstart = int(max(sxind - wind, 1))
    fstop = int(min(sxind + wind, len(n)))

    sxfit = sx[fstart:fstop + 1]
    syfit = sy[fstart:fstop + 1]
    x = z[fstart:fstop + 1]

    psx = np.polyfit(x, sxfit, 2)
    psy = np.polyfit(x, syfit, 2)

    s_df = max(0, len(syfit) - 3)
    s_norm = np.linalg.norm(syfit - np.polyval(psy, x))

    fmin = -psx[1] / 2 / psx[0]
    zsx = fmin
    fminy = -psy[1] / 2 / psy[0]
    zsy = fminy

    nfit = n[fstart:fstop + 1]
    p = np.polyfit(x, nfit, 2)

    fmin = -p[1] / 2 / p[0]
    zn = fmin

    ro = np.roots(psy - psx)

    if s_df < 3 or s_norm > 150 or len(x) < 5 or len(ro) < 2:
        zas = np.NaN
    else:
        mp = z[fstop]
        d = abs(ro - mp)
        mi = np.argmin(d)
        zas = ro[mi]
        if np.imag(zas) != 0:
            zas = np.NaN
    return zas

