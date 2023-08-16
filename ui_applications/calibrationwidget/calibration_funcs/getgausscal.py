# getgausszal function
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import numpy as np
from csaps import csaps
from scipy.optimize import leastsq, least_squares
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def getgausscal(p, curves, spline_fig = None):
    gauss = dict()
    gauss["Sx2_Sy2"], fig_list = cal_Sx2_Sy2(curves, p)
    gauss["fitzpar"] = cal_fitzpar(curves, p, spline_fig)
    return gauss, fig_list


def cal_Sx2_Sy2(curves, p):
    z = np.squeeze(np.vstack(curves['z']))
    sx = np.squeeze(np.vstack(curves['sx']))
    sy = np.squeeze(np.vstack(curves['sy']))
    zrange = p['gaussrange']
    spline, ds_range, fig_list = fitsx2sy2(sx, sy, z, zrange)
    return {"function": spline, "ds2range": ds_range}, fig_list


def fitsx2sy2(sx: np.ndarray, sy: np.ndarray, z: np.ndarray, zrange: np.ndarray):
    indf = np.where((zrange[0] < z) & (z < zrange[1]), 1, 0)
    ds = sx ** 2 - sy ** 2
    q1 = np.quantile(ds[indf != 0], 0.05, method="hazen") - 2
    q2 = np.quantile(ds[indf != 0], 0.95, method="hazen") + 2

    inds = np.where(np.logical_and(q1 < ds, ds < q2), 1, 0)
    inds = indf & inds

    spline = []
    ds_range = []
    if np.sum(indf) > 5:
        spline = csaps(sx[inds != 0] ** 2 - sy[inds != 0] ** 2, z[inds != 0], smooth=0.95, normalizedsmooth=True)
        indgood = np.where(np.logical_and(zrange[0] < spline(ds), spline(ds) < zrange[1]), 1, 0)
        ds_range = [np.min(ds[indgood != 0]), np.max(ds[indgood != 0])]

        sxsort = np.sort(sx ** 2 - sy ** 2)
        zsort = spline(sxsort)

        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.set_xlabel('sx^2-sy^2')
        ax.set_ylabel('z (nm)')
        ax.set_title('Calibration for Gaussian fit')
        ax.set_xlim([zrange[0], zrange[1]])
        ax.set_ylim([q1, q2])
        ax.set_axis_on()
        ax.plot(z[inds != 0], sx[inds != 0] ** 2 - sy[inds != 0] ** 2, marker='.')
        ax.plot(zsort, sxsort, color='k')
        ax.plot(zrange, [ds_range[0], ds_range[0]], zrange, [ds_range[1],ds_range[1]])
        #ax.legend()

    return spline, ds_range, [['Gauss cal', figure]]


def cal_fitzpar(curves, p, spline_fig):
    z = np.squeeze(np.vstack(curves['z']))
    sx = np.squeeze(np.vstack(curves['sx']))
    sy = np.squeeze(np.vstack(curves['sy']))
    zrange = p['gaussrange']
    fitzpar = getzfitpar(sx, sy, z, zrange, 0, True, spline_fig)
    return fitzpar


def getzfitpar(sx, sy, znm, zrange, midpoint, B0, spline_fig):
    startp = np.array([0.3, 1.0, 1.0000, 0, 0, 0, 0, 0.307, - midpoint / 1000])
    startp[1] = np.quantile(sx, 0.01)
    startp[2] = np.quantile(sy, 0.01)

    ind = np.where(np.logical_and(zrange[0] < znm, znm < zrange[1]), 1, 0)
    sx = sx[ind != 0]
    sy = sy[ind != 0]
    znm = znm[ind != 0]
    z = znm / 1000

    fitp = least_squares(sbothfromsigmaerr, startp, args=(np.hstack([z, z]), np.hstack([sx, sy]), 0))
    if B0:
        fitp = least_squares(sbothfromsigmaerr, startp, args=(np.hstack([z, z]), np.hstack([sx, sy]), True))

    zt = np.linspace(np.min(z), np.max(z), int(np.floor((np.max(z) - np.min(z)) / 0.01)) + 1)
    zfit = np.real(fitp['x'][[1, 3, 4, 5, 6, 7, 0, 2]])
    # if spline_fig is not None:
    #     #ax = spline_fig.subplots()
    #     ax = spline_fig.add_axes([0, 0, 1, 1])
    #     sxf = sigmafromz(fitp['x'][[0, 1, 3, 5, 7, 8]], zt, B0)
    #
    #     ax.plot(z * 1000, sx, color='b', marker='*', label='average PSF')
    #     ax.plot(z * 1000, sy, color='b', marker='*', label='average PSF')
    #     ax.plot(zt * 1000, sxf, color='b', linestyle='-', label='Gauss zfit')
    #     fpy = fitp['x'][[0, 2, 4, 6, 7, 8]]
    #     fpy[4] = -fpy[4]
    #     syf = sigmafromz(fpy, zt, B0)
    #     ax.plot(zt * 1000, syf, color='b', linestyle='-', label='Gauss zfit')
    return zfit


def sbothfromsigma(par, z, B0):
    px = par[[0, 1, 3, 5, 7, 8]]
    py = par[[0, 2, 4, 6, 7, 8]]
    py[4] = -py[4]
    zh = z[0:int(z.size / 2)]
    sx = sigmafromz(px, zh, B0)
    sy = sigmafromz(py, zh, B0)
    s = np.hstack([sx, sy])
    return s


def sbothfromsigmaerr(par, z, sx, B0):
    sf = sbothfromsigma(par, z, B0)
    err = sf - sx
    err = err / np.sqrt(abs(err))
    return err


def sigmafromz(par, z, B0):
    s0 = par[1]
    d = par[0]
    A = par[2]
    B = par[3] * B0
    g = par[4]
    mp = par[5]
    v = 1 + (z - g + mp) ** 2 / (d ** 2) + A * ((z - g + mp) ** 3)/ (d ** 3) + B * ((z - g + mp) ** 4) / (d ** 4)
    s = s0 * np.sqrt(v)
    return s
