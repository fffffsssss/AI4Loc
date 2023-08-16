from .ext_round import *
from csaps import csaps
from .registerPSF3D_so import robustMean
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def getspline_so(beads, p):
    curves = np.empty(len(beads), dtype=dict)
    for B in range(len(beads) - 1, -1, -1):
        beadz0 = (beads[B]['f0']) * p['dz']

        if 'astig' in p['zcorr'] or 'corr' in p['zcorr']:
            beadz = (beads[B]['loc']['frames'] * p['dz']) - beadz0
        else:
            beadz = (beads[B]['loc']['frames'] - p['midpoint']) * p['dz']
        sx = beads[B]['loc']['PSFxpix']
        sy = beads[B]['loc']['PSFypix']
        z = beadz
        phot = beads[B]['loc']['phot']
        inzr = np.nonzero((z >= float(p['gaussrange'][0])) & (z <= float(p['gaussrange'][1])))
        curves[B] = dict()
        curves[B]['sx'] = sx[inzr]
        curves[B]['sy'] = sy[inzr]
        curves[B]['z'] = z[inzr]
        curves[B]['phot'] = phot[inzr]
        curves[B]['xpos'] = beads[B]['pos'][0]
        curves[B]['ypos'] = beads[B]['pos'][1]

    # get calibrations
    spline, indgood, fig_list = getcleanspline(curves, p)
    return spline, indgood, curves, fig_list


def getcleanspline(curves, p):
    fig_list = []
    za = np.hstack([i['z'] for i in curves if 'z' in i])
    Sxa = np.hstack([i['sx'] for i in curves if 'sx' in i])
    Sya = np.hstack([i['sy'] for i in curves if 'sy' in i])

    indz = np.nonzero((za >= p['gaussrange'][0]) & (za <= p['gaussrange'][1]))
    z = za[indz]
    Sx = Sxa[indz]
    Sy = Sya[indz]

    # warn_log('curvefit:fit:iterationLimitReached')
    splinex = np.poly1d(np.polyfit(z, Sx, 6))
    spliney = np.poly1d(np.polyfit(z, Sy, 6))
    # warn_log('curvefit:fit:iterationLimitReached')

    err = np.zeros(len(curves))
    err2 = np.zeros(len(curves))
    err3 = np.zeros(len(curves))
    for k in range(len(curves) - 1, -1, -1):
        w = 1
        zh = curves[k]['z']
        indzh = np.where((zh >= p['gaussrange'][0]) & (zh <= p['gaussrange'][1]), 1, 0)
        w = w * indzh
        errh = np.power(curves[k]['sx'] - splinex(zh), 2) * w + np.power(curves[k]['sy'] - spliney(zh), 2) * w
        err[k] = np.sqrt(np.sum(errh) / np.sum(w))
        errh2 = (curves[k]['sx'] - splinex(zh)) * w / curves[k]['sx']
        errh3 = (curves[k]['sy'] - spliney(zh)) * w / curves[k]['sy']
        err2[k] = np.abs(np.sum(errh2) / np.sum(w))
        err3[k] = np.abs(np.sum(errh3) / np.sum(w))

    em, es = robustMean(err)
    if es == np.NaN:
        es = em
    indgood2 = np.where((err < em + 2.5 * es) & (err2 + err3 < 0.2), 1, 0)
    if np.sum(indgood2) == 0:
        indgood2 = np.ones([1, len(curves)], dtype=bool)
    zg = np.hstack([i['z'] for i in curves[indgood2 != 0] if 'z' in i])
    indz = np.nonzero((zg > p['gaussrange'][0]) & (zg < p['gaussrange'][1]))
    zg = zg[indz]
    sxg = np.hstack([i['sx'] for i in curves[indgood2 != 0] if 'sx' in i])
    syg = np.hstack([i['sy'] for i in curves[indgood2 != 0] if 'sy' in i])
    sxg = sxg[indz]
    syg = syg[indz]

    splinex2 = getspline(sxg, zg, 1.0 / (abs(sxg - splinex(zg)) + 0.1))
    spliney2 = getspline(syg, zg, 1.0 / (abs(syg - spliney(zg)) + 0.1))

    zt = np.arange(min(zg), max(zg) + 0.01, 0.01, dtype=np.float64)
    z1a = np.array([])
    z2a = np.array([])
    x1a = np.array([])
    x2a = np.array([])

    for k in range(0, len(curves)):
        if indgood2[k]:
            z1a = np.append(z1a, curves[k]['z'])
            z1a = np.append(z1a, curves[k]['z'])
            x1a = np.append(x1a, curves[k]['sx'])
            x1a = np.append(x1a, curves[k]['sy'])
        else:
            z2a = np.append(z2a, curves[k]['z'])
            z2a = np.append(z2a, curves[k]['z'])
            x2a = np.append(x2a, curves[k]['sx'])
            x2a = np.append(x2a, curves[k]['sy'])

    if len(z2a) == 0:
        z2a = 0
        x2a = 0

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.set_xlabel('z (nm)')
    ax.set_ylabel('PSFx, PSFy (pixel)')
    ax.set_xlim([zt[0], zt[-1]])
    ax.set_ylim([0, min(5, max(np.max(splinex2(zt)), np.max(spliney2(zt))))])
    ax.set_axis_on()
    ax.set_title("Lateral size of the PSF")
    ax.plot(z2a, x2a, 'o', color='green', label='bad bead data')
    ax.plot(z1a, x1a, 'o', color='red', label='good bead data')
    ax.plot(zt, splinex2(zt), color='black', label='spline fix sx')
    ax.plot(zt, spliney2(zt), color='black', label='spline fix sy')
    ax.legend()
    fig_list.append(['sx(z),sy(z)', figure])

    s = dict()
    s['x'] = splinex2
    s['y'] = spliney2
    s['zrange'] = [zt[0], zt[-1]]
    # title(p.ax, 'Lateral size of the PSF')

    zr = np.arange(zt[0], zt[-1], 1)
    midp = int(ext_round(len(zr) / 8))

    ind1x = np.argmax(s['x'](zr[0:midp - 1]))
    ind1y = np.argmax(s['y'](zr[0:midp - 1]))
    ind2x = np.argmax(s['x'](zr[midp - 1:-1]))
    ind2y = np.argmax(s['y'](zr[midp - 1:-1]))

    z1 = max(zr[ind1x], zr[ind1y])
    z2 = min(zr[ind2x + midp - 1], zr[ind2y + midp - 1])

    s['maxmaxrange'] = [z1, z2]
    return s, indgood2, fig_list


def getspline(S, z, w, p=0.96):
    zs = np.sort(z)
    zind = np.argsort(z)
    Ss = S[zind]
    ws = w[zind]
    spline = csaps(zs, Ss, weights=ws, smooth=p, normalizedsmooth=True)
    return spline
