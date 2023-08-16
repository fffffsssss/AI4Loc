# Image to beads
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from .mat2py import *
from .ext_round import *


def image2beads(p: dict):
    """
    Images to beads
    :param p: Parameters
    :return: The beads result
    """
    # filter size
    fs = p['filtersize']
    h = fspecial_gauss(2 * ext_round(fs * 3 / 2) + 1, fs)
    fmax = 0
    roisize = p['ROIxy']
    roisizeh = ext_round(1.5 * (p['ROIxy'] - 1) / 2)
    rsr = np.arange(-roisizeh, roisizeh + 1)

    b = []
    p['images'] = []
    fig_list = []
    for k in range(0, len(p['filelist'])):
        imstack = read_tiff(p['filelist'][k])
        imstack = imstack - imstack.min()
        mim = imstack.max(0)
        mim = filter2(h, mim)
        maxima = maximum_find(mim)

        intv = maxima[:, 2]
        try:
            mimc = mim[roisize-1:-roisize, roisize-1:-roisize]
            mmed = quantile(mimc, 0.3)
            mimc_f = mimc.reshape(mimc.size, order='F')
            imt = mimc_f[mimc_f < mmed]
            sm = np.sort(intv)
            mv = np.mean(sm[-6:])
            cutoff = np.mean(imt) + max(2.5 * np.std(imt), (mv - np.mean(imt)) / 15)
        except:
            cutoff = quantile(mimc, 0.95)

        cutoff *= p['relative_cutoff']

        if np.any(intv > cutoff):
            maxima = maxima[intv > cutoff, :]
        else:
            indm = np.argmax(intv)
            maxima = maxima[indm, :]

        if 'beadpos' in p:
            maxima = ext_round(p['beadpos'])

        # figure
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title("Maximum intensity projection")
        ax.imshow(mim, cmap='jet', vmin=np.min(mim), vmax=4 * np.max(mim) / 3)
        ax.scatter(maxima[:, 0], maxima[:, 1], marker="o", edgecolors="red", facecolors="none",alpha=0.5)
        fig_list.append(['File %d' % (k + 1), figure])

        numframes = imstack.shape[0]
        bind = len(b) + maxima.shape[0] - 1
        for i in range(0, maxima.shape[0]):
            b.append(dict())

        for i in range(0, maxima.shape[0]):
            b[bind]['loc'] = dict()
            b[bind]['loc']['frames'] = np.arange(1, numframes + 1).T
            b[bind]['loc']['filenumber'] = np.zeros(numframes, order='F') + k
            b[bind]['filenumber'] = k
            b[bind]['pos'] = maxima[i, 0:2]

            try:
                b[bind]['stack'] = dict()
                idx1 = np.array(b[bind]['pos'][1] + rsr).astype(int)
                idx2 = np.array(b[bind]['pos'][0] + rsr).astype(int)
                img = imstack[:, idx1, :][:, :, idx2]
                b[bind]['stack']['image'] = img.transpose(1, 2, 0).copy()
                b[bind]['stack']['framerange'] = np.arange(1, numframes + 1)
                b[bind]['isstack'] = True
            except:
                b[bind]['isstack'] = False

            if 'files' in p:
                b[bind]['roi'] = p['files'][k]['info']['roi']
            else:
                b[bind]['roi'] = [0, 0, imstack.shape[1], imstack.shape[2]]

            bind -= 1
        fmax = max(fmax, numframes)

    r = [x for x in b if x['isstack']]

    p['fminmax'] = [1, fmax]

    if 'files' in p:
        p['cam_pixelsize_um'] = p['files'][-1]['info']['cam_piexlsize_um']
    else:
        p['cam_pixelsize_um'] = np.ones(2) / 1000.  # FIXME

    p['pathhere'] = os.path.dirname(p['filelist'][0])

    return r, p, fig_list