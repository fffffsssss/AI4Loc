# Simple fitter with cubic spline
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import os
from .logger import *
from .mat2py import *
from .mlefit import *
import matplotlib.pyplot as plt


def fitspline(im, peakcoordinates, p, varstack):
    """
    Fit spline
    """
    if p['isspline']:
        if p['bidirectional']:
            fitmode = 6
        else:
            fitmode = 5
        fitpar = np.transpose(p['coeff']).copy()
    else:
        if p['bidirectional']:
            fitmode = 2
        else:
            fitmode = 4
        fitpar = np.float32(1)

    thispath = os.path.dirname(os.path.abspath(__file__))
    cpupath = os.path.abspath(thispath + '\\..\\..\\lib\\dll\\CPUmleFit_LM.dll')

    dll = mleFit(usecuda=0, cpu_path=cpupath)

    data = np.transpose(np.float32(im)).copy()
    r = dll.fit_coef(data, fitmode, 50, fitpar)
    Pcspline = r[0].T
    CRLB = r[1].T
    LL = r[2].T

    results = np.zeros((data.shape[0], 12))
    results[:, 0] = peakcoordinates[:, 2]
    if p['mirror']:
        results[:, 1] = p['dx'] - Pcspline[:, 1] + peakcoordinates[:, 0]
    else:
        results[:, 1] = Pcspline[:, 1] - p['dx'] + peakcoordinates[:, 0]

    if p['isspline']:
        results[:, 2] = Pcspline[:, 0] - p['dx'] + peakcoordinates[:, 1]
        results[:, 3] = (Pcspline[:, 4] - p['z0']) * p['dz']
        results[:, 4:6] = Pcspline[:, 2:4]
        results[:, 6:8] = np.real(np.sqrt(CRLB[:, 1::-1]))
        results[:, 8] = np.real(np.sqrt(CRLB[:, 4] * p['dz']))
        results[:, 9:11] = np.real(np.sqrt(CRLB[:, 2:4]))
        results[:, 11] = LL
    else:
        results[:, 2] = Pcspline[:, 0] - p['dx'] + peakcoordinates[:, 1]
        results[:, 3] = Pcspline[:, 4]
        if p['bidirectional']:
            results[:, 4] = Pcspline[:, 3]
        else:
            results[:, 4] = Pcspline[:, 5]
        results[:, 5:7] = Pcspline[:, 2:4]
        results[:, 7:9] = np.real(np.sqrt(CRLB[:, 1::-1]))
        results[:, 9:11] = np.real(np.sqrt(CRLB[:, 2:4]))
        results[:, 11] = LL

    return results


def write_csv(result, filename, isspline):
    """
    Write to CSV.
    """
    if isspline:
        header = "frame,x_pix,y_pix,z_nm,photons,background,crlb_x,crlb_y,crlb_z,crlb_photons,crlb_background," + \
                "logLikelyhood,x_nm,y_nm,crlb_xnm,crlb_ynm"
    else:
        header = "frame,x_pix,y_pix,sx_pix,sy_pix,photons,background, crlb_x,crlb_y,crlb_photons,crlb_background," + \
                 "logLikelyhood,x_nm,y_nm,crlb_xnm,crlb_ynm"
    fmt = ["%d", "%.18f", "%.2f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f", "%.18f",
           "%.18f", "%d", "%.18f", "%.18f"]
    np.savetxt(filename, result, fmt=fmt, header=header, delimiter=',', comments="")


def simplefitter_cspline(p: dict):
    """
    Simple fitter with cubic spline
    :param p: Parameters
    :return: True or False
    """

    fittime = 0
    fitsperblock = 50000

    imstack = np.zeros((p['roifit'], p['roifit'], fitsperblock))
    peakcoordinates = np.zeros((fitsperblock, 3), dtype=int)

    indstack = 0
    resultsind = 0
    results = np.empty((0, 12))

    bgmode = 0
    if "avelet" in p['backgroundmode']:
        bgmode = 2
    elif "aussian" in p['backgroundmode']:
        bgmode = 1

    # scmos
    varmap = np.empty((0,))
    varstack = 0
    if p['isscmos']:
        varstack = np.ones((p['roifit'],  p['roifit'], fitsperblock))
        if p['scmosfile'][-4:] == ".tif":
            varmap = read_tiff(p['scmosfile'])
        elif p['scmosfile'][-4:] == ".mat":
            varmap = load_mat(p['scmosfile'])
            if 'varmap' in varmap:
                varmap = varmap['varmap']
        else:
            error_log("Could not load variance map. No sCMOS noise model used.")
            p['isscmos'] = False

    varmap = varmap * p['conversion'] ** 2

    # results
    # load calibration
    if len(p['calfile']) > 0:
        cal = load_mat(p['calfile'])

        if('SXY' in cal.keys()):
            p['dz'] = cal['SXY']['cspline'][0][0]['dz'][0][0][0][0]  # coordinate system of spline PSF is corner based and in units pixels / planes
            p['z0'] = cal['SXY']['cspline'][0][0]['z0'][0][0][0][0]
            p['coeff'] = cal['SXY']['cspline'][0][0]['coeff'][0][0][0][0]
            p['isspline'] = True
        else:
            p['dz'] = cal['cspline']['dz'][0][0]  # coordinate system of spline PSF is corner based and in units pixels / planes
            p['z0'] = cal['cspline']['z0'][0][0]
            p['coeff'] = cal['cspline']['coeff'][0][0]
            p['isspline'] = True
    else:
        warn_log("3D calibration file could not be loaded. Using Gaussian fitter instead!")
        p['isspline'] = False

    p['dx'] = int(np.floor(p['roifit'] / 2))

    p['status'] = "Open tiff file"

    numframes = 0
    images = []
    if "TIF" in p['loader']:
        images = read_tiff(p['imagefile'])
        numframes = images.shape[0]
    else:
        warn_log("Unsupport loader: " + p['loader'])
        return False

    frames = range(0, numframes)
    if p['preview']:
        frame = min(p['previewframe'], numframes)
        frames = (frame - 1,)

    # loop over frames, do filtering/peakfinding
    hgauss = fspecial_gauss(max(3, int(np.ceil(3 * p['peakfilter'] + 1))), p['peakfilter'])
    rsize = max(int(np.ceil(6 * p['peakfilter'] + 1)), 3)
    hdog = fspecial_gauss(rsize, p['peakfilter']) - fspecial_gauss(rsize, max(1, int(2.5 * p['peakfilter'])))

    for i in frames:
        image = images[i]
        sim = image.shape
        imphot = (image - p['offset']) * p['conversion']

        if bgmode == 1:
            impf = filter2(hdog, (imphot - min(imphot[:, 0])))
        elif bgmode == 2:
            impf = filter2(hgauss, imphot)

        maxima = maximum_find(impf)

        indmgood = maxima[:, 2] > p['peakcutoff']
        for j in range(indmgood.shape[0]):
            if indmgood[j]:
                if p['dx'] <= maxima[j][0] < (sim[1] - p['dx']) and p['dx'] <= maxima[j][1] < (sim[0] - p['dx']):
                    indmgood[j] = True
                else:
                    indmgood[j] = False

        maxgood = maxima[indmgood, :]

        if p['preview']:
            if maxgood.shape[0] > 2000:
                p['status'] = "Warning: too many localizations found, increase cutoff please!"
                return False
            elif maxgood.shape[0] == 0:
                p['status'] = "Warning: No localizations found, decrease cutoff please!"
                return False

        # Cut out images
        for j in range(0, maxgood.shape[0]):
            if p['dx'] <= maxgood[j][0] < (sim[1] - p['dx']) and p['dx'] <= maxgood[j][1] < (sim[0] - p['dx']):
                if p['mirror']:
                    imstack[:, :, indstack] = imphot[int(maxgood[j][1] - p['dx']):int(maxgood[j][1] + p['dx'] + 1),
                                                     int(maxgood[j][0] + p['dx'] + 1):-1:int(maxgood[j][0] - p['dx'])]
                else:
                    imstack[:, :, indstack] = imphot[int(maxgood[j][1] - p['dx']):int(maxgood[j][1] + p['dx'] + 1),
                                                     int(maxgood[j][0] - p['dx']):int(maxgood[j][0] + p['dx'] + 1)]

                if p['isscmos']:
                    varstack[:, :, indstack] = varmap[int(maxgood[j][1] - p['dx']):int(maxgood[j][1] + p['dx'] + 1),
                                                      int(maxgood[j][0] - p['dx']):int(maxgood[j][0] + p['dx'] + 1)]

                peakcoordinates[indstack, 0:2] = maxgood[j, 0:2]
                peakcoordinates[indstack, 2] = i

                if indstack == fitsperblock:
                    p['status'] = "Fitting..."
                    resultsh = fitspline(imstack, peakcoordinates, p, varstack)
                    results = np.vstack((results, resultsh))
                    resultsind += fitsperblock
                    indstack = 0
                else:
                    indstack += 1

    p['status'] = "Fitting last stack..."
    if indstack < 1:
        p['status'] = "No localizations found. Increase cutoff?"

    if p['isscmos']:
        varh = varstack[:, :, 0:indstack]
    else:
        varh = 0

    resultsh = fitspline(imstack[:, :, 0:indstack], peakcoordinates[0:indstack, :], p, varh)
    results = np.vstack((results, resultsh))


    if p['preview']:
        p['images'] = []
        # figure
        figure = plt.figure()
        ax = figure.add_axes([0.1, 0.05, 0.75, 0.9])
        ax.set_aspect('equal')
        im = ax.imshow(impf, cmap='jet', vmin=np.min(impf), vmax=4 * np.max(impf) / 3)
        ax.scatter(maxgood[:, 0], maxgood[:, 1], marker="o", edgecolors="white", facecolors="none")
        ax.scatter(results[:, 1], results[:, 2], marker="+", facecolors="black")
        ax = figure.add_axes([0.88, 0.05, 0.05, 0.9])
        figure.colorbar(im, cax=ax)
        p['images'].append(figure)
        p['results'] = results
    else:
        tmp = results[:, 1] * p['pixelsize']
        results = np.column_stack((results, tmp))

        tmp = results[:, 2] * p['pixelsize']
        results = np.column_stack((results, tmp))

        tmp = results[:, 6] * p['pixelsize']
        results = np.column_stack((results, tmp))

        tmp = results[:, 7] * p['pixelsize']
        results = np.column_stack((results, tmp))

        write_csv(results, p['outputfile'], p['isspline'])
        p['results'] = results

    return True
