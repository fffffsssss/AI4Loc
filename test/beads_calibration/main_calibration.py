import numpy as np
import matplotlib.pyplot as plt

from .image2beads import image2beads
from .zernike_calibrate import zernike_calibrate_beads
from .ext_round import *
from .getspline_so import getspline_so
from .get_stackcal_so import get_stackcal_so


def calibrate_data(parameters: dict):

    beads, parameters, fig_list = image2beads(parameters)
    plt.show()

    beadsposx = np.zeros(len(beads))
    beadsposy = np.zeros(len(beads))
    for k in range(len(beads) - 1, -1, -1):
        beadsposx[k] = beads[k]["pos"][0]
        beadsposy[k] = beads[k]["pos"][1]

    if 'fov' in parameters and parameters['fov'] is not None:
        indgood = np.nonzero((beadsposx >= parameters['fov'][0]) & \
                             (beadsposx <= parameters['fov'][2]) & \
                             (beadsposy >= parameters['fov'][1]) & \
                             (beadsposx <= parameters['fov'][3]))
        beads = beads[indgood]

    if 'mindistance' in parameters and parameters['mindistance'] is not None:
        indgood = np.ones(len(beads), dtype=bool)
        for k in range(0, len(beads)):
            for l in range(k + 1, len(beads)):
                dis = np.sum(np.power(beads[k]['pos'] - beads[l]['pos'], 2))
                if beads[k]['filenumber'] == beads[l]['filenumber'] and \
                        dis < np.power(parameters['mindistance'], 2):
                    indgood[k] = False
                    indgood[l] = False
        beads = np.array(beads)[indgood]

    if len(beads) == 0:
        return "error: could not find and segment any bead..."

    parameters['midpoint'] = ext_round((beads[0]['stack']['image'].shape[0]) / 2)
    parameters['ploton'] = False

    f0g = parameters['midpoint']
    for k in range(0, len(beads)):
        beads[k]['f0'] = f0g

    beadsposxs = np.zeros(len(beads))
    beadsposys = np.zeros(len(beads))
    beadfilenumber = np.zeros(len(beads))
    for k in range(len(beads) - 1, -1, -1):
        beadsposxs[k] = beads[k]['pos'][0]
        beadsposys[k] = beads[k]['pos'][1]
        beadfilenumber[k] = beads[k]['filenumber']

    imageRoi = parameters['imageRoi']
    SXY = np.empty([len(parameters['xrange']) - 1, len(parameters['yrange']) - 1], dtype=dict)
    for X in range(0, len(parameters['xrange']) - 1):
        for Y in range(0, len(parameters['yrange']) - 1):
            indgood = np.nonzero((beadsposxs + imageRoi[0] < parameters['xrange'][X + 1]) & \
                                 (beadsposxs + imageRoi[0] > parameters['xrange'][X]) & \
                                 (beadsposys + imageRoi[1] < parameters['yrange'][Y + 1]) & \
                                 (beadsposys + imageRoi[1] > parameters['yrange'][Y]))
            beadsh = beads[indgood]

            if len(beadsh) == 0:
                continue

            gausscal = dict()
            spline_fig = None

            indgoodc = np.ones(len(beadsh), dtype=bool)

            csplinecal, indgoods, shift, fig_list = get_stackcal_so(beadsh[indgoodc != 0], parameters)
            plt.show()

            icf = np.where(indgoodc)[0]
            icfs = icf[indgoods]
            cspline = dict()
            coeff = np.zeros((1,), dtype=object)
            coeff[0] = csplinecal['cspline']['coeff']
            cspline['coeff'] = coeff
            cspline['dz'] = np.float64(csplinecal['cspline']['dz'])
            cspline['z0'] = np.float64(csplinecal['cspline']['z0'])
            cspline['x0'] = np.float64(csplinecal['cspline']['x0'])

            gausscal = []
            gauss_sx2_sy2 = []
            gauss_zfit = []
            parameters['ax_sxsy'] = []

            cspline_all = csplinecal
            PSF = csplinecal['PSF']
            SXY[X - 1, Y - 1] = {'gausscal': gausscal, 'cspline_all': cspline_all, 'gauss_sx2_sy2': gauss_sx2_sy2, \
                                 'gauss_zfit': gauss_zfit, 'cspline': cspline,
                                 'Xrangeall': parameters['xrange'] + imageRoi[0],
                                 'Yrangeall': parameters['yrange'] + imageRoi[1],
                                 'Xrange': parameters['xrange'][X:X + 1] + imageRoi[0],
                                 'Yrange': parameters['yrange'][Y:Y + 1] + imageRoi[1],
                                 'posind': [X, Y], 'EMon': parameters['emgain'], 'PSF': PSF}

    if(np.size(SXY )==1):
        SXY = SXY[0, 0]

    # zernike fitting process
    if parameters['use_zernike_fit']:
        stack = PSF.transpose(2, 0, 1).copy()
        rxy = np.floor(parameters['ROIxy'] / 2).astype(int)
        mp = np.ceil(np.shape(stack)[1] / 2).astype(int)
        zborder = np.round(100 / parameters['dz']).astype(int)
        stack = stack[zborder + 1:-zborder + 1, mp - rxy:mp + rxy + 1, mp - rxy:mp + rxy + 1]
        stack *= 1000
        zernike_fit_results, fig_list = zernike_calibrate_beads(stack, parameters)
        plt.show()

    # if parameters['smap']:
    result = {'SXY': SXY, 'parameters': parameters}
    if parameters['use_zernike_fit']:
        result['zernike_fit_results'] = zernike_fit_results

    # else:
    #     result = {'gausscal': gausscal, 'cspline_all': cspline_all, 'gauss_sx2_sy2': gauss_sx2_sy2, \
    #               'gauss_zfit': gauss_zfit, 'cspline': cspline, 'parameters': parameters}

    return result