# Calibrate3D
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import time
import matplotlib
matplotlib.use('Agg')
from PyQt5.QtCore import *

from .image2beads import *
from .getgausscal import *
from .get_stackcal_so import *
from .stackas2z import *
from .getspline_so import *
from .zernike_calibrate import *


class Calibrate3DProcess(QObject):
    """
    Image processing.
    """
    raw_data_recv = pyqtSignal(dict)
    figure_show = pyqtSignal(list)
    message_send = pyqtSignal(str)
    data_calibrated = pyqtSignal(dict)

    def __init__(self):
        """
        Constructor.
        """
        QObject.__init__(self)
        self.thispath = os.path.dirname(os.path.abspath(__file__))
        cpupath = self.thispath + '/CPUmleFit_LM.dll'
        gpupath = self.thispath + '/GPUmleFit_LM.dll'
        self.dll = mleFit(usecuda=0, cpu_path=cpupath, gpu_path=gpupath)
        # self.dll = mleFit(usecuda=1, gpu_path=gpupath)

    @pyqtSlot(dict)
    def calibrate_data(self, parameters:dict):
        beads, parameters, fig_list = image2beads(parameters)
        self.figure_show.emit(fig_list)

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
            # warn_log('Could not find and segment any bead. ROI size too large?')
            self.message_send.emit("error: could not find and segment any bead...")
            return

        parameters['midpoint'] = ext_round((beads[0]['stack']['image'].shape[0]) / 2)
        parameters['ploton'] = False

        if 'astig' in parameters['modality']:
            t = time.time()
            self.message_send.emit("Gaussian fit of beads to get spatial parameters")
            for k in range(0, len(beads)):
                stackh = beads[k]['stack']['image'].astype(float)
                s = stackh.shape
                d = int(ext_round((s[1] - parameters['gaussroi']) / 2)) - 1  # index 1 to 0
                stack = stackh[d + 1:s[0] - 1 - d, d + 1:s[1] - 1 - d, :].transpose(2, 1, 0).copy()
                P, CRLB, LL, coeff = self.dll.fit_coef(stack, 4, 100, 1)

                beads[k]['loc']['PSFxpix'] = P[4, :]
                beads[k]['loc']['PSFypix'] = P[5, :]
                beads[k]['loc']['phot'] = P[2, :]
                beads[k]['f0'] = stackas2z(beads[k]['loc']['PSFxpix'], beads[k]['loc']['PSFypix'],
                                           beads[k]['loc']['frames'], beads[k]['loc']['phot'], parameters)
                beads[k]['loc']['bg'] = P[3, :]
                ind_all = np.where(beads[k]['loc']['frames'] <= beads[k]['f0'])
                if beads[k]['f0'] == np.NaN or len(ind_all) == 0 or len(ind_all[0]) == 0:
                    ind = 1
                else:
                    ind = ind_all[0][-1]
                beads[k]['psfx0'] = beads[k]['loc']['PSFxpix'][ind]
                beads[k]['psfy0'] = beads[k]['loc']['PSFypix'][ind]
                if (time.time() - t) > 1:
                    self.message_send.emit('Gaussian fit of beads to get spatial parameters: %d of %d' % (k, len(beads)))
                    t = time.time()
                if beads[k]['f0'] == np.NaN:
                    beads[k] = []
        else:
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
                    self.message_send.emit('no beads found in part X(%s, %s), Y(%s, %s)' % \
                                           (str(parameters['xrange'][X]), \
                                            str(parameters['xrange'][X + 1]), \
                                            str(parameters['yrange'][Y]), \
                                            str(parameters['yrange'][Y:Y + 1])))
                    continue

                gausscal = dict()
                spline_fig = None
                if 'astig' in parameters['modality']:
                    self.message_send.emit('get spline approximation')
                    spline_curves, indgoodc, curves, fig_list = getspline_so(beadsh, parameters)
                    spline_fig = fig_list[0][1]
                    self.figure_show.emit(fig_list)
                    gausscal['spline_curves'] = spline_curves
                else:
                    indgoodc = np.ones(len(beadsh), dtype=bool)

                self.message_send.emit('start cspline calibration')

                csplinecal, indgoods, shift, fig_list = get_stackcal_so(beadsh[indgoodc != 0], parameters)
                self.figure_show.emit(fig_list)

                icf = np.where(indgoodc)[0]
                icfs = icf[indgoods]
                cspline = dict()
                coeff = np.zeros((1,), dtype=object)
                coeff[0] = csplinecal['cspline']['coeff']
                cspline['coeff'] = coeff
                cspline['dz'] = np.float64(csplinecal['cspline']['dz'])
                cspline['z0'] = np.float64(csplinecal['cspline']['z0'])
                cspline['x0'] = np.float64(csplinecal['cspline']['x0'])

                if 'astig' in parameters['modality']:
                    photbead = 10 ** 5
                    stackb = csplinecal['PSF']
                    stackb = (stackb) * photbead
                    mp = int(np.ceil(np.shape(stackb)[0] / 2))
                    dx = int(np.floor(parameters['gaussroi'] / 2))

                    ch = dict()
                    stack = stackb[mp - dx - 1:mp + dx + 1, mp - dx - 1: mp + dx + 1, :].transpose(2, 1, 0).copy()
                    P, CRLB, LL, coeff = self.dll.fit_coef(stack, 4, 200, 1)
                    ch['sx'] = P[4, :]
                    ch['sy'] = P[5, :]
                    f0m = np.median([i['f0'] for i in beadsh[icfs] if 'f0' in i])
                    ch['z'] = (np.arange(1, np.size(stack, 0) + 1) - f0m) * parameters['dz']

                    self.message_send.emit('get Gauss model calibration')
                    gausscalh, fig_list = getgausscal(parameters, ch, spline_fig)
                    self.figure_show.emit(fig_list)

                    gauss_zfit = gausscalh['fitzpar']
                    gauss_sx2_sy2 = gausscalh['Sx2_Sy2']
                    gausscal['fitzpar'] = gauss_zfit
                    gausscal['Sx2_Sy2'] = gauss_sx2_sy2
                else:
                    gausscal = []
                    gauss_sx2_sy2 = []
                    gauss_zfit = []
                    parameters['ax_sxsy'] = []

                cspline_all = csplinecal
                PSF = csplinecal['PSF']
                SXY[X - 1, Y - 1] = {'gausscal': gausscal, 'cspline_all': cspline_all, 'gauss_sx2_sy2': gauss_sx2_sy2, \
                                     'gauss_zfit': gauss_zfit, 'cspline': cspline, \
                                     'Xrangeall': parameters['xrange'] + imageRoi[0], \
                                     'Yrangeall': parameters['yrange'] + imageRoi[1], \
                                     'Xrange': parameters['xrange'][X:X + 1] + imageRoi[0], \
                                     'Yrange': parameters['yrange'][Y:Y + 1] + imageRoi[1], \
                                     'posind': [X, Y], 'EMon': parameters['emgain'], 'PSF': PSF}

        if(np.size(SXY)==1):
            SXY = SXY[0, 0]

        # zernike fitting process
        if parameters['use_zernike_fit']:
            stack = PSF.transpose(2, 0, 1).copy()
            rxy = np.floor(parameters['ROIxy']/2).astype(int)
            mp = np.ceil(np.shape(stack)[1]/2).astype(int)
            zborder = np.round(100/parameters['dz']).astype(int)
            stack = stack[zborder+1:-zborder+1, mp-rxy:mp+rxy+1, mp-rxy:mp+rxy+1]
            stack *= 1000
            self.message_send.emit('start zernike fitting')
            zernike_fit_results, fig_list = zernike_calibrate_beads(stack, parameters)
            self.figure_show.emit(fig_list)

        self.message_send.emit('emit calibrated result')

        if parameters['smap']:
            result = {'SXY': SXY, 'parameters': parameters}
            if parameters['use_zernike_fit']:
                result['zernike_fit_results'] = zernike_fit_results
            self.data_calibrated.emit(result)
        else:
            result = {'gausscal': gausscal, 'cspline_all': cspline_all, 'gauss_sx2_sy2': gauss_sx2_sy2, \
                      'gauss_zfit': gauss_zfit, 'cspline': cspline, 'parameters': parameters}
            self.data_calibrated.emit(result)
