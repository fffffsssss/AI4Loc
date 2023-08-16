from ctypes import *
#from tqdm import tqdm
from .psf2cspline import *
import numpy.ctypeslib as ctl
import numpy as np


class mleFit:
    def __init__(self,usecuda=0,gpu_path=None,cpu_path=None):
        if usecuda==1:
            self.psf_gpu = CDLL(gpu_path, winmode=0)
            self._mleFit = self.psf_gpu.GPUmleFit_LM
        else:
            self.psf_cpu = CDLL(cpu_path, winmode=0)
            self._mleFit = self.psf_cpu.CPUmleFit_LM
        self._mleFit.argtypes = [
            ctl.ndpointer(np.float32),  # data
            c_int32,  # fittype: 1 fixed sigma; 2 free sigma; 4 sigmax and sigmay; 5 spline
            c_int32,  # iterations
            ctl.ndpointer(np.float32),  # spline_coeff or PSF sigma
            ctl.ndpointer(np.float32),  # varim
            c_float,  # init_z
            ctl.ndpointer(np.int32),  # datasize
            ctl.ndpointer(np.int32),  # spline_size
            ctl.ndpointer(np.float32),  # P
            ctl.ndpointer(np.float32),  # CRLB
            ctl.ndpointer(np.float32)  # LL
        ]


    def fit_coef(self, psf_data, fittype,iterations = 30, coeff=None, pixelsize_z=None):
        rsz = psf_data.shape[-1]

        Nfit = psf_data.shape[-3]
        fittype = np.int32(fittype)
        if fittype == 5:
            coeff = coeff.astype(np.float32)
            splinesize = np.array(np.flip(coeff.shape))
            ccz = coeff.shape[-3] // 2
            #initparam = np.array([-3, -2, -1, 0, 1, 2, 3]) * 0.3 / pixelsize_z + ccz
            initparam = ccz
            paramstart = (initparam,)
        elif fittype==3:
            coeff = coeff.astype(np.float32)
            paramstart = np.array([1.5]).astype(np.float32)
            splinesize = np.array([0])
        else:
            paramstart = np.array([coeff]).astype(np.float32)
            coeff = paramstart
            splinesize = np.array([0])


        if fittype == 1:
            Nparam = 4
        elif fittype == 4:
            Nparam = 6
        else:
            Nparam = 5
        data = psf_data.astype(np.float32)

        datasize = np.array(np.flip(data.shape))

        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam + 1, Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam, Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)

        P = np.zeros((Nparam + 1, Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam, Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32) - 1e10

        #pbar = tqdm()
        for param in paramstart:
            #pbar.set_description("localization")
            self._mleFit(data, fittype, iterations, coeff, varim, param, datasize, splinesize, Pk, CRLBk, LLk)
            mask = (LLk - LL) > 1e-4
            LL[mask] = LLk[mask]
            P[:, mask] = Pk[:, mask]
            CRLB[:, mask] = CRLBk[:, mask]
            #pbar.update(1)

        #pbar.refresh()
        #pbar.close()

        return P, CRLB, LL, coeff

    def fit(self, psf_data, fittype,iterations = 30, I_model=None, pixelsize_z=None):
        rsz = psf_data.shape[-1]

        Nfit = psf_data.shape[-3]
        fittype = np.int32(fittype)
        if fittype == 5:
            Imd = I_model
            #pbar = tqdm()
            #pbar.set_description("calculating spline coefficients")
            coeff = psf2cspline_np(Imd)
            #pbar.update(1)
            #pbar.refresh()
            #pbar.close()
            coeff = coeff.astype(np.float32)
            splinesize = np.array(np.flip(coeff.shape))
            ccz = coeff.shape[-3] // 2
            initparam = np.array([-3, -2, -1, 0, 1, 2, 3]) * 0.3 / pixelsize_z + ccz
            paramstart = initparam.astype(np.float32)
        else:
            paramstart = np.array([1]).astype(np.float32)
            coeff = paramstart
            splinesize = np.array([0])

        if fittype == 1:
            Nparam = 4
        elif fittype == 4:
            Nparam = 6
        else:
            Nparam = 5
        data = psf_data.astype(np.float32)
        # bxsz = np.min((rsz, 20))
        # data = data[:, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2, rsz // 2 - bxsz // 2:rsz // 2 + bxsz // 2].astype(
        #     np.float32)
        # data = np.maximum(data, 0.0)

        datasize = np.array(np.flip(data.shape))

        varim = np.array((0)).astype(np.float32)
        Pk = np.zeros((Nparam + 1, Nfit)).astype(np.float32)
        CRLBk = np.zeros((Nparam, Nfit)).astype(np.float32)
        LLk = np.zeros((Nfit)).astype(np.float32)

        P = np.zeros((Nparam + 1, Nfit)).astype(np.float32)
        CRLB = np.zeros((Nparam, Nfit)).astype(np.float32)
        LL = np.zeros((Nfit)).astype(np.float32) - 1e10

        #pbar = tqdm()
        for param in paramstart:
            #pbar.set_description("localization")
            self._mleFit(data, fittype, iterations, coeff, varim, param, datasize, splinesize, Pk, CRLBk, LLk)
            mask = (LLk - LL) > 1e-4
            LL[mask] = LLk[mask]
            P[:, mask] = Pk[:, mask]
            CRLB[:, mask] = CRLBk[:, mask]
            #pbar.update(1)

        #pbar.refresh()
        #pbar.close()

        return P, CRLB, LL, coeff
