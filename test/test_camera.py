import napari
import numpy as np
import torch
from torch import linalg as LA
import time
import matplotlib.pyplot as plt

import ailoc.common
from ailoc.simulation.vectorpsf import VectorPSFCUDA, VectorPSFTorch
from ailoc.simulation.camera import *


def test_IdeaCamera():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 70, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
    #                       50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
    #                       80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0, 0]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2], 'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)

    # set emitter positions
    n_mol = 1000
    x = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    y = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-600, 600, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    # robust_training means adding gaussian noise to the zernike coef of each psf, unit nm
    zernike_coefs = torch.tile(ailoc.common.gpu(psf_cuda.zernike_coef), dims=(1000, 1))  # unit nm
    zernike_coefs += torch.normal(mean=0, std=3.33, size=(1000, psf_cuda.zernike_mode.shape[0]),
                                  device='cuda')  # different zernike coef for each psf

    raw_data = psf_cuda.simulate(x,y,z,photons, zernike_coefs=None)

    camera_idea = IdeaCamera()

    data_camera = camera_idea.forward(raw_data)
    data_photon = camera_idea.backward(data_camera)

    ailoc.common.cmpdata_napari(raw_data, data_camera)
    ailoc.common.cmpdata_napari(raw_data, data_photon)

    print(f"the relative root of squared error is: "
          f"{LA.norm((raw_data - data_photon).flatten(), ord=2) / LA.norm(raw_data.flatten(), ord=2) * 100}%")


def test_SCMOS():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 70, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
    #                       50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
    #                       80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0, 0]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2], 'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)

    # set emitter positions
    n_mol = 1000
    x = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    y = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-600, 600, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    # robust_training means adding gaussian noise to the zernike coef of each psf, unit nm
    zernike_coefs = torch.tile(ailoc.common.gpu(psf_cuda.zernike_coef), dims=(1000, 1))  # unit nm
    zernike_coefs += torch.normal(mean=0, std=3.33, size=(1000, psf_cuda.zernike_mode.shape[0]),
                                  device='cuda')  # different zernike coef for each psf

    raw_data = psf_cuda.simulate(x, y, z, photons, zernike_coefs=None)

    camera_params_dict = {'qe': 0.95, 'spurious_charge': 0.002,
                          'read_noise_sigma': 1.6, 'read_noise_map': None,
                          'e_per_adu': 0.5, 'baseline': 100.0}

    # camera_params_dict = {'qe': 0.95, 'spurious_charge': 0.002,
    #                       'read_noise_sigma': None, 'read_noise_map': np.ones([256, 512]),
    #                       'e_per_adu': 0.5, 'baseline': 100.0}

    camera_scmos = SCMOS(camera_params_dict)

    data_camera = camera_scmos.forward(raw_data, fov_xy=[0, 50, 0, 50])
    data_photon = camera_scmos.backward(data_camera)

    ailoc.common.cmpdata_napari(raw_data, data_camera)
    ailoc.common.cmpdata_napari(raw_data, data_photon)

    print(f"the relative root of squared error is: "
          f"{LA.norm((raw_data - data_photon).flatten(), ord=2) / LA.norm(raw_data.flatten(), ord=2) * 100}%")


def test_EMCCD():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 70, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
    #                       50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
    #                       80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0, 0]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2], 'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)

    # set emitter positions
    n_mol = 1000
    x = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    y = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-600, 600, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    # robust_training means adding gaussian noise to the zernike coef of each psf, unit nm
    zernike_coefs = torch.tile(ailoc.common.gpu(psf_cuda.zernike_coef), dims=(1000, 1))  # unit nm
    zernike_coefs += torch.normal(mean=0, std=3.33, size=(1000, psf_cuda.zernike_mode.shape[0]),
                                  device='cuda')  # different zernike coef for each psf

    raw_data = psf_cuda.simulate(x, y, z, photons, zernike_coefs=None)

    camera_params_dict = {'qe': 0.9, 'spurious_charge': 0.002,
                          'em_gain': 300, 'read_noise_sigma': 74.4,
                          'e_per_adu': 45, 'baseline': 100.0}

    camera_emccd = EMCCD(camera_params_dict)

    data_camera = camera_emccd.forward(raw_data)
    data_photon = camera_emccd.backward(data_camera)

    ailoc.common.cmpdata_napari(raw_data, data_camera)
    ailoc.common.cmpdata_napari(raw_data, data_photon)

    print(f"the relative root of squared error is: "
          f"{LA.norm((raw_data - data_photon).flatten(), ord=2) / LA.norm(raw_data.flatten(), ord=2) * 100}%")


if __name__ == '__main__':
    # test_IdeaCamera()
    # test_SCMOS()
    test_EMCCD()
