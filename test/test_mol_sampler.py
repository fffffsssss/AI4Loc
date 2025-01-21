import napari
import numpy as np
import torch
from torch import linalg as LA
import time
import matplotlib.pyplot as plt

import ailoc.common
from ailoc.simulation.vectorpsf import VectorPSFCUDA, VectorPSFTorch
from ailoc.simulation.camera import *
from ailoc.simulation.mol_sampler import *


def test_MoleculeSampler():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 70, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0, 0]
    npupil = 64
    psf_size = 51

    # psf_params_dict = {'na': na, 'wavelength': wavelength,
    #                    'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
    #                    'zernike_mode': zernike_aber[:, 0:2],
    #                    'zernike_coef': zernike_aber[:, 2], 'zernike_coef_map': None,
    #                    'objstage0': objstage0, 'zemit0': zemit0,
    #                    'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
    #                    'npupil': npupil, 'psf_size': psf_size}
    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': None, 'zernike_coef_map': np.random.rand(21, 256, 512),
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)

    # camera_params_dict = {'qe': 0.95, 'spurious_charge': 0.002,
    #                       'read_noise_sigma': 1.6, 'read_noise_map': None,
    #                       'e_per_adu': 0.5, 'baseline': 100.0}
    camera_params_dict = {'qe': 0.95, 'spurious_charge': 0.002,
                          'read_noise_sigma': None, 'read_noise_map': np.ones([256, 512]),
                          'e_per_adu': 0.5, 'baseline': 100.0}
    camera_scmos = SCMOS(camera_params_dict)

    sampler_params_dict = {'local_context': True, 'robust_training': True,
                           'train_size': 128, 'num_em_avg': 10,
                           'photon_range': [1000, 8000], 'z_range': [-700, 700], 'bg_range': [50, 100],
                           'bg_perlin': True}

    sampler = MoleculeSampler(sampler_params_dict, psf_cuda.zernike_coef_map, camera_scmos.read_noise_map)

    p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
    p_map_gt, xyzph_array_gt, mask_array_gt = sampler.sample_for_train(batch_size=16, psf_model=psf_cuda, iter_train=0)

    p_map_sample, xyzph_map_sample, bg_map_sample, sub_fov_xy_list, zernike_coefs, xyzph_array_gt, mask_array_gt = \
    sampler.sample_for_evaluation(num_image=1000, psf_model=psf_cuda)

    print('fs')

if __name__ == '__main__':
    test_MoleculeSampler()
