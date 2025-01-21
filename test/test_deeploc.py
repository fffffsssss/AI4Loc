import numpy as np
import torch

import ailoc.deeploc
import ailoc.common
ailoc.common.setup_seed(42)


def test_train_DeepLoc():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = (100, 100)
    otf_rescale_xy = (0.5, 0.5)
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2], 'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}

    # camera_params_dict = {'camera_type': 'idea'}

    # camera_params_dict = {'camera_type': 'emccd',
    #                       'qe': 0.9, 'spurious_charge': 0.002, 'em_gain': 300,
    #                       'read_noise_sigma': 74.4, 'e_per_adu': 45, 'baseline': 100.0}

    camera_params_dict = {'camera_type': 'scmos',
                          'qe': 0.9, 'spurious_charge': 0.002,
                          'read_noise_sigma': 1.6, 'read_noise_map': None,
                          'e_per_adu': 0.5, 'baseline': 100.0}

    sampler_params_dict = {'local_context': False, 'robust_training': False,
                           'train_size': 128, 'num_em_avg': 10, 'num_evaluation_data': 1000,
                           'photon_range': (1000, 8000), 'z_range': (-700, 700), 'bg_range': (50, 100)}

    deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    deeploc_model.check_training_psf()

    deeploc_model.check_training_data()

    deeploc_model.build_evaluation_dataset(napari_plot=True)

    deeploc_model.online_train(batch_size=16, max_iterations=50000, eval_freq=500)

    print('fs')


def test_train_ckpoint_DeepLoc():
    with open('E:\projects\FS_work\AI-Loc\AI-Loc_project/test/2023-07-05-20DeepLoc.pt', 'rb') as f:
        deeploc_model = torch.load(f)
    deeploc_model.online_train(batch_size=16, max_iterations=50000, eval_freq=500)


if __name__ == '__main__':
    test_train_DeepLoc()
    # test_train_ckpoint_DeepLoc()
