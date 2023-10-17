import numpy as np
import torch
import sys
sys.path.append('../../')
import datetime
import os
import tifffile
from IPython.display import display
import scipy.io as sio

import ailoc.deeploc
import ailoc.common
ailoc.common.setup_seed(42)


def deeploc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for photon range and bg range estimation
    calib_file = 'E:/projects/FS_work/AI-Loc/AI-Loc_project/datasets/NPC_DMO6_alpha30/beads-DMO6_alpha30_+-3umstep50nm_5/DMO6_alpha30_+-3umstep50nm_5_MMStack_Default_calib_results.pt'
    experiment_file = 'E:/projects/FS_work/AI-Loc/AI-Loc_project/datasets/NPC_DMO6_alpha30/NPC_DMO6_alpha30_2_MMStack_Default.ome.tif'

    calib_dict = torch.load(calib_file)
    psf_torch_calib = calib_dict['psf_torch_fitted']
    camera_calib = calib_dict['calib_params_dict']['camera_model']
    if os.path.isfile(experiment_file):
        experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 1)
        experimental_images = ailoc.common.cpu(camera_calib.backward(torch.tensor(experimental_images.astype(np.float32))))
        bg_range = ailoc.common.get_bg_stats_gauss(experimental_images, percentile=10, plot=True)

        # # TODO: maybe we can use single molecule data to estimate the photon range and zernike aberrations
        # #  training density, etc.
        # molecule_images = ailoc.common.segment_local_max_smlm_data(images=experimental_images,
        #                                                            camera=camera_calib,
        #                                                            filter_sigma=3,
        #                                                            roi_size=51,
        #                                                            threshold_abs=20)
        # ailoc.common.viewdata_napari(molecule_images)

    # # manually set psf parameters
    # zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
    #                          4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
    #                          5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
    #                         dtype=np.float32).reshape([21, 3])
    # psf_params_dict = {'na': 1.49,
    #                    'wavelength': 660,  # unit: nm
    #                    'refmed': 1.518,
    #                    'refcov': 1.518,
    #                    'refimm': 1.518,
    #                    'zernike_mode': zernike_aber[:, 0:2],
    #                    'zernike_coef': zernike_aber[:, 2],
    #                    'zernike_coef_map': None,
    #                    'pixel_size_xy': (100, 100),
    #                    'otf_rescale_xy': (0, 0),
    #                    'npupil': 64,
    #                    'psf_size': 51,
    #                    'objstage0': 0,
    #                    'zemit0': 0,
    #                    }

    psf_params_dict = {'na': ailoc.common.cpu(psf_torch_calib.na),
                       'wavelength': ailoc.common.cpu(psf_torch_calib.wavelength),
                       'refmed': ailoc.common.cpu(psf_torch_calib.refmed),
                       'refcov': ailoc.common.cpu(psf_torch_calib.refcov),
                       'refimm': ailoc.common.cpu(psf_torch_calib.refimm),
                       'zernike_mode': ailoc.common.cpu(psf_torch_calib.zernike_mode),
                       'zernike_coef': ailoc.common.cpu(psf_torch_calib.zernike_coef),
                       'pixel_size_xy': tuple(ailoc.common.cpu(psf_torch_calib.pixel_size_xy)),
                       'otf_rescale_xy': tuple(ailoc.common.cpu(psf_torch_calib.otf_rescale_xy)),
                       'npupil': psf_torch_calib.npupil,
                       'psf_size': psf_torch_calib.psf_size,
                       'objstage0': -3000, 
                       }

    # # manually set camera parameters
    # camera_params_dict = {'camera_type': 'idea'}
    # camera_params_dict = {'camera_type': 'emccd',
    #                       'qe': 0.9, 
    #                       'spurious_charge': 0.002, 
    #                       'em_gain': 300,
    #                       'read_noise_sigma': 74.4, 
    #                       'e_per_adu': 45, 
    #                       'baseline': 100.0,
    #                       }
    # camera_params_dict = {'camera_type': 'scmos',
    #                       'qe': 0.9,
    #                       'spurious_charge': 0.002,
    #                       'read_noise_sigma': 1.6,
    #                       'read_noise_map': None,
    #                       'e_per_adu': 0.5,
    #                       'baseline': 100.0,
    #                       }

    camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    # manually set sampler parameters
    sampler_params_dict = {'local_context': False,
                           'robust_training': True,
                           'train_size': 128,
                           'num_em_avg': 5,
                           'num_evaluation_data': 1000,
                           'photon_range': (1000, 10000), 
                           'z_range': (-3000, 3000),
                           'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
                           'bg_perlin': True,
                           }

    # print learning parameters
    for params_dict in [psf_params_dict, camera_params_dict, sampler_params_dict]:
        for keys in params_dict.keys():
            params = params_dict[keys].transpose() if keys == 'zernike_mode' else params_dict[keys]
            print(f"{keys}: {params}")

    deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    deeploc_model.check_training_psf()

    deeploc_model.check_training_data()

    deeploc_model.build_evaluation_dataset(napari_plot=True)

    file_name = '../../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H') + 'DeepLoc.pt'
    deeploc_model.online_train(batch_size=10,
                               max_iterations=30000,
                               eval_freq=500,
                               file_name=file_name)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)


def deeploc_ckpoint_train():
    model_name = '../../results/2023-09-09-16DeepLoc.pt'
    with open(model_name, 'rb') as f:
        deeploc_model = torch.load(f)
    deeploc_model.online_train(batch_size=10,
                               max_iterations=30000,
                               eval_freq=500,
                               file_name=model_name)


if __name__ == '__main__':
    deeploc_train()
    # deeploc_ckpoint_train()
