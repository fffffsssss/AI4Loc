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
import ailoc.simulation
ailoc.common.setup_seed(42)


def deeploc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = '../../datasets/sw_npc_20211028/beads/astigmatism/astigmatism_beads_2um_20nm_512x512_hamm_1/astigmatism_beads_2um_20nm_512x512_hamm_1_MMStack_Default_calib_results.mat'
    experiment_file = '../../datasets/sw_npc_20211028/NUP96_SNP647_3D_512_20ms_hama_mm_1800mW_3/NUP96_SNP647_3D_512_20ms_hama_mm_1800mW_3_MMStack_Default.ome.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['objstage0'] = -700  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        psf_params_dict = {'na': 1.49,
                           'wavelength': 660,  # unit: nm
                           'refmed': 1.518,
                           'refcov': 1.518,
                           'refimm': 1.518,
                           'zernike_mode': zernike_aber[:, 0:2],
                           'zernike_coef': zernike_aber[:, 2],
                           'pixel_size_xy': (100, 100),
                           'otf_rescale_xy': (0, 0),
                           'npupil': 64,
                           'psf_size': 51,
                           'objstage0': 0,
                           'zemit0': 0,
                           }

        # manually set camera parameters
        camera_params_dict = {'camera_type': 'idea'}
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

    # estimate the background range from experimental images
    if experiment_file is not None:
        experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)
        camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
        experimental_images = ailoc.common.cpu(
            camera_calib.backward(torch.tensor(experimental_images.astype(np.float32))))
        bg_range = ailoc.common.get_bg_stats_gauss(experimental_images, percentile=10, plot=True)

    # manually set sampler parameters
    sampler_params_dict = {'local_context': True,
                           'robust_training': True,
                           'train_size': 64,
                           'num_em_avg': 10,
                           'num_evaluation_data': 1000,
                           'photon_range': (1000, 10000), 
                           'z_range': (-700, 700),
                           'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
                           'bg_perlin': True,
                           }

    # print learning parameters
    for params_dict in [psf_params_dict, camera_params_dict, sampler_params_dict]:
        for keys in params_dict.keys():
            params = params_dict[keys].transpose() if keys == 'zernike_mode' else params_dict[keys]
            print(f"{keys}: {params}")

    # instantiate the DeepLoc model and start to train
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
