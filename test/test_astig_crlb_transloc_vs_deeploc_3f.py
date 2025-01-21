import time
import numpy as np
import torch
import sys
sys.path.append('../../')
import datetime
import os
import tifffile
from IPython.display import display
import scipy.io as sio

import ailoc.common
import ailoc.simulation
import ailoc.transloc
import ailoc.deeploc
ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True


def main_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = None
    experiment_file = None

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
        psf_params_dict = {'na': 1.5,
                           'wavelength': 670,  # unit: nm
                           'refmed': 1.518,
                           'refcov': 1.518,
                           'refimm': 1.518,
                           'zernike_mode': zernike_aber[:, 0:2],
                           'zernike_coef': zernike_aber[:, 2],
                           'pixel_size_xy': (108, 108),
                           'otf_rescale_xy': (0.5, 0.5),
                           'npupil': 64,
                           'psf_size': 25,
                           'objstage0': -1000,
                           # 'zemit0': 0,
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
        #                       'qe': 0.81,
        #                       'spurious_charge': 0.002,
        #                       'read_noise_sigma': 1.61,
        #                       'read_noise_map': None,
        #                       'e_per_adu': 0.47,
        #                       'baseline': 100.0,
        #                       }

    # estimate the background range from experimental images
    if experiment_file is not None:
        experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)
        camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
        experimental_images_photon = ailoc.common.cpu(
            camera_calib.backward(torch.tensor(experimental_images.astype(np.float32))))
        bg_range = ailoc.common.get_bg_stats_gauss(experimental_images_photon, percentile=10, plot=True)

    # manually set sampler parameters
    sampler_params_dict = {
        'temporal_attn': True,
        'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
        'context_size': 10,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-700, 700),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
        'bg_perlin': False,
    }

    # print learning parameters
    for params_dict in [psf_params_dict, camera_params_dict, sampler_params_dict]:
        for keys in params_dict.keys():
            params = params_dict[keys].transpose() if keys == 'zernike_mode' else params_dict[keys]
            print(f"{keys}: {params}")

    # instantiate the TransLoc model and start to train
    transloc_model = ailoc.transloc.TransLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    transloc_model.check_training_psf()

    transloc_model.check_training_data()

    transloc_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'TransLoc.pt'
    transloc_model.online_train(batch_size=1,
                                max_iterations=40000,
                                eval_freq=500,
                                file_name=file_name)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(transloc_model)

    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(loc_model=transloc_model,
                                              psf_params=transloc_model.dict_psf_params,
                                              xy_range=(-50, 50),
                                              z_range=np.array(transloc_model.dict_sampler_params['z_range']) * 0.98,
                                              photon=np.mean(transloc_model.dict_sampler_params['photon_range']),
                                              bg=np.mean(transloc_model.dict_sampler_params['bg_range']),
                                              num_z_step=31,
                                              num_repeat=1000,
                                              show_res=True)



    # for deeploc model
    sampler_params_dict = {
        'local_context': True,
        'robust_training': False,
        'context_size': 10,
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-700, 700),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
        'bg_perlin': False,
    }
    deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict, camera_params_dict, sampler_params_dict)
    deeploc_model.check_training_psf()
    deeploc_model.check_training_data()
    deeploc_model.build_evaluation_dataset(napari_plot=False)
    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
    deeploc_model.online_train(batch_size=1,
                               max_iterations=40000,
                               eval_freq=500,
                               file_name=file_name)
    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)
    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(loc_model=deeploc_model,
                                              psf_params=deeploc_model.dict_psf_params,
                                              xy_range=(-50, 50),
                                              z_range=np.array(deeploc_model.dict_sampler_params['z_range']) * 0.98,
                                              photon=np.mean(deeploc_model.dict_sampler_params['photon_range']),
                                              bg=np.mean(deeploc_model.dict_sampler_params['bg_range']),
                                              num_z_step=31,
                                              num_repeat=1000,
                                              show_res=True)


if __name__ == "__main__":
    main_train()