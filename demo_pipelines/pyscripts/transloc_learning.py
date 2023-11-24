import numpy as np
import torch
import sys
sys.path.append('../../')
import datetime
import os
import tifffile
from IPython.display import display
import scipy.io as sio

import ailoc.transloc
import ailoc.common
import ailoc.simulation
ailoc.common.setup_seed(42)


def transloc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = '../../datasets/NPC_DMO6_alpha30/beads-DMO6_alpha30_+-3umstep50nm_5/DMO6_alpha30_+-3umstep50nm_5_MMStack_Default_calib_results.mat'
    experiment_file = '../../datasets/NPC_DMO6_alpha30/NPC_DMO6_alpha30_2_MMStack_Default.ome.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['objstage0'] = -3000  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, -20, 3, 1, 10, 4, 0, -15, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        psf_params_dict = {'na': 1.5,
                           'wavelength': 670,  # unit: nm
                           'refmed': 1.406,
                           'refcov': 1.524,
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
        # camera_params_dict = {'camera_type': 'idea'}
        # camera_params_dict = {'camera_type': 'emccd',
        #                       'qe': 0.9,
        #                       'spurious_charge': 0.002,
        #                       'em_gain': 300,
        #                       'read_noise_sigma': 74.4,
        #                       'e_per_adu': 45,
        #                       'baseline': 100.0,
        #                       }
        camera_params_dict = {'camera_type': 'scmos',
                              'qe': 0.81,
                              'spurious_charge': 0.002,
                              'read_noise_sigma': 1.61,
                              'read_noise_map': None,
                              'e_per_adu': 0.47,
                              'baseline': 100.0,
                              }

    # estimate the background range from experimental images
    if experiment_file is not None:
        experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)
        camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
        experimental_images_photon = ailoc.common.cpu(
            camera_calib.backward(torch.tensor(experimental_images.astype(np.float32))))
        bg_range = ailoc.common.get_bg_stats_gauss(experimental_images_photon, percentile=10, plot=True)

    # manually set sampler parameters
    sampler_params_dict = {
        'local_context': False,
        'robust_training': False,
        'train_size': 128,
        'num_em_avg': 10,
        'eval_batch_size': 100,
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

    # instantiate the TransLoc model and start to train
    transloc_model = ailoc.transloc.TransLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    transloc_model.check_training_psf()

    transloc_model.check_training_data()

    transloc_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H') + 'TransLoc.pt'
    transloc_model.online_train(batch_size=1,
                                max_iterations=30000,
                                eval_freq=500,
                                file_name=file_name)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(transloc_model)


    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)
    save_path = '../../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    print(save_path)
    transloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=transloc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=16,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)
    transloc_analyzer.check_single_frame_output(frame_num=3)
    preds_array, preds_rescale_array = transloc_analyzer.divide_and_conquer()


def transloc_ckpoint_train():
    model_name = '../../results/2023-09-09-16DeepLoc.pt'
    with open(model_name, 'rb') as f:
        transloc_model = torch.load(f)
    transloc_model.online_train(batch_size=10,
                                max_iterations=30000,
                                eval_freq=500,
                                file_name=model_name)


if __name__ == '__main__':
    transloc_train()
    # deeploc_ckpoint_train()
