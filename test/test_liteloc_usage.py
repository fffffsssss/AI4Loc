import numpy as np
import torch
import sys
sys.path.append('../../')
import datetime
import os
import tifffile
from IPython.display import display
import scipy.io as sio
import time

import ailoc.common
import ailoc.simulation
import ailoc.liteloc
ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True


def liteloc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = '../../datasets/20231011-npc-astig-sw/20231011_beads/beads_1/beads_1_MMStack_Pos0_calib_results.mat'
    experiment_file = '../../datasets/20231011-npc-astig-sw/fluorescence_20ms_2/fluorescence_20ms_2_MMStack_Pos0.ome.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        # psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        # psf_params_dict['refmed'] = 1.406  # refractive index of sample medium may be different from beads
        # psf_params_dict['objstage0'] = -1500  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
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
        'local_context': True,
        'robust_training': True,
        'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-700, 700),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
        'bg_perlin': True,
    }

    # print learning parameters
    ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

    # instantiate the DeepLoc model and start to train
    liteloc_model = ailoc.liteloc.LiteLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    liteloc_model.check_training_psf()

    liteloc_model.check_training_data()

    liteloc_model.build_evaluation_dataset(napari_plot=False)
    # gt_csv = '../datasets/Snow_dataset_fs/snow_astig_md/activations.csv'
    # liteloc_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :]
    # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
    # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
    # liteloc_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LiteLoc.pt'
    liteloc_model.online_train(batch_size=2,
                               max_iterations=40000,
                               eval_freq=500,
                               file_name=file_name)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(liteloc_model)

    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(
        loc_model=liteloc_model,
        psf_params=liteloc_model.dict_psf_params,
        xy_range=(-50, 50),
        z_range=np.array(liteloc_model.dict_sampler_params['z_range']) * 0.98,
        photon=np.mean(liteloc_model.dict_sampler_params['photon_range']),
        bg=np.mean(liteloc_model.dict_sampler_params['bg_range']),
        num_z_step=31,
        num_repeat=1000,
        show_res=True
    )

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)
    save_path = '../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    liteloc_analyzer = ailoc.common.SmlmDataAnalyzer(
        loc_model=liteloc_model,
        tiff_path=image_path,
        output_path=save_path,
        time_block_gb=1,
        batch_size=32,
        sub_fov_size=256,
        over_cut=8,
        num_workers=0
    )
    liteloc_analyzer.check_single_frame_output(frame_num=3)
    preds_array, preds_rescale_array = liteloc_analyzer.divide_and_conquer()


def liteloc_analysis():
    model_name = '../results/2024-06-19-12-13LiteLoc.pt'
    with open(model_name, 'rb') as f:
        liteloc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(liteloc_model)

    # analyze the experimental data
    image_path_list = ["../datasets/NPC_DMO6_alpha30",  # can be a tiff file path or a folder path
                        ]

    for image_path in image_path_list:
        save_path = '../results/' + \
                    os.path.split(model_name)[-1].split('.')[0] + \
                    '_' + os.path.basename(image_path) + '_predictions.csv'

        liteloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=liteloc_model,
                                                         tiff_path=image_path,
                                                         output_path=save_path,
                                                         time_block_gb=1,
                                                         batch_size=32,
                                                         sub_fov_size=256,
                                                         over_cut=8,
                                                         num_workers=0)
        liteloc_analyzer.check_single_frame_output(frame_num=3)
        preds_array, preds_rescale_array = liteloc_analyzer.divide_and_conquer()


if __name__ == '__main__':
    liteloc_train()
    # liteloc_analysis()

