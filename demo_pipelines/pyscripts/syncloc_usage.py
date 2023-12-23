import numpy as np
import torch
import sys
sys.path.append('../../')
import datetime
import os
import tifffile
from IPython.display import display
import imageio
import scipy.io as sio

import ailoc.syncloc
import ailoc.common
import ailoc.simulation
ailoc.common.setup_seed(25)


def syncloc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = None
    experiment_file = '../../datasets/mismatch_data2/1.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['refmed'] = 1.406  # refractive index of sample medium may be different from beads
        psf_params_dict['objstage0'] = -0  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 60, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
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
                           'otf_rescale_xy': (0.0, 0.0),
                           'npupil': 64,
                           'psf_size': 25,
                           'objstage0': -0,
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
        'local_context': False,
        'robust_training': False,
        'context_size': 10,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 1000,
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

    # instantiate the SyncLoc model and start to train
    syncloc_model = ailoc.syncloc.SyncLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    syncloc_model.check_training_psf()

    syncloc_model.check_training_data()

    syncloc_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'SyncLoc.pt'
    syncloc_model.online_train(batch_size=1,
                               max_iterations=30000,
                               eval_freq=500,
                               file_name=file_name,
                               real_data=experimental_images,
                               num_sample=50,
                               wake_interval=2,
                               max_recon_psfs=5000,
                               online_build_eval_set=True,)

    # plot evaluation performance during the training
    phase_record = ailoc.common.plot_syncloc_record(syncloc_model)
    # save the phase learned during the training
    imageio.mimsave('../../results/' + os.path.split(file_name)[-1].split('.')[0] + '_phase.gif',
                    phase_record,
                    duration=200)

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)  # can be a tiff file path or a folder path
    save_path = '../../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    syncloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=syncloc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=32,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)
    syncloc_analyzer.check_single_frame_output(frame_num=3)
    preds_array, preds_rescale_array = syncloc_analyzer.divide_and_conquer()


def syncloc_ckpoint_train():
    model_name = '../../results/2023-12-21-15-09SyncLoc.pt'
    with open(model_name, 'rb') as f:
        syncloc_model = torch.load(f)

    experiment_file = '../../datasets/mismatch_data2/1.tif'
    experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)

    # torch.autograd.set_detect_anomaly(True)
    syncloc_model.online_train(batch_size=1,
                               max_iterations=30000,
                               eval_freq=500,
                               file_name=model_name,
                               real_data=experimental_images,
                               num_sample=50,
                               wake_interval=2,
                               max_recon_psfs=5000,
                               online_build_eval_set=True,)

    # plot evaluation performance during the training
    phase_record = ailoc.common.plot_syncloc_record(syncloc_model)
    # save the phase learned during the training
    imageio.mimsave('../../results/' + os.path.split(model_name)[-1].split('.')[0] + '_phase.gif',
                    phase_record,
                    duration=200)

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)  # can be a tiff file path or a folder path
    save_path = '../../results/' + \
                os.path.split(model_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    syncloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=syncloc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=32,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)
    syncloc_analyzer.check_single_frame_output(frame_num=3)
    preds_array, preds_rescale_array = syncloc_analyzer.divide_and_conquer()


def syncloc_analyze():
    loc_model_path = '../../results/2023-10-11-11SyncLoc.pt'
    image_path = '../../datasets/mismatch_data2/'   # can be a tiff file path or a folder path
    save_path = '../../results/' + \
                os.path.split(loc_model_path)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        syncloc_model = torch.load(f)

    # plot evaluation performance during the training
    phase_record = ailoc.common.plot_syncloc_record(syncloc_model)
    # save the phase learned during the training
    imageio.mimsave('../../results/'+os.path.split(loc_model_path)[-1].split('.')[0]+'_phase.gif',
                    phase_record,
                    duration=200)

    syncloc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=syncloc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=32,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)

    syncloc_analyzer.check_single_frame_output(frame_num=3)

    preds_array, preds_rescale_array = syncloc_analyzer.divide_and_conquer()

    # read the ground truth and calculate metrics
    gt_array = ailoc.common.read_csv_array("../../datasets/mismatch_data2/activations.csv")

    metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_array,
                                                                ground_truth=gt_array,
                                                                frame_num=syncloc_analyzer.tiff_dataset.end_frame_num,
                                                                fov_xy_nm=syncloc_analyzer.fov_xy_nm,
                                                                print_info=True)
    # # write the paired localizations to csv file
    # save_paried_path = '../../results/'+os.path.split(save_path)[-1].split('.')[0]+'_paired.csv'
    # ailoc.common.write_csv_array(input_array=paired_array,
    #                              filename=save_paried_path,
    #                              write_mode='write paired localizations')


if __name__ == '__main__':
    syncloc_train()
    # syncloc_ckpoint_train()
    # syncloc_analyze()
