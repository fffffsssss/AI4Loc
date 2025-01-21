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
import time
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import ailoc.lunar
import ailoc.common
import ailoc.simulation
ailoc.common.setup_seed(81)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def lunar_loclearning():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = '../datasets/dmo4Spectrin_sw20220923/crop_beads/1_calib_results.mat'
    experiment_file = '../datasets/dmo4Spectrin_sw20220923/crop_neuron/1.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['objstage0'] = -2000  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        psf_params_dict = {
            'na': 1.5,
            'wavelength': 680,  # unit: nm
            'refmed': 1.352,
            'refcov': 1.524,
            'refimm': 1.518,
            'zernike_mode': zernike_aber[:, 0:2],
            'zernike_coef': zernike_aber[:, 2],
            'pixel_size_xy': (108, 108),
            'otf_rescale_xy': (0.5, 0.5),
            'npupil': 64,
            'psf_size': 35,
            'objstage0': -4660,
            'zemit0': 3700,
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
        camera_params_dict = {
            'camera_type': 'scmos',
            'qe': 0.9,
            'spurious_charge': 0.2,
            'read_noise_sigma': 1.1,
            'read_noise_map': None,
            'e_per_adu': 0.43,
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
        'temporal_attn': True,
        'robust_training': True,  # if True, the training data will be added with some random Zernike aberrations
        'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 128,
        'num_em_avg': 20,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-2000, 2000),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
        'bg_perlin': True,
    }

    # print learning parameters
    ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

    # instantiate the LUNAR model and start to train
    lunar_model = ailoc.lunar.Lunar_LocLearning(psf_params_dict, camera_params_dict, sampler_params_dict)

    lunar_model.check_training_psf()

    lunar_model.check_training_data()

    lunar_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_LL.pt'
    lunar_model.online_train(
        batch_size=2,
        max_iterations=40000,
        eval_freq=500,
        file_name=file_name
    )

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(lunar_model)

    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(
        loc_model=lunar_model,
        psf_params=None,
        xy_range=(-50, 50),
        z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
        photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
        bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
        num_z_step=31,
        num_repeat=1000,
        show_res=True,
    )

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)  # can be a tiff file path or a folder path
    save_path = '../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'

    lunar_analyzer = ailoc.common.SmlmDataAnalyzer(
        loc_model=lunar_model,
        tiff_path=image_path,
        output_path=save_path,
        time_block_gb=1,
        batch_size=32,
        sub_fov_size=256,
        over_cut=8,
        num_workers=0
    )

    lunar_analyzer.check_single_frame_output(frame_num=11)

    t0 = time.time()
    preds_array, preds_rescale_array = lunar_analyzer.divide_and_conquer(degrid=True)
    print(f'Prediction time cost: {time.time() - t0} s')


def lunar_synclearning():
    # set the file paths, calibration file is necessary,
    # experiment file is needed for background range estimation and synchronized learning
    calib_file = '../datasets/dmo4Spectrin_sw20220923/crop_beads/1_calib_results.mat'
    experiment_file = '../datasets/dmo4Spectrin_sw20220923/crop_neuron/1.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['objstage0'] = -2000  # the objective stage position is different from beads calibration, unit: nm

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, -200, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        psf_params_dict = {
            'na': 1.35,
            'wavelength': 670,  # unit: nm
            'refmed': 1.406,
            'refcov': 1.524,
            'refimm': 1.406,
            'zernike_mode': zernike_aber[:, 0:2],
            'zernike_coef': zernike_aber[:, 2],
            'pixel_size_xy': (108, 108),
            'otf_rescale_xy': (0.5, 0.5),
            'npupil': 64,
            'psf_size': 61,
            'objstage0': -3000,
            'zemit0': 3000,  # will be automatically set, if needs to be specified, don't call est_z0_zrange func
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
        camera_params_dict = {
            'camera_type': 'scmos',
            'qe': 0.9,
            'spurious_charge': 0.01,
            'read_noise_sigma': 1.535,
            'read_noise_map': None,
            'e_per_adu': 0.747,
            'baseline': 100.0,
        }

    # manually set sampler parameters
    sampler_params_dict = {
        'temporal_attn': True,
        'robust_training': False,
        'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 128,
        'num_em_avg': 20,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-2000, 2000),
        'bg_range': (40, 60),  # will be automatically adjusted if provided experimental images
        'bg_perlin': True,
    }

    # estimate the background range from experimental images
    if experiment_file is not None:
        experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)
        camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
        experimental_images_photon = ailoc.common.cpu(
            camera_calib.backward(torch.tensor(experimental_images.astype(np.float32))))

        print('experimental images provided, automatically adjust training parameters')

        sampler_params_dict['bg_range'] = ailoc.common.get_bg_stats_gauss(experimental_images_photon, percentile=10,
                                                                          plot=True)
        print(f'Adjusted bg_range: {sampler_params_dict["bg_range"]}')

        # psf_params_dict['zemit0'], sampler_params_dict['z_range'] = ailoc.common.est_z0_zrange(psf_params_dict,
        #                                                                                        sampler_params_dict)
        # print(f'Adjusted PSF zemit0: {psf_params_dict["zemit0"]}, training z_range: {sampler_params_dict["z_range"]}')

    # print learning parameters
    ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

    # instantiate the LUNAR model and start to train
    lunar_model = ailoc.lunar.Lunar_SyncLearning(psf_params_dict, camera_params_dict, sampler_params_dict)

    lunar_model.check_training_psf()

    lunar_model.check_training_data()

    lunar_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_SL.pt'
    # torch.autograd.set_detect_anomaly(True)
    lunar_model.online_train(
        batch_size=2,
        max_iterations=40000,
        eval_freq=1000,
        file_name=file_name,
        real_data=experimental_images,
        num_sample=100,
        wake_interval=2,
        max_recon_psfs=5000,
        online_build_eval_set=True,
    )

    # plot evaluation performance during the training
    phase_record = ailoc.common.plot_synclearning_record(lunar_model)
    # save the phase learned during the training
    imageio.mimsave('../results/' + os.path.split(file_name)[-1].split('.')[0] + '_phase.gif',
                    phase_record,
                    duration=200)

    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(
        loc_model=lunar_model,
        psf_params=None,
        xy_range=(-50, 50),
        z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
        photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
        bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
        num_z_step=31,
        num_repeat=1000,
        show_res=True,
    )

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)  # can be a tiff file path or a folder path
    save_path = '../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    lunar_analyzer = ailoc.common.SmlmDataAnalyzer(
        loc_model=lunar_model,
        tiff_path=image_path,
        output_path=save_path,
        time_block_gb=1,
        batch_size=32,
        sub_fov_size=256,
        over_cut=8,
        num_workers=0,
    )
    lunar_analyzer.check_single_frame_output(frame_num=12)
    preds_array, preds_rescale_array = lunar_analyzer.divide_and_conquer(degrid=True)


def lunar_synclearning_ckpoint():
    model_name = '../results/2024-12-12-19-46LUNAR_SL.pt'
    with open(model_name, 'rb') as f:
        lunar_model = torch.load(f)

    experiment_file = '../datasets/dmo4Spectrin_sw20220923/crop_neuron/1.tif'
    # experimental_images = ailoc.common.read_first_size_gb_tiff(experiment_file, 2)
    #
    # # torch.autograd.set_detect_anomaly(True)
    # lunar_model.online_train(
    #     batch_size=2,
    #     max_iterations=40000,
    #     eval_freq=1000,
    #     file_name=model_name,
    #     real_data=experimental_images,
    #     num_sample=100,
    #     wake_interval=2,
    #     max_recon_psfs=5000,
    #     online_build_eval_set=True,
    # )

    # # plot evaluation performance during the training
    # phase_record = ailoc.common.plot_synclearning_record(lunar_model)
    # # save the phase learned during the training
    # imageio.mimsave('../results/' + os.path.split(model_name)[-1].split('.')[0] + '_phase.gif',
    #                 phase_record,
    #                 duration=200)

    # test single emitter localization accuracy with CRLB
    # ailoc.common.test_single_emitter_accuracy(
    #     loc_model=lunar_model,
    #     psf_params=None,
    #     xy_range=(-50, 50),
    #     z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
    #     photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
    #     bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
    #     num_z_step=31,
    #     num_repeat=1000,
    #     show_res=True
    # )

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)  # can be a tiff file path or a folder path
    save_path = '../results/' + \
                os.path.split(model_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    lunar_analyzer = ailoc.common.SmlmDataAnalyzer(
        loc_model=lunar_model,
        tiff_path=image_path,
        output_path=save_path,
        time_block_gb=1,
        batch_size=32,
        sub_fov_size=256,
        over_cut=8,
        num_workers=0
    )
    lunar_analyzer.check_single_frame_output(frame_num=12940)
    preds_array, preds_rescale_array = lunar_analyzer.divide_and_conquer()


if __name__ == '__main__':
    # lunar_loclearning()
    # lunar_loclearning_ckpoint()
    # lunar_synclearning()
    lunar_synclearning_ckpoint()
    # lunar_analyze()
    # lunar_competitive_analyze()
