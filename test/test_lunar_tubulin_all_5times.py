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

import ailoc.lunar
import ailoc.common
import ailoc.simulation
# ailoc.common.setup_seed(17)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def lunar_loclearning_tubulin_astig_test():
    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_astig_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_astig_md',
                       '../datasets/simu_tubulin/simu_tubulin_astig_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_astig_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_astig_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_astig_hd/activations.csv', ]
    repeat_times = 5
    num_datasets = len(image_path_list)
    metric_dict_list = [[] for i in range(repeat_times)]
    metric_avg_dict_list = []
    metric_std_dict_list = []

    for i in range(repeat_times):
        print(f'--------------------------------repeat: {i}----------------------------------------')

        # set the file paths, calibration file is necessary,
        # experiment file is optional for background range estimation
        calib_file = None
        experiment_file = '../datasets/simu_tubulin/simu_tubulin_astig_md/1.tif'
        # experiment_file = None

        if calib_file is not None:
            # using the same psf parameters and camera parameters as beads calibration
            calib_dict = sio.loadmat(calib_file, simplify_cells=True)
            psf_params_dict = calib_dict['psf_params_fitted']
            psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
            psf_params_dict['objstage0'] = -700  # the objective stage position is different from beads calibration, unit: nm

            camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

        else:
            # manually set psf parameters
            zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, -20, 3, 1, 20, 4, 0, -10, 3, -3, 0, 3, 3, 0,
                                     4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                     5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                    dtype=np.float32).reshape([21, 3])
            psf_params_dict = {
                'na': 1.5,
                'wavelength': 670,  # unit: nm
                'refmed': 1.518,
                'refcov': 1.524,
                'refimm': 1.518,
                'zernike_mode': zernike_aber[:, 0:2],
                'zernike_coef': zernike_aber[:, 2],
                'pixel_size_xy': (108, 108),
                'otf_rescale_xy': (0.5, 0.5),
                'npupil': 64,
                'psf_size': 35,
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
            camera_params_dict = {
                'camera_type': 'scmos',
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
            'temporal_attn': True,
            'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
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

        # instantiate the TransLoc model and start to train
        lunar_model = ailoc.lunar.Lunar_LocLearning(psf_params_dict,
                                                    camera_params_dict,
                                                    sampler_params_dict,
                                                    attn_length=5)

        # lunar_model.check_training_psf()

        # lunar_model.check_training_data()

        lunar_model.build_evaluation_dataset(napari_plot=False)
        # gt_csv = '../datasets/Snow_dataset_fs/snow_astig_md/activations.csv'
        # deeploc_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :]
        # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
        # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
        # deeploc_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])

        file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_LL.pt'
        lunar_model.online_train(
            batch_size=2,
            max_iterations=40000,
            eval_freq=1000,
            file_name=file_name
        )

        # plot evaluation performance during the training
        ailoc.common.plot_train_record(lunar_model)

        # test single emitter localization accuracy with CRLB
        ailoc.common.test_single_emitter_accuracy(
            loc_model=lunar_model,
            psf_params=lunar_model.dict_psf_params,
            xy_range=(-50, 50),
            z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
            bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=1000,
            show_res=False
        )

        for image_path, gt_path in zip(image_path_list, gt_path_list):
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

            # lunar_analyzer.check_single_frame_output(frame_num=11)

            t0 = time.time()
            preds_array, preds_array_rescaled = lunar_analyzer.divide_and_conquer()
            print(f'Prediction time cost: {time.time() - t0} s')

            # read the ground truth and calculate metrics
            gt_array = ailoc.common.read_csv_array(gt_path)

            metric_dict, paired_array = ailoc.common.pair_localizations(
                prediction=preds_array,
                ground_truth=gt_array,
                frame_num=lunar_analyzer.tiff_dataset.end_frame_num,
                fov_xy_nm=lunar_analyzer.fov_xy_nm,
                print_info=True
            )
            metric_dict_list[i].append(metric_dict)
    print(f'----------------------------------finished------------------------------------------')


    for i in range(num_datasets):
        metric_avg_dict = {}
        metric_std_dict = {}
        for key in metric_dict_list[0][0].keys():
            tmp_list = []
            for j in range(repeat_times):
                tmp_list.append(metric_dict_list[j][i][key])
            metric_avg_dict[key] = np.mean(np.array(tmp_list))
            metric_std_dict[key] = np.std(np.array(tmp_list))
        metric_avg_dict_list.append(metric_avg_dict)
        metric_std_dict_list.append(metric_std_dict)
    for i in range(num_datasets):
        print(f'------------avg metric ± (std) for dataset {image_path_list[i]}----------------')
        for key in metric_avg_dict_list[i].keys():
            print(f'{key}: {metric_avg_dict_list[i][key]:.3f} ± {metric_std_dict_list[i][key]:.3f}')


def lunar_loclearning_tubulin_tetra3_test():
    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra3_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_tetra3_md',
                       '../datasets/simu_tubulin/simu_tubulin_tetra3_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra3_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra3_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra3_hd/activations.csv', ]
    repeat_times = 5
    num_datasets = len(image_path_list)
    metric_dict_list = [[] for i in range(repeat_times)]
    metric_avg_dict_list = []
    metric_std_dict_list = []

    for i in range(repeat_times):
        print(f'--------------------------------repeat: {i}----------------------------------------')

        # set the file paths, calibration file is necessary,
        # experiment file is optional for background range estimation
        calib_file = None
        experiment_file = '../datasets/simu_tubulin/simu_tubulin_tetra3_md/1.tif'
        # experiment_file = None

        if calib_file is not None:
            # using the same psf parameters and camera parameters as beads calibration
            calib_dict = sio.loadmat(calib_file, simplify_cells=True)
            psf_params_dict = calib_dict['psf_params_fitted']
            psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
            psf_params_dict['objstage0'] = -700  # the objective stage position is different from beads calibration, unit: nm

            camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

        else:
            # manually set psf parameters
            zernike_aber = np.array([2, -2, 120, 2, 2, 0, 3, -1, -20, 3, 1, 20, 4, 0, -50, 3, -3, 0, 3, 3, 0,
                                     4, -2, -130, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                     5, -3, 0, 5, 3, 0, 6, -2, -6, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
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
                'psf_size': 51,
                'objstage0': -2000,
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
            camera_params_dict = {
                'camera_type': 'scmos',
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
            'temporal_attn': True,
            'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
            'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
            'train_size': 128,
            'num_em_avg': 10,
            'eval_batch_size': 100,
            'photon_range': (1000, 10000),
            'z_range': (-1500, 1500),
            'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
            'bg_perlin': True,
        }

        # print learning parameters
        ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

        # instantiate the TransLoc model and start to train
        lunar_model = ailoc.lunar.Lunar_LocLearning(psf_params_dict, camera_params_dict, sampler_params_dict)

        # lunar_model.check_training_psf()

        # lunar_model.check_training_data()

        lunar_model.build_evaluation_dataset(napari_plot=False)
        # gt_csv = '../datasets/Snow_dataset_fs/snow_astig_md/activations.csv'
        # deeploc_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :]
        # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
        # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
        # deeploc_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])

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
            psf_params=lunar_model.dict_psf_params,
            xy_range=(-50, 50),
            z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
            bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=1000,
            show_res=False
        )

        for image_path, gt_path in zip(image_path_list, gt_path_list):
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

            # lunar_analyzer.check_single_frame_output(frame_num=11)

            t0 = time.time()
            preds_array, preds_array_rescaled = lunar_analyzer.divide_and_conquer()
            print(f'Prediction time cost: {time.time() - t0} s')

            # read the ground truth and calculate metrics
            gt_array = ailoc.common.read_csv_array(gt_path)

            metric_dict, paired_array = ailoc.common.pair_localizations(
                prediction=preds_array,
                ground_truth=gt_array,
                frame_num=lunar_analyzer.tiff_dataset.end_frame_num,
                fov_xy_nm=lunar_analyzer.fov_xy_nm,
                print_info=True
            )
            metric_dict_list[i].append(metric_dict)
    print(f'----------------------------------finished------------------------------------------')


    for i in range(num_datasets):
        metric_avg_dict = {}
        metric_std_dict = {}
        for key in metric_dict_list[0][0].keys():
            tmp_list = []
            for j in range(repeat_times):
                tmp_list.append(metric_dict_list[j][i][key])
            metric_avg_dict[key] = np.mean(np.array(tmp_list))
            metric_std_dict[key] = np.std(np.array(tmp_list))
        metric_avg_dict_list.append(metric_avg_dict)
        metric_std_dict_list.append(metric_std_dict)
    for i in range(num_datasets):
        print(f'------------avg metric ± (std) for dataset {image_path_list[i]}----------------')
        for key in metric_avg_dict_list[i].keys():
            print(f'{key}: {metric_avg_dict_list[i][key]:.3f} ± {metric_std_dict_list[i][key]:.3f}')


def lunar_loclearning_tubulin_tetra6_test():
    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_md',
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_hd/activations.csv', ]
    repeat_times = 5
    num_datasets = len(image_path_list)
    metric_dict_list = [[] for i in range(repeat_times)]
    metric_avg_dict_list = []
    metric_std_dict_list = []

    for i in range(repeat_times):
        print(f'--------------------------------repeat: {i}----------------------------------------')

        # set the file paths, calibration file is necessary,
        # experiment file is optional for background range estimation
        calib_file = None
        experiment_file = '../datasets/simu_tubulin/simu_tubulin_tetra6_md/1.tif'
        # experiment_file = None

        if calib_file is not None:
            # using the same psf parameters and camera parameters as beads calibration
            calib_dict = sio.loadmat(calib_file, simplify_cells=True)
            psf_params_dict = calib_dict['psf_params_fitted']
            psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
            psf_params_dict['objstage0'] = -700  # the objective stage position is different from beads calibration, unit: nm

            camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

        else:
            # manually set psf parameters
            zernike_aber = np.array([2, -2, 230, 2, 2, 0, 3, -1, -30, 3, 1, 30, 4, 0, -70, 3, -3, 0, 3, 3, 0,
                                     4, -2, -240, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                     5, -3, 0, 5, 3, 0, 6, -2, -17, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
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
            camera_params_dict = {
                'camera_type': 'scmos',
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
            'temporal_attn': True,
            'robust_training': True,  # if True, the training data will be added with some random Zernike aberrations
            'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
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
        lunar_model = ailoc.lunar.Lunar_LocLearning(psf_params_dict, camera_params_dict, sampler_params_dict)

        # lunar_model.check_training_psf()

        # lunar_model.check_training_data()

        lunar_model.build_evaluation_dataset(napari_plot=False)
        # gt_csv = '../datasets/Snow_dataset_fs/snow_astig_md/activations.csv'
        # deeploc_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :]
        # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
        # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
        # deeploc_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])

        file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_LL.pt'
        lunar_model.online_train(
            batch_size=2,
            max_iterations=40000,
            eval_freq=500,
            file_name=file_name
        )

        # # plot evaluation performance during the training
        # ailoc.common.plot_train_record(lunar_model)

        # test single emitter localization accuracy with CRLB
        ailoc.common.test_single_emitter_accuracy(
            loc_model=lunar_model,
            psf_params=lunar_model.dict_psf_params,
            xy_range=(-50, 50),
            z_range=np.array(lunar_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(lunar_model.dict_sampler_params['photon_range']),
            bg=np.mean(lunar_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=1000,
            show_res=False
        )

        for image_path, gt_path in zip(image_path_list, gt_path_list):
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

            # lunar_analyzer.check_single_frame_output(frame_num=11)

            t0 = time.time()
            preds_array, preds_array_rescaled = lunar_analyzer.divide_and_conquer()
            print(f'Prediction time cost: {time.time() - t0} s')

            # read the ground truth and calculate metrics
            gt_array = ailoc.common.read_csv_array(gt_path)

            metric_dict, paired_array = ailoc.common.pair_localizations(
                prediction=preds_array,
                ground_truth=gt_array,
                frame_num=lunar_analyzer.tiff_dataset.end_frame_num,
                fov_xy_nm=lunar_analyzer.fov_xy_nm,
                print_info=True
            )
            metric_dict_list[i].append(metric_dict)
    print(f'----------------------------------finished------------------------------------------')

    for i in range(num_datasets):
        metric_avg_dict = {}
        metric_std_dict = {}
        for key in metric_dict_list[0][0].keys():
            tmp_list = []
            for j in range(repeat_times):
                tmp_list.append(metric_dict_list[j][i][key])
            metric_avg_dict[key] = np.mean(np.array(tmp_list))
            metric_std_dict[key] = np.std(np.array(tmp_list))
        metric_avg_dict_list.append(metric_avg_dict)
        metric_std_dict_list.append(metric_std_dict)
    for i in range(num_datasets):
        print(f'------------avg metric ± (std) for dataset {image_path_list[i]}----------------')
        for key in metric_avg_dict_list[i].keys():
            print(f'{key}: {metric_avg_dict_list[i][key]:.3f} ± {metric_std_dict_list[i][key]:.3f}')


if __name__ == '__main__':
    # lunar_loclearning_tubulin_astig_test()
    lunar_loclearning_tubulin_tetra3_test()
    # lunar_loclearning_tubulin_tetra6_test()
