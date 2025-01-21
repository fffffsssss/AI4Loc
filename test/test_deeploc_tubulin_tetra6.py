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
import ailoc.deeploc
ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def deeploc_train():
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
            'psf_size': 71,
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
        'local_context': False,
        'robust_training': False,
        'context_size': 8,
        'train_size': 128,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (1000, 10000),
        'z_range': (-3000, 3000),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (49, 51),
        'bg_perlin': True,
    }

    # print learning parameters
    ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

    # instantiate the DeepLoc model and start to train
    deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict, camera_params_dict, sampler_params_dict)

    deeploc_model.check_training_psf()

    deeploc_model.check_training_data()

    deeploc_model.build_evaluation_dataset(napari_plot=False)
    # gt_csv = '../datasets/Snow_dataset_fs/snow_astig_md/activations.csv'
    # deeploc_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :]
    # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
    # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
    # deeploc_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
    deeploc_model.online_train(
        batch_size=2,
        max_iterations=40000,
        eval_freq=500,
        file_name=file_name
    )

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    # test single emitter localization accuracy with CRLB
    _, _, _ = ailoc.common.test_single_emitter_accuracy(
        loc_model=deeploc_model,
        psf_params=None,
        xy_range=(-50, 50),
        z_range=np.array(deeploc_model.dict_sampler_params['z_range']) * 0.98,
        photon=np.mean(deeploc_model.dict_sampler_params['photon_range']),
        bg=np.mean(deeploc_model.dict_sampler_params['bg_range']),
        num_z_step=31,
        num_repeat=1000,
        show_res=True
    )

    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_md',
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_hd/activations.csv', ]

    for image_path, gt_path in zip(image_path_list, gt_path_list):
        save_path = '../results/' + \
                    os.path.split(file_name)[-1].split('.')[0] + \
                    '_' + os.path.basename(image_path) + '_predictions.csv'

        deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(
            loc_model=deeploc_model,
            tiff_path=image_path,
            output_path=save_path,
            time_block_gb=1,
            batch_size=32,
            sub_fov_size=256,
            over_cut=8,
            num_workers=0
        )

        deeploc_analyzer.check_single_frame_output(frame_num=3)

        t0 = time.time()
        preds_array, preds_rescale_array = deeploc_analyzer.divide_and_conquer()
        print(f'Prediction time cost: {time.time() - t0} s')

        # read the ground truth and calculate metrics
        gt_array = ailoc.common.read_csv_array(gt_path)

        metric_dict, paired_array = ailoc.common.pair_localizations(
            prediction=preds_array,
            ground_truth=gt_array,
            frame_num=deeploc_analyzer.tiff_dataset.end_frame_num,
            fov_xy_nm=deeploc_analyzer.fov_xy_nm,
            print_info=True
        )


def deeploc_analysis():
    model_name = '../results/2024-08-09-10-09DeepLoc.pt'
    with open(model_name, 'rb') as f:
        deeploc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_md',
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_hd/activations.csv', ]

    for image_path, gt_path in zip(image_path_list, gt_path_list):
        save_path = '../results/' + \
                    os.path.split(model_name)[-1].split('.')[0] + \
                    '_' + os.path.basename(image_path) + '_predictions.csv'
        deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(
            loc_model=deeploc_model,
            tiff_path=image_path,
            output_path=save_path,
            time_block_gb=1,
            batch_size=32,
            sub_fov_size=256,
            over_cut=8,
            num_workers=0
        )
        deeploc_analyzer.check_single_frame_output(frame_num=3)
        preds_array, preds_rescale_array = deeploc_analyzer.divide_and_conquer()

        # read the ground truth and calculate metrics
        gt_array = ailoc.common.read_csv_array(gt_path)

        metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_array,
                                                                    ground_truth=gt_array,
                                                                    frame_num=deeploc_analyzer.tiff_dataset.end_frame_num,
                                                                    fov_xy_nm=deeploc_analyzer.fov_xy_nm,
                                                                    print_info=True)


def deeploc_competitive_analyze():
    model_name = '../results/2024-06-18-10-45DeepLoc.pt'
    with open(model_name, 'rb') as f:
        deeploc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    # analyze the experimental data
    image_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld',  # can be a tiff file path or a folder path
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_md',
                       '../datasets/simu_tubulin/simu_tubulin_tetra6_hd', ]
    gt_path_list = ['../datasets/simu_tubulin/simu_tubulin_tetra6_ld/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_md/activations.csv',
                    '../datasets/simu_tubulin/simu_tubulin_tetra6_hd/activations.csv', ]

    for image_path, gt_path in zip(image_path_list, gt_path_list):
        save_path = '../results/' + \
                    os.path.split(model_name)[-1].split('.')[0] + \
                    '_' + os.path.basename(image_path) + '_predictions.csv'

        deeploc_analyzer = ailoc.common.CompetitiveSmlmDataAnalyzer(
            loc_model=deeploc_model,
            tiff_path=image_path,
            output_path=save_path,
            time_block_gb=1,
            batch_size=32,
            sub_fov_size=256,
            over_cut=8,
            multi_GPU=True,
            end_frame_num=None,
        )

        t0 = time.time()
        deeploc_analyzer.start()
        print(f'Prediction time cost: {time.time() - t0} s')

        # read the ground truth and calculate metrics
        gt_array = ailoc.common.read_csv_array(gt_path)
        preds_array = ailoc.common.read_csv_array(save_path)

        metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_array,
                                                                    ground_truth=gt_array,
                                                                    frame_num=None,
                                                                    fov_xy_nm=deeploc_analyzer.fov_xy_nm,
                                                                    print_info=True)


if __name__ == '__main__':
    deeploc_train()
    # deeploc_analysis()
    # deeploc_competitive_analyze()

