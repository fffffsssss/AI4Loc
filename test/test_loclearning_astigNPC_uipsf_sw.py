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
ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def lunar_loclearning():
    # set the file paths, calibration file is necessary,
    # experiment file is needed for background range estimation and synchronized learning
    calib_file = None
    experiment_file = '../datasets/uipsf-data/Fig3_nup96_zpos5um/Nup96SNAP_BME_i50SM_ex200_zpos5um_NC_1_MMStack_Default.ome.tif'

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 680  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['refmed'] = 1.352  # refractive index of sample medium may be different from beads
        psf_params_dict['objstage0'] = -4660  # the objective stage position is different from beads calibration, unit: nm
        psf_params_dict['psf_size'] = 51  # the size of the PSF needs larger when large spherical aberration exists

        camera_params_dict = calib_dict['calib_params_dict']['camera_params_dict']

    else:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        zernike_aber[:, 2] = [-0.865293522938792,
                              50.2402387494785,
                              -1.50367792976949,
                              1.91149748387304,
                              -22.6777181997294,
                              2.08060645749109,
                              -0.710395296926919,
                              1.22264321176102,
                              -2.10604771930759,
                              1.91825379088753,
                              -0.331338571487232,
                              3.80041794591443,
                              -0.949889315110660,
                              -1.55270526467370,
                              -1.01292748705404,
                              -0.980513764653257,
                              -1.86096187688476,
                              3.04570736652086,
                              -1.14870236529330,
                              -3.83271764707233,
                              3.56771545948601, ]
        psf_params_dict = {
            'na': 1.43,
            'wavelength': 680,  # unit: nm
            'refmed': 1.516,
            'refcov': 1.516,
            'refimm': 1.516,
            'zernike_mode': zernike_aber[:, 0:2],
            'zernike_coef': zernike_aber[:, 2],
            'pixel_size_xy': (127, 116),
            'otf_rescale_xy': (0.5, 0.5),
            'npupil': 64,
            'psf_size': 35,
            'objstage0': -0,
            # 'zemit0': 0,
        }

        # manually set camera parameters
        # camera_params_dict = {'camera_type': 'idea'}
        camera_params_dict = {'camera_type': 'emccd',
                              'qe': 1.0,
                              'spurious_charge': 0.0015,
                              'em_gain': 100,
                              'read_noise_sigma': 58.8,
                              'e_per_adu': 5,
                              'baseline': 398.6,
                              }
        # camera_params_dict = {
        #     'camera_type': 'scmos',
        #     'qe': 0.9,
        #     'spurious_charge': 0.002,
        #     'read_noise_sigma': 1.61,
        #     'read_noise_map': None,
        #     'e_per_adu': 0.05,
        #     'baseline': 398.6,
        # }

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
        'robust_training': True,
        'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (1000, 20000),
        'z_range': (-700, 700),
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

    # # using the simulated data with ground truth as evaluation dataset
    # gt_csv = '../datasets/mismatch_data2/activations.csv'
    # lunar_model.evaluation_dataset['data'] = ailoc.common.cpu(experimental_images)[0:2000, None, :, :].reshape((-1,lunar_model.context_size,experimental_images.shape[-2],experimental_images.shape[-1]))
    # molecule_list_gt = sorted(ailoc.common.read_csv_array(gt_csv), key=lambda x: x[0])
    # end_idx = ailoc.common.find_frame(molecule_list_gt, frame_nbr=2000)
    # lunar_model.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt[:end_idx])
    # ailoc.common.viewdata_napari(lunar_model.evaluation_dataset['data'])

    file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_LL.pt'
    # torch.autograd.set_detect_anomaly(True)
    lunar_model.online_train(
        batch_size=2,
        max_iterations=40000,
        eval_freq=1000,
        file_name=file_name,
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
        num_workers=0,
    )
    lunar_analyzer.check_single_frame_output(frame_num=12)
    preds_array, preds_rescale_array = lunar_analyzer.divide_and_conquer(degrid=True)

    # # read the ground truth and calculate metrics
    # gt_array = ailoc.common.read_csv_array("../datasets/simu_tubulin/simu_tubulin_astig_hd/activations.csv")
    #
    # metric_dict, paired_array = ailoc.common.pair_localizations(
    #     prediction=preds_array,
    #     ground_truth=gt_array,
    #     frame_num=lunar_analyzer.tiff_dataset.end_frame_num,
    #     fov_xy_nm=lunar_analyzer.fov_xy_nm,
    #     print_info=True,
    # )


def lunar_analyze():
    loc_model_path = '../results/2024-09-25-13-30LUNAR_SL.pt'
    image_path = '../datasets/uipsf-data/Fig3_nup96_zpos5um'  # can be a tiff file path or a folder path
    save_path = '../results/' + \
                os.path.split(loc_model_path)[-1].split('.')[0] + \
                '_'+os.path.split(image_path)[-1].split('.')[0]+'_predictions.csv'

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        lunar_model = torch.load(f)

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

    lunar_analyzer.check_single_frame_output(frame_num=12)
    # lunar_analyzer.check_single_frame_output(frame_num=17)
    # lunar_analyzer.check_single_frame_output(frame_num=32)

    preds_array, preds_rescale_array = lunar_analyzer.divide_and_conquer(degrid=True)


if __name__ == '__main__':
    lunar_loclearning()
    # lunar_loclearning_ckpoint()
    # lunar_synclearning()
    # lunar_synclearning_ckpoint()
    # lunar_analyze()
    # lunar_competitive_analyze()
