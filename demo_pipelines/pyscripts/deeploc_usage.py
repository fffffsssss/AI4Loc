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
torch.backends.cudnn.benchmark = True


def deeploc_train():
    # set the file paths, calibration file is necessary,
    # experiment file is optional for background range estimation
    calib_file = None
    experiment_file = None

    if calib_file is not None:
        # using the same psf parameters and camera parameters as beads calibration
        calib_dict = sio.loadmat(calib_file, simplify_cells=True)
        psf_params_dict = calib_dict['psf_params_fitted']
        psf_params_dict['wavelength'] = 670  # wavelength of sample may be different from beads, unit: nm
        psf_params_dict['refmed'] = 1.518  # refractive index of sample medium may be different from beads
        psf_params_dict['objstage0'] = -0  # the objective stage position is different from beads calibration, unit: nm

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
                           'psf_size': 25,
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
    sampler_params_dict = {
        'local_context': False,
        'robust_training': False,
        'context_size': 1,  # for each batch unit, simulate several frames share the same photophysics and bg to train
        'train_size': 64,
        'num_em_avg': 10,
        'eval_batch_size': 100,
        'photon_range': (4000, 6000),
        'z_range': (-700, 700),
        'bg_range': bg_range if 'bg_range' in locals().keys() else (40, 60),
        'bg_perlin': False,
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

    deeploc_model.build_evaluation_dataset(napari_plot=False)

    file_name = '../../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
    deeploc_model.online_train(batch_size=10,
                               max_iterations=40000,
                               eval_freq=500,
                               file_name=file_name)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    # test single emitter localization accuracy with CRLB
    ailoc.common.test_single_emitter_accuracy(loc_model=deeploc_model,
                                              psf_params=deeploc_model.dict_psf_params,
                                              xy_range=(-50, 50),
                                              z_range=(-700, 700),
                                              photon=5000,
                                              bg=50,
                                              num_z_step=31,
                                              num_repeat=1000,
                                              show_res=True)

    # analyze the experimental data
    image_path = os.path.dirname(experiment_file)
    save_path = '../../results/' + \
                os.path.split(file_name)[-1].split('.')[0] + \
                '_' + os.path.basename(image_path) + '_predictions.csv'
    deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=deeploc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=32,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)
    deeploc_analyzer.check_single_frame_output(frame_num=3)
    preds_array, preds_rescale_array = deeploc_analyzer.divide_and_conquer()


def deeploc_ckpoint_train():
    model_name = '../../results/2023-09-09-16DeepLoc.pt'
    with open(model_name, 'rb') as f:
        deeploc_model = torch.load(f)
    deeploc_model.online_train(batch_size=1,
                               max_iterations=30000,
                               eval_freq=500,
                               file_name=model_name)


def deeploc_analyze():
    loc_model_path = '../../results/2023-12-05-20-02DeepLoc.pt'
    # can be a tiff file path or a folder path
    image_path = '../../datasets/sw_npc_20211028/NUP96_SNP647_3D_512_20ms_hama_mm_1800mW_3/'
    save_path = '../../results/' + \
                os.path.split(loc_model_path)[-1].split('.')[0] + \
                '_'+os.path.basename(image_path)+'_predictions.csv'

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        deeploc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=deeploc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=1,
                                                     batch_size=32,
                                                     sub_fov_size=256,
                                                     over_cut=8,
                                                     num_workers=0)

    deeploc_analyzer.check_single_frame_output(frame_num=3)

    preds_array, preds_rescale_array = deeploc_analyzer.divide_and_conquer()

    # # read the ground truth and calculate metrics
    # gt_array = ailoc.common.read_csv_array("../../datasets/match_data/activations.csv")
    #
    # metric_dict, paired_array = ailoc.common.pair_localizations(prediction=preds_rescale_array,
    #                                                             ground_truth=gt_array,
    #                                                             frame_num=deeploc_analyzer.tiff_dataset.end_frame_num,
    #                                                             fov_xy_nm=deeploc_analyzer.fov_xy_nm,
    #                                                             print_info=True)
    # # write the paired localizations to csv file
    # save_paried_path = '../../results/'+os.path.split(save_path)[-1].split('.')[0]+'_paired.csv'
    # ailoc.common.write_csv_array(input_array=paired_array,
    #                              filename=save_paried_path,
    #                              write_mode='write paired localizations')



if __name__ == '__main__':
    deeploc_train()
    # deeploc_ckpoint_train()
    # deeploc_analyze()
