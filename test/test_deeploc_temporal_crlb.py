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

import ailoc.deeploc
import ailoc.common
import ailoc.simulation
# ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def test_deeploc_temporal_crlb_astig():
    attn_length_test = [9,7,5,3,1]
    for attn_length in attn_length_test:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
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
        camera_params_dict = {'camera_type': 'idea'}
        # camera_params_dict = {'camera_type': 'emccd',
        #                       'qe': 0.9,
        #                       'spurious_charge': 0.002,
        #                       'em_gain': 300,
        #                       'read_noise_sigma': 74.4,
        #                       'e_per_adu': 45,
        #                       'baseline': 100.0,
        #                       }
        # camera_params_dict = {
        #     'camera_type': 'scmos',
        #     'qe': 0.81,
        #     'spurious_charge': 0.002,
        #     'read_noise_sigma': 1.61,
        #     'read_noise_map': None,
        #     'e_per_adu': 0.47,
        #     'baseline': 100.0,
        # }

        # manually set sampler parameters
        sampler_params_dict = {
            'local_context': True if (attn_length != 1) else False,
            'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
            'context_size': 8,  # for each batch unit, simulate several frames share the same photophysics and bg to train
            'train_size': 128,
            'num_em_avg': 10,
            'eval_batch_size': 100,
            'photon_range': (1000, 10000),
            'z_range': (-700, 700),
            'bg_range': (40, 60),
            'bg_perlin': False,
        }

        # print learning parameters
        ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

        # instantiate the DeepLoc model and start to train
        deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict,
                                              camera_params_dict,
                                              sampler_params_dict,
                                              attn_length)

        deeploc_model.build_evaluation_dataset(napari_plot=False)

        file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
        deeploc_model.online_train(
            batch_size=2,
            max_iterations=40000,
            eval_freq=1000,
            file_name=file_name
        )

        # test single emitter localization accuracy with CRLB
        _, paired_array, z_xyz_crlb = ailoc.common.test_single_emitter_accuracy(
            loc_model=deeploc_model,
            psf_params=None,
            xy_range=(-50, 50),
            z_range=np.array(deeploc_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(deeploc_model.dict_sampler_params['photon_range']),
            bg=np.mean(deeploc_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=3000,
            show_res=True
        )

        # write the paired localizations to csv file
        save_paried_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb_test.csv'
        ailoc.common.write_csv_array(input_array=paired_array,
                                     filename=save_paried_path,
                                     write_mode='write paired localizations')

        # write the crlb to csv file
        save_crlb_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb.csv'
        ailoc.common.write_csv_array(input_array=z_xyz_crlb,
                                     filename=save_crlb_path,
                                     write_mode='write paired localizations')


def test_deeploc_temporal_crlb_tetra3():
    attn_length_test = [9,7,5,3,1]
    for attn_length in attn_length_test:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 120, 2, 2, 0, 3, -1, -0, 3, 1, 0, 4, 0, -0, 3, -3, 0, 3, 3, 0,
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
        camera_params_dict = {'camera_type': 'idea'}
        # camera_params_dict = {'camera_type': 'emccd',
        #                       'qe': 0.9,
        #                       'spurious_charge': 0.002,
        #                       'em_gain': 300,
        #                       'read_noise_sigma': 74.4,
        #                       'e_per_adu': 45,
        #                       'baseline': 100.0,
        #                       }
        # camera_params_dict = {
        #     'camera_type': 'scmos',
        #     'qe': 0.81,
        #     'spurious_charge': 0.002,
        #     'read_noise_sigma': 1.61,
        #     'read_noise_map': None,
        #     'e_per_adu': 0.47,
        #     'baseline': 100.0,
        # }

        # manually set sampler parameters
        sampler_params_dict = {
            'local_context': True if (attn_length != 1) else False,
            'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
            'context_size': 8,
            # for each batch unit, simulate several frames share the same photophysics and bg to train
            'train_size': 128,
            'num_em_avg': 10,
            'eval_batch_size': 100,
            'photon_range': (1000, 10000),
            'z_range': (-1500, 1500),
            'bg_range': (40, 60),
            'bg_perlin': False,
        }

        # print learning parameters
        ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

        # instantiate the DeepLoc model and start to train
        deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict,
                                              camera_params_dict,
                                              sampler_params_dict,
                                              attn_length)

        deeploc_model.build_evaluation_dataset(napari_plot=False)

        file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
        deeploc_model.online_train(
            batch_size=2,
            max_iterations=40000,
            eval_freq=1000,
            file_name=file_name
        )

        # test single emitter localization accuracy with CRLB
        _, paired_array, z_xyz_crlb = ailoc.common.test_single_emitter_accuracy(
            loc_model=deeploc_model,
            psf_params=None,
            xy_range=(-50, 50),
            z_range=np.array(deeploc_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(deeploc_model.dict_sampler_params['photon_range']),
            bg=np.mean(deeploc_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=3000,
            show_res=True
        )

        # write the paired localizations to csv file
        save_paried_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb_test.csv'
        ailoc.common.write_csv_array(input_array=paired_array,
                                     filename=save_paried_path,
                                     write_mode='write paired localizations')

        # write the crlb to csv file
        save_crlb_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb.csv'
        ailoc.common.write_csv_array(input_array=z_xyz_crlb,
                                     filename=save_crlb_path,
                                     write_mode='write paired localizations')


def test_deeploc_temporal_crlb_tetra6():
    # attn_length_test = [9,7,5,3,1]
    attn_length_test = [3, 1]
    for attn_length in attn_length_test:
        # manually set psf parameters
        zernike_aber = np.array([2, -2, 230, 2, 2, 0, 3, -1, -0, 3, 1, 0, 4, 0, -0, 3, -3, 0, 3, 3, 0,
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
        camera_params_dict = {'camera_type': 'idea'}
        # camera_params_dict = {'camera_type': 'emccd',
        #                       'qe': 0.9,
        #                       'spurious_charge': 0.002,
        #                       'em_gain': 300,
        #                       'read_noise_sigma': 74.4,
        #                       'e_per_adu': 45,
        #                       'baseline': 100.0,
        #                       }
        # camera_params_dict = {
        #     'camera_type': 'scmos',
        #     'qe': 0.81,
        #     'spurious_charge': 0.002,
        #     'read_noise_sigma': 1.61,
        #     'read_noise_map': None,
        #     'e_per_adu': 0.47,
        #     'baseline': 100.0,
        # }

        # manually set sampler parameters
        sampler_params_dict = {
            'local_context': True if (attn_length != 1) else False,
            'robust_training': False,  # if True, the training data will be added with some random Zernike aberrations
            'context_size': 8,
            # for each batch unit, simulate several frames share the same photophysics and bg to train
            'train_size': 128,
            'num_em_avg': 10,
            'eval_batch_size': 100,
            'photon_range': (1000, 10000),
            'z_range': (-3000, 3000),
            'bg_range': (40, 60),
            'bg_perlin': False,
        }

        # print learning parameters
        ailoc.common.print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict)

        # instantiate the DeepLoc model and start to train
        deeploc_model = ailoc.deeploc.DeepLoc(psf_params_dict,
                                              camera_params_dict,
                                              sampler_params_dict,
                                              attn_length)

        deeploc_model.build_evaluation_dataset(napari_plot=False)

        file_name = '../results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt'
        deeploc_model.online_train(
            batch_size=2,
            max_iterations=40000,
            eval_freq=1000,
            file_name=file_name
        )

        # test single emitter localization accuracy with CRLB
        _, paired_array, z_xyz_crlb = ailoc.common.test_single_emitter_accuracy(
            loc_model=deeploc_model,
            psf_params=None,
            xy_range=(-50, 50),
            z_range=np.array(deeploc_model.dict_sampler_params['z_range']) * 0.98,
            photon=np.mean(deeploc_model.dict_sampler_params['photon_range']),
            bg=np.mean(deeploc_model.dict_sampler_params['bg_range']),
            num_z_step=31,
            num_repeat=3000,
            show_res=True
        )

        # write the paired localizations to csv file
        save_paried_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb_test.csv'
        ailoc.common.write_csv_array(input_array=paired_array,
                                     filename=save_paried_path,
                                     write_mode='write paired localizations')

        # write the crlb to csv file
        save_crlb_path = '../results/' + os.path.split(file_name)[-1].split('.')[0] + '_crlb.csv'
        ailoc.common.write_csv_array(input_array=z_xyz_crlb,
                                     filename=save_crlb_path,
                                     write_mode='write paired localizations')


if __name__ == '__main__':
    # test_deeploc_temporal_crlb_astig()
    # test_deeploc_temporal_crlb_tetra3()
    test_deeploc_temporal_crlb_tetra6()
