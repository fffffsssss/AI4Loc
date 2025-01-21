import os
import napari
import numpy as np
import torch
from torch import linalg as LA
import time
import matplotlib.pyplot as plt

import ailoc.common
from ailoc.simulation.vectorpsf import VectorPSFCUDA, VectorPSFTorch
ailoc.common.setup_seed(25)
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def test_loc_model_accuracy():
    # set the loc model to test
    # model_name = '../results/2024-11-04-16-27DeepLoc.pt'
    model_name = '../results/2024-11-09-14-56LUNAR_LL.pt'
    with open(model_name, 'rb') as f:
        loc_model = torch.load(f)

    # # manually set psf parameters
    # zernike_aber = np.array(
    #     [2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
    #      4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
    #      5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
    #     dtype=np.float32).reshape([21, 3])
    # psf_params_dict = {'na': 1.5,
    #                    'wavelength': 670,  # unit: nm
    #                    'refmed': 1.518,
    #                    'refcov': 1.518,
    #                    'refimm': 1.518,
    #                    'zernike_mode': zernike_aber[:, 0:2],
    #                    'zernike_coef': zernike_aber[:, 2],
    #                    'pixel_size_xy': (100, 100),
    #                    'otf_rescale_xy': (0, 0),
    #                    'npupil': 64,
    #                    'psf_size': 51,
    #                    'objstage0': -0,
    #                    # 'zemit0': 0,
    #                    }

    _, paired_array, z_xyz_crlb = ailoc.common.test_single_emitter_accuracy(
        loc_model=loc_model,
        psf_params=None,
        xy_range=(-50, 50),
        z_range=np.array(loc_model.dict_sampler_params['z_range']) * 0.98,
        photon=np.mean(loc_model.dict_sampler_params['photon_range']),
        bg=np.mean(loc_model.dict_sampler_params['bg_range']),
        num_z_step=31,
        num_repeat=3000,
        show_res=True
    )

    # write the paired localizations to csv file
    save_paried_path = '../results/'+os.path.split(model_name)[-1].split('.')[0]+'_crlb_test.csv'
    ailoc.common.write_csv_array(input_array=paired_array,
                                 filename=save_paried_path,
                                 write_mode='write paired localizations')

    # write the crlb to csv file
    save_crlb_path = '../results/' + os.path.split(model_name)[-1].split('.')[0] + '_crlb.csv'
    ailoc.common.write_csv_array(input_array=z_xyz_crlb,
                                 filename=save_crlb_path,
                                 write_mode='write paired localizations')

if __name__ == "__main__":
    test_loc_model_accuracy()
