import numpy as np
import sys
sys.path.append('../../')
import time
import torch
from torch import linalg as LA
import tifffile
import os

import ailoc.deeploc
import ailoc.common
import ailoc.simulation


def beads_stack_calibrate():
    """
    Bead stack calibration.
    """

    # load bead stack
    beads_file_name = "../../datasets/npc_DMO1.2__5/beads_DMO1.2_+-1um_10nm_2/DMO1.2_+-1um_10nm_2_MMStack_Default.ome.tif"
    beads_data = tifffile.imread(beads_file_name)

    # set psf parameters
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    psf_params_dict = {'na': 1.5,
                       'wavelength': 680,  # unit: nm
                       'refmed': 1.518,
                       'refcov': 1.524,
                       'refimm': 1.518,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': 0,
                       'zemit0': 0,
                       'pixel_size_xy': (108, 108),
                       'otf_rescale_xy': (0., 0.),  # this is an empirical value, which maybe due to the pixelation
                       'npupil': 64,
                       'psf_size': 51}

    # set camera parameters
    camera_params_dict = {'camera_type': 'scmos',
                          'qe': 0.81, 'spurious_charge': 0.002,
                          'read_noise_sigma': 1.6, 'read_noise_map': None,
                          'e_per_adu': 0.47, 'baseline': 100.0}
    camera = ailoc.simulation.SCMOS(camera_params_dict)

    # set calibration parameters
    calib_params_dict = {'raw_images': beads_data,
                         'camera_model': camera,
                         'camera_params_dict': camera_params_dict,
                         'z_step': 10,
                         'roi_size': 25,
                         'filter_sigma': 3,
                         'threshold_abs': 20,
                         'fit_brightest': False,
                         'psf_params_dict': psf_params_dict}
    psf_params_dict['psf_size'] = calib_params_dict['roi_size']

    # preprocess the bead stack
    beads_seg = ailoc.common.segment_local_max_beads(calib_params_dict)

    # fit the noised beads
    t0 = time.time()
    results = ailoc.common.zernike_calibrate_3d_beads_stack(calib_params_dict)
    print(f"the fitting time is: {time.time() - t0}s")

    # print the fitting results
    print(f"the relative root of squared error is: "
          f"{LA.norm((results['data_fitted'] - results['data_stacked']).flatten(), ord=2) / LA.norm(results['data_fitted'].flatten(), ord=2) * 100}%")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    for i in range(len(results['psf_torch_fitted'].zernike_coef)):
        print(f"{ailoc.common.cpu(results['psf_torch_fitted'].zernike_mode[i])}: "
              f"{ailoc.common.cpu(results['psf_torch_fitted'].zernike_coef[i])}")

    ailoc.common.cmpdata_napari(results['data_stacked'], results['data_fitted'])

    # save the calibration results
    save_path = os.path.dirname(beads_file_name)+'/'+os.path.basename(beads_file_name).split('.')[0] + '_calib_results.pt'
    torch.save(results, save_path)
    print(f"the calibration results are saved in: {save_path}")


if __name__ == "__main__":
    beads_stack_calibrate()
