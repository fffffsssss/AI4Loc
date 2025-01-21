import numpy as np
import ailoc.simulation
import ailoc.common

def test_Simulator():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
    #                       50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
    #                       80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0.5, 0.5]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': None, 'zernike_coef_map': np.random.rand(21, 512, 256),
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}

    # camera_params_dict = {'camera_type': 'emccd',
    #                       'qe': 0.9, 'spurious_charge': 0.002, 'em_gain': 300,
    #                       'read_noise_sigma': 74.4, 'e_per_adu': 45, 'baseline': 100.0}

    camera_params_dict = {'camera_type': 'scmos',
                          'qe': 0.9, 'spurious_charge': 0.002,
                          'read_noise_sigma': 1.6, 'read_noise_map': None,
                          'e_per_adu': 0.5, 'baseline': 100.0}

    sampler_params_dict = {'local_context': True,
                           'robust_training': True,
                           'train_size': 128,
                           'num_em_avg': 10,
                           'photon_range': [1000, 8000],
                           'z_range': [-700, 700],
                           'bg_range': [50, 100],
                           'bg_perlin': True}

    simulator = ailoc.simulation.Simulator(psf_params_dict, camera_params_dict, sampler_params_dict)

    train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, curr_sub_fov_xy = \
        simulator.sample_training_data(batch_size=16, iter_train=0)

    ailoc.common.viewdata_napari(train_data)

    eval_data, molecule_list_gt, sub_fov_xy_list = simulator.sample_evaluation_data(num_image=500)

    ailoc.common.viewdata_napari(eval_data)


if __name__ == '__main__':
    test_Simulator()
