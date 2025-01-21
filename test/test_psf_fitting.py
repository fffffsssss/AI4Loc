import napari
import numpy as np
import torch
from torch import linalg as LA
import torch.nn as nn
import time
import matplotlib.pyplot as plt

import ailoc.common
from ailoc.simulation.vectorpsf import VectorPSFCUDA, VectorPSFTorch
ailoc.common.setup_seed(42)


def test_psf_fitting():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = zernike_aber[:, 2] + np.random.normal(0, 30, zernike_aber.shape[0])
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = (100, 100)
    otf_rescale_xy = (0., 0.)
    npupil = 128
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_torch = VectorPSFTorch(psf_params_dict)

    # set random emitter positions
    n_zstack = 21
    x = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    x += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
    y = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    y += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
    z = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    objstage = ailoc.common.gpu(torch.linspace(-1000, 1000, n_zstack))  # unit nm
    objstage += ailoc.common.gpu(torch.randn(1) * 10)
    photons = ailoc.common.gpu(torch.ones(n_zstack) * 10000)  # unit photons
    photons += ailoc.common.gpu(torch.rand(n_zstack) * 2000)
    bg = ailoc.common.gpu(torch.ones(n_zstack) * 10)  # unit photons
    bg += ailoc.common.gpu(torch.rand(n_zstack) * 20)

    # run the VectorPSF generation
    psfs_data_torch = psf_torch.simulate(x, y, z, photons, objstage)

    # set the camera noise
    camera = ailoc.simulation.IdeaCamera()
    data_noised = camera.forward(psfs_data_torch + bg[:, None, None])
    # data_noised = torch.ceil(psfs_data_torch + bg[:, None, None])

    # view data
    # ailoc.common.viewdata_napari(data_noised)

    # torch.autograd.set_detect_anomaly(True)
    # fit the data, first prepare the parameters
    # reset the zernike_coef for fitting test
    psf_params_dict['zernike_coef'] = np.zeros(psf_params_dict['zernike_coef'].shape)
    # psf_params_dict['otf_rescale_xy'] = (0.01, 0.01)
    psf_torch_fitted = VectorPSFTorch(psf_params_dict, req_grad=True)
    x_fitted = nn.parameter.Parameter(ailoc.common.gpu(0.0))  # 1 parameter
    y_fitted = nn.parameter.Parameter(ailoc.common.gpu(0.0))  # 1 parameter
    objstage_fitted = nn.parameter.Parameter(ailoc.common.gpu(0.0))  # 1 parameter
    photons_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.ones(n_zstack) * 5))  # n_zstack parameters
    bg_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.ones(n_zstack) * 10))  # n_zstack parameters

    # # AdamW
    # tolerance = 1e-7
    # old_loss = 1e10
    # optimizer = torch.optim.AdamW([
    #     psf_torch_fitted.zernike_coef,
    #     x_fitted,
    #     y_fitted,
    #     objstage_fitted,
    #     photons_fitted,
    #     bg_fitted
    #     ], lr=5)
    #
    # for iterations in range(1000):
    #     x_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + x_fitted
    #     y_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + y_fitted
    #     z_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))
    #     objstage_tmp = ailoc.common.gpu(torch.linspace(-1000, 1000, n_zstack)) + objstage_fitted
    #     photons_tmp = torch.clamp(photons_fitted, min=1) * 1000
    #     bg_tmp = torch.clamp(bg_fitted, min=1)
    #
    #     psf_torch_fitted._pre_compute()
    #     model_fitted = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]
    #     model = torch.distributions.Poisson(model_fitted)
    #     loss = -model.log_prob(data_noised).sum()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f'iter:{iterations}, nll:{loss.item()}, '
    #           f'zernike sum abs err:{np.sum(np.abs(ailoc.common.cpu(psf_torch_fitted.zernike_coef - psf_torch.zernike_coef)))}')
    #     if abs(old_loss - loss.item()) < tolerance:
    #         break
    #     old_loss = loss.item()

    # LBFGS
    tolerance = 1e-7
    old_loss = 1e10
    optimizer = torch.optim.LBFGS([
        psf_torch_fitted.zernike_coef,
        # psf_torch_fitted.otf_rescale_xy,  # this is difficult to fit, does not converge
        x_fitted,
        y_fitted,
        objstage_fitted,
        photons_fitted,
        bg_fitted
    ])

    def closure():
        x_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + x_fitted
        y_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + y_fitted
        z_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))
        objstage_tmp = ailoc.common.gpu(torch.linspace(-1000, 1000, n_zstack)) + objstage_fitted
        photons_tmp = torch.clamp(photons_fitted, min=1) * 1000
        bg_tmp = torch.clamp(bg_fitted, min=1)

        # psf_torch_fitted._pre_compute()
        model_fitted = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]
        model = torch.distributions.Poisson(model_fitted)
        loss = -model.log_prob(data_noised).sum()
        # loss = torch.sum(2*((model_fitted-data_noised)-data_noised*torch.log(model_fitted/data_noised)))
        optimizer.zero_grad()
        loss.backward()
        return loss

    for iteration in range(75):
        loss = closure()
        print(f'iter:{iteration}, nll:{loss.item()}, '
              f'zernike sum abs err:{np.sum(np.abs(ailoc.common.cpu(psf_torch_fitted.zernike_coef - psf_torch.zernike_coef)))}')
        if abs(old_loss - loss.item()) < tolerance:
            break
        old_loss = loss.item()
        optimizer.step(closure)

    # view the fitted data
    x_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + x_fitted
    y_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack)) + y_fitted
    z_tmp = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))
    objstage_tmp = ailoc.common.gpu(torch.linspace(-1000, 1000, n_zstack)) + objstage_fitted
    photons_tmp = photons_fitted * 1000
    bg_tmp = bg_fitted

    model_fitted = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]

    print(f"the relative root of squared error is: "
          f"{LA.norm((model_fitted - data_noised).flatten(), ord=2) / LA.norm(model_fitted.flatten(), ord=2) * 100}%")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(ailoc.common.cpu(psf_torch_fitted.zernike_coef), '\n', ailoc.common.cpu(psf_torch.zernike_coef))

    # ailoc.common.cmpdata_napari(data_noised, model_fitted)
    ailoc.common.plot_image_stack_difference(data_noised, model_fitted)


def test_single_beads_fitting():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 70, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 20, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = zernike_aber[:, 2] + np.random.normal(0, 30, zernike_aber.shape[0])
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = (100, 100)
    otf_rescale_xy = (0, 0)
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    # psf_cuda = VectorPSFCUDA(psf_params_dict)
    psf_torch = VectorPSFTorch(psf_params_dict)

    # set random emitter positions
    n_zstack = 201
    x = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    x += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
    y = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    y += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
    z = ailoc.common.gpu(torch.linspace(0, 0, n_zstack))  # unit nm
    objstage = ailoc.common.gpu(torch.linspace(-1000, 1000, n_zstack))  # unit nm
    objstage += ailoc.common.gpu(torch.randn(1) * 10)
    photons = ailoc.common.gpu(torch.ones(n_zstack) * 10000)  # unit photons
    photons += ailoc.common.gpu(torch.rand(n_zstack) * 2000)
    bg = ailoc.common.gpu(torch.ones(n_zstack) * 10)  # unit photons
    bg += ailoc.common.gpu(torch.rand(n_zstack) * 20)

    # run the VectorPSF generation
    psfs_data_torch = psf_torch.simulate(x, y, z, photons, objstage)

    # set the camera noise
    camera = ailoc.simulation.IdeaCamera()
    data_noised = camera.forward(psfs_data_torch + bg[:, None, None])

    # view data
    # ailoc.common.viewdata_napari(data_noised)

    # set the beads calibration parameters dictionary
    calib_psf_params_dict = psf_params_dict.copy()
    calib_params_dict = {'raw_images': data_noised,
                         'camera_model': camera,
                         'z_step': 10,
                         'roi_size': 25,
                         'filter_sigma': 2,
                         'threshold_rel': 0.8,
                         'fit_brightest': False,
                         'psf_params_dict': calib_psf_params_dict}
    calib_psf_params_dict['zernike_coef'] = np.zeros(21, dtype=np.float32)
    calib_psf_params_dict['psf_size'] = calib_params_dict['roi_size']

    # fit the noised beads
    t0 = time.time()
    results = ailoc.common.zernike_calibrate_3d_beads_stack(calib_params_dict)
    print(f"the fitting time is: {time.time() - t0}s")

    print(f"the relative root of squared error is: "
          f"{LA.norm((results['data_fitted'] - results['data_stacked']).flatten(), ord=2) / LA.norm(results['data_fitted'].flatten(), ord=2) * 100}%")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(ailoc.common.cpu(results['psf_torch_fitted'].zernike_coef), '\n', ailoc.common.cpu(psf_torch.zernike_coef))

    # ailoc.common.cmpdata_napari(results['data_stacked'], results['data_fitted'])
    ailoc.common.plot_image_stack_difference(results['data_stacked'], results['data_fitted'])


def test_multi_beads_fitting():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 50, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 20, 3, -3, 0, 3, 3, 0,
                             4, -2, 200, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = zernike_aber[:, 2] + np.random.normal(0, 30, zernike_aber.shape[0])
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = (100, 100)
    otf_rescale_xy = (0.5, 0.5)
    npupil = 64
    psf_size = 71

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    # psf_cuda = VectorPSFCUDA(psf_params_dict)
    psf_torch = VectorPSFTorch(psf_params_dict)

    # simulate several beads on an image
    n_beads = 5
    n_zstack = 121
    fov_size = 512
    data = ailoc.common.gpu(torch.zeros(n_zstack, fov_size, fov_size))
    delta = data.detach().clone()
    delata_indices_i = torch.floor(torch.rand(n_beads) * fov_size)
    delata_indices_j = torch.floor(torch.rand(n_beads) * fov_size)
    for indices in zip(delata_indices_i, delata_indices_j):
        delta[:, int(indices[0]), int(indices[1])] = 1

    x_list = []
    y_list = []
    z_list = []
    objstage_list = []
    photons_list = []
    for beads_idx in range(n_beads):
        # set random emitter positions for each beads
        x = ailoc.common.gpu(torch.zeros(n_zstack))  # unit nm
        x += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
        y = ailoc.common.gpu(torch.zeros(n_zstack))  # unit nm
        y += ailoc.common.gpu(torch.randn(1) * pixel_size_xy[0])
        z = ailoc.common.gpu(torch.zeros(n_zstack))  # unit nm
        objstage_prior = ailoc.common.gpu(torch.linspace(3000, -3000, n_zstack))  # unit nm
        objstage = objstage_prior + ailoc.common.gpu(torch.randn(1) * 10)
        photons = ailoc.common.gpu(torch.ones(n_zstack) * 100000)  # unit photons
        photons += ailoc.common.gpu(torch.rand(n_zstack) * 2000)

        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
        objstage_list.append(objstage)
        photons_list.append(photons)

    # background is uniform
    bg = ailoc.common.gpu(torch.ones(n_zstack) * 20)  # unit photons
    bg += ailoc.common.gpu(torch.rand(n_zstack) * 20)

    # run the VectorPSF generation
    x_list = torch.stack(x_list, dim=1).view(-1)
    y_list = torch.stack(y_list, dim=1).view(-1)
    z_list = torch.stack(z_list, dim=1).view(-1)
    photons_list = torch.stack(photons_list, dim=1).view(-1)
    objstage_list = torch.stack(objstage_list, dim=1).view(-1)
    psfs_data_torch = psf_torch.simulate(x_list, y_list, z_list, photons_list, objstage_list)

    data = ailoc.simulation.Simulator.place_psfs(delta=delta, psf_patches=psfs_data_torch)

    # set the camera noise
    camera_params_dict = {'camera_type': 'idea'}
    camera = ailoc.simulation.instantiate_camera(camera_params_dict)
    data_noised = ailoc.common.cpu(camera.forward(data + bg[:, None, None]))

    # set the beads calibration parameters dictionary
    calib_psf_params_dict = psf_params_dict.copy()
    calib_params_dict = {'raw_images': data_noised,
                         'camera_params_dict': camera_params_dict,
                         'z_step': 50,
                         'roi_size': psf_size,
                         'filter_sigma': 2,
                         'threshold_abs': 20,
                         'fit_brightest': False,
                         'psf_params_dict': calib_psf_params_dict}
    calib_psf_params_dict['zernike_coef'] = np.zeros(21, dtype=np.float32)
    # calib_psf_params_dict['psf_size'] = calib_params_dict['roi_size']

    # preprocess the bead stack
    beads_seg = ailoc.common.segment_local_max_beads(calib_params_dict)

    # fit the noised beads
    t0 = time.time()
    results = ailoc.common.zernike_calibrate_3d_beads_stack(calib_params_dict)
    print(f"the fitting time is: {time.time() - t0}s")

    print(f"the relative root of squared error is: "
          f"{np.linalg.norm((results['data_fitted'] - results['data_measured']).flatten(), ord=2) / np.linalg.norm(results['data_measured'].flatten(), ord=2) * 100}%")

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(ailoc.common.cpu(results['psf_params_fitted']['zernike_coef']), '\n',
          ailoc.common.cpu(psf_torch.zernike_coef))

    # ailoc.common.cmpdata_napari(results['data_measured'], results['data_fitted'])
    ailoc.common.plot_image_stack_difference(results['data_measured'], results['data_fitted'])


if __name__ == '__main__':
    # test_psf_fitting()
    # test_single_beads_fitting()
    test_multi_beads_fitting()
