import napari
import numpy as np
import torch
from torch import linalg as LA
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import tifffile
import skimage
from torch.cuda.amp import autocast

import ailoc.common
import ailoc.simulation


def segment_local_max_smlm_data(images, camera, filter_sigma, roi_size, threshold_abs):

    molecule_images = []
    for i in range(images.shape[0]):
        raw_images = ailoc.common.gpu(images[i].astype(np.float32))
        raw_images_photons = ailoc.common.cpu(camera.backward(raw_images))
        # remove the nonuniform background for better local max extraction, maybe unnecessary
        bg = skimage.restoration.rolling_ball(raw_images_photons, radius=roi_size // 2)
        images_bg = np.clip(raw_images_photons - bg, a_min=0, a_max=None)

        images_bg_filtered = skimage.filters.gaussian(images_bg, sigma=filter_sigma)
        peak_coords = skimage.feature.peak_local_max(images_bg_filtered,
                                                     min_distance=roi_size // 2,
                                                     # threshold_rel=threshold_rel,
                                                     threshold_abs=threshold_abs,
                                                     exclude_border=True)

        # # Display the results
        # fig, ax = plt.subplots()
        # # img_tmp = ax.imshow(raw_images_photons, cmap='gray')
        # img_tmp = ax.imshow(images_bg_filtered, cmap='gray')
        # for i in range(peak_coords.shape[0]):
        #     rect = plt.Rectangle((peak_coords[i, 1] - roi_size // 2, peak_coords[i, 0] - roi_size // 2), roi_size,
        #                          roi_size,
        #                          edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # # ax.plot(peak_coords[:, 1], peak_coords[:, 0], 'rx')
        # # ax.axis('off')
        # plt.colorbar(mappable=img_tmp, ax=ax, fraction=0.046, pad=0.04)
        # plt.show()

        for peak_coord in peak_coords:
            x, y = peak_coord
            mol_seg = raw_images_photons[int(x - roi_size / 2):int(x + roi_size / 2),
                                         int(y - roi_size / 2):int(y + roi_size / 2)]
            molecule_images.append(mol_seg)

    return np.vstack(molecule_images)


def segment_local_max_beads(params_dict: dict) -> list:
    # set parameters
    raw_images = ailoc.common.gpu(params_dict['raw_images'].astype(np.float32))
    camera_model = params_dict['camera_model']
    roi_size = params_dict['roi_size']
    filter_sigma = params_dict['filter_sigma']
    # threshold_rel = params_dict['threshold_rel']
    threshold_abs = params_dict['threshold_abs']

    # transform the raw images to photons
    raw_images_photons = ailoc.common.cpu(camera_model.backward(raw_images))

    # remove the nonuniform background for better local max extraction, maybe unnecessary
    bg = skimage.restoration.rolling_ball(np.mean(raw_images_photons, axis=0), radius=roi_size//2)
    images_bg = np.clip(raw_images_photons-bg, a_min=0, a_max=None)

    images_bg_filtered = skimage.filters.gaussian(np.mean(images_bg, axis=0), sigma=filter_sigma)
    peak_coords = skimage.feature.peak_local_max(images_bg_filtered,
                                                 min_distance=roi_size//2,
                                                 # threshold_rel=threshold_rel,
                                                 threshold_abs=threshold_abs,
                                                 exclude_border=True)

    # Display the results
    fig, ax = plt.subplots()
    img_tmp = ax.imshow(raw_images_photons.mean(axis=0), cmap='gray')
    # img_tmp = ax.imshow(images_bg_filtered, cmap='gray')
    for i in range(peak_coords.shape[0]):
        rect = plt.Rectangle((peak_coords[i, 1] - roi_size//2, peak_coords[i, 0] - roi_size//2), roi_size, roi_size,
                             edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # ax.plot(peak_coords[:, 1], peak_coords[:, 0], 'rx')
    # ax.axis('off')
    plt.colorbar(mappable=img_tmp, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    # segment the beads using the peak coordinates and roi_size
    beads_seg = []
    for peak_coord in peak_coords:
        x, y = peak_coord
        bead_seg = raw_images_photons[:, int(x-roi_size/2):int(x+roi_size/2), int(y-roi_size/2):int(y+roi_size/2)]
        beads_seg.append(bead_seg)

    params_dict['beads_seg'] = beads_seg

    return beads_seg


def zernike_calibrate_3d_beads_stack(params_dict: dict) -> dict:
    # torch.autograd.set_detect_anomaly(True)

    # set parameters
    beads_seg = params_dict['beads_seg']
    roi_size = params_dict['roi_size']
    z_step = params_dict['z_step']
    fit_brightest = params_dict['fit_brightest']
    psf_params_dict = params_dict['psf_params_dict']

    # calibrate the segmented beads, first prepare the parameters
    data_stacked = torch.clamp(ailoc.common.gpu(np.vstack(beads_seg)), min=1e-6)
    n_beads = len(beads_seg)
    n_zstack = beads_seg[0].shape[0]
    bg_estimated = data_stacked.view(n_zstack * n_beads, -1).min(dim=1).values
    photons_estimated = torch.sum((data_stacked - bg_estimated[:, None, None]), dim=(1, 2)) / 1000
    # fit all beads or only fit the brightest one
    if fit_brightest:
        tmp_brightness = -1
        final_slice = None
        for i in range(n_beads):
            slice_tmp = slice(i * n_zstack, (i + 1) * n_zstack)
            if photons_estimated[slice_tmp].mean() > tmp_brightness:
                tmp_brightness = photons_estimated[slice_tmp].mean()
                final_slice = slice_tmp
        n_beads = 1
        data_stacked = data_stacked[final_slice]

    # first we move the objective away from the beads and step closer
    objstage_prior = ailoc.common.gpu(torch.linspace(n_zstack//2*z_step, -(n_zstack//2*z_step), n_zstack))
    psf_torch_fitted = ailoc.simulation.VectorPSFTorch(psf_params_dict, req_grad=True, data_type=torch.float32)

    # n_beads parameters
    x_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    y_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    objstage_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))

    # n_zstack * n_beads parameters
    photons_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))
    bg_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))

    # initialize the parameters
    x_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size-1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       (roi_size-1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       roi_size))
    y_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size-1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       (roi_size-1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       roi_size))
    x_mesh, y_mesh = torch.meshgrid(x_image_linspace, y_image_linspace, indexing='xy')
    with torch.no_grad():
        bg_fitted += data_stacked.view(n_zstack * n_beads, -1).min(dim=1).values
        photons_fitted += torch.sum((data_stacked-bg_fitted[:, None, None]), dim=(1, 2))/1000
        for i in range(n_beads):
            slice_tmp = slice(i*n_zstack, (i+1)*n_zstack)
            x_fitted[i] += torch.sum(x_mesh*torch.mean(data_stacked[slice_tmp], dim=0))/(photons_fitted.mean()*1000)
            y_fitted[i] += torch.sum(y_mesh*torch.mean(data_stacked[slice_tmp], dim=0))/(photons_fitted.mean()*1000)

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
    #     x_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     y_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     z_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     objstage_tmp = ailoc.common.gpu(torch.zeros(n_beads * n_zstack))
    #     for i in range(n_beads):
    #         x_tmp[i * n_zstack:(i + 1) * n_zstack] = x_fitted[i]
    #         y_tmp[i * n_zstack:(i + 1) * n_zstack] = y_fitted[i]
    #         objstage_tmp[i * n_zstack:(i + 1) * n_zstack] = objstage_fitted[i] + objstage_prior
    #
    #     photons_tmp = torch.clamp(photons_fitted*1000, min=0)
    #     bg_tmp = torch.clamp(bg_fitted, min=0)
    #
    #     psf_torch_fitted._pre_compute()
    #     model_fitted = psf_torch_fitted.simulate(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]
    #     model = torch.distributions.Poisson(model_fitted)
    #     loss = -model.log_prob(data_stacked).sum()
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     print(f'iter:{iterations}, nll:{loss.item()}')
    #     if abs(old_loss - loss.item()) < tolerance:
    #         break
    #     old_loss = loss.item()

    # LBFGS
    tolerance = 1e-5
    old_loss = 1e10
    optimizer = torch.optim.LBFGS([
         psf_torch_fitted.zernike_coef,
         x_fitted,
         y_fitted,
         objstage_fitted,
         photons_fitted,
         bg_fitted
         ],
         lr=0.5,)
    # lambda_reg = 0
    lambda_reg = 1e4

    def closure():
        x_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        y_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        z_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        objstage_tmp = ailoc.common.gpu(torch.zeros(n_beads*n_zstack))
        for i in range(n_beads):
            x_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(x_fitted[i],
                                                           min=x_image_linspace.min(),
                                                           max=x_image_linspace.max())
            y_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(y_fitted[i],
                                                           min=y_image_linspace.min(),
                                                           max=y_image_linspace.max())
            objstage_tmp[i*n_zstack:(i+1)*n_zstack] = torch.clamp(objstage_fitted[i],
                                                                  min=objstage_prior.min(),
                                                                  max=objstage_prior.max()) + objstage_prior
        photons_tmp = torch.clamp(photons_fitted*1000, min=0)
        bg_tmp = torch.clamp(bg_fitted, min=0)

        psf_torch_fitted._pre_compute()
        mu = psf_torch_fitted.simulate_parallel(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]

        assert torch.min(mu) >= 0, 'fitting failed, mu should be positive'
        if photons_fitted.min() <= 0 or bg_fitted.min() <= 0:
            print('photons or bg <= 0, using torch.clamp and regularization')
            indices_photons = torch.nonzero(photons_fitted <= 0)
            indices_bg = torch.nonzero(bg_fitted <= 0)
            model = torch.distributions.Poisson(mu)
            loss = -model.log_prob(data_stacked).sum() - \
                   lambda_reg*(photons_fitted[indices_photons].sum() + bg_fitted[indices_bg].sum())
            # loss = torch.sum(2*((mu-data_stacked)-data_stacked*torch.log(mu/data_stacked))) - \
            #        lambda_reg*(photons_fitted[indices_photons].sum() + bg_fitted[indices_bg].sum())
        else:
            model = torch.distributions.Poisson(mu)
            loss = -model.log_prob(data_stacked).sum()
            # loss = torch.sum(2 * ((mu - data_stacked) - data_stacked * torch.log(mu / data_stacked)))

        optimizer.zero_grad()
        loss.backward()
        return loss.detach()

    # with autocast():
    for iteration in range(75):
        loss = closure()
        print(f'iter:{iteration}, nll:{loss.item()}')
        if abs(old_loss - loss.item()) < tolerance:
            break
        old_loss = loss.item()
        optimizer.step(closure)

    # view the fitted data
    x_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    y_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    z_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    objstage_tmp = ailoc.common.gpu(torch.zeros_like(photons_fitted))
    for i in range(n_beads):
        x_tmp[i * n_zstack:(i + 1) * n_zstack] = x_fitted[i]
        y_tmp[i * n_zstack:(i + 1) * n_zstack] = y_fitted[i]
        objstage_tmp[i * n_zstack:(i + 1) * n_zstack] = objstage_fitted[i] + objstage_prior
    photons_tmp = photons_fitted * 1000
    bg_tmp = bg_fitted

    mu = psf_torch_fitted.simulate_parallel(x_tmp, y_tmp, z_tmp, photons_tmp, objstage_tmp) + bg_tmp[:, None, None]

    results = {'calib_params_dict': params_dict, 'psf_torch_fitted': psf_torch_fitted,
               'data_stacked': data_stacked, 'data_fitted': mu}

    # ailoc.common.cmpdata_napari(results['data_stacked'], results['data_fitted'])

    return results


def zernike_calibrate_3d_beads_stack_cuda_ext(params_dict: dict) -> dict:
    # TODO: need to modify the cuda c zernike fit programme to provide both shared,
    #  unshared and patially shared mechanism, then rewrite this api

    # set parameters
    raw_images = params_dict['raw_images']
    camera_model = params_dict['camera_model']
    roi_size = params_dict['roi_size']
    z_step = params_dict['z_step']
    filter_sigma = params_dict['filter_sigma']
    threshold_rel = params_dict['threshold_rel']
    psf_params_dict = params_dict['psf_params_dict']

    # pre-process the raw images
    raw_images_photons = camera_model.backward(raw_images)
    beads_seg = segment_local_max(raw_images_photons, roi_size, filter_sigma, threshold_rel)

    # calibrate the segmented beads, first prepare the parameters
    data_stacked = torch.clamp(ailoc.common.gpu(np.vstack(beads_seg)), min=1e-6)
    n_beads = len(beads_seg)
    n_zstack = beads_seg[0].shape[0]
    objstage_prior = ailoc.common.gpu(torch.linspace(-(n_zstack // 2 * z_step), n_zstack // 2 * z_step, n_zstack))
    psf_torch_fitted = ailoc.simulation.VectorPSFTorch(psf_params_dict, req_grad=True, data_type=torch.float32)
    # n_beads parameters
    x_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    y_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    objstage_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_beads)))
    # n_zstack * n_beads parameters
    photons_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))
    bg_fitted = nn.parameter.Parameter(ailoc.common.gpu(torch.zeros(n_zstack * n_beads)))

    # initialize the parameters
    x_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size - 1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       (roi_size - 1) * psf_params_dict['pixel_size_xy'][0] / 2,
                                                       roi_size))
    y_image_linspace = ailoc.common.gpu(torch.linspace(-(roi_size - 1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       (roi_size - 1) * psf_params_dict['pixel_size_xy'][1] / 2,
                                                       roi_size))
    x_mesh, y_mesh = torch.meshgrid(x_image_linspace, y_image_linspace, indexing='xy')
    with torch.no_grad():
        bg_fitted += data_stacked.view(n_zstack * n_beads, -1).min(dim=1).values
        photons_fitted += torch.sum((data_stacked - bg_fitted[:, None, None]), dim=(1, 2)) / 1000
        for i in range(n_beads):
            slice_tmp = slice(i * n_zstack, (i + 1) * n_zstack)
            x_fitted[i] += torch.sum(x_mesh * torch.mean(data_stacked[slice_tmp], dim=0)) / (
                        photons_fitted.mean() * 1000)
            y_fitted[i] += torch.sum(y_mesh * torch.mean(data_stacked[slice_tmp], dim=0)) / (
                        photons_fitted.mean() * 1000)
    pass

