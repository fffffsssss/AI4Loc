import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

import ailoc.common
import ailoc.simulation


def spatial_integration_v1(p_pred, thre_candi_1=0.3, thre_candi_2=0.6):
    """
    Extract local maximum from the probability map. Two situations are considered

    Args:
        p_pred: the original probability map output from the network
        thre_candi_1: the threshold for the original probability value for situation 1
        thre_candi_2: the threshold for the original probability value for situation 2

    Returns:
        torch.Tensor: The spatially integrated probability map with only candidate pixels non-zero
    """

    with torch.no_grad():
        p_pred = ailoc.common.gpu(p_pred)[:, None]

        # Situation 1: Local maximum with probability values > candi_thre (normally 0.3)
        # are regarded as candidates
        p_pred_clip = torch.where(p_pred > thre_candi_1, p_pred, torch.zeros_like(p_pred))
        # localize local maximum within a 3x3 patch
        pool = F.max_pool2d(p_pred_clip, kernel_size=3, stride=1, padding=1)
        candidate_mask1 = torch.eq(p_pred, pool).float()

        # Situation 2: In order do be able to identify two emitters in adjacent pixels we look for
        # probability values > 0.6 that are not part of the first mask
        p_pred_except_candi1 = p_pred * (1 - candidate_mask1)
        candidate_mask2 = torch.where(p_pred_except_candi1 > thre_candi_2,
                                      torch.ones_like(p_pred_except_candi1),
                                      torch.zeros_like(p_pred_except_candi1))

        # Add probability values from the 4 adjacent pixels to the center pixel
        diag = 0.  # 1/np.sqrt(2)
        filter_ = ailoc.common.gpu(torch.tensor([[diag, 1., diag],
                                                 [1, 1, 1],
                                                 [diag, 1, diag]])).view([1, 1, 3, 3])
        p_integrated = F.conv2d(p_pred, filter_, padding=1)

        p_integrated_candidate_1 = candidate_mask1 * p_integrated
        p_integrated_candidate_2 = candidate_mask2 * p_integrated

        # This is our final integrated probability map which we then threshold (normally > 0.7)
        # to get our final discrete locations
        p_integrated_candidate = torch.clamp(p_integrated_candidate_1 + p_integrated_candidate_2, 0., 1.)

        return ailoc.common.cpu(p_integrated_candidate[:, 0])


def spatial_integration(p_pred, thre_candi_1=0.3, thre_candi_2=0.6):
    """
    Extract local maximum from the probability map. Two situations are considered

    Args:
        p_pred: the original probability map output from the network
        thre_candi_1: the threshold for the original probability value for situation 1
        thre_candi_2: the threshold for the original probability value for situation 2

    Returns:
        torch.Tensor: The spatially integrated probability map with only candidate pixels non-zero
    """

    with torch.no_grad():
        p_pred = ailoc.common.gpu(p_pred)[:, None]

        # Situation 1: Local maximum with probability values > candi_thre (normally 0.3)
        # are regarded as candidates
        p_pred_clip = torch.where(p_pred > thre_candi_1, p_pred, torch.zeros_like(p_pred))
        # localize local maximum within a 3x3 patch
        pool = F.max_pool2d(p_pred_clip, kernel_size=3, stride=1, padding=1)
        candidate_mask1 = torch.eq(p_pred, pool).float()

        # Situation 2: In order do be able to identify two emitters in adjacent pixels we look for
        # probability values > 0.6 that are not part of the first mask
        p_pred_except_candi1 = p_pred * (1 - candidate_mask1)
        candidate_mask2 = torch.where(p_pred_except_candi1 > thre_candi_2,
                                      torch.ones_like(p_pred_except_candi1),
                                      torch.zeros_like(p_pred_except_candi1))

        # Add probability values from the 4 adjacent pixels to the center pixel
        diag = 0.  # 1/np.sqrt(2)
        filter_ = ailoc.common.gpu(torch.tensor([[diag, 1., diag],
                                                 [1, 1, 1],
                                                 [diag, 1, diag]])).view([1, 1, 3, 3])
        p_integrated = F.conv2d(p_pred, filter_, padding=1)

        p_integrated_candidate_1 = candidate_mask1 * p_integrated
        p_integrated_candidate_2 = candidate_mask2 * p_integrated

        # This is our final integrated probability map which we then threshold (normally > 0.7)
        # to get our final discrete locations
        p_integrated_candidate = torch.clamp(p_integrated_candidate_1 + p_integrated_candidate_2, 0., 1.)

        return p_integrated_candidate[:, 0]


def sample_prob_v1(p_pred, batch_size, thre_integrated=0.7, thre_candi_1=0.3, thre_candi_2=0.6):
    """
    Sample the probability map to get the binary map that indicates the existence of a molecule.

    Args:
        p_pred (torch.Tensor): the original probability map output from the network.
        batch_size (int): the batch size used for spatial integration.
        thre_integrated (float): the threshold for the integrated probability value.

    Returns:
        np.ndarray: The binary map that indicates the existence of a molecule.
    """

    num_img = len(p_pred)
    p_integrated = np.zeros_like(p_pred)
    for i in range(int(np.ceil(num_img/batch_size))):
        slice_tmp = np.index_exp[i*batch_size: min((i+1)*batch_size, num_img)]
        p_integrated[slice_tmp] = spatial_integration_v1(p_pred[slice_tmp], thre_candi_1, thre_candi_2)

    p_sampled = np.where(p_integrated > thre_integrated, 1, 0)

    return p_sampled


def sample_prob(p_pred, batch_size, thre_integrated=0.7, thre_candi_1=0.3, thre_candi_2=0.6):
    """
    Sample the probability map to get the binary map that indicates the existence of a molecule.

    Args:
        p_pred (torch.Tensor): the original probability map output from the network.
        batch_size (int): the batch size used for spatial integration.
        thre_integrated (float): the threshold for the integrated probability value.

    Returns:
        np.ndarray: The binary map that indicates the existence of a molecule.
    """

    num_img = len(p_pred)
    p_integrated = torch.zeros_like(p_pred)
    for i in range(int(np.ceil(num_img/batch_size))):
        slice_tmp = np.index_exp[i*batch_size: min((i+1)*batch_size, num_img)]
        p_integrated[slice_tmp] = spatial_integration(p_pred[slice_tmp], thre_candi_1, thre_candi_2)

    p_sampled = torch.where(p_integrated > thre_integrated, 1, 0)

    return p_sampled


def inference_map_to_localizations_v1(inference_dict, pixel_size_xy, z_scale, photon_scale):
    """
    Convert inference map to a list of molecules. The returned list is in the format of
    [frame, x, y, z, photon, integrated prob,
     x uncertainty, y uncertainty, z uncertainty, photon uncertainty,
     x_offset_pixel, y_offset_pixel]. The xy position is based on the size of input maps, if the map size is
     64 pixels, the xy position will be in the range of [0,64] * pixel size.

     Returns:
        np.ndarray: Molecule array
    """

    pred_mol_list = []

    for i_frame in range(inference_dict['prob'].shape[0]):
        prob_sampled_tmp = inference_dict['prob_sampled'][i_frame]
        prob_tmp = inference_dict['prob'][i_frame]
        x_offset_tmp = inference_dict['x_offset'][i_frame]
        y_offset_tmp = inference_dict['y_offset'][i_frame]
        z_offset_tmp = inference_dict['z_offset'][i_frame]
        photon_tmp = inference_dict['photon'][i_frame]
        x_sig_tmp = inference_dict['x_sig'][i_frame]
        y_sig_tmp = inference_dict['y_sig'][i_frame]
        z_sig_tmp = inference_dict['z_sig'][i_frame]
        photon_sig_tmp = inference_dict['photon_sig'][i_frame]

        rc_inds = np.nonzero(prob_sampled_tmp)

        for j in range(len(rc_inds[0])):
            pred_mol_list.append([i_frame + 1,
                                  0.5 + rc_inds[1][j] + x_offset_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  0.5 + rc_inds[0][j] + y_offset_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  z_offset_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  photon_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  prob_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  x_sig_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  y_sig_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  z_sig_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  photon_sig_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  x_offset_tmp[rc_inds[0][j], rc_inds[1][j]],
                                  y_offset_tmp[rc_inds[0][j], rc_inds[1][j]]
                                  ])

    pred_mol_array = np.array(pred_mol_list)
    if len(pred_mol_array) > 0:
        pred_mol_array[:, 1] *= pixel_size_xy[0]
        pred_mol_array[:, 2] *= pixel_size_xy[1]
        pred_mol_array[:, 3] *= z_scale
        pred_mol_array[:, 4] *= photon_scale
        pred_mol_array[:, 6] *= pixel_size_xy[0]
        pred_mol_array[:, 7] *= pixel_size_xy[1]
        pred_mol_array[:, 8] *= z_scale
        pred_mol_array[:, 9] *= photon_scale

    return pred_mol_array


def inference_map_to_localizations(p_sampled,
                                   p_pred,
                                   xyzph_pred,
                                   xyzph_sig_pred,
                                   bg_pred,
                                   pixel_size_xy,
                                   z_scale,
                                   photon_scale):
    """
    Convert inference map to a list of molecules. The returned list is in the format of
    [frame, x, y, z, photon, integrated prob,
     x uncertainty, y uncertainty, z uncertainty, photon uncertainty,
     x_offset, y_offset]. The xy position is based on the size of input maps, if the map size is
     64 pixels, the xy position will be in the range of [0,64] * pixel size.

     Returns:
        np.ndarray: Molecule array
    """

    idxs_3d = torch.nonzero(p_sampled)
    pred_mol_tensor = torch.zeros([len(idxs_3d), 12])

    pred_mol_tensor[:, 0] = idxs_3d[:, 0] + 1
    pred_mol_tensor[:, 1] = idxs_3d[:, 2] + 0.5 + xyzph_pred[:, 0][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 2] = idxs_3d[:, 1] + 0.5 + xyzph_pred[:, 1][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 3] = xyzph_pred[:, 2][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 4] = xyzph_pred[:, 3][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 5] = p_pred[idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 6] = xyzph_sig_pred[:, 0][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 7] = xyzph_sig_pred[:, 1][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 8] = xyzph_sig_pred[:, 2][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 9] = xyzph_sig_pred[:, 3][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 10] = xyzph_pred[:, 0][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]
    pred_mol_tensor[:, 11] = xyzph_pred[:, 1][idxs_3d[:, 0], idxs_3d[:, 1], idxs_3d[:, 2]]

    if len(pred_mol_tensor)>0:
        pred_mol_tensor[:, 1] *= pixel_size_xy[0]
        pred_mol_tensor[:, 2] *= pixel_size_xy[1]
        pred_mol_tensor[:, 3] *= z_scale
        pred_mol_tensor[:, 4] *= photon_scale
        pred_mol_tensor[:, 6] *= pixel_size_xy[0]
        pred_mol_tensor[:, 7] *= pixel_size_xy[1]
        pred_mol_tensor[:, 8] *= z_scale
        pred_mol_tensor[:, 9] *= photon_scale
        pred_mol_tensor[:, 10] *= pixel_size_xy[0]
        pred_mol_tensor[:, 11] *= pixel_size_xy[1]
    pred_mol_array = ailoc.common.cpu(pred_mol_tensor)

    return pred_mol_array


def gmm_to_localizations_v1(inference_dict, thre_integrated, pixel_size_xy, z_scale, photon_scale, bg_scale, batch_size):
    """
    Postprocess the GMM posterior to the molecule list.

    Args:
        inference_dict (dict): the inference result from the network, contains the GMM components.
        thre_integrated (float): the threshold for the integrated probability value.
        pixel_size_xy (list of int): [int int], the pixel size in the xy plane.
        z_scale (float): the scale factor for the z axis.
        photon_scale (float): the scale factor for the photon count.
        bg_scale (float): the scale factor for the background.
        batch_size (int): the batch size used for spatial integration.

    Returns:
        (np.ndarray, dict): the molecule list with frame ordered based on the input dict, the modified inference dict.
    """

    inference_dict['prob_sampled'] = sample_prob_v1(inference_dict['prob'], batch_size, thre_integrated)
    molecule_array = inference_map_to_localizations_v1(inference_dict, ailoc.common.cpu(pixel_size_xy), z_scale, photon_scale)
    inference_dict['bg_sampled'] = inference_dict['bg'] * bg_scale

    return molecule_array, inference_dict


def gmm_to_localizations(p_pred,
                         xyzph_pred,
                         xyzph_sig_pred,
                         bg_pred,
                         thre_integrated,
                         pixel_size_xy,
                         z_scale,
                         photon_scale,
                         bg_scale,
                         batch_size,
                         return_infer_map):
    """
    Postprocess the GMM posterior to the molecule list.

    Args:
        inference_dict (dict): the inference result from the network, contains the GMM components.
        thre_integrated (float): the threshold for the integrated probability value.
        pixel_size_xy (list of int): [int int], the pixel size in the xy plane.
        z_scale (float): the scale factor for the z axis.
        photon_scale (float): the scale factor for the photon count.
        bg_scale (float): the scale factor for the background.
        batch_size (int): the batch size used for spatial integration.

    Returns:
        (np.ndarray, dict): the molecule list with frame ordered based on the input dict, the modified inference dict.
    """

    p_sampled = sample_prob(p_pred, batch_size, thre_integrated)
    molecule_array = inference_map_to_localizations(p_sampled,
                                                    p_pred,
                                                    xyzph_pred,
                                                    xyzph_sig_pred,
                                                    bg_pred,
                                                    ailoc.common.cpu(pixel_size_xy),
                                                    z_scale,
                                                    photon_scale)
    bg_sampled = bg_pred * bg_scale

    if return_infer_map:
        inference_dict = {'prob': [], 'x_offset': [], 'y_offset': [], 'z_offset': [], 'photon': [],
                          'bg': [], 'x_sig': [], 'y_sig': [], 'z_sig': [], 'photon_sig': [],
                          'prob_sampled': [], 'bg_sampled': []}

        inference_dict['prob'].append(ailoc.common.cpu(p_pred))
        inference_dict['x_offset'].append(ailoc.common.cpu(xyzph_pred[:, 0, :, :]))
        inference_dict['y_offset'].append(ailoc.common.cpu(xyzph_pred[:, 1, :, :]))
        inference_dict['z_offset'].append(ailoc.common.cpu(xyzph_pred[:, 2, :, :]))
        inference_dict['photon'].append(ailoc.common.cpu(xyzph_pred[:, 3, :, :]))
        inference_dict['x_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 0, :, :]))
        inference_dict['y_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 1, :, :]))
        inference_dict['z_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 2, :, :]))
        inference_dict['photon_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 3, :, :]))
        inference_dict['bg'].append(ailoc.common.cpu(bg_pred))
        inference_dict['prob_sampled'].append(ailoc.common.cpu(p_sampled))
        inference_dict['bg_sampled'].append(ailoc.common.cpu(bg_sampled))
        for k in inference_dict.keys():
            inference_dict[k] = np.vstack(inference_dict[k])
    else:
        inference_dict = None

    return molecule_array, inference_dict


def histogram_equalization(x):
    """
    Histogram equalization for the sub-pixel offsets in the range (-1, 1) to
    uniform distribution in the range (-0.5, 0.5).
    """

    x = np.clip(x, -0.99, 0.99)

    x_pdf = np.histogram(x, bins=np.linspace(-1, 1, 201))
    x_cdf = np.cumsum(x_pdf[0]) / sum(x_pdf[0])

    # linear transform x[-0.99, 0.99] into (0,198], namely the independent variable of the CDF function
    ind = (x + 1) / 2 * 200 - 1.

    # decimal part of transformed x
    dec = ind - np.floor(ind)

    # get the CDF(transformed x), considering the discrete CDF function, so take two neighbour values and interpolate
    # CDF( ceil(ind) -- [1,199] )
    tmp1 = x_cdf[[int(i) + 1 for i in ind]]
    # CDF( floor(ind) -- [0,198] )
    tmp2 = x_cdf[[int(i) for i in ind]]

    # weighted average of cdf( floor(ind) ) and CDF( ceil(ind) ), equal to linear interpolation
    x_re = dec * tmp1 + (1 - dec) * tmp2 - 0.5

    # fig, axes = plt.subplots(3, 1)
    # axes[0].hist(x, bins=np.linspace(-1, 1, 201))
    # axes[1].plot(x_pdf[1][1:], x_cdf)
    # axes[2].hist(x_re, bins=np.linspace(-1, 1, 201))
    # plt.show()

    return x_re


def rescale_offset(preds_array, pixel_size=None, rescale_bins=20, sig_3d=False):
    """
    Rescales x and y offsets so that they are distributed uniformly within [-0.5, 0.5] to
    correct for biased outputs. All molecules are binned based on their uncertainties and then
    do histogram equalization within each bin.

    Args:
        preds_array (np.ndarray): the molecule list with format [frame, x, y, z, photon, integrated prob, x uncertainty,
            y uncertainty, z uncertainty, photon uncertainty, x_offset, y_offset].
        sig_3d (bool): If True, the z uncertainty will be used to bin the molecules.
        rescale_bins (int): The bias scales with the uncertainty of the localization. All molecules
            are binned according to their predicted uncertainty. Detections within different bins are then
            rescaled seperately. This specifies the number of bins.
        pixel_size (list of int): [int int], the pixel size in the xy plane.

    Returns:
        np.ndarray: the rescaled molecule list, the rescaled xo, yo, xnm, ynm are stored in the last four columns.
    """

    if pixel_size is None:
        pixel_size = [100, 100]

    xo = preds_array[:, -2].copy() / pixel_size[0]
    yo = preds_array[:, -1].copy() / pixel_size[1]

    x_sig = preds_array[:, -6].copy()
    y_sig = preds_array[:, -5].copy()
    z_sig = preds_array[:, -4].copy()

    x_sig_var = np.var(x_sig)
    y_sig_var = np.var(y_sig)
    z_sig_var = np.var(z_sig)

    xo_rescale = xo.copy()
    yo_rescale = yo.copy()

    tot_sig = x_sig ** 2 + (np.sqrt(x_sig_var / y_sig_var) * y_sig) ** 2
    if sig_3d:
        tot_sig += (np.sqrt(x_sig_var / z_sig_var) * z_sig) ** 2

    bins = np.interp(np.linspace(0, len(tot_sig), rescale_bins + 1), np.arange(len(tot_sig)), np.sort(tot_sig))

    for i in range(rescale_bins):
        inds = np.where((tot_sig > bins[i]) & (tot_sig < bins[i + 1]) & (tot_sig != 0))
        xo_rescale[inds] = histogram_equalization(xo[inds]) + np.mean(xo[inds])
        yo_rescale[inds] = histogram_equalization(yo[inds]) + np.mean(yo[inds])

        # fig, ax = plt.subplots(1, 2, constrained_layout=True)
        # ax[0].hist(xo[inds], bins=100)
        # ax[1].hist(xo_rescale[inds], bins=100)
        # plt.show()

    x_rescale = preds_array[:, 1] + (xo_rescale-xo) * pixel_size[0]
    y_rescale = preds_array[:, 2] + (yo_rescale-yo) * pixel_size[1]

    # preds_array_rescale = np.column_stack((preds_array, xo_rescale, yo_rescale, x_rescale, y_rescale))
    preds_array_rescale = preds_array.copy()
    preds_array_rescale[:, 1:3] = np.column_stack((x_rescale, y_rescale))

    # plot the histogram of the original and rescaled offsets
    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax[0, 0].hist(xo, bins=100, label='original x offset')
    ax[0, 0].legend()
    ax[0, 0].set_xlim([-0.5, 0.5])
    ax[0, 1].hist(xo_rescale, bins=100, label='rescaled x offset')
    ax[0, 1].legend()
    ax[0, 1].set_xlim([-0.5, 0.5])
    ax[1, 0].hist(yo, bins=100, label='original y offset')
    ax[1, 0].legend()
    ax[1, 0].set_xlim([-0.5, 0.5])
    ax[1, 1].hist(yo_rescale, bins=100, label='rescaled y offset')
    ax[1, 1].legend()
    ax[1, 1].set_xlim([-0.5, 0.5])
    plt.show()

    return preds_array_rescale


def resample_offset(preds_array, pixel_size=None, threshold=0.3):
    """
    Resample localizations' xy offsets with large uncertainties to avoid that zero expectation with large uncertainty
    results in grid effects, the original xy will be replaced by the resampled xy.

    Args:
        preds_array (np.ndarray): the molecule list with format [frame, x, y, z, photon, integrated prob, x uncertainty,
            y uncertainty, z uncertainty, photon uncertainty, x_offset, y_offset].
        threshold (float): the threshold of the uncertainty to resample, if the uncertainty relative to pixel size
            is larger than the threshold, the xy offset will be resampled.
        pixel_size (tuple of int): (int int), the pixel size in the xy plane.

    Returns:
        np.ndarray: the resampled molecule list, with the original xo, yo, xnm, ynm replaced by the resampled ones.
    """

    if pixel_size is None:
        pixel_size = (100, 100)

    xo = preds_array[:, -2].copy()
    yo = preds_array[:, -1].copy()

    x_sig = preds_array[:, -6].copy()
    y_sig = preds_array[:, -5].copy()
    z_sig = preds_array[:, -4].copy()

    xo_resample = xo.copy()
    yo_resample = yo.copy()

    tot_sig = np.sqrt(x_sig ** 2 + y_sig ** 2)

    # sig_thre = threshold * np.sqrt(pixel_size[0] ** 2 + pixel_size[1] ** 2)
    sig_thre_x = threshold * pixel_size[0]
    sig_thre_y = threshold * pixel_size[1]

    # inds = np.where(tot_sig > sig_thre)
    inds_x = np.where(x_sig > sig_thre_x)
    inds_y = np.where(y_sig > sig_thre_y)

    # xo_resample[inds] = np.random.normal(0, 1, len(inds[0])) * x_sig[inds] / pixel_size[0] + xo[inds]
    # yo_resample[inds] = np.random.normal(0, 1, len(inds[0])) * y_sig[inds] / pixel_size[1] + yo[inds]

    xo_resample[inds_x] = np.random.normal(0, 1, len(inds_x[0])) * x_sig[inds_x] + xo[inds_x]
    yo_resample[inds_y] = np.random.normal(0, 1, len(inds_y[0])) * y_sig[inds_y] + yo[inds_y]

    # xo_resample[inds_x] = histogram_equalization(xo[inds_x]) + np.mean(xo[inds_x])
    # yo_resample[inds_y] = histogram_equalization(yo[inds_y]) + np.mean(yo[inds_y])

    x_resample = preds_array[:, 1] + (xo_resample - xo)
    y_resample = preds_array[:, 2] + (yo_resample - yo)

    # preds_array_resample = np.column_stack((preds_array, xo_resample, yo_resample, x_resample, y_resample))
    preds_array_resample = preds_array.copy()
    preds_array_resample[:, 1:3] = np.column_stack((x_resample, y_resample))

    # plot the histogram of the original and resampled offsets
    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    ax[0, 0].hist(xo, bins=100, label='original x offset')
    ax[0, 0].legend()
    ax[0, 0].set_xlim([-0.5 * pixel_size[0], 0.5 * pixel_size[0]])
    ax[0, 1].hist(xo_resample, bins=100, label='resampled x offset')
    ax[0, 1].legend()
    ax[0, 1].set_xlim([-0.5 * pixel_size[0], 0.5 * pixel_size[0]])
    ax[1, 0].hist(yo, bins=100, label='original y offset')
    ax[1, 0].legend()
    ax[1, 0].set_xlim([-0.5 * pixel_size[1], 0.5 * pixel_size[1]])
    ax[1, 1].hist(yo_resample, bins=100, label='resampled y offset')
    ax[1, 1].legend()
    ax[1, 1].set_xlim([-0.5 * pixel_size[1], 0.5 * pixel_size[1]])
    plt.show()

    return preds_array_resample
