import torch
import numpy as np
import torch.nn.functional as F

import ailoc.common
import ailoc.simulation


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

        return ailoc.common.cpu(p_integrated_candidate[:, 0])


def sample_prob(p_pred, batch_size, thre_integrated=0.7):
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
        p_integrated[slice_tmp] = spatial_integration(p_pred[slice_tmp])

    p_sampled = np.where(p_integrated > thre_integrated, 1, 0)

    return p_sampled


def inference_map_to_localizations(inference_dict, pixel_size_xy, z_scale, photon_scale):
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


def gmm_to_localizations(inference_dict, thre_integrated, pixel_size_xy, z_scale, photon_scale, bg_scale, batch_size):
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

    # todo: implement this function, we can treat the localizations with low/high uncertainty differently
    #  to avoid the grid effects. For example, we can sample the distribution with high uncertainty
    #  instead of taking the mean value.

    inference_dict['prob_sampled'] = sample_prob(inference_dict['prob'], batch_size, thre_integrated)
    molecule_array = inference_map_to_localizations(inference_dict, pixel_size_xy, z_scale, photon_scale)
    inference_dict['bg_sampled'] = inference_dict['bg'] * bg_scale

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
            y uncertainty, z uncertainty, photon uncertainty, x_offset_pixel, y_offset_pixel].
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

    xo = preds_array[:, -2].copy()
    yo = preds_array[:, -1].copy()

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

    x_rescale = preds_array[:, 1] + (xo_rescale-xo) * pixel_size[0]
    y_rescale = preds_array[:, 2] + (yo_rescale-yo) * pixel_size[1]

    preds_array_rescale = np.column_stack((preds_array, xo_rescale, yo_rescale, x_rescale, y_rescale))

    return preds_array_rescale

