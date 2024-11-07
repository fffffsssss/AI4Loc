import copy

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndi
import scipy
import warnings

import ailoc.common
import ailoc.simulation


def build_roi_library(images,
                      raw_images,
                      psf_params_dict,
                      sampler_params_dict,
                      max_candi_num=50000,
                      min_signal_num=5000,
                      sparse_first=True,
                      attn_length=1):
    """
    Given raw SMLM data, use DoG and maximum finding to build a library of ROIs for synchronized learning
    of SyncLoc. Each ROI does not need to have only one single molecule, but can have multiple molecules.
    But this function will preferentially select the sparser roi, then the overlapping roi.

    Args:
        images (np.ndarray): SMLM data in photon unit, shape (n, h, w)
        raw_images (np.ndarray): SMLM data in adu unit, shape (n, h, w)
        psf_params_dict (dict): use the initial PSF parameters of SyncLoc to automatically determine the
            ROI extraction parameters
        sampler_params_dict (dict): use the initial sampler parameters of SyncLoc to automatically determine the
            ROI extraction parameters
        max_candi_num (int): maximum number of candidate signals to extract from raw data after border removal
        min_signal_num (int): minimum number of extracted signals in ROI library
        sparse_first (bool): whether to select the sparse ROI first, if False, directly extract ROIs with
           multiple signals
        attn_length (int): default is 1 for 2D ROIs,
            for SyncLoc should be consistent with the syncloc.attn_length (normally 7)

    Returns:
        Dict: Dictionary containing the sparse and dense ROI library
    """
    print('-' * 200)
    print('Building ROI library for synchronized learning...')

    # prepare the parameters for ROI extraction
    psf_size = psf_params_dict['psf_size']
    img_height, img_width = images.shape[-2:]
    # set the sparse roi size and dense roi size, make sure the roi size is a
    # multiple of 4 and should not be larger than the image size
    sparse_roi_size = psf_size
    factor = 4
    if (sparse_roi_size % 4 != 0):
        sparse_roi_size = (sparse_roi_size // factor + 1) * factor
    assert sparse_roi_size <= min(img_height, img_width), \
        f"sparse roi size larger than the image size! Please check the PSF size and image size."
    dense_roi_size = sparse_roi_size * 2
    # make the dense_roi_size compatible with image size and multiple of 4
    if dense_roi_size > min(img_height, img_width):
        dense_roi_size = (min(img_height, img_width) // factor) * factor
        if dense_roi_size <= sparse_roi_size:
            sparse_first = False
            dense_roi_size = sparse_roi_size
    # set the peak finding param
    dof_range = (sampler_params_dict['z_range'][1] - sampler_params_dict['z_range'][0])/1000
    dog_sigma = max(4, dof_range*2)
    find_max_kernel = dog_sigma + 1
    print(f'sparst first: {sparse_first}, '
          f'sparse_roi_size: {sparse_roi_size}, '
          f'dense_roi_size: {dense_roi_size}, '
          f'image size: {img_height}x{img_width}, '
          f'dog_sigma: {dog_sigma}, '
          f'find_max_kernel: {find_max_kernel}')

    # extract the peaks
    all_peaks_list, all_frames_list = [], []
    sparse_peaks_list, sparse_frames_list = [], []
    dense_peaks_list, dense_frames_list = [], []
    peaks_num = 0
    for frame, image in enumerate(images):
        following = np.min((frame + 100, images.shape[0]))
        image_nobg = image - np.mean(images[frame:following], axis=0)
        peaks = extract_smlm_peaks(image_nobg=image_nobg,
                                   dog_sigma=(dog_sigma, dog_sigma),
                                   find_max_thre=0.3,
                                   find_max_kernel=(find_max_kernel, find_max_kernel),)

        #  remove peaks that are too close to border
        if len(peaks) > 0:
            peaks = remove_border_peaks(peaks,
                                        sparse_roi_size // 2 + 1,
                                        image.shape)

        #  remove peaks that are too close to each other
        if len(peaks) > 0:
            all_peaks_list.append(peaks)
            all_frames_list.append(np.array([frame] * peaks.shape[0])[:, None])
            peaks_num += peaks.shape[0]
            tmp_sparse_peaks, tmp_dense_peaks = remove_close_peaks(peaks, np.hypot(sparse_roi_size, sparse_roi_size))
            if tmp_sparse_peaks.shape[0] > 0:
                sparse_peaks_list.append(tmp_sparse_peaks)
                sparse_frames_list.append(np.array([frame] * tmp_sparse_peaks.shape[0])[:, None])
            if tmp_dense_peaks.shape[0] > 0:
                dense_peaks_list.append(tmp_dense_peaks)
                dense_frames_list.append(np.array([frame] * tmp_dense_peaks.shape[0])[:, None])

        if peaks_num >= max_candi_num:
            break

    # extract the roi using the peaks list
    if sparse_first:
        #  select the sparse roi first
        sparse_rois, sparse_rois_yxt, sparse_roi_peaks_num = roi_extract_smlm(
            raw_images,
            sparse_peaks_list,
            sparse_frames_list,
            sparse_roi_size,
            edge_dist=sparse_roi_size // 2,
            sparse=True,
            attn_length=attn_length,
        )

        dense_rois, dense_start_yxt, dense_roi_peaks_num = roi_extract_smlm(
            raw_images,
            dense_peaks_list,
            dense_frames_list,
            dense_roi_size,
            edge_dist=sparse_roi_size // 2,
            sparse=False,
            attn_length=attn_length,
        )

    else:
        # directly extract the roi without selecting the sparse roi
        sparse_rois, sparse_rois_yxt, sparse_roi_peaks_num = [], [], 0
        dense_rois, dense_start_yxt, dense_roi_peaks_num = roi_extract_smlm(
            raw_images,
            all_peaks_list,
            all_frames_list,
            dense_roi_size,
            edge_dist=sparse_roi_size // 2,
            sparse=False,
            attn_length=attn_length,
        )

    if sparse_roi_peaks_num + dense_roi_peaks_num < min_signal_num:
        warnings.warn(f'Not enough signals ({sparse_roi_peaks_num + dense_roi_peaks_num}) '
                      f'found in the dataset to build the ROI library, '
                      'please provide more data or check the parameters, '
                      'the algorithm may not work properly.')

    roi_library = {}
    roi_library['sparse'] = [sparse_rois, sparse_rois_yxt, sparse_roi_peaks_num]
    roi_library['dense'] = [dense_rois, dense_start_yxt, dense_roi_peaks_num]
    all_peaks_yxt = np.concatenate([np.concatenate(all_peaks_list, axis=0),
                                    np.concatenate(all_frames_list, axis=0)],
                                   axis=1)
    roi_library['all_peaks'] = {'all_peaks_yxt': all_peaks_yxt,
                                'all_peaks_img_shape': (frame+1, img_height, img_width),
                                'find_max_kernel': find_max_kernel}

    print(f'ROI library built successfully! '
          f'Found {len(all_peaks_yxt)} peaks in the dataset, \n'
          f'Sparse ROI number: {len(sparse_rois)}, containing {sparse_roi_peaks_num} signals,\n'
          f'Dense ROI number: {len(dense_rois)}, containing {dense_roi_peaks_num} signals.')

    if len(sparse_rois) > 0:
        print('plot example sparse ROIs')
        ailoc.common.plot_image_stack(sparse_rois[:, attn_length//2, :, :])
    print('plot example dense ROIs')
    ailoc.common.plot_image_stack(dense_rois[:, attn_length // 2, :, :])

    return roi_library


def extract_smlm_peaks(image_nobg,
                       dog_sigma=None,
                       find_max_thre=0.3,
                       find_max_kernel=(3, 3), ):
    """
    Extracts the peak coordinates (row, column) from a SMLM image using DoG and maximum finding
    """

    if dog_sigma is not None and np.linalg.norm(dog_sigma) > 0:
        im2 = ndi.gaussian_filter(image_nobg, list(np.array(dog_sigma) * 0.75)) - ndi.gaussian_filter(image_nobg,
                                                                                                      dog_sigma)
    else:
        im2 = image_nobg
    coordinates = find_local_max(im2, threshold_rel=find_max_thre, kernel=find_max_kernel)
    coordinates = np.array(coordinates)

    centers = np.round(coordinates).astype(np.int32)

    return centers


def find_local_max(img,
                   threshold_rel,
                   kernel):
    """
    Find the local maxima in an image using a maximum filter and a threshold.
    """

    img_filtered = ndi.maximum_filter(img, size=kernel)
    img_max = (img_filtered == img) * img  # extracts only the local maxima but leaves the values in
    mask = (img_max == img)

    thresh = np.quantile(img[mask], 1 - 1e-4) * threshold_rel
    labels, num_labels = ndi.label(img_max > thresh)

    # Get the positions of the maxima
    coords = ndi.measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))

    return coords


def remove_border_peaks(peaks, border_dist, image_shape):
    """
    Removes peaks that are too close to the border of the image.
    """

    keep_idxs = (np.all(peaks - border_dist >= 0, axis=1) &
                 np.all(image_shape - peaks - border_dist >= 0, axis=1))
    return peaks[keep_idxs]


def remove_close_peaks(peaks, min_dist):
    """
    Calculates the distance between all peaks and removes the ones
    that are to close to each other in order to ensure that there is only
    one signal per roi.
    Note! If two beads are close together and one is close to border,
    in this case only the roi that is not close to the border is cut,
    but since the other one is not filtered out here,
    so it could be possible that there are two signals visible in one roi...
    """

    dist_matrix = scipy.spatial.distance_matrix(peaks, peaks)
    keep_matrix_idxs = np.where((0 == dist_matrix) | (dist_matrix > min_dist))
    unique, counts = np.unique(keep_matrix_idxs[0], return_counts=True)
    keep_idxs = unique[counts == peaks.shape[0]]
    return peaks[keep_idxs], peaks[np.setdiff1d(np.arange(peaks.shape[0]), keep_idxs)]


def roi_extract_smlm(images,
                     peaks_list,
                     frames_list,
                     roi_size,
                     edge_dist,
                     sparse=True,
                     attn_length=1,):
    """
    Extracts the 3D ROIs (peaks and temporal context) from the SMLM data.
    If sparse is True, directly extract the ROIs from the peaks_list,
    otherwise extract the multi-emitter ROIs considering multiple peaks a time.
    """
    roi_list = []
    roi_yxt_list = []
    roi_peaks_num = 0
    extra_length = attn_length//2
    if sparse:
        for frame_num, frame_peaks in zip(frames_list, peaks_list):
            image_tmp = images[frame_num[0, 0]-extra_length: frame_num[0, 0]+extra_length+1, :, :]
            if len(image_tmp) != attn_length:
                continue
            for peak in frame_peaks:
                # make sure the roi is not out of the image, after remove_border_peaks, this may not happen
                start_row = max(peak[0] - edge_dist, 0)
                start_col = max(peak[1] - edge_dist, 0)
                end_row = min(start_row + roi_size, image_tmp.shape[-2])
                end_col = min(start_col + roi_size, image_tmp.shape[-1])
                if end_row - start_row != roi_size or end_col - start_col != roi_size:
                    continue
                tmp_slice = (slice(0, image_tmp.shape[0]), slice(start_row, end_row), slice(start_col, end_col))
                # tmp_slice = roi_slice_3d(peak, roi_size, edge_dist, image_tmp.shape)
                # if tmp_slice is None:
                #     continue
                roi_tmp = image_tmp[tmp_slice]
                roi_list.append(roi_tmp)
                roi_yxt_list.append((start_row, start_col, frame_num[0, 0]))
                roi_peaks_num += 1
    else:
        for frame_num, frame_peaks in zip(frames_list, peaks_list):
            image_tmp = images[frame_num[0, 0]-extra_length: frame_num[0, 0]+extra_length+1, :, :]
            if len(image_tmp) != attn_length:
                continue
            processed_peaks = set()
            for peak in frame_peaks:
                if (peak[0], peak[1]) in processed_peaks:
                    continue
                start_row = max(peak[0]-edge_dist, 0)
                start_col = max(peak[1]-edge_dist, 0)
                end_row = min(start_row+roi_size, image_tmp.shape[-2])
                end_col = min(start_col+roi_size, image_tmp.shape[-1])
                if end_row-start_row != roi_size or end_col-start_col != roi_size:
                    continue
                roi_peaks = [(peak_row, peak_col) for peak_row, peak_col in frame_peaks if start_row <= peak_row < end_row and start_col <= peak_col < end_col]
                valid_flag = True
                for roi_peak in roi_peaks:
                    if not is_peak_valid(roi_peak, start_row, start_col, end_row, end_col, edge_dist):
                        valid_flag = False
                        break
                if valid_flag:
                    for roi_peak in roi_peaks:
                        processed_peaks.add((roi_peak[0], roi_peak[1]))
                    tmp_slice = (slice(0, image_tmp.shape[0]), slice(start_row, end_row), slice(start_col, end_col))
                    roi_tmp = image_tmp[tmp_slice]
                    roi_list.append(roi_tmp)
                    roi_yxt_list.append((start_row, start_col, frame_num[0, 0]))
                    roi_peaks_num += len(roi_peaks)

    return (np.array(roi_list),
            np.array(roi_yxt_list),
            roi_peaks_num)


def roi_slice_3d(center, roi_size, edge_dist, image_shape):
    """
    Constructs a 3D slice object for 3D array access given 2D center coordinates,
    ensuring that the ROI is not out of the image.
    If the ROI is out of the image, the slice will be adjusted to fit the image.
    If cannot get suitable ROI in the image, return None.
    """

    slices = []
    slices.append(slice(0, image_shape[0]))
    for d in range(2):
        # make sure the roi is not out of the image, after remove_border_peaks, this may not happen
        start = max(center[d] - edge_dist, 0)
        stop = min(start + roi_size, image_shape[1+d])
        if stop - start != roi_size:
            return None
        slices.append(slice(start, stop))

    return tuple(slices)


def is_peak_valid(roi_peak, start_row, start_col, end_row, end_col, edge_dist):
    """
    Judge if peak is not in the ROI borders
    """

    peak_row, peak_col = roi_peak
    return (start_row + edge_dist <= peak_row < end_row - edge_dist and
            start_col + edge_dist <= peak_col < end_col - edge_dist)


def est_training_density(images, roi_library, sampler_params_dict):
    """
    Estimate the emitter number per frame for training using the detected peaks in raw data
    """

    all_peaks_yxt = roi_library['all_peaks']['all_peaks_yxt']
    frame_num, img_height, img_width = roi_library['all_peaks']['all_peaks_img_shape']
    coords_xy = all_peaks_yxt[:, 1::-1]
    train_size = sampler_params_dict['train_size']

    # Create a grid to count points
    grid_step = roi_library['all_peaks']['find_max_kernel']  # unit: pixel
    grid_size_x, grid_size_y = int(img_width // grid_step), int(img_height // grid_step)
    x_bins, x_step = np.linspace(0, img_width, grid_size_x+1, retstep=True)
    y_bins, y_step = np.linspace(0, img_height, grid_size_y+1, retstep=True)

    # Count the number of points in each bin
    heatmap, _, _ = np.histogram2d(coords_xy[:, 0],
                                   coords_xy[:, 1],
                                   bins=(x_bins, y_bins))

    # Normalize the heatmap into density (counts/pixel^2/frame)
    heatmap /= frame_num
    heatmap /= (x_step * y_step)

    # find the 1% largest value of the heatmap as the threshold
    mask = heatmap > np.quantile(heatmap, 0.99)
    # use the median value of the heatmap to compute the average number of emitters per training frame
    num_em_avg = max(np.median(heatmap[mask]) * train_size**2, 10)  # get the train density

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),)
    # Plot heatmap
    axes[0].imshow(heatmap.T, cmap='hot', extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]])
    axes[0].set_title('Density Distribution of 2D Coordinates')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    fig.colorbar(axes[0].images[0], ax=axes[0], label='Signal number density')

    # Plot scatter plot
    axes[1].plot(coords_xy[:, 0], coords_xy[:, 1], 'o', markersize=8, markerfacecolor='none')
    axes[1].set_xlim([0, img_width])  # Set the x-axis range
    axes[1].set_ylim([0, img_height])  # Set the y-axis range
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].invert_yaxis()
    axes[1].set_title('Scatter Plot of 2D Coordinates')
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')

    # set the figure title
    plt.suptitle(f'Detected signals distribution')
    plt.show()

    return num_em_avg


def est_z0_zrange(psf_params_dict, sampler_params_dict):
    """
    Given the psf parameters and sampler parameters, this function will calculate the initial PSF,
    then adjust the actual focus z0 to the brightest position. Considering that the signals in the
    raw data may be far away from the focus z0, we also adjust the z_range to cover more signals,
    this is done by computing the SBR with a threshold.
    """

    psf_params = copy.deepcopy(psf_params_dict)
    sampler_params = copy.deepcopy(sampler_params_dict)

    # first we calculate the initial PSF in the axial range centered at approximate focus z0
    psf_params['zemit0'] = -psf_params['objstage0']/psf_params['refimm']*psf_params['refmed']
    # photon_avg = np.average(sampler_params['photon_range'])
    photon_avg = 5000  # use a fixed photon budget
    bg_avg = np.average(sampler_params['bg_range'])
    init_z_range = sampler_params['z_range']

    # adjust the z0 to ensure the molecule z position has physical meaning
    if init_z_range[0] + psf_params['zemit0'] < 0:
        psf_params['zemit0'] -= (init_z_range[0] + psf_params['zemit0'])

    z_bins = 61
    z_eval = ailoc.common.gpu(torch.linspace(*init_z_range, steps=z_bins))
    x_eval = ailoc.common.gpu(torch.zeros_like(z_eval))
    y_eval = ailoc.common.gpu(torch.zeros_like(z_eval))
    photon_eval = ailoc.common.gpu(torch.ones_like(x_eval) * photon_avg)

    # calculate the psfs at different positions
    init_psf_model = ailoc.simulation.VectorPSFTorch(psf_params)
    init_psfs = ailoc.common.cpu(init_psf_model.simulate(x_eval, y_eval, z_eval, photon_eval))

    # find the brightest position, add the offset (relative to the range center) to zemit0
    max_idx = np.argmax(np.max(init_psfs, axis=(-1, -2)))
    psf_params['zemit0'] += (z_eval[max_idx].item() - z_eval[z_bins//2].item())

    # adjust the z0 to ensure the molecule z position has physical meaning
    if init_z_range[0] + psf_params['zemit0'] < 0:
        psf_params['zemit0'] -= (init_z_range[0] + psf_params['zemit0'])

    # then re-calculate the psfs, and adjust the z_range to cover more signals probably in the raw data
    psf_model_1 = ailoc.simulation.VectorPSFTorch(psf_params)

    adjust_step = 100  # nm
    max_extended_step_num = 10  # the maximum extended range is 10*100=1000 nm
    left_adjusts = ailoc.common.gpu(
        np.concatenate([[init_z_range[0] - adjust_step*i] for i in range(max_extended_step_num, 0, -1)])
    )
    right_adjusts = ailoc.common.gpu(
        np.concatenate([[init_z_range[1] + adjust_step * i] for i in range(1, max_extended_step_num + 1)])
    )

    psfs_left = ailoc.common.cpu(
        psf_model_1.simulate(x_eval[0:max_extended_step_num],
                                      y_eval[0:max_extended_step_num],
                                      left_adjusts,
                                      photon_eval[0:max_extended_step_num])
    )

    psfs_right = ailoc.common.cpu(
        psf_model_1.simulate(x_eval[0:max_extended_step_num],
                                      y_eval[0:max_extended_step_num],
                                      right_adjusts,
                                      photon_eval[0:max_extended_step_num])
    )

    # calculate the peak SBR at different z
    peak_sbr_left = np.max(psfs_left / bg_avg, axis=(-1, -2))
    peak_sbr_right = np.max(psfs_right / bg_avg, axis=(-1, -2))

    # use the SBR threshold to determine the extended range
    sbr_threshold = 0.5
    left_side = left_adjusts[np.where(peak_sbr_left > sbr_threshold)[0][0]].item() \
        if np.any(peak_sbr_left > sbr_threshold) else init_z_range[0]
    right_side = right_adjusts[np.where(peak_sbr_right > 0.5)[0][-1]].item() \
        if np.any(peak_sbr_right > sbr_threshold) else init_z_range[1]
    sampler_params['z_range'] = (left_side, right_side)

    # adjust the z0 again to ensure the molecule z position has physical meaning
    if sampler_params['z_range'][0] + psf_params['zemit0'] < 0:
        psf_params['zemit0'] -= (sampler_params['z_range'][0] + psf_params['zemit0'])

    # plot the PSF data at extended z
    data_left = np.random.poisson(psfs_left + bg_avg)
    data_right = np.random.poisson(psfs_right + bg_avg)

    cmap = 'magma'
    num_z = data_left.shape[0]
    fig, ax_arr = plt.subplots(num_z,
                               2,
                               figsize=(1 * 6, 2 * num_z),
                               constrained_layout=True)
    ax = []
    plts = []
    for i in ax_arr:
        ax.append(i[0])
    for i in ax_arr:
        ax.append(i[1])

    for i in range(num_z):
        plts.append(ax[i].imshow(data_left[i], cmap=cmap))
        ax[i].set_title(f"Z: {ailoc.common.cpu(left_adjusts[i])} nm, Peak SBR: {peak_sbr_left[i]:.2f}")
    for i in range(num_z):
        plts.append(ax[i+num_z].imshow(data_right[i], cmap=cmap))
        ax[i+num_z].set_title(f"Z: {ailoc.common.cpu(right_adjusts[i])} nm, Peak SBR: {peak_sbr_right[i]:.2f}")
    plt.show()

    return psf_params['zemit0'], sampler_params['z_range']
