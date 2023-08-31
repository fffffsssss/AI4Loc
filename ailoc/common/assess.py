import numpy as np
from scipy.spatial.distance import cdist


def find_frame(molecule_list, frame_nbr, frame_parm_pos=0):
    """
    Find the list index of the last molecule with frame number not larger than the specified frame number and plus 1.

    Args:
        molecule_list (list): list of molecule parameters, sorted by frame number
        frame_nbr (int): frame number to find the molecule index
        frame_parm_pos (int): position of the frame number within each row of molecule list, start from 0

    Returns:
        list_index (int): the list index of the last molecule with frame number
            not larger than the specified frame number and plus 1, start from 0
    """

    end_index = None
    for i, molecule in enumerate(molecule_list):
        if molecule[frame_parm_pos] > frame_nbr:
            end_index = i
            break
    return end_index


def pair_localizations(prediction, ground_truth, frame_num=None, fov_xy_nm=None, border=450, tolerance_xy=250, tolerance_z=500,
                       print_info=False):
    """
    Pair the predicted localizations with the ground truth and calculate the assessment
    metrics in the challenge.

    Args:
        prediction (np.ndarray):
            the predicted molecule list with first 5 columns are [frame, x, y, z, photon]
        ground_truth (np.ndarray):
            the ground truth molecule list with first 5 columns are [frame, x, y, z, photon]
        frame_num (int or None):
            the number of frames whose localizations will be paired, should be looped through
            because the ground truth or predictions on the last frame may be empty. If None, use the last
            frame number of the ground truth, may have a little bias.
        fov_xy_nm (tuple or None):
            (x_start, x_end, y_start, y_end) in nm unit, localizations in where will be paired. If None,
            all localizations are paired.
        border (int or float):
            If fov_xy_range is not None, localizations close to the margin of the FOV will be
            excluded considering the incomplete PSFs. If fov_xy_range is None, this argument is meaningless.
        tolerance_xy (int):
            localizations are paired when they are within a circle of this radius tolerance, unit nm.
        tolerance_z (int):
            localizations are paired when they are within this axial distance tolerance, unit nm.
        print_info (bool):
            whether to print the assessment metrics.

    Returns:
        (dict, np.ndarray): the assessment metrics dict and the paired molecule array
            [frame, gt_x, gt_y, gt_z, gt_photon, pred_x, pred_y, pred_z, pred_photon].
    """

    pred_list = sorted(prediction, key=lambda x: x[0])
    gt_list = sorted(ground_truth, key=lambda x: x[0])

    if frame_num is None:
        frame_num = int(gt_list[-1][0])

    pred_list = pred_list[: find_frame(pred_list, frame_num)]
    gt_list = gt_list[: find_frame(gt_list, frame_num)]

    # get the molecules in the specified FOV
    if fov_xy_nm is not None:
        gt_array = np.array(gt_list)
        pred_array = np.array(pred_list)

        gt_idx = np.where(
            (gt_array[:, 1] < fov_xy_nm[0] + border) | (gt_array[:, 1] > fov_xy_nm[1] - border) |
            (gt_array[:, 2] < fov_xy_nm[2] + border) | (gt_array[:, 2] > fov_xy_nm[3] - border))[0]
        pred_idx = np.where(
            (pred_array[:, 1] < fov_xy_nm[0] + border) | (pred_array[:, 1] > fov_xy_nm[1] - border) |
            (pred_array[:, 2] < fov_xy_nm[2] + border) | (pred_array[:, 2] > fov_xy_nm[3] - border))[0]

        for idx_tmp in reversed(np.sort(gt_idx)):
            del gt_list[idx_tmp]
        for idx_tmp in reversed(np.sort(pred_idx)):
            del pred_list[idx_tmp]

    print(f"FOV={fov_xy_nm} nm, border={border}, tolerance_xy={tolerance_xy}, tolerance_z={tolerance_z}\n"
          f"pairing localizations on {frame_num} images, ground truth: {len(gt_list)}, predictions: {len(pred_list)}, "
          f"please waiting...")

    if (len(pred_list) == 0) or (len(gt_list) == 0):
        metric_dict = {'recall': np.nan, 'precision': np.nan, 'jaccard': np.nan, 'rmse_lat': np.nan, 'rmse_ax': np.nan,
                       'rmse_vol': np.nan, 'jor': np.nan, 'eff_lat': np.nan, 'eff_ax': np.nan, 'eff_3d': np.nan,
                       'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'rmse_ph': np.nan}
        return metric_dict, []

    tp = 0
    fp = 0
    fn = 0
    mse_lat = 0
    mse_ax = 0
    mse_vol = 0

    paired_list = []

    for i in range(1, frame_num+1):
        gt_frame = np.array([gt for gt in gt_list if gt[0] == i])
        pred_frame = np.array([pred for pred in pred_list if pred[0] == i])

        if len(gt_frame) == 0:
            fp += len(pred_frame)
            continue
        if len(pred_frame) == 0:
            fn += len(gt_frame)
            continue

        gt_frame = gt_frame[:, 1:]
        pred_frame = pred_frame[:, 1:]

        num_gt_remain = len(gt_frame)
        num_pred_remain = len(pred_frame)

        dist_lat = cdist(gt_frame[:, :2], pred_frame[:, :2])
        dist_ax = cdist(gt_frame[:, 2:3], pred_frame[:, 2:3])
        dist_vol = np.sqrt(dist_lat**2 + dist_ax**2) if tolerance_z != np.inf else dist_lat

        while np.min(dist_lat) < tolerance_xy:
            row, col = np.unravel_index(np.argmin(dist_vol), dist_vol.shape)
            if (dist_ax[row, col] < tolerance_z) and (dist_lat[row, col] < tolerance_xy):
                mse_lat += dist_lat[row, col]**2
                mse_ax += dist_ax[row, col]**2
                mse_vol += dist_vol[row, col]**2
                tp += 1
                num_gt_remain -= 1
                num_pred_remain -= 1
                paired_list.append([i, gt_frame[row, 0], gt_frame[row, 1], gt_frame[row, 2], gt_frame[row, 3],
                                    pred_frame[col, 0], pred_frame[col, 1], pred_frame[col, 2], pred_frame[col, 3]])
                dist_lat[row, :] = np.inf
                dist_lat[:, col] = np.inf
                dist_vol[row, :] = np.inf
                dist_vol[:, col] = np.inf
            else:
                dist_lat[row, col] = np.inf
                dist_vol[row, col] = np.inf
        fp += num_pred_remain
        fn += num_gt_remain

    # after pairing all localizations, calculate the metrics
    precision = tp / (tp + fp) if tp + fp != 0 else np.nan
    recall = tp / (tp + fn) if tp + fn != 0 else np.nan
    jaccard = tp / (tp + fp + fn) if tp + fp + fn != 0 else np.nan
    rmse_lat = np.sqrt(mse_lat / tp) if tp != 0 else np.nan
    rmse_ax = np.sqrt(mse_ax / tp) if tp != 0 else np.nan
    rmse_vol = np.sqrt(mse_vol / tp) if tp != 0 else np.nan
    jor = 100 * jaccard / rmse_lat if rmse_lat != 0 else np.nan

    eff_lat = 100 - np.sqrt((100 - 100 * jaccard)**2 + 1**2 * rmse_lat**2)
    eff_ax = 100 - np.sqrt((100 - 100 * jaccard)**2 + 0.5**2 * rmse_ax**2)
    eff_3d = (eff_lat + eff_ax) / 2

    paired_array = np.array(paired_list)
    if len(paired_array):
        rmse_x = np.sqrt(np.mean((paired_array[:, 1] - paired_array[:, 5])**2))
        rmse_y = np.sqrt(np.mean((paired_array[:, 2] - paired_array[:, 6])**2))
        rmse_z = np.sqrt(np.mean((paired_array[:, 3] - paired_array[:, 7])**2))
        rmse_ph = np.sqrt(np.mean((paired_array[:, 4] - paired_array[:, 8])**2))
    else:
        rmse_x = np.nan
        rmse_y = np.nan
        rmse_z = np.nan
        rmse_ph = np.nan
        print('no paired molecules')

    metric_dict = {'recall': recall, 'precision': precision, 'jaccard': jaccard, 'rmse_lat': rmse_lat,
                   'rmse_ax': rmse_ax, 'rmse_vol': rmse_vol, 'rmse_x': rmse_x, 'rmse_y': rmse_y,
                   'rmse_z': rmse_z, 'rmse_ph': rmse_ph, 'jor': jor, 'eff_lat': eff_lat, 'eff_ax': eff_ax,
                   'eff_3d': eff_3d}

    if print_info:
        print(f"Recall: {recall:0.3f}\n"
              f"Precision: {precision:0.3f}\n"
              f"Jaccard: {100 * jaccard:0.3f}\n"
              f"RMSE_lat: {rmse_lat:0.3f}\n"
              f"RMSE_ax: {rmse_ax:0.3f}\n"
              f"RMSE_vol: {rmse_vol:0.3f}\n"
              f"Eff_lat: {eff_lat:0.3f}\n"
              f"Eff_ax: {eff_ax:0.3f}\n"
              f"Eff_3d: {eff_3d:0.3f}\n"
              f"FN: {fn}, FP: {fp}")

    return metric_dict, paired_array
