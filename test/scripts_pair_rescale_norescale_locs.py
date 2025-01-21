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
import matplotlib.pyplot as plt

import ailoc.common
import ailoc.simulation
ailoc.common.setup_seed(25)


def compare_rescale_metrics():
    # load ground truth
    gt_array = ailoc.common.read_csv_array('../datasets/simu_tubulin/simu_tubulin_tetra6_hd/activations.csv')
    # load predictions
    preds_array = ailoc.common.read_csv_array('../results/2024-08-30-15-49DeepLoc_simu_tubulin_tetra6_hd_predictions.csv')

    # compare ground truth and predictions
    metric_dict, paired_array = ailoc.common.pair_localizations(
        prediction=preds_array,
        ground_truth=gt_array,
        frame_num=None,
        fov_xy_nm=(0, 256*108, 0, 256*108),
        print_info=True
    )

    # rescale predictions
    preds_array_re = ailoc.common.rescale_offset(
        preds_array,
        pixel_size=[108, 108],
        rescale_bins=20,
        threshold=0.01,
    )

    # compare rescaled predictions and ground truth
    metric_dict, paired_array = ailoc.common.pair_localizations(
        prediction=preds_array_re,
        ground_truth=gt_array,
        frame_num=None,
        fov_xy_nm=(0, 256*108, 0, 256*108),
        print_info=True
    )

if __name__ == '__main__':
    compare_rescale_metrics()
