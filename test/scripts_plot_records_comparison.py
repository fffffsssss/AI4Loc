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


def plot_records_comparison():
    # # astigmatism 1 frame
    # loc_model_path_1 = '../results/2024-11-13-11-04LUNAR_LL.pt'
    # loc_model_path_2 = '../results/2024-11-13-10-56DeepLoc.pt'
    # # astigmatism 3 frame
    # loc_model_path_1 = '../results/2024-11-13-14-27LUNAR_LL.pt'
    # loc_model_path_2 = '../results/2024-11-13-14-05DeepLoc.pt'
    # # astigmatism 7 frame
    # loc_model_path_1 = '../results/2024-11-13-18-08LUNAR_LL.pt'
    # loc_model_path_2 = '../results/2024-11-13-17-42DeepLoc.pt'

    # # tetra6 psf 1 frame
    # loc_model_path_1 = '../results/2024-11-26-13-09LUNAR_LL.pt'
    # loc_model_path_2 = '../results/2024-08-21-07-50DeepLoc.pt'
    # # tetra6 psf 3 frame
    # loc_model_path_1 = '../results/2024-11-26-14-45LUNAR_LL.pt'
    # loc_model_path_2 = '../results/2024-08-23-14-33DeepLoc.pt'
    # tetra6 psf 7 frame
    loc_model_path_1 = '../results/2024-11-26-18-51LUNAR_LL.pt'
    loc_model_path_2 = '../results/2024-08-26-13-34DeepLoc.pt'

    # load the completely trained model
    with open(loc_model_path_1, 'rb') as f:
        loc_model_1 = torch.load(f)
    with open(loc_model_path_2, 'rb') as f:
        loc_model_2 = torch.load(f)

    # plot evaluation performance during the training
    recorder_1 = loc_model_1.evaluation_recorder
    recorder_2 = loc_model_2.evaluation_recorder

    # # plot the RMSE of the localization
    # plt.figure(figsize=(8, 6), constrained_layout=True)
    # plt.plot(*zip(*sorted(recorder_1['rmse_vol'].items())),
    #          label='LUNAR', color='blue', linestyle='-', linewidth=2)
    # plt.plot(*zip(*sorted(recorder_2['rmse_vol'].items())),
    #          label='FD-DeepLoc', color='orange', linestyle='--', linewidth=2)
    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('RMSE$_{3D}$', fontsize=14)
    # # plt.title('Comparison of RMSE during Training', fontsize=16)
    # plt.grid(True)
    # plt.legend(fontsize=12)
    # plt.show(block=True)
    #
    # # plot the jaccard index during the training
    # plt.figure(figsize=(8, 6), constrained_layout=True)
    # plt.plot(*zip(*sorted(recorder_1['jaccard'].items())),
    #          label='LUNAR', color='blue', linestyle='-', linewidth=2)
    # plt.plot(*zip(*sorted(recorder_2['jaccard'].items())),
    #             label='FD-DeepLoc', color='orange', linestyle='--', linewidth=2)
    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('Jaccard Index', fontsize=14)
    # # plt.title('Comparison of Jaccard Index during Training', fontsize=16)
    # plt.grid(True)
    # plt.legend(fontsize=12)
    # plt.show(block=True)
    #
    # # plot the 3D efficiency during the training
    # plt.figure(figsize=(8, 6), constrained_layout=True)
    # plt.plot(*zip(*sorted(recorder_1['eff_3d'].items())),
    #          label='LUNAR', color='blue', linestyle='-', linewidth=2)
    # plt.plot(*zip(*sorted(recorder_2['eff_3d'].items())),
    #             label='FD-DeepLoc', color='orange', linestyle='--', linewidth=2)
    # plt.xlabel('Iterations', fontsize=14)
    # plt.ylabel('3D Efficiency', fontsize=14)
    # # plt.title('Comparison of 3D Efficiency during Training', fontsize=16)
    # plt.grid(True)
    # plt.legend(fontsize=12)
    # plt.show(block=True)

    # Create a single figure with 3 subplots
    fontsize = 20
    tick_fontsize = 16

    fig, axs = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True, sharex=True, dpi=300)

    # Plot the 3D efficiency during the training
    axs[0].plot(*zip(*sorted(recorder_1['eff_3d'].items())), label='LUNAR', color='blue', linestyle='-', linewidth=2)
    axs[0].plot(*zip(*sorted(recorder_2['eff_3d'].items())), label='FD-DeepLoc', color='orange', linestyle='--',
                linewidth=2)
    axs[0].set_xlabel('Iterations', fontsize=fontsize)
    axs[0].set_ylabel('3D Efficiency', fontsize=fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].grid(True)
    axs[0].legend(fontsize=tick_fontsize, loc='lower right')
    # axs[0].set_title('Comparison of 3D Efficiency during Training', fontsize=16)

    # Plot the Jaccard index during the training
    axs[1].plot(*zip(*sorted(recorder_1['jaccard'].items())), label='LUNAR', color='blue', linestyle='-', linewidth=2)
    axs[1].plot(*zip(*sorted(recorder_2['jaccard'].items())), label='FD-DeepLoc', color='orange', linestyle='--',
                linewidth=2)
    axs[1].set_xlabel('Iterations', fontsize=fontsize)
    axs[1].set_ylabel('Jaccard index', fontsize=fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].grid(True)
    # axs[1].legend(fontsize=tick_fontsize)
    # axs[1].set_title('Comparison of Jaccard Index during Training', fontsize=16)

    # Plot the RMSE of the localization
    axs[2].plot(*zip(*sorted(recorder_1['rmse_vol'].items())), label='LUNAR', color='blue', linestyle='-', linewidth=2)
    axs[2].plot(*zip(*sorted(recorder_2['rmse_vol'].items())), label='FD-DeepLoc', color='orange', linestyle='--',
                linewidth=2)
    axs[2].set_xlabel('Iterations', fontsize=fontsize)
    axs[2].set_ylabel('RMSE$_{\mathrm{3D}}$', fontsize=fontsize)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[2].grid(True)
    # axs[2].legend(fontsize=tick_fontsize)
    # axs[2].set_title('Comparison of RMSE during Training', fontsize=16)

    # Show the combined figure
    plt.show(block=True)


if __name__ == '__main__':
    plot_records_comparison()
