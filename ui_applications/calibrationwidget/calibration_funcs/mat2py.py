# Matlab function to ptyon
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import numpy as np
import tifffile as tiff
import scipy


def read_tiff(file):
    """
    Read tiff file list.
    :param file: TIFF file
    :return: TIFF pixels
    """
    return tiff.imread(file)


def read_tiff_info(file):
    """
    Read tiff header information
    :param file: TIFF file
    :return: The information
    """
    with tiff.TiffFile(file) as tif:
        tags = {}
        for tag in tif.pages[0].tags.values():
            tags[tag.name] = tag.value
        return tags


def load_mat(file):
    """
    Load Matlab file.
    :param file: The MATLAB file
    :return: The MATLAB file data
    """
    return scipy.io.loadmat(file)


def fspecial_gauss(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function.
    :param size: Size of matrix
    :param sigma: Gauss sigma
    :return: Gauss filter
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def maximum_find(mat):
    """
    Find maximum value.
    :param mat: 2D matrix
    :return: 3D matrix [y, x, intensity]
    """
    x = []
    y = []
    value = []

    (r, c) = mat.shape
    i = 1
    while i < c - 1:
        j = 1
        while j < r - 1:
            if mat[j, i] > mat[j - 1, i - 1] and \
                    mat[j, i] > mat[j - 1, i] and \
                    mat[j, i] > mat[j - 1, i + 1] and \
                    mat[j, i] > mat[j, i - 1] and \
                    mat[j, i] > mat[j, i + 1] and \
                    mat[j, i] > mat[j + 1, i - 1] and \
                    mat[j, i] > mat[j + 1, i] and \
                    mat[j, i] > mat[j + 1, i + 1]:
                x.append(j)
                y.append(i)
                value.append(mat[j, i])
                j += 1
            j += 1
        i += 1
    return np.column_stack((y, x, value))


def quantile(mat, p):
    """
    Function to mimic the 'quantile' gaussian MATLAB function.
    :param mat: Matrix
    :param p: probability
    :return: quantiles of the elements in mat for the cumulative probability p in the interval [0,1].
    """
    return scipy.stats.mstats.mquantiles(mat, p, alphap=0.5, betap=0.5)


def filter2(h, x):
    """
    Function to mimic the 'filter2' MATLAB function
    :param h: Coefficients matrix
    :param x: Matrix x to apply
    :return: Filter result.
    """
    return scipy.signal.correlate2d(x, h, mode='same')