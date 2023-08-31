# evalBSpline function
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import numpy as np
import numba

@numba.jit(nopython=True) 
def evalBSpline(xi, deg):
    if deg == 0:
        bi = np.zeros(xi.shape)
        k = np.where((xi >= -0.5) & (xi < 0.5))
        if len(k[0]) > 0:
            bi[k] = 1
    elif deg == 1:
        bi = np.zeros(xi.shape)
        k = np.where((xi >= 0) & (xi < 1))
        if len(k[0]) > 0:
            bi[k] = 1 - xi[k]
        k = np.where((xi >= -1) & (xi < 0))
        if len(k[0]) > 0:
            bi[k] = 1 + xi[k]
    elif deg == 2:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        k = np.where((xi >= 0.5) & (xi < 1.5))
        if len(k[0]) > 0:
            bi[k] = 9 / 8 - xi[k] * 1.5 + x2[k] * 0.5
        k = np.where((xi >= -0.5) & (xi < 0.5))
        if len(k[0]) > 0:
            bi[k] = 4 / 3 - x2[k]
        k = np.where((xi >= -1.5) & (xi < -0.5))
        if len(k[0]) > 0:
            bi[k] = 9 / 8 + xi[k] * 1.5 + x2[k] * 0.5
    elif deg == 3:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        x3 = x2 * xi
        k = np.where((xi >= 1) & (xi < 2))
        if len(k[0]) > 0:
            bi[k] = 4 / 3 - xi[k] * 2 + x2[k] - x3[k] * 1 / 6
        k = np.where((xi >= 0) & (xi < 1))
        if len(k[0]) > 0:
            bi[k] = 2 / 3 - x2[k] + x3[k] * 1 / 2
        k = np.where((xi >= -1) & (xi < 0))
        if len(k[0]) > 0:
            bi[k] = 2 / 3 - x2[k] - x3[k] * 1 / 2
        k = np.where((xi >= -2) & (xi < -1))
        if len(k[0]) > 0:
            bi[k] = 4 / 3 + xi[k] * 2 + x2[k] + x3[k] * 1 / 6
    elif deg == 4:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        x3 = x2 * xi
        x4 = x3 * xi
        k = np.where((xi >= 3/2) & (xi < 5/2))
        if len(k[0]) > 0:
            bi[k] = 625/384 - (125/48) * xi[k] + (25/16) * x2[k] - (5/12) * x3[k] + (1/24) * x4[k]
        k = np.where((xi >= 1/2) & (xi < 3/2))
        if len(k[0]) > 0:
            bi[k] = 55/96 + (5/24) * xi[k] - (5/4) * x2[k] + (5/6) * x3[k] - (1/6) * x4[k]
        k = np.where((xi >= -1/2) & (xi < 1/2))
        if len(k[0]) > 0:
            bi[k] = 115/192 - (5/8) * x2[k] + (1/4) * x4[k]
        k = np.where((xi >= -3/2) & (xi < -1/2))
        if len(k[0]) > 0:
            bi[k] = 55/96 - (5/24) * xi[k] - (5/4) * x2[k] - (5/6) * x3[k] - (1/6) * x4[k]
        k = np.where((xi >= -5/2) & (xi < -3/2))
        if len(k[0]) > 0:
            bi[k] = 625/384 + (125/48) * xi[k] + (25/16) * x2[k] + (5/12) * x3[k] + (1/24) * x4[k]
    elif deg == 5:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        x3 = x2 * xi
        x4 = x3 * xi
        x5 = x4 * xi
        k = np.where((xi >= 2) & (xi < 3))
        if len(k[0]) > 0:
            bi[k] = 81 / 40 - (27 / 8) * xi[k] + (9 / 4) * x2[k] - (3 / 4) * x3[k] + (1 / 8) * x4[k] - (1 / 120) * x5[k]
        k = np.where((xi >= 1) & (xi < 2))
        if len(k[0]) > 0:
            bi[k] = 17 / 40 + (5 / 8) * xi[k] - (7 / 4) * x2[k] + (5 / 4) * x3[k] - (3 / 8) * x4[k] + (1 / 24) * x5[k]
        k = np.where((xi >= 0) & (xi < 1))
        if len(k[0]) > 0:
            bi[k] = 11 / 20 - (1 / 2) * x2[k] + (1 / 4) * x4[k] - (1 / 12) * x5[k]
        k = np.where((xi >= -1) & (xi < 0))
        if len(k[0]) > 0:
            bi[k] = 11 / 20 - (1 / 2) * x2[k] + (1 / 4) * x4[k] + (1 / 12) * x5[k]
        k = np.where((xi >= -2) & (xi < -1))
        if len(k[0]) > 0:
            bi[k] = 17 / 40 - (5 / 8) * xi[k] - (7 / 4) * x2[k] - (5 / 4) * x3[k] - (3 / 8) * x4[k] - (1 / 24) * x5[k]
        k = np.where((xi >= -3) & (xi < -2))
        if len(k[0]) > 0:
            bi[k] = 81 / 40 + (27 / 8) * xi[k] + (9 / 4) * x2[k] + (3 / 4) * x3[k] + (1 / 8) * x4[k] + (1 / 120) * x5[k]
    elif deg == 6:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        x3 = x2 * xi
        x4 = x3 * xi
        x5 = x4 * xi
        x6 = x5 * xi
        k = np.where((xi >= 5/2) & (xi < 7/2))
        if len(k[0]) > 0:
            bi[k] = 117649 / 46080 - (16807 / 3840) * xi[k] + (2401 / 768) * x2[k] - (343 / 288) * x3[k] + (49 / 192) * x4[k] - (7 / 240) * x5[k] + (1 / 720) * x6[k]
        k = np.where((xi >= 3/2) & (xi < 5/2))
        if len(k[0]) > 0:
            bi[k] = 1379 / 7680 + (1267 / 960) * xi[k] - (329 / 128) * x2[k] + (133 / 72) * x3[k] - (21 / 32) * x4[k] + (7 / 60) * x5[k] - (1 / 120) * x6[k]
        k = np.where((xi >= 1/2) & (xi < 3/2))
        if len(k[0]) > 0:
            bi[k] = 7861 / 15360 - (7 / 768) * xi[k] - (91 / 256) * x2[k] - (35 / 288) * x3[k] + (21 / 64) * x4[k] - (7 / 48) * x5[k] + (1 / 48) * x6[k]
        k = np.where((xi >= -1/2) & (xi < 1/2))
        if len(k[0]) > 0:
            bi[k] = 5887 / 11520 - (77 / 192) * x2[k] + (7 / 48) * x4[k] - (1 / 36) * x6[k]
        k = np.where((xi >= -3/2) & (xi < -1/2))
        if len(k[0]) > 0:
            bi[k] = 7861 / 15360 + (7 / 768) * xi[k] - (91 / 256) * x2[k] + (35 / 288) * x3[k] + (21 / 64) * x4[k] + (7 / 48) * x5[k] + (1 / 48) * x6[k]
        k = np.where((xi >= -5/2) & (xi < -3/2))
        if len(k[0]) > 0:
            bi[k] = 1379 / 7680 - (1267 / 960) * xi[k] - (329 / 128) * x2[k] - (133 / 72) * x3[k] - (21 / 32) * x4[k] - (7 / 60) * x5[k] - (1 / 120) * x6[k]
        k = np.where((xi >= -7/2) & (xi < -5/2))
        if len(k[0]) > 0:
            bi[k] = 117649 / 46080 + (16807 / 3840) * xi[k] + (2401 / 768) * x2[k] + (343 / 288) * x3[k] + (49 / 192) * x4[k] + (7 / 240) * x5[k] + (1 / 720) * x6[k]
    elif deg == 7:
        bi = np.zeros(xi.shape)
        x2 = xi * xi
        x3 = x2 * xi
        x4 = x3 * xi
        x5 = x4 * xi
        x6 = x5 * xi
        x7 = x6 * xi
        k = np.where((xi >= 3) & (xi < 4))
        if len(k[0]) > 0:
            bi[k] = 6405119470038039 / 1970324836974592 - (672537544353994073 / 118219490218475520) * xi[k] + (64 / 15) * x2[k] - (16 / 9) * x3[k] + (4 / 9) * x4[k] - (1 / 15) * x5[k] + (1 / 180) * x6[k] - (1 / 5040) * x7[k]
        k = np.where((xi >= 2) & (xi < 3))
        if len(k[0]) > 0:
            bi[k] = -2173612320154509 / 9851624184872960 + (855120979246972939 / 354658470655426560) * xi[k] - (23 / 6) * x2[k] + (49 / 18) * x3[k] - (19 / 18) * x4[k] + (7 / 30) * x5[k] - (1 / 36) * x6[k] + (1 / 720) * x7[k]
        k = np.where((xi >= 1) & (xi < 2))
        if len(k[0]) > 0:
            bi[k] = 103 / 210 - (7 / 90) * xi[k] - (1 / 10) * x2[k] - (7 / 18) * x3[k] + (1 / 2) * x4[k] - (7 / 30) * x5[k] + (1 / 20) * x6[k] - (1 / 240) * x7[k]
        k = np.where((xi >= 0) & (xi < 1))
        if len(k[0]) > 0:
            bi[k] = 151 / 315 - (1 / 3) * x2[k] + (1 / 9) * x4[k] - (1 / 36) * x6[k] + (1 / 144) * x7[k]
        k = np.where((xi >= -1) & (xi < 0))
        if len(k[0]) > 0:
            bi[k] = 151 / 315 - (1 / 3) * x2[k] + (1 / 9) * x4[k] - (1 / 36) * x6[k] - (1 / 144) * x7[k]
        k = np.where((xi >= -2) & (xi < -1))
        if len(k[0]) > 0:
            bi[k] = 103 / 210 + (7 / 90) * xi[k] - (1 / 10) * x2[k] + (7 / 18) * x3[k] + (1 / 2) * x4[k] + (7 / 30) * x5[k] + (1 / 20) * x6[k] + (1 / 240) * x7[k]
        k = np.where((xi >= -3) & (xi < -2))
        if len(k[0]) > 0:
            bi[k] = -2173612320154509 / 9851624184872960 - (855120979246972939 / 354658470655426560) * xi[k] - (23 / 6) * x2[k] - (49 / 18) * x3[k] - (19 / 18) * x4[k] - (7 / 30) * x5[k] - (1 / 36) * x6[k] - (1 / 720) * x7[k]
        k = np.where((xi >= -4) & (xi < -3))
        if len(k[0]) > 0:
            bi[k] = 6405119470038039 / 1970324836974592 + (672537544353994073 / 118219490218475520) * xi[k] + (64 / 15) * x2[k] + (16 / 9) * x3[k] + (4 / 9) * x4[k] + (1 / 15) * x5[k] + (1 / 180) * x6[k] + (1 / 5040) * x7[k]
    else:
        bi = ((xi + (deg + 1) / 2) * evalBSpline(xi + (1 / 2), deg - 1) + ((deg + 1) / 2 - xi) * evalBSpline(xi - (1 / 2), deg - 1)) / deg
    return bi