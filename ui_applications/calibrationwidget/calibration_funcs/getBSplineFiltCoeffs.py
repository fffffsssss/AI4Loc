import numpy as np
from .logger import *


def getBSplineFiltCoeffs(varargin: list):
    F = None
    C0 = None
    D, CFlag, direction, lam = parseInputs(varargin)
    if direction == "indirect":
        C0 = 1
        if D == 0:
            if CFlag:
                if lam == 0:
                    F = 1
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = 1
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 1:
            if CFlag:
                if lam == 0:
                    F = 1
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 1]) / 2
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 2:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 6, 1]) / 8
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 1]) / 2
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 3:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 4, 1]) / 6
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 23, 23, 1]) / 48
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 4:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 76, 230, 76, 1]) / 384
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 11, 11, 1]) / 24
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 5:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 26, 66, 26, 1]) / 120
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 237, 1682, 1682, 237, 1]) / 3840
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 6:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 772, 10543, 23548, 10543, 722, 1]) / 46080
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 57, 302, 302, 57, 1]) / 720
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
        elif D == 7:
            if CFlag:
                if lam == 0:
                    F = np.array([1, 120, 1191, 2416, 1191, 120, 1]) / 5040
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
            else:
                if lam == 0:
                    F = np.array([1, 2179, 60657, 259723, 259723, 60657, 2179, 1]) / 645120
                else:
                    error_log("Indirect BSpline transform not supported for nonzero lambda.")
    elif direction == "direct":
        if D == 0:
            if CFlag:
                if lam == 0:
                    F = []
                    C0 = 1
                else:
                    error_log("Direct BSpline transform of even degree not supported for nonzero lambda")
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 1:
            if CFlag:
                if lam == 0:
                    F = []
                    C0 = 1
                else:
                    F = 1 + 1 / (2 * lam) - np.sqrt(1 + 4 * lam) / (2 * lam )
                    C0 = -1 / lam
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 2:
            if CFlag:
                if lam == 0:
                    F = 2 * np.sqrt(2) - 3
                    C0 = 8
                else:
                    error_log("Direct BSpline transform of even degree not supported for nonzero lambda.")
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 3:
            if CFlag:
                if lam == 0:
                    F = np.sqrt(3) - 2
                    C0 = 6
                else:
                    p = np.array([1, -4, 6, -4, 1]).astype(float)
                    p[1] = p[1] + 1 / (6.0 * lam)
                    p[2] = p[2] + 2 / (3.0 * lam)
                    p[3] = p[3] + 1 / (6.0 * lam)
                    F = np.roots(p)
                    F = F[abs(F) <= 1]
                    C0 = 1 / lam
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 4:
            if CFlag:
                if lam == 0:
                    F = np.array([-0.36134122590022, -0.0137254292973391])
                    C0 = 384
                else:
                    error_log("Direct BSpline transform of even degree not supported for nonzero lambda.")
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 5:
            if CFlag:
                if lam == 0:
                    F = np.array([-0.430575347099973, -0.0430962882032647])
                    C0 = 120
                else:
                    p = np.array([1,-6, 15, -20, 15, -6, 1]).astype(float)
                    p[1] = p[1] - 1 / (120.0 * lam)
                    p[2] = p[2] - 13 / (60.0 * lam)
                    p[3] = p[3] - 11 / (20.0 * lam)
                    p[4] = p[4] - 13 / (60.0 * lam)
                    p[5] = p[1] - 1 / (120.0 * lam)
                    F = np.roots(p)
                    F = F[abs(F) <= 1]
                    C0 = -1 / lam
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 6:
            if CFlag:
                if lam == 0:
                    F = np.array([-0.488294589303046, -0.0816792710762375, -0.00141415180832582])
                    C0 = 46080
                else:
                    error_log("Direct BSpline transform of even degree not supported for nonzero lambda.")
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
        elif D == 7:
            if CFlag:
                if lam == 0:
                    F = np.array([-0.535280430796439, -0.122554615192327, -0.00914869480960828])
                    C0 = 5040
                else:
                    p = np.array([1, -8, 28, -56, 70, -56, 28, -8, 1]).astype(float)
                    p[1] = p[1] + 1 / (5040 * lam)
                    p[2] = p[2] + 1 / (42 * lam)
                    p[3] = p[3] + 397 / (1680 * lam)
                    p[4] = p[4] + 151 / (315 * lam)
                    p[5] = p[5] + 397 / (1680 * lam)
                    p[6] = p[6] + 1 / (42 * lam)
                    p[7] = p[7] + 1 / (5040 * lam)
                    F = np.roots(p)
                    F = F[abs(F) <= 1]
                    C0 = 1 / lam
            else:
                error_log("Direct BSpline transform not supported for shifted basis functions.")
    return F, C0

def parseInputs(varargin: list):
    if len(varargin) < 1:
        D = 3
    else:
        D = varargin[0]
    if D.size > 1 or D < 0 or D > 7 :
        error_log("D must be an integer in the interval [0,7]")

    if len(varargin) < 2:
        CFlag = True
    else:
        CFlag = varargin[1]
    if not isinstance(CFlag, bool):
        error_log("CFlag must be a logical scalar")

    if len(varargin) < 3:
        direction = 'indirect'
    else:
        direction = varargin[2]
    if not isinstance(direction, str) or not ('direct' in direction.lower() or \
            'indirect' in direction.lower()):
        error_log("Direction must be either ''direct'' or ''indirect''.")

    if len(varargin) < 4:
        lam = 0
    else:
        lam = varargin[3]
    if lam < 0:
        error_log("lambda must be a nonnegative real number.")
    return D, CFlag, direction.lower(), lam