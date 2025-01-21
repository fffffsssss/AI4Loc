import os
import time
import scipy.io as scio
import numpy as np
from scipy import ndimage
import numba

def psf2cspline_np(psf):
	# calculate A
	A = np.zeros((64, 64))
	for i in range(1, 5):
		dx = (i - 1) / 3
		for j in range(1, 5):
			dy = (j - 1) / 3
			for k in range(1, 5):
				dz = (k - 1) / 3
				for l in range(1, 5):
					for m in range(1, 5):
						for n in range(1, 5):
							A[(i - 1) * 16 + (j - 1) * 4 + k - 1, (l - 1) * 16 + (m - 1) * 4 + n - 1] = dx ** (
										l - 1) * dy ** (m - 1) * dz ** (n - 1)

	# upsample psf with factor of 3
	psf_up = ndimage.zoom(psf, 3.0, mode='grid-constant', grid_mode=True)[1:-1, 1:-1, 1:-1].astype(np.float32)
	A = np.float32(A)
	coeff = calsplinecoeff(A, psf, psf_up)
	return coeff

# @numba.jit(nopython=True) 
def calsplinecoeff(A, psf, psf_up):
	coeff = np.zeros((64, psf.shape[0] - 1, psf.shape[1] - 1, psf.shape[2] - 1))
	for i in range(coeff.shape[1]):
		for j in range(coeff.shape[2]):
			for k in range(coeff.shape[3]):
				temp = psf_up[i * 3: 3 * (i + 1) + 1, j * 3: 3 * (j + 1) + 1, k * 3: 3 * (k + 1) + 1]
				temp1 = np.reshape(temp.transpose([2,1,0]),[64,])
				x = np.linalg.solve(A, temp1)
				coeff[:, i, j, k] = x
	return coeff.transpose([1,2,3,0])

if __name__ == '__main__':
	thispath = os.path.dirname(os.path.abspath(__file__))
	dataFile = thispath + '\\..\\..\\spline_interp3.mat'
	data = scio.loadmat(dataFile)
	corrPSFhd = data['corrPSFhd'].astype(np.double)
	print(time.time())
	coeff = psf2cspline_np(corrPSFhd)
	print(time.time())
