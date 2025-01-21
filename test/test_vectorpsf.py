import napari
import numpy as np
import torch
from torch import linalg as LA
import time
import matplotlib.pyplot as plt

import ailoc.common
from ailoc.simulation.vectorpsf import VectorPSFCUDA, VectorPSFTorch


def test_specified_psf():
    na = 1.5645674897
    wavelength = 670  # unit: nm
    refmed = 1.406
    refcov = 1.524
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
                          50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
                          80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = -5000
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = (100, 100)
    otf_rescale_xy = (0.5, 0.5)
    npupil = 128
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)
    psf_torch = VectorPSFTorch(psf_params_dict, data_type=torch.float64, req_grad=True)

    # set emitter positions
    n_mol = 1000
    x = ailoc.common.gpu(torch.linspace(-5 * pixel_size_xy[0], 5 * pixel_size_xy[0], n_mol))  # unit nm
    y = ailoc.common.gpu(torch.linspace(5 * pixel_size_xy[1], -5 * pixel_size_xy[1], n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-1000, 1000, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    # robust_training means adding gaussian noise to the zernike coef of each psf, unit nm
    zernike_coefs = torch.tile(ailoc.common.gpu(psf_cuda.zernike_coef), dims=(n_mol, 1))  # unit nm
    zernike_coefs += torch.normal(mean=0, std=3.33, size=(n_mol, psf_cuda.zernike_mode.shape[0]),
                                  device='cuda')  # different zernike coef for each psf

    # run the VectorPSF generation
    psfs_data_cuda = psf_cuda.simulate(x, y, z, photons, zernike_coefs=None)
    psfs_data_torch = psf_torch.simulate(x, y, z, photons)

    print(f"the relative root of squared error is: "
          f"{LA.norm((psfs_data_cuda - psfs_data_torch).flatten(), ord=2) / LA.norm(psfs_data_cuda.flatten(), ord=2) * 100}%")

    # view data
    # ailoc.common.viewdata_napari(psfs_data_cuda, psfs_data_torch)
    ailoc.common.cmpdata_napari(psfs_data_cuda, psfs_data_torch)
    # ailoc.common.plot_image_stack_difference(psfs_data_cuda,psfs_data_torch)


def test_random_psf():
    for i_loop in range(30):
        na = 1.5 + np.random.rand() * 0.1
        wavelength = 670 + (np.random.rand() - 0.5) * 100  # unit: nm
        refmed = 1.518 + np.random.rand() * 0.1
        refcov = 1.518 + np.random.rand() * 0.1
        refimm = 1.518 + np.random.rand() * 0.1
        zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                 4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                 5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                                dtype=np.float32).reshape([21, 3])
        zernike_aber[:, 2] = np.random.rand(zernike_aber.shape[0]) * 100
        objstage0 = np.random.randn(1) * 1000
        zemit0 = -1 * refmed / refimm * objstage0
        pixel_size_xy = [100 + (np.random.rand() - 0.5) * 100, 100 + (np.random.rand() - 0.5) * 100]
        otf_rescale_xy = [0 + np.random.rand(), 0 + np.random.rand()]
        npupil = 64
        psf_size = 51

        psf_params_dict = {'na': na, 'wavelength': wavelength,
                           'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                           'zernike_mode': zernike_aber[:, 0:2],
                           'zernike_coef': zernike_aber[:, 2],
                           'zernike_coef_map': None,
                           'objstage0': objstage0, 'zemit0': zemit0,
                           'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                           'npupil': npupil, 'psf_size': psf_size}
        psf_cuda = VectorPSFCUDA(psf_params_dict)
        psf_torch = VectorPSFTorch(psf_params_dict)

        # set emitter positions
        x = ailoc.common.gpu((np.random.rand(1000) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        y = ailoc.common.gpu((np.random.rand(1000) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        z = ailoc.common.gpu(torch.linspace(-1000, 1000, 1000))  # unit nm
        photons = ailoc.common.gpu(np.ones(1000) * 10000)  # unit photons

        # run the VectorPSF generation
        psfs_data_cuda = psf_cuda.simulate(x, y, z, photons)
        psfs_data_torch = psf_torch.simulate_parallel(x, y, z, photons)

        print(f"the relative root of squared error is: "
              f"{LA.norm((psfs_data_cuda - psfs_data_torch).flatten(), ord=2) / LA.norm(psfs_data_cuda.flatten(), ord=2) * 100}%")

        # view data
        ailoc.common.cmpdata_napari(psfs_data_cuda, psfs_data_torch)


def test_simulate_speed():
    na = 1.5 + np.random.rand() * 0.1
    wavelength = 670 + (np.random.rand() - 0.5) * 100  # unit: nm
    refmed = 1.518 + np.random.rand() * 0.1
    refcov = 1.518 + np.random.rand() * 0.1
    refimm = 1.518 + np.random.rand() * 0.1
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = np.random.rand(zernike_aber.shape[0]) * 100
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100 + (np.random.rand() - 0.5) * 100, 100 + (np.random.rand() - 0.5) * 100]
    otf_rescale_xy = [0 + np.random.rand(), 0 + np.random.rand()]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)
    psf_torch_nograd = VectorPSFTorch(psf_params_dict, req_grad=False)
    psf_torch_grad = VectorPSFTorch(psf_params_dict, req_grad=True)

    cuda_time = []
    torch_time_nograd = []
    torch_time_nograd_parallel = []
    torch_time_grad = []
    torch_time_grad_parallel = []
    nmol_test = list(np.linspace(10, 1010, 11, dtype=int))
    for nmol in nmol_test:
        # set emitter positions
        x = ailoc.common.gpu((np.random.rand(nmol) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        y = ailoc.common.gpu((np.random.rand(nmol) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        z = ailoc.common.gpu(torch.linspace(-1000, 1000, nmol))  # unit nm
        photons = ailoc.common.gpu(np.ones(nmol) * 10000)  # unit photons

        # run the VectorPSF generation
        t_start = time.time()
        psfs_data_cuda = psf_cuda.simulate(x, y, z, photons)
        cuda_time.append(time.time() - t_start)

        t_start = time.time()
        psfs_data_torch_nograd = psf_torch_nograd.simulate(x, y, z, photons)
        torch_time_nograd.append(time.time() - t_start)

        t_start = time.time()
        psfs_data_torch_nograd_parallel = psf_torch_nograd.simulate_parallel(x, y, z, photons)
        torch_time_nograd_parallel.append(time.time() - t_start)

        t_start = time.time()
        psfs_data_torch_grad = psf_torch_grad.simulate(x, y, z, photons)
        torch_time_grad.append(time.time() - t_start)

        t_start = time.time()
        psfs_data_torch_grad_parallel = psf_torch_grad.simulate_parallel(x, y, z, photons)
        torch_time_grad_parallel.append(time.time() - t_start)

        print(f"the relative root of squared error is: "
              f"{LA.norm((psfs_data_cuda - psfs_data_torch_nograd_parallel).flatten(), ord=2) / LA.norm(psfs_data_cuda.flatten(), ord=2) * 100}%")

    plt.figure(constrained_layout=True)
    plt.scatter(nmol_test, cuda_time, label='CUDA')
    plt.scatter(nmol_test, torch_time_nograd, label='Torch nograd')
    plt.scatter(nmol_test, torch_time_nograd_parallel, label='Torch nograd parallel')
    plt.scatter(nmol_test, torch_time_grad, label='Torch grad')
    plt.scatter(nmol_test, torch_time_grad_parallel, label='Torch grad parallel')
    plt.legend()
    plt.xlabel('number of molecules')
    plt.ylabel('seconds')
    plt.show()


def test_precompute_speed():
    na = 1.5 + np.random.rand() * 0.1
    wavelength = 670 + (np.random.rand() - 0.5) * 100  # unit: nm
    refmed = 1.518 + np.random.rand() * 0.1
    refcov = 1.518 + np.random.rand() * 0.1
    refimm = 1.518 + np.random.rand() * 0.1
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = np.random.rand(zernike_aber.shape[0]) * 100
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100 + (np.random.rand() - 0.5) * 100, 100 + (np.random.rand() - 0.5) * 100]
    otf_rescale_xy = [0 + np.random.rand(), 0 + np.random.rand()]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_cuda = VectorPSFCUDA(psf_params_dict)
    psf_torch_nograd = VectorPSFTorch(psf_params_dict, req_grad=False)
    psf_torch_grad = VectorPSFTorch(psf_params_dict, req_grad=True)

    torch_time_nograd = []
    torch_time_grad = []
    torch_time_nograd_parallel = []
    torch_time_grad_parallel = []
    for i in range(100):
        t_start = time.time()
        psf_torch_nograd._pre_compute_v1()
        torch_time_nograd.append(time.time() - t_start)

        t_start = time.time()
        psf_torch_grad._pre_compute_v1()
        torch_time_grad.append(time.time() - t_start)

        t_start = time.time()
        psf_torch_nograd._pre_compute()
        torch_time_nograd_parallel.append(time.time() - t_start)

        t_start = time.time()
        psf_torch_grad._pre_compute()
        torch_time_grad_parallel.append(time.time() - t_start)

    print(f"psf_torch_nograd:{np.mean(torch_time_nograd)} \n"
          f"psf_torch_grad:{np.mean(torch_time_grad)} \n"
          f"psf_torch_nograd_parallel:{np.mean(torch_time_nograd_parallel)} \n"
          f"psf_torch_grad_parallel:{np.mean(torch_time_grad_parallel)} ")


def test_derivative_speed():
    na = 1.5 + np.random.rand() * 0.1
    wavelength = 670 + (np.random.rand() - 0.5) * 100  # unit: nm
    refmed = 1.518 + np.random.rand() * 0.1
    refcov = 1.518 + np.random.rand() * 0.1
    refimm = 1.518 + np.random.rand() * 0.1
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = np.random.rand(zernike_aber.shape[0]) * 100
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100 + (np.random.rand() - 0.5) * 100, 100 + (np.random.rand() - 0.5) * 100]
    otf_rescale_xy = [0 + np.random.rand(), 0 + np.random.rand()]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_torch_nograd = VectorPSFTorch(psf_params_dict, req_grad=False)
    psf_torch_grad = VectorPSFTorch(psf_params_dict, req_grad=True)

    torch_time_nograd = []
    torch_time_nograd_parallel = []
    torch_time_grad = []
    torch_time_grad_parallel = []
    nmol_test = list(np.linspace(10, 1010, 11, dtype=int))
    for nmol in nmol_test:
        # set emitter positions
        x = ailoc.common.gpu((np.random.rand(nmol) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        y = ailoc.common.gpu((np.random.rand(nmol) - 0.5) * 10 * pixel_size_xy[0])  # unit nm
        z = ailoc.common.gpu(torch.linspace(-1000, 1000, nmol))  # unit nm
        photons = ailoc.common.gpu(np.ones(nmol) * 10000)  # unit photons
        bgs = ailoc.common.gpu(np.ones(nmol) * 100)

        t_start = time.time()
        ders_nograd, psfs_nograd = psf_torch_nograd._compute_derivative(x, y, z, photons, bgs)
        torch_time_nograd.append(time.time() - t_start)

        t_start = time.time()
        ders_nograd_parallel, psfs_nograd_parallel = psf_torch_nograd._compute_derivative_parallel(x, y, z, photons, bgs)
        torch_time_nograd_parallel.append(time.time() - t_start)

        t_start = time.time()
        ders_grad, psfs_grad = psf_torch_grad._compute_derivative(x, y, z, photons, bgs)
        torch_time_grad.append(time.time() - t_start)

        t_start = time.time()
        ders_grad_parallel, psfs_grad_parallel = psf_torch_grad._compute_derivative_parallel(x, y, z, photons, bgs)
        torch_time_grad_parallel.append(time.time() - t_start)

        print(f"the relative root of squared error is: "
              f"{LA.norm((ders_grad - ders_grad_parallel).flatten(), ord=2) / LA.norm(ders_grad.flatten(), ord=2) * 100}%")

    plt.figure(constrained_layout=True)
    plt.scatter(nmol_test, torch_time_nograd, label='Torch nograd')
    plt.scatter(nmol_test, torch_time_nograd_parallel, label='Torch nograd parallel')
    plt.scatter(nmol_test, torch_time_grad, label='Torch grad')
    plt.scatter(nmol_test, torch_time_grad_parallel, label='Torch grad parallel')
    plt.legend()
    plt.xlabel('number of molecules')
    plt.ylabel('seconds')
    plt.show()


def test_crlb():
    # manually set psf parameters
    zernike_aber = np.array(
                     [2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    psf_params_dict = {'na': 1.5,
                       'wavelength': 670,  # unit: nm
                       'refmed': 1.518,
                       'refcov': 1.518,
                       'refimm': 1.518,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'pixel_size_xy': (100, 100),
                       'otf_rescale_xy': (0, 0),
                       'npupil': 64,
                       'psf_size': 51,
                       'objstage0': -0,
                       # 'zemit0': 0,
                       }

    psf_torch = VectorPSFTorch(psf_params_dict)

    # set emitter positions
    n_mol = 21
    x = ailoc.common.gpu(torch.linspace(-5 * psf_params_dict['pixel_size_xy'][0], 5 * psf_params_dict['pixel_size_xy'][0], n_mol))  # unit nm
    y = ailoc.common.gpu(torch.linspace(5 * psf_params_dict['pixel_size_xy'][1], -5 * psf_params_dict['pixel_size_xy'][1], n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-700, 700, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    bgs = ailoc.common.gpu(torch.ones(n_mol) * 50)  # unit photons

    xyz_crlb, model_torch = psf_torch.compute_crlb(x, y, z, photons, bgs)

    xyz_crlb_np = ailoc.common.cpu(xyz_crlb)
    print('average 3D CRLB is:',
          np.sum(xyz_crlb_np[:, 0] ** 2 + xyz_crlb_np[:, 1] ** 2 + xyz_crlb_np[:, 2] ** 2) / x.shape[0])
    plt.figure(constrained_layout=True)
    plt.plot(ailoc.common.cpu(z), xyz_crlb_np[:, 0], 'b', ailoc.common.cpu(z), xyz_crlb_np[:, 1], 'g',
             ailoc.common.cpu(z), xyz_crlb_np[:, 2], 'r')
    plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$'))
    plt.xlim([np.min(ailoc.common.cpu(z)), np.max(ailoc.common.cpu(z))])
    plt.show()

    plt.figure(constrained_layout=True)
    for i in range(n_mol):
        plt.subplot(int(np.ceil(np.sqrt(n_mol))), int(np.ceil(np.sqrt(n_mol))), i + 1)
        plt.imshow(ailoc.common.cpu(model_torch[i]))
    plt.show()

    # ailoc.common.viewdata_napari(model_torch)


def test_optimize_crlb():
    na = 1.5
    wavelength = 670  # unit: nm
    refmed = 1.518
    refcov = 1.518
    refimm = 1.518
    zernike_aber = np.array([2, -2, 70, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    # zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
    #                       50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
    #                       80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = 0
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0, 0]
    npupil = 64
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_torch = VectorPSFTorch(psf_params_dict, req_grad=True)

    # set emitter positions
    n_mol = 21
    x = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    y = ailoc.common.gpu(torch.zeros(n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-600, 600, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 2000)  # unit photons
    bgs = ailoc.common.gpu(torch.ones(n_mol) * 20)  # unit photons

    # show crlb at the start
    xyz_crlb, model_torch = psf_torch.compute_crlb(x, y, z, photons, bgs)
    xyz_crlb_np = ailoc.common.cpu(xyz_crlb)
    print('average 3D CRLB is:',
          np.sum(xyz_crlb_np[:, 0] ** 2 + xyz_crlb_np[:, 1] ** 2 + xyz_crlb_np[:, 2] ** 2) / x.shape[0])
    plt.figure(constrained_layout=True)
    plt.plot(ailoc.common.cpu(z), xyz_crlb_np[:, 0], 'b', ailoc.common.cpu(z), xyz_crlb_np[:, 1], 'g',
             ailoc.common.cpu(z), xyz_crlb_np[:, 2], 'r')
    plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$'))
    plt.xlim([np.min(ailoc.common.cpu(z)), np.max(ailoc.common.cpu(z))])
    plt.show()
    plt.figure(constrained_layout=True)
    for i in range(n_mol):
        plt.subplot(int(np.ceil(np.sqrt(n_mol))), int(np.ceil(np.sqrt(n_mol))), i + 1)
        plt.imshow(ailoc.common.cpu(model_torch[i]))
    plt.show()
    print(torch.concat([psf_torch.zernike_mode, psf_torch.zernike_coef[:, None]], dim=1))

    # begin optimization with respect to the zernike coefficients
    psf_torch.optimize_crlb(x, y, z, photons, bgs, tolerance=1e-5)

    # show crlb after optimization
    xyz_crlb, model_torch = psf_torch.compute_crlb(x, y, z, photons, bgs)
    xyz_crlb_np = ailoc.common.cpu(xyz_crlb)
    print('average 3D CRLB is:',
          np.sum(xyz_crlb_np[:, 0] ** 2 + xyz_crlb_np[:, 1] ** 2 + xyz_crlb_np[:, 2] ** 2) / x.shape[0])
    plt.figure(constrained_layout=True)
    plt.plot(ailoc.common.cpu(z), xyz_crlb_np[:, 0], 'b', ailoc.common.cpu(z), xyz_crlb_np[:, 1], 'g',
             ailoc.common.cpu(z), xyz_crlb_np[:, 2], 'r')
    plt.legend(('$CRLB_x^{1/2}$', '$CRLB_y^{1/2}$', '$CRLB_z^{1/2}$'))
    plt.xlim([np.min(ailoc.common.cpu(z)), np.max(ailoc.common.cpu(z))])
    plt.show()
    plt.figure(constrained_layout=True)
    for i in range(n_mol):
        plt.subplot(int(np.ceil(np.sqrt(n_mol))), int(np.ceil(np.sqrt(n_mol))), i + 1)
        plt.imshow(ailoc.common.cpu(model_torch[i]))
    plt.show()
    print(torch.concat([psf_torch.zernike_mode, psf_torch.zernike_coef[:, None]], dim=1))


def test_psftorch_with_zernike_input():
    na = 1.5645674897
    wavelength = 670  # unit: nm
    refmed = 1.406
    refcov = 1.524
    refimm = 1.518
    zernike_aber = np.array([2, -2, 0, 2, 2, 80, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    zernike_aber[:, 2] = [62.05453, 59.11847, 89.83418, 17.44882, 59.83409, 66.87828, 38.73564,
                          50.43868, 87.31123, 16.99177, 65.38263, 10.89043, 29.75010, 46.28417,
                          80.85130, 13.92037, 99.03661, 0.66625, 89.16090, 31.50599, 66.74316]
    objstage0 = -5000
    zemit0 = -1 * refmed / refimm * objstage0
    pixel_size_xy = [100, 100]
    otf_rescale_xy = [0.5, 0.5]
    npupil = 128
    psf_size = 51

    psf_params_dict = {'na': na, 'wavelength': wavelength,
                       'refmed': refmed, 'refcov': refcov, 'refimm': refimm,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'zernike_coef_map': None,
                       'objstage0': objstage0, 'zemit0': zemit0,
                       'pixel_size_xy': pixel_size_xy, 'otf_rescale_xy': otf_rescale_xy,
                       'npupil': npupil, 'psf_size': psf_size}
    psf_torch_1 = VectorPSFTorch(psf_params_dict, data_type=torch.float64, req_grad=True)
    psf_torch_1._pre_compute_v2()
    psf_torch_2 = VectorPSFTorch(psf_params_dict, data_type=torch.float64, req_grad=True)

    # set emitter positions
    n_mol = 1000
    x = ailoc.common.gpu(torch.linspace(-5 * pixel_size_xy[0], 5 * pixel_size_xy[0], n_mol))  # unit nm
    y = ailoc.common.gpu(torch.linspace(5 * pixel_size_xy[1], -5 * pixel_size_xy[1], n_mol))  # unit nm
    z = ailoc.common.gpu(torch.linspace(-1000, 1000, n_mol))  # unit nm
    photons = ailoc.common.gpu(torch.ones(n_mol) * 5000)  # unit photons
    # robust_training means adding gaussian noise to the zernike coef of each psf, unit nm
    zernike_coefs = torch.tile(ailoc.common.gpu(psf_torch_2.zernike_coef), dims=(n_mol, 1))  # unit nm
    zernike_coefs += torch.normal(mean=0, std=3.33, size=(n_mol, psf_torch_2.zernike_mode.shape[0]),
                                  device='cuda')  # different zernike coef for each psf

    # run the VectorPSF generation
    psfs_data_torch_1 = psf_torch_1.simulate_v2(x, y, z, photons)
    psfs_data_torch_2 = psf_torch_2.simulate(x, y, z, photons, zernike_coefs=None)

    print(f"the relative root of squared error is: "
          f"{LA.norm((psfs_data_torch_1 - psfs_data_torch_2).flatten(), ord=2) / LA.norm(psfs_data_torch_1.flatten(), ord=2) * 100}%")

    # view data
    ailoc.common.plot_image_stack_difference(psfs_data_torch_1, psfs_data_torch_2)


if __name__ == '__main__':
    test_specified_psf()
    # test_random_psf()
    # test_simulate_speed()
    # test_precompute_speed()
    # test_derivative_speed()
    # test_crlb()
    # test_optimize_crlb()
    # test_psftorch_with_zernike_input()
