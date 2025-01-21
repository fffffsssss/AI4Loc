import matplotlib.pyplot as plt
import numpy as np
import ctypes
import napari
import os


def gauss2D_kernel(shape=(3, 3), sigmax=0.5, sigmay=0.5):
    """
    2D gaussian mask for VectorPSF otf rescale
    """

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.meshgrid(np.arange(-m, m + 1, 1), np.arange(-n, n + 1, 1), indexing='ij')
    h = np.exp(-(x * x) / (2. * sigmax * sigmax + 1e-6) - (y * y) / (2. * sigmay * sigmay + 1e-6))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# def view_npdata_napari(*args):
#     viewer = napari.view_image(args[0], colormap='turbo')
#     for i in range(1, len(args)):
#         viewer.add_image(args[i], colormap='turbo')
#     napari.run()
#
#
# def cmp_npdata_napari(data1, data2):
#     assert data1.shape == data2.shape, "data1 and data2 must have the same shape"
#     width = data1.shape[-1]
#     data1 = np.pad(data1, ((0, 0), (0, 0), (0, int(0.05 * width))), constant_values=np.nan)
#     data2 = np.pad(data2, ((0, 0), (0, 0), (0, int(0.05 * width))), constant_values=np.nan)
#     data3 = np.concatenate((data1, data2, data1-data2), axis=2)
#     viewer = napari.view_image(data3, colormap='turbo')
#     napari.run()

def zernike_calibrate_beads(beads_data, parameters: dict):

    # prepare the fitting input/output parameters and buffers
    class PSFFitStructure(ctypes.Structure):
        _fields_ = [
            ('NA', ctypes.c_double),
            ('refmed', ctypes.c_double),
            ('refcov', ctypes.c_double),
            ('refimm', ctypes.c_double),
            ('lambda0', ctypes.c_double),
            ('zeimt0', ctypes.c_double),
            ('objStage0', ctypes.c_double),
            ('pixelSizeX', ctypes.c_double),
            ('pixelSizeY', ctypes.c_double),
            ('Npupil', ctypes.c_double),
            ('sizeX', ctypes.c_double),
            ('sizeY', ctypes.c_double),
            ('sizeZ', ctypes.c_double),
            ('aberrations', ctypes.POINTER(ctypes.c_double)),
            ('maxJump', ctypes.POINTER(ctypes.c_double)),
            ('numparams', ctypes.c_double),
            ('numAberrations', ctypes.c_double),
            ('zemitStack', ctypes.POINTER(ctypes.c_double)),
            ('objStageStack', ctypes.POINTER(ctypes.c_double)),
            ('ztype', ctypes.c_char_p),  # stage or emitter
            ('map', ctypes.POINTER(ctypes.c_double)),
            ('Nitermax', ctypes.c_double)
        ]

    psf_fit_structure = PSFFitStructure()
    psf_fit_structure.NA = parameters['na']
    psf_fit_structure.refmed = parameters['refmed']
    psf_fit_structure.refcov = parameters['refcov']
    psf_fit_structure.refimm = parameters['refimm']
    psf_fit_structure.lambda0 = parameters['lambda']
    psf_fit_structure.objStage0 = 0
    psf_fit_structure.zeimt0 = 50
    psf_fit_structure.pixelSizeX = parameters['pixelSizeY']
    psf_fit_structure.pixelSizeY = parameters['pixelSizeX']
    psf_fit_structure.Npupil = 64

    aberrations = np.array([[2, 2, 3, 3, 4, 3, 3, 4, 4, 5, 5, 6, 4, 4, 5, 5, 6, 6, 7, 7, 8],
                            [-2, 2, -1, 1, 0, -3, 3, -2, 2, -1, 1, 0, -4, 4, -3, 3, -2, 2, 1, -1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                           dtype=np.float64)
    psf_fit_structure.aberrations = aberrations.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    psf_fit_structure.sizeX = beads_data.shape[2]
    psf_fit_structure.sizeY = beads_data.shape[1]
    psf_fit_structure.sizeZ = beads_data.shape[0]
    psf_fit_structure.Nitermax = parameters['iterations']
    nmol = beads_data.shape[0]

    num_aberrations = aberrations.shape[1]
    shared = np.concatenate((np.ones(num_aberrations), np.array([1, 1, 1, 0, 0])), axis=0, dtype=np.float64)
    sumshared = np.sum(shared)
    numparams = int(26 * nmol - sumshared * (nmol - 1))

    psf_fit_structure.numparams = numparams
    psf_fit_structure.numAberrations = num_aberrations

    dz = parameters['dz']  # nm
    zmax = (nmol - 1) * dz / 2
    zemitStack = np.zeros((nmol, 1), dtype=np.float64)  # nm
    psf_fit_structure.zemitStack = zemitStack.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    objStageStack = np.linspace(zmax, -zmax, nmol, dtype=np.float64)
    psf_fit_structure.objStageStack = objStageStack.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    ztype = "stage"
    psf_fit_structure.ztype = ctypes.c_char_p(ztype.encode('utf-8'))

    # init theta
    thetainit = np.zeros((int(numparams), 1), dtype=np.float64)

    bg0 = np.zeros((1, nmol))
    Nph = np.zeros((1, nmol))
    x0 = np.zeros((1, nmol))
    y0 = np.zeros((1, nmol))
    z0 = np.zeros((1, nmol))

    # center of mass with nm unit
    ImageSizex = parameters['pixelSizeX'] * psf_fit_structure.sizeX / 2
    ImageSizey = parameters['pixelSizeY'] * psf_fit_structure.sizeY / 2

    DxImage = 2 * ImageSizex / psf_fit_structure.sizeX
    DyImage = 2 * ImageSizey / psf_fit_structure.sizeY
    ximagelin = np.arange(-ImageSizex + DxImage / 2, ImageSizex, DxImage)
    yimagelin = np.arange(-ImageSizey + DyImage / 2, ImageSizey, DyImage)
    YImage, XImage = np.meshgrid(yimagelin, ximagelin)
    for i in range(nmol):
        dTemp = beads_data[i, :, :]
        bg0[0, i] = np.max([np.min(dTemp), 1])
        Nph[0, i] = np.sum(dTemp - bg0[0, i])
        x0[0, i] = np.sum(np.multiply(dTemp, XImage)) / Nph[0, i]
        y0[0, i] = np.sum(np.multiply(dTemp, YImage)) / Nph[0, i]
        z0[0, i] = 0

    allTheta = np.zeros((num_aberrations + 5, nmol), dtype=np.float64)
    allTheta[num_aberrations, :] = x0[0]
    allTheta[num_aberrations + 1, :] = y0[0]
    allTheta[num_aberrations + 2, :] = z0[0]
    allTheta[num_aberrations + 3, :] = Nph[0]
    allTheta[num_aberrations + 4, :] = bg0[0]
    allTheta[0:num_aberrations,:] = np.tile(aberrations[2:3, :], [nmol, 1]).transpose()

    # first column/row denotes shared or not shared, the second column denotes the parameter type,
    # the third row denotes the data index corresponding to the parameter, 0 means shared
    map = np.zeros((3, int(numparams)), np.float64, order='C')
    n = 1
    for i in range(num_aberrations + 5):
        if shared[i] == 1:
            map[0, n - 1] = 1
            map[1, n - 1] = i + 1
            map[2, n - 1] = 0
            n += 1
        elif shared[i] == 0:
            for j in range(nmol):
                map[0, n - 1] = 0
                map[1, n - 1] = i + 1
                map[2, n - 1] = j + 1
                n += 1

    for i in range(numparams):
        if map[0, i] == 1:
            thetainit[i] = np.mean(allTheta[int(map[1, i]) - 1, :])
        elif map[0, i] == 0:
            thetainit[i] = allTheta[int(map[1, i]) - 1, int(map[2, i]) - 1]

    psf_fit_structure.map = map.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    zernikecoefsmax = 0.25 * parameters['lambda'] * np.ones((num_aberrations, 1))
    maxJump = np.concatenate([zernikecoefsmax,
                              psf_fit_structure.pixelSizeX * np.ones(
                                  (np.max([nmol * int(shared[num_aberrations] == 0), 1]), 1)),
                              psf_fit_structure.pixelSizeY * np.ones(
                                  (np.max([nmol * int(shared[num_aberrations + 1] == 0), 1]), 1)),
                              500 * np.ones((np.max([nmol * int(shared[num_aberrations + 2] == 0), 1]), 1)),
                              2 * np.max(Nph) * np.ones(
                                  (np.max([nmol * int(shared[num_aberrations + 3] == 0), 1]), 1)),
                              100 * np.ones((np.max([nmol * int(shared[num_aberrations + 4] == 0), 1]), 1))
                              ], axis=0, dtype=np.float64)
    psf_fit_structure.maxJump = maxJump.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    pout = np.ones((numparams + 1, 1), dtype=np.float64, order='C') * np.nan
    modelout = np.ones(beads_data.shape, dtype=np.float64, order='C') * np.nan
    errout = np.ones((int(psf_fit_structure.Nitermax), 1), dtype=np.float64, order='C') * np.nan

    kernel = np.zeros([5, 5], dtype=np.float64)
    kernel += gauss2D_kernel(shape=[5, 5], sigmax=parameters['otf_rescale'], sigmay=parameters['otf_rescale']).reshape([5, 5])

    thispath = os.path.dirname(os.path.abspath(__file__))
    dllpath = thispath + '/Vectorial_PSF_fit_CUDA_py_ext_FS.dll'
    beads_calibrator = ctypes.CDLL(dllpath, winmode=0)

    beads_data_1 = np.zeros_like(beads_data,dtype=np.float64)
    beads_data_1 += beads_data
    beads_calibrator.fit_zernike_py_api(beads_data_1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        thetainit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        psf_fit_structure,
                                        shared.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        ctypes.c_double(0.1),
                                        kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        pout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        modelout.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                        errout.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                                        )

    aberrations_estimated = aberrations.copy()
    aberrations_estimated[2:3, :] = pout[0:num_aberrations, 0].transpose()
    np.set_printoptions(suppress=True)
    # print(f"{aberrations_estimated.transpose()}")
    print(f"the relative root of squared error is: "
          f"{np.linalg.norm((modelout - beads_data).flatten(), ord=2) / np.linalg.norm(beads_data.flatten(), ord=2) * 100}%")

    #  plot the results
    fig_list = []

    if nmol <= 24:
        ncolumns = 4
        nrows = int(np.floor(nmol / ncolumns))
        nidx = np.arange(nmol)
    else:
        ncolumns = 4
        nrows = 6
        dn = int(np.floor(nmol / (nrows * ncolumns)))
        nidx = np.arange(0, (nrows * ncolumns) * dn, dn)
    figure, ax = plt.subplots(nrows, ncolumns)
    i_slice = 0
    for i in range(nrows):
        for j in range(ncolumns):
            beads_tmp = beads_data[nidx[i_slice]]
            model_tmp = modelout[nidx[i_slice]]
            mismatch_tmp = beads_tmp - model_tmp
            image_tmp = np.concatenate([beads_tmp, model_tmp, mismatch_tmp], axis=1)
            img_tmp = ax[i, j].imshow(image_tmp, cmap='turbo')
            # plt.colorbar(mappable=img_tmp, ax=ax[i, j], fraction=0.015, pad=0.005)
            # ax[i,j].set_title(f"data, model, error")
            i_slice += 1
    ax[0, 0].set_title(f"data, model, error")

    fig_list.append(['ZernikeModel',figure])

    figure, ax = plt.subplots(1, 1)
    aberrations_names = []
    for i in range(aberrations_estimated.shape[1]):
        aberrations_names.append(f"{aberrations_estimated[0, i]:.0f}, {aberrations_estimated[1, i]:.0f}")
    plt.xticks(np.arange(aberrations_estimated.shape[1]), labels=aberrations_names, rotation=30)
    bar_tmp = ax.bar(np.arange(aberrations_estimated.shape[1]), aberrations_estimated[2], width=0.9, color='orange',
                     edgecolor='k')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x(), 1.01 * height, '%.2f' % height, )

    autolabel(bar_tmp)
    ax.tick_params(axis='both',
                   direction='out'
                   )
    ax.set_ylabel('Zernike coefficients (nm)')
    ax.set_xlabel('Zernike modes')

    fig_list.append(['ZernikeModes', figure])

    # output the results
    zernike_fit_results = {}
    zernike_fit_results['na'] = parameters['na']
    zernike_fit_results['refmed'] = parameters['refmed']
    zernike_fit_results['refcov'] = parameters['refcov']
    zernike_fit_results['refimm'] = parameters['refimm']
    zernike_fit_results['wavelength'] = parameters['lambda']
    zernike_fit_results['objstage0'] = psf_fit_structure.objStage0
    zernike_fit_results['zemit0'] = psf_fit_structure.zeimt0
    zernike_fit_results['pixelSizeX'] = parameters['pixelSizeX']
    zernike_fit_results['pixelSizeY'] = parameters['pixelSizeY']
    zernike_fit_results['otf_rescale'] = parameters['otf_rescale']
    zernike_fit_results['Npupil'] = 64
    zernike_fit_results['Nitermax'] = psf_fit_structure.Nitermax
    zernike_fit_results['zernike_coefficients'] = aberrations_estimated
    zernike_fit_results['pout'] = pout
    zernike_fit_results['modelout'] = modelout
    zernike_fit_results['beads_data'] = beads_data
    zernike_fit_results['errout'] = errout

    return zernike_fit_results, fig_list
