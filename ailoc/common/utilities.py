import csv
import random
from operator import itemgetter
import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt
import napari
import tifffile
import os
import cv2
import sys

# import ailoc.common.local_tifffile
import ailoc.common
import ailoc.simulation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # this seems affect the speed of some networks
    torch.backends.cudnn.benchmark = False


def gpu(x, data_type=torch.float32):
    """
    Transforms numpy array or torch tensor to torch.cuda.FloatTensor
    """

    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device='cuda', dtype=data_type)
    return x.to(device='cuda', dtype=data_type)


def cpu(x, data_type=np.float32):
    """
    Transforms torch tensor into numpy array
    """

    if not isinstance(x, torch.Tensor):
        return np.array(x, dtype=data_type)
    return x.cpu().detach().numpy().astype(data_type)


def softp(x):
    """
    Returns softplus(x)
    """

    return np.log(1 + np.exp(x))


def sigmoid(x):
    """
    Returns sigmoid(x)
    """

    return 1 / (1 + np.exp(-x))


def inv_softp(x):
    """
    Returns inverse softplus(x)
    """

    return np.log(np.exp(x) - 1)


def inv_sigmoid(x):
    """
    Returns inverse sigmoid(x)
    """

    return -np.log(1 / x - 1)


def torch_arctanh(x):
    """
    Returns arctanh(x) for tensor input
    """

    return 0.5 * torch.log(1 + x) - 0.5 * torch.log(1 - x)


def torch_softp(x):
    """
    Returns softplus(x) for tensor input
    """

    return torch.log(1 + torch.exp(x))


def flip_filt(filt):
    """
    Returns filter flipped over x and y dimension
    """

    return np.ascontiguousarray(filt[..., ::-1, ::-1])


def get_bg_stats_gamma(images, percentile=10, plot=False, xlim=None, floc=0):
    """Infers the parameters of a gamma distribution that fit the background of SMLM recordings.
    Identifies the darkest pixels from the averaged images as background and fits a gamma distribution to the histogram of intensity values.

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
        plot (bool): If true produces a plot of the histogram and fit
        xlim (list of float): Sets xlim of the plot
        floc (float): Baseline for the gamma fit. Equal to fitting gamma to (x - floc)

    Returns:
        (float, float): (Mean, scale) parameter of the gamma fit
    """

    # ensure positive
    ind = np.where(images <= 0)
    images[ind] = 1

    # get the positions where the mean intensity is below the percentile
    map_empty = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, map_empty[0], map_empty[1]].reshape(-1)
    # fit the gamma distribution, return the alpha and scale=1/beta
    fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(pixel_vals, floc=floc)

    if plot:
        plt.figure(constrained_layout=True)
        if xlim is None:
            low, high = pixel_vals.min(), pixel_vals.max()
        else:
            low, high = xlim[0], xlim[1]

        _ = plt.hist(pixel_vals, bins=np.linspace(low, high), histtype='step', label='data')
        _ = plt.hist(np.random.gamma(shape=fit_alpha, scale=fit_beta, size=len(pixel_vals)) + floc,
                     bins=np.linspace(low, high), histtype='step', label='fit')
        plt.xlim(low, high)
        plt.legend()
        # plt.tight_layout()
        plt.show()
    return fit_alpha * fit_beta, fit_beta  # return the expectation and scale


def get_bg_stats_gauss(images, percentile=10, plot=False):
    """Infers the parameters of a gauss distribution that fit the background of SMLM recordings.
    Identifies the darkest pixels from the averaged images as background.

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
        plot (bool): If true produces a plot of the histogram and fit

    Returns:
        (float, float): (bg_min, bg_max) background parameters
    """

    # get the positions where the mean intensity is below the percentile
    roi_idx = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, roi_idx[0], roi_idx[1]].reshape(-1)

    # fit the gauss distribution
    result = scipy.stats.norm.fit(pixel_vals)

    if plot:
        plt.figure(constrained_layout=True)
        _ = plt.hist(pixel_vals,
                     bins=np.linspace(pixel_vals.min(), pixel_vals.max(), 50),
                     # histtype='step',
                     alpha=0.5,
                     label='bg data')
        _ = plt.hist(np.random.normal(loc=result[0], scale=result[1], size=len(pixel_vals)),
                     bins=np.linspace(pixel_vals.min(), pixel_vals.max()),
                     # histtype='step',
                     alpha=0.5,
                     label='bg fit')
        plt.legend()
        plt.show()

    bg_range = tuple(np.clip([result[0] - 2 * result[1], result[0] + 2 * result[1]],
                             a_min=0, a_max=None))
    return bg_range

def get_gain_bg_empirical(images,
                          camera_params_dict,
                          adjust_gain=True,
                          percentile=50,
                          plot=False):
    """
    This is an empirical method to estimate gain and training background range from data with uneven background,
    use the 1% darkest pixels to estimate the variance/mean ratio as the gain, and adjust the e_per_adu to make the
    variance/mean ratio to be 1.0 for Poisson assumption, a little mismatch is allowed.
    Then use the pixels with intensity higher than the percentile to estimate the background range.
    The estimated bg range maybe a little higher than the real one on uniform background data, but it is more robust
    as the network will not predict too many false positive signals.

    Args:
        images (np.ndarray): 3D array of recordings in photon counts
        camera_params_dict (dict): dict containing the camera parameters.
        adjust_gain (bool): If true, adjust the e_per_adu to make the variance/mean ratio to be 1.0 for Poisson assumption
        percentile (float): Percentile between 0 and 100.
            Sets the percentage of pixels that are assumed to have signals,
            and the training background range is estimated from these pixels,
            the background estimated from the rest is often lower and may cause false positives.
        plot (bool): If true produces a plot of the histogram and fit

    Returns:
        (float, float): (bg_min, bg_max) background parameters
    """
    n, h, w = images.shape

    camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
    images_photon = ailoc.common.cpu(camera_calib.backward(torch.tensor(images.astype(np.float32))))

    if adjust_gain:
        # get the minimal 1% pixels as the background to estimate the gain
        min_idx = np.where(images_photon.mean(0) < np.quantile(images_photon.mean(0), 0.005))

        # split the images into 1000 frames as time chunks
        images_electron = camera_calib.qe * images_photon.reshape(-1, 1000, h, w) if (n % 1000 == 0) \
            else (images_photon[:-(n % 1000)].reshape(-1, 1000, h, w))

        pix_vals = images_electron[:, :, min_idx[0], min_idx[1]]

        pix_mean = pix_vals.mean(1).reshape(-1)
        pix_var = pix_vals.var(1).reshape(-1)
        pix_gain = ((pix_var-camera_calib.read_noise_sigma**2)/pix_mean)
        # remove the outliers, 95% remain
        used_idx = np.logical_and(pix_gain > pix_gain.mean() - 2 * pix_gain.std(),
                                  pix_gain < pix_gain.mean() + 2 * pix_gain.std())
        pix_mean_used = pix_mean[used_idx]
        pix_var_used = pix_var[used_idx]
        pix_gain_used = pix_gain[used_idx]
        est_gain = pix_gain_used.mean()

        print(f'The variance/mean ratio of data is estimated as {est_gain:.2f} using the provided QE and e_per_adu.')

        if est_gain > 1.2 or est_gain < 0.9:
            e_per_adu_new = ((pix_mean_used.mean() +
                             np.sqrt(pix_mean_used.mean()**2-4*pix_var_used.mean()*
                                     (est_gain*pix_mean_used.mean()-pix_var_used.mean())))/(2*pix_var_used.mean())
                             *camera_calib.e_per_adu)
            # e_per_adu_new = camera_calib.e_per_adu / est_gain
            e_per_adu_new = np.around(e_per_adu_new, decimals=2)
            print(f'This might be unreliable, '
                  f'automatically change the e_per_adu from {camera_calib.e_per_adu:.2f} to '
                  f'{e_per_adu_new:.2f} to make the variance/mean ratio 1.0 for Poisson noise assumption.')
            camera_calib.e_per_adu = e_per_adu_new
            images_photon = ailoc.common.cpu(camera_calib.backward(torch.tensor(images.astype(np.float32))))
        else:
            e_per_adu_new = camera_calib.e_per_adu
    else:
        e_per_adu_new = camera_calib.e_per_adu

    # get the region of interest that has both background and signals
    sample_mask = np.where(images_photon.mean(0) > np.percentile(images_photon.mean(0), percentile))
    # average the images for every 1000 frames
    images_avg = images_photon.reshape(-1, 1000, h, w).mean(1) if (n % 1000 == 0) \
        else images_photon[:-(n % 1000)].reshape(-1, 1000, h, w).mean(1)

    # pixel values containing both bg and signals
    pixel_vals = images_avg[:, sample_mask[0], sample_mask[1]].reshape(-1)

    # fit the gauss distribution
    result = scipy.stats.norm.fit(pixel_vals)

    # get the background range, the lower bound is the mean minus 2 times std, the upper bound is the mean
    bg_range = tuple(np.clip([result[0] - np.clip(2 * result[1], 20, 200), result[0]],
                             a_min=0, a_max=None))
    print(f'Estimated bg_range: {bg_range}')

    if plot:
        # Create the figure and GridSpec
        fig = plt.figure(figsize=(12, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        if adjust_gain:
            # First plot: Histogram with star marker
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(images_photon.mean(0) < np.quantile(images_photon.mean(0), 0.001), cmap='turbo')
            ax0.set_title('Pixels to estimate var/mean')

            ax1 = fig.add_subplot(gs[0, 1:])
            ax1.hist(pix_gain_used,
                     bins=np.linspace(pix_gain_used.min(), pix_gain_used.max(), 50),
                     alpha=0.5,
                     label='0.5% pixel var/mean')
            ax1.plot(est_gain, 1, '*', markersize=20, label=f'extra gain: {est_gain:.1f}')
            ax1.legend()

        # Second plot: Mean image
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(images_photon.mean(0), cmap='turbo')
        ax2.set_title('mean image')

        # Third plot: Masked area
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(images_photon.mean(0) > np.percentile(images_photon.mean(0), percentile))
        ax3.set_title(f'masked area for bg fit \n(mean image>{percentile}%)')

        # Fourth plot: Dual histogram
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(pixel_vals,
                 density=True,
                 bins=20,
                 alpha=0.5,
                 label='masked pixels including bg and signals')
        ax4.hist(np.random.uniform(low=bg_range[0], high=bg_range[1], size=len(pixel_vals)),
                 density=True,
                 bins=20,
                 alpha=0.5,
                 label=f'training bg range ({bg_range[0]:.1f}, {bg_range[1]:.1f})')
        ax4.legend()

        plt.show()

    return bg_range, e_per_adu_new


def get_mean_percentile(images, percentile=10):
    """
    Returns the mean of the pixels at where their mean values are less than the given percentile of the average image

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Used to calculate the mean of the percentile of the images
    """

    idx_2d = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, idx_2d[0], idx_2d[1]]

    return pixel_vals.mean()


def get_window_map(img, winsize=40, percentile=20):
    """Helper function

    Parameters
    ----------
    images: array
        3D array of recordings
    percentile: float
        Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
    plot: bool
        If true produces a plot of the histogram and fit
    xlim: list of floats
        Sets xlim of the plot
    floc: float
        Baseline for the the gamma fit. Equal to fitting gamma to (x - floc)

    Returns
    -------
    binmap: array
        Mean and scale parameter of the gamma fit
    """

    img = img.mean(0)  # 按第一维求平均,得到[64 64]
    res = np.zeros([int(img.shape[0] - winsize), int(img.shape[1] - winsize)])  # [64-40，64-40]的零矩阵
    for i in range(res.shape[0]):  # 0-24
        for j in range(res.shape[1]):
            res[i, j] = img[i:i + int(winsize), j:j + int(winsize)].mean()  # 以i j出发求[40 40]区域内的平均值
    thresh = np.percentile(res, percentile)  # 从小到大，第percentile%的值，也就是还有percentile%比这个值小
    binmap = np.zeros_like(res)
    binmap[res > thresh] = 1  # 图像中intensity大于20%的都设为1，应该是表示该处有分子荧光
    return binmap


def get_pixel_truth(eval_csv, ind, field_size, pixel_size):
    """
    draw a pixel-wise binary map of the groudtruth
    """

    eval_list = []
    if isinstance(eval_csv, str):
        with open(eval_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    eval_list.append([float(r) for r in row])
    else:
        for r in eval_csv:
            eval_list.append([i for i in r])
    eval_list = sorted(eval_list, key=itemgetter(1))  # csv文件按frame升序来排列

    molecule_list = []
    for i in range(len(eval_list)):
        if eval_list[i][1] == ind:
            molecule_list.append([round(eval_list[i][2] / pixel_size[0]), round(eval_list[i][3] / pixel_size[1])])
        if eval_list[i][1] > ind:
            break

    truth_map = np.zeros([round(field_size[0] / pixel_size[0]), round(field_size[1] // pixel_size[1])])
    for molecule in molecule_list:
        truth_map[molecule[0], molecule[1]] = 1

    return truth_map


def read_first_size_gb_tiff(image_path, size_gb=4):
    # with ailoc.common.local_tifffile.TiffFile(image_path, is_ome=False) as tif:
    with tifffile.TiffFile(image_path, is_ome=False) as tif:
        total_shape = tif.series[0].shape
        # occu_mem = total_shape[0] * total_shape[1] * total_shape[2] * 16 / (1024 ** 3) / 8
        occu_mem = tif.series[0].size * tif.series[0].dtype.itemsize / (1024 ** 3)  # GBytes
        if occu_mem < size_gb:
            index_img = total_shape[0]
        else:
            index_img = int(size_gb / occu_mem * total_shape[0])
        images = tif.asarray(key=range(0, index_img), series=0)
    print('-' * 200)
    print(f"read first {images.shape} images")
    return images


def find_file(name, path_list):
    for path in path_list:
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)


def viewdata_napari(*args):
    viewer = napari.view_image(cpu(args[0]), colormap='turbo')
    for i in range(1, len(args)):
        viewer.add_image(cpu(args[i]), colormap='turbo')
    napari.run()


def cmpdata_napari(data1, data2):
    assert data1.shape == data2.shape, "data1 and data2 must have the same shape"
    n_dim = len(data1.shape)
    width = data1.shape[-1]
    pad_width = [(0,0) for i in range(n_dim-1)]
    pad_width.append((0, int(0.05 * width)))
    data1 = np.pad(cpu(data1), tuple(pad_width), constant_values=np.nan)
    data2 = np.pad(cpu(data2), tuple(pad_width), constant_values=np.nan)
    data3 = np.concatenate((data1, data2, data1-data2), axis=-1)
    viewer = napari.view_image(data3, colormap='turbo')
    napari.run()


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict):
    print('-' * 70, 'learning parameters', '-' * 100)
    for params_dict in [psf_params_dict, camera_params_dict, sampler_params_dict]:
        for keys in params_dict.keys():
            if keys == 'zernike_mode':
                params = params_dict[keys].transpose()
                print(f"{keys}:\n {params}")
            elif keys == 'zernike_coef':
                params = np.around(params_dict[keys], decimals=1)
                print(f"{keys}:\n {params}")
            else:
                params = params_dict[keys]
                print(f"{keys}: {params}")


class TrainLogger(object):
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == "__main__":
    # Generate a figure with matplotlib</font>
    figure = plt.figure()
    plot = figure.add_subplot(111)

    # draw a cardinal sine plot
    x = np.arange(1, 100, 0.1)
    y = np.sin(x) / x
    plot.plot(x, y)
    plt.show()
    ##
    image = fig2data(figure)
    cv2.imshow('image', image)
    cv2.waitKey(0)