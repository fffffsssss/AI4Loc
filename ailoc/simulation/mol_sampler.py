import torch
import numpy as np
# from deprecated import deprecated

import ailoc.common


class MoleculeSampler:
    """
    A sampler for generating x, y, z, photon, background, sub-fov (divide and conquer strategy),
    local field-dependent aberration maps, local read noise maps for training data simulation
    """

    def __init__(self, sampler_params, aberration_map=None, read_noise_map=None):
        """

        Args:
            sampler_params (dict): dict of parameters for the sampler.
                local_context: flag whether to use local context
                robust_training: flag whether to use robust training,
                    namely add noise to the sampled zernike coefficients
                train_size: size of the square training image
                train_density: number of molecules in the training image
                photon_range: list of [min, max]
                z_range: list of [min, max]
                bg_range: list of [min, max]
            aberration_map (None or np.ndarray): aberration map of the psf_model, used to determine the whole FOV size
                (num of zernike, FOV_size_x, FOV_size_y)
            read_noise_map (None or np.ndarray): read noise map of the scmos camera model, used to determine the whole
                FOV size (FOV_size_x, FOV_size_y)

        Returns:
            MoleculeSampler: a sampler for training data simulation
        """

        self.local_context = sampler_params['local_context']
        self.robust_training = sampler_params['robust_training']
        self.train_size = int(sampler_params['train_size'])
        self.num_em_avg = sampler_params['num_em_avg']
        self.train_prob_map = self._compute_prob_map(self.train_size, self.train_size, self.num_em_avg)

        self.photon_range = sampler_params['photon_range']
        self.photon_scale = np.max(np.abs(self.photon_range))
        self.photon_range_scaled = (self.photon_range[0] / self.photon_scale, self.photon_range[1] / self.photon_scale)
        assert self.photon_range_scaled[0] >= 0 and self.photon_range_scaled[1] <= 1, \
            "scaled photon range should be included in [0, 1]"

        self.z_range = sampler_params['z_range']
        self.z_scale = np.max(np.abs(self.z_range))
        self.z_range_scaled = (self.z_range[0] / self.z_scale, self.z_range[1] / self.z_scale)
        assert self.z_range_scaled[0] >= -1 and self.z_range_scaled[1] <= 1, \
            "scaled z range should be included in [-1, 1]"

        self.bg_range = sampler_params['bg_range']
        self.bg_scale = np.max(np.abs(self.bg_range))
        self.bg_range_scaled = (self.bg_range[0] / self.bg_scale, self.bg_range[1] / self.bg_scale)
        assert self.bg_range_scaled[0] >= 0 and self.bg_range_scaled[1] <= 1, \
            "scaled bg range should be included in [0, 1]"

        # determine the FOV size and sliding windows provided the aberration map and read noise map
        if aberration_map is not None and read_noise_map is not None:
            assert aberration_map.shape[-2:] == read_noise_map.shape[-2:], \
                "aberration map and read noise map should have the same size as whole FOV"
            self.fov_size = aberration_map.shape[-2:]
        elif aberration_map is not None and read_noise_map is None:
            self.fov_size = aberration_map.shape[-2:]
        elif read_noise_map is not None and aberration_map is None:
            self.fov_size = read_noise_map.shape[-2:]
        else:
            self.fov_size = torch.Size([self.train_size, self.train_size])
        assert self.fov_size[0] >= self.train_size and self.fov_size[1] >= self.train_size, \
            "train_size should not be larger than FOV size for proper sampling"

        self.sliding_windows = self._compute_sliding_windows(self.train_size, self.fov_size)

    def sample_for_train(self, batch_size, psf_model, iter_train):
        """
        Sample x, y, z, photon, background, sub-fov coordinate, zernike coefs for each PSF in the batch,
        and ground truth. All images in the batch are sampled from the same sub-fov.

        Args:
            batch_size (int): batch size
            psf_model (ailoc.simulation.vectorpsf.VectorPSFCUDA): a vector PSF model used for zernike sampling
            iter_train (int): the number of training iterations, used for sequentially select the current sub-fov

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                1) for training simulation, one-hot 4d map for p_s
                    (batch_size, local_context, train_size, train_size);
                2) for training simulation, 5d map for x,y,z,photon
                    (4 parameters, batch_size, local_context, train_size, train_size);
                3) background map (batch_size, train_size, train_size);
                4) current sub-fov coordinate [x_start, x_end, y_start, y_end]
                5) zernike coefs for each PSF in the batch (num_PSFs, num_zernike),
                    could be field-dependent with robust training;
                6) probability map ground truth for the middle frame,
                    (batch_size,1,train_size,train_size)
                7) molecule ground truth of middle frame, 3d array for x,y,z,photon padding with 0
                    (batch_size, max num_emitters, 4);
                8) binary mask to indicate the valid emitters in the molecule ground truth
                    (batch_size, max num_emitters);
        """

        curr_sub_fov_xy = self.sliding_windows[iter_train % len(self.sliding_windows)]

        p_map_sample, xyzph_map_sample = self.sample_p_xyz_phot(batch_size, self.train_prob_map)

        bg_map_sample = self.sample_bg(batch_size, self.train_prob_map)

        zernike_coefs = self.sample_zernike_coefs(p_map_sample, psf_model, curr_sub_fov_xy, self.robust_training)

        # generate the ground truth array with valid mask for loss calculation
        p_map_gt, xyzph_array_gt, mask_array_gt = self.generate_gt_array(xyzph_map_sample, p_map_sample)

        return p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
               p_map_gt, xyzph_array_gt, mask_array_gt

    def sample_for_evaluation(self, num_image, psf_model):
        """
        Sample x y z photon bg for num_image images, the difference from sample_for_train is that each image
        in the batch has different sub-fov coordinates and so zernike_coefs if field-dependent aberration is used
        """

        p_map_sample, xyzph_map_sample = self.sample_p_xyz_phot(num_image, self.train_prob_map)

        bg_map_sample = self.sample_bg(num_image, self.train_prob_map)

        # generate the ground truth array with valid mask for loss calculation
        p_map_gt, xyzph_array_gt, mask_array_gt = self.generate_gt_array(xyzph_map_sample, p_map_sample)

        sub_fov_xy_list = []
        zernike_coefs_list = []
        for i in range(num_image):
            curr_sub_fov_xy = self.sliding_windows[i % len(self.sliding_windows)]
            curr_zernike_coefs = self.sample_zernike_coefs(p_map_sample[i], psf_model, curr_sub_fov_xy, False)
            sub_fov_xy_list.append(curr_sub_fov_xy)
            zernike_coefs_list.append(curr_zernike_coefs)
        zernike_coefs = torch.concat(zernike_coefs_list[:], 0)

        return p_map_sample, xyzph_map_sample, bg_map_sample, sub_fov_xy_list, zernike_coefs, xyzph_array_gt, mask_array_gt

    @staticmethod
    def sample_zernike_coefs(delta_maps, psf_model, fov_xy, robust_training):
        """
        Sample zernike coefficients for each PSF in the batch
        """

        if psf_model.zernike_coef_map is None:
            curr_aber_map = None
        else:
            curr_aber_map = ailoc.common.gpu(psf_model.zernike_coef_map[:,
                                                                        fov_xy[2]:fov_xy[3] + 1,
                                                                        fov_xy[0]:fov_xy[1] + 1])

        pix_inds = tuple(delta_maps.nonzero().transpose(1, 0))
        num_psf = pix_inds[0].shape[0]
        num_zernike = psf_model.zernike_mode.shape[0]

        zernike_coefs = curr_aber_map[:, pix_inds[-2], pix_inds[-1]].T \
            if curr_aber_map is not None else torch.tile(ailoc.common.gpu(psf_model.zernike_coef), dims=(num_psf, 1))

        if robust_training:
            zernike_coefs += ailoc.common.gpu(torch.normal(mean=0, std=3.33, size=(num_psf, num_zernike)))

        return zernike_coefs

    def sample_p_xyz_phot(self, batch_size, train_prob_map):
        """
        Sample x, y, z, photon for training data simulation
        """

        blink_p = train_prob_map
        blink_p = blink_p.reshape(1, 1, blink_p.shape[-2], blink_p.shape[-1]).expand(batch_size, -1, -1, -1)

        # every pixel has a probability blink_p of existing a molecule, following binomial distribution
        p1_s = ailoc.common.gpu(torch.distributions.Binomial(1, blink_p).sample())
        zeros = ailoc.common.gpu(torch.zeros_like(p1_s))

        # z position follows a uniform distribution with predefined range
        z_s = ailoc.common.gpu(
            torch.distributions.Uniform(zeros + self.z_range_scaled[0], zeros + self.z_range_scaled[1]).sample())

        # xy offset follow uniform distribution
        x_s = ailoc.common.gpu(torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample())
        y_s = ailoc.common.gpu(torch.distributions.Uniform(zeros - 0.5, zeros + 0.5).sample())

        if self.local_context:
            # the probability that a molecule survives in the next frame
            surv_p = 0.5

            # The probability that a single molecule in the previous frame is still present in the next frame
            a11 = 1 - (1 - blink_p) * (1 - surv_p)

            # correction of the number of emitters on three frames, now is (num_em_avg, ~1.5x, ~2.25x)
            # p2_s = ailoc.common.gpu(torch.distributions.Binomial(1, (1 - p1_s) * blink_p + p1_s * a11).sample())
            # p3_s = ailoc.common.gpu(torch.distributions.Binomial(1, (1 - p2_s) * blink_p + p2_s * a11).sample())

            p2_s = ailoc.common.gpu(torch.distributions.Binomial(1,(1-p1_s)*blink_p*(1-a11) + p1_s * a11).sample())
            p3_s = ailoc.common.gpu(torch.distributions.Binomial(1,(1-p2_s)*blink_p*(1-a11) + p2_s * a11).sample())

            p_s = torch.cat([p1_s, p2_s, p3_s], 1)
            x_s = x_s.repeat(1, 3, 1, 1)
            y_s = y_s.repeat(1, 3, 1, 1)
            z_s = z_s.repeat(1, 3, 1, 1)
        else:
            p_s = p1_s

        #  photon number is sampled from a uniform distribution
        ph_s = ailoc.common.gpu(torch.distributions.Uniform(torch.zeros_like(p_s) + self.photon_range_scaled[0],
                                                            torch.zeros_like(p_s) + self.photon_range_scaled[1]).sample())
        x_s *= p_s
        y_s *= p_s
        z_s *= p_s
        ph_s *= p_s

        xyzph_s = torch.cat([x_s[None], y_s[None], z_s[None], ph_s[None]], 0)

        return p_s, xyzph_s

    def sample_bg(self, batch_size, train_prob_map):
        """
        Sample background for training data simulation
        """

        ones = ailoc.common.gpu(torch.ones(batch_size))
        bg_s = ailoc.common.gpu(torch.distributions.Uniform(ones * self.bg_range_scaled[0],
                                                            ones * self.bg_range_scaled[1]).sample())
        bg_s = bg_s.reshape(batch_size, 1, 1).expand(-1, train_prob_map.shape[-2], train_prob_map.shape[-1])
        return bg_s

    # @deprecated(reason='aberration map is no longer a property of the MoleculeSampler')
    # def sample_aberration(self, sub_fov_xy):
    #     """
    #     Sample the aberration map for the current sub-fov
    #
    #     Args:
    #         sub_fov_xy (list): [x_start, x_end, y_start, y_end]
    #
    #     Returns:
    #         torch.Tensor or None: aberration map for the current sub-fov
    #     """
    #
    #     if self.aberration_map is None:
    #         return None
    #     return self.aberration_map[:, sub_fov_xy[2]:sub_fov_xy[3] + 1, sub_fov_xy[0]:sub_fov_xy[1] + 1]
    #
    # @deprecated(reason='read noise map is no longer a property of the MoleculeSampler')
    # def sample_read_noise(self, sub_fov_xy):
    #     """
    #     Sample the read noise map for the current sub-fov
    #
    #     Args:
    #         sub_fov_xy (list): [x_start, x_end, y_start, y_end]
    #
    #     Returns:
    #         torch.Tensor or None: read noise map for the current sub-fov
    #     """
    #
    #     if self.read_noise_map is None:
    #         return None
    #     return self.read_noise_map[sub_fov_xy[2]:sub_fov_xy[3] + 1, sub_fov_xy[0]:sub_fov_xy[1] + 1]

    @staticmethod
    def _compute_prob_map(size_row, size_col, num_em_avg):
        prob_map = ailoc.common.gpu(torch.ones([size_row, size_col]))
        prob_map = prob_map / prob_map.sum() * num_em_avg
        return prob_map

    @staticmethod
    def _compute_sliding_windows(train_size, fov_size, overlap=10):
        """
        Compute the sliding windows for divide and conquer strategy on a large FOV

        Args:
            train_size (int): size of the square training image, unit pixel
            fov_size (torch.Size): size of the full FOV, unit pixel
            overlap (int): for divide and conquer strategy with position sensibility (FD-DeepLoc),
                there should be overlap pixels between neighbour sliding windows to avoid incomplete
                PSF learning at specific position

        Returns:
            list: list of sliding windows on the whole FOV, [x_start, x_end, y_start, y_end]
        """

        row_num = int(np.ceil((fov_size[0] - overlap) / (train_size - overlap)))
        column_num = int(np.ceil((fov_size[1] - overlap) / (train_size - overlap)))

        sliding_windows = []
        for idx in range(0, row_num * column_num):
            x_start = idx % column_num * (train_size - overlap) \
                if idx % column_num * (train_size - overlap) + train_size <= fov_size[1] else fov_size[1] - train_size

            y_start = idx // column_num % row_num * (train_size - overlap) \
                if idx // column_num % row_num * (train_size - overlap) + train_size <= fov_size[0] else fov_size[0] - train_size

            # it represents the [x, y] position, not [row, column], numerical range is [0, fov_size-1]
            sliding_windows.append((x_start, x_start + train_size - 1, y_start, y_start + train_size - 1))

        return sliding_windows

    @staticmethod
    def generate_gt_array(xyzph_map, p_map):
        """
        Generate the ground truth array based on the sampled maps
        """

        curr_batch_size = p_map.shape[0]
        local_context = True if p_map.shape[1] == 3 else False

        p_map_gt = p_map[:, 1] if local_context else p_map[:, 0]
        xyzph_map_gt = xyzph_map[:, :, 1] if local_context else xyzph_map[:, :, 0]
        xyzph_gt = ailoc.common.gpu(torch.zeros([curr_batch_size, 0, 4]))
        mask_gt = ailoc.common.gpu(torch.zeros([curr_batch_size, 0]))

        if p_map_gt.sum():
            # get the emitters' pixel indices (n_emitter, 3), [image index in a batch, row, column]
            inds = tuple(p_map_gt.nonzero().transpose(1, 0))

            # get corresponding xyz photons and build a gt matrix with shape (n_emitters in this batch, 4)
            xyzph_list_gt = xyzph_map_gt[:, inds[0], inds[1], inds[2]]
            xyzph_list_gt[0] += inds[2] + 0.5
            xyzph_list_gt[1] += inds[1] + 0.5
            xyzph_list_gt = xyzph_list_gt.transpose(1, 0)

            # get the number of emitters in each image
            em_num = torch.unique_consecutive(inds[0], return_counts=True)[1]
            em_num_max = em_num.max()

            # build a gt matrix with shape (batch_size, em_num, 4)
            xyzph_arr_gt = ailoc.common.gpu(torch.zeros([curr_batch_size, em_num_max, 4]))
            mask_arr_gt = ailoc.common.gpu(torch.zeros([curr_batch_size, em_num_max]))

            # order all emitters on their own image respectively, e.g. [0, 1, 2, 0, 1, 2, 3, 0, 1...],
            # with shape (n_emitters in this batch)
            em_inds = torch.cat([torch.arange(num_on_each) for num_on_each in em_num], dim=0)

            # fill the gt matrix using the image index and the order on each image
            xyzph_arr_gt[inds[0], em_inds] = xyzph_list_gt
            mask_arr_gt[inds[0], em_inds] = 1

            xyzph_gt = torch.cat([xyzph_gt, xyzph_arr_gt], dim=1)
            mask_gt = torch.cat([mask_gt, mask_arr_gt], dim=1)

        return p_map_gt, xyzph_gt, mask_gt
