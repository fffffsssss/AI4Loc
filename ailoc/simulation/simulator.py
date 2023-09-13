import torch
import numpy as np
import torch.nn as nn

import ailoc.simulation
import ailoc.common


class Simulator:
    """
    Simulator class for generating simulated images of single molecules.
    """

    def __init__(self, psf_params, camera_params, sampler_params):
        """
        Args:
            psf_params (dict): parameters for the PSF
            camera_params (dict): parameters for the camera
            sampler_params (dict): parameters for the molecule sampler
        """

        self.psf_model = ailoc.simulation.VectorPSFCUDA(psf_params)
        if camera_params['camera_type'].upper() == 'EMCCD':
            self.camera = ailoc.simulation.EMCCD(camera_params)
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map)
        elif camera_params['camera_type'].upper() == 'SCMOS':
            self.camera = ailoc.simulation.SCMOS(camera_params)
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map,
                                                                self.camera.read_noise_map)
        elif camera_params['camera_type'].upper() == 'IDEA':
            self.camera = ailoc.simulation.IdeaCamera()
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map)
        else:
            raise NotImplementedError('Camera type not supported.')

    def sample_training_data(self, batch_size, iter_train):
        """
        Sample a batch of training data.

        Args:
            batch_size (int): batch size
            iter_train (int): the number of training iterations, used for sequentially select the
                sub-fov to simulate images

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list):
                a batch of simulated images and corresponding ground truth
        """

        p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
        p_map_gt, xyzph_array_gt, mask_array_gt = self.mol_sampler.sample_for_train(batch_size, self.psf_model, iter_train)

        data = self.gen_noiseless_data(self.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)

        data_cam = self.camera.forward(data, curr_sub_fov_xy) \
            if isinstance(self.camera, ailoc.simulation.SCMOS) else self.camera.forward(data)

        return data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy

    def sample_evaluation_data(self, num_image):
        """
        sample dataset for online evaluation, the data should be generated the same as the
        training data except robust training, and each group of images (local context) in the dataset has a sub-fov.

        Args:
            num_image (int): number of evaluation images to generate.

        Returns:
            (torch.Tensor, list, list):
                evaluation images with shape (num_image, local context, train_size, train_size),
                ground truth molecule list (frame, x, y, z, photon) and
                sub-fov list [x_start, x_end, y_start, y_end] (each evaluation image is from a specific sub-fov).
        """

        p_map_sample, xyzph_map_sample, bg_map_sample, sub_fov_xy_list, zernike_coefs, \
        xyzph_array_gt, mask_array_gt = self.mol_sampler.sample_for_evaluation(num_image, self.psf_model)

        molecule_list_gt = []
        eval_data = self.gen_noiseless_data(self.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)

        for i in range(num_image):
            eval_data[i] = self.camera.forward(eval_data[i], sub_fov_xy_list[i]) \
                if isinstance(self.camera, ailoc.simulation.SCMOS) else self.camera.forward(eval_data[i])

            for j in range(mask_array_gt.shape[1]):
                if mask_array_gt[i, j] == 1:
                    molecule_list_gt.\
                        append([i+1,
                                float(ailoc.common.cpu(xyzph_array_gt[i, j, 0] * self.psf_model.pixel_size_xy[0])),
                                float(ailoc.common.cpu(xyzph_array_gt[i, j, 1] * self.psf_model.pixel_size_xy[1])),
                                float(ailoc.common.cpu(xyzph_array_gt[i, j, 2] * self.mol_sampler.z_scale)),
                                float(ailoc.common.cpu(xyzph_array_gt[i, j, 3] * self.mol_sampler.photon_scale))]
                               )

        return eval_data, molecule_list_gt, sub_fov_xy_list

    def gen_noiseless_data(self, psf_model, delta_map, xyzph_map, bg, zernike_coefs):
        """
        Generate noiseless data from the given delta map and psf parameters.
        """

        batch_size, channels, height, width = delta_map.shape[0], delta_map.shape[1], delta_map.shape[2], delta_map.shape[3]

        bg = torch.reshape(bg, [batch_size, 1, height, width]).repeat(1, channels, 1, 1)*self.mol_sampler.bg_scale

        x,y,z,photons = self._translate_maps(delta_map.reshape([-1, height, width]), xyzph_map.reshape([4, -1, height, width]))

        psf_patches = psf_model.simulate(x, y, z, photons, zernike_coefs) \
            if zernike_coefs is not None else psf_model.simulate(x, y, z, photons)

        raw_data = self.place_psfs(delta_map.reshape([-1, height, width]), psf_patches) + bg.reshape([-1, height, width])

        return torch.reshape(raw_data, [batch_size, channels, height, width])

    def _translate_maps(self, delta, xyzph):
        pix_inds = tuple(delta.nonzero().transpose(1, 0))

        x_offsets = xyzph[0][pix_inds] * self.psf_model.pixel_size_xy[0]
        y_offsets = xyzph[1][pix_inds] * self.psf_model.pixel_size_xy[1]
        z_offsets = xyzph[2][pix_inds] * self.mol_sampler.z_scale
        photons = xyzph[3][pix_inds] * self.mol_sampler.photon_scale

        return x_offsets, y_offsets, z_offsets, photons

    @staticmethod
    def place_psfs(delta, psf_patches):
        """
        Place the psf_patches on the canvas according to the delta map.
        """

        psf_size = psf_patches.shape[-1]

        canvas = torch.zeros_like(delta)
        canvas_h, canvas_w = delta.shape[1], delta.shape[2]

        delta_inds = tuple(delta.nonzero().transpose(1, 0))
        relu = nn.ReLU()

        # all row and column indices and all possible row and column indices
        row_col_inds = delta.nonzero()[:, 1:]
        unique_inds = delta.sum(0).nonzero()

        # indices on canvas, corresponding to the upper-left corner of the psf_patch on the canvas
        canvas_row_start = relu(unique_inds[:, 0] - psf_size // 2)
        canvas_col_start = relu(unique_inds[:, 1] - psf_size // 2)

        # indices on psf_patches, cut the psf_patches if it is out of the canvas
        psf_row_start = relu(psf_size // 2 - unique_inds[:, 0])
        psf_row_end = psf_size - (unique_inds[:, 0] + psf_size // 2 - canvas_h) - 1
        psf_col_start = relu(psf_size // 2 - unique_inds[:, 1])
        psf_col_end = psf_size - (unique_inds[:, 1] + psf_size // 2 - canvas_w) - 1

        # convert row and column indices to linear indices, just to give each delta a unique index
        row_col_inds_linear = canvas_w * row_col_inds[:, 0] + row_col_inds[:, 1]
        unique_inds_linear = canvas_w * unique_inds[:, 0] + unique_inds[:, 1]

        for i in range(len(unique_inds)):
            # traverse the unique indices, get all candidates that should be put the
            # same way as the current unique index
            curr_inds = torch.nonzero(ailoc.common.gpu(row_col_inds_linear == unique_inds_linear[i]))[:, 0]

            # indices on psf_patches, make sure it can be put on the canvas
            w_cut = psf_patches[curr_inds, psf_row_start[i]: psf_row_end[i], psf_col_start[i]: psf_col_end[i]]

            canvas[delta_inds[0][curr_inds],
                   canvas_row_start[i]:canvas_row_start[i] + w_cut.shape[1],
                   canvas_col_start[i]:canvas_col_start[i] + w_cut.shape[2]] += w_cut

        return canvas
