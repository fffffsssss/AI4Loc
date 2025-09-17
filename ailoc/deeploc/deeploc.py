import torch
import numpy as np
import time
import collections
import matplotlib.pyplot as plt
import datetime

import ailoc.common
import ailoc.simulation
import ailoc.deeploc


class DeepLoc(ailoc.common.XXLoc):
    """
    DeepLoc class, can only process spatially invariant PSF, its performance should be similar to DECODE.
    """

    def __init__(self, psf_params_dict, camera_params_dict, sampler_params_dict, attn_length=3):
        self.dict_psf_params, self.dict_camera_params, self.dict_sampler_params = \
            psf_params_dict, camera_params_dict, sampler_params_dict

        self._data_simulator = ailoc.simulation.Simulator(psf_params_dict, camera_params_dict, sampler_params_dict)
        self.scale_ph_offset = np.mean(self.dict_sampler_params['bg_range'])
        self.scale_ph_factor = self.dict_sampler_params['photon_range'][1]/50

        self.local_context = self.dict_sampler_params['local_context']
        # attn_length only useful when local_context=True, should be odd,
        # using the same number of frames before and after the target frame
        self.attn_length = attn_length
        assert self.attn_length % 2 == 1, 'attn_length should be odd'
        # add frames at the beginning and end to provide context
        self.context_size = sampler_params_dict['context_size'] + 2*(self.attn_length//2) if self.local_context else sampler_params_dict['context_size']
        self._network = ailoc.deeploc.DeepLocNet(self.local_context,
                                                 self.attn_length,
                                                 self.context_size,)

        self.evaluation_dataset = {}
        self.evaluation_recorder = self._init_recorder()

        self._iter_train = 0

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=6e-4, weight_decay=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

    @staticmethod
    def _init_recorder():
        recorder = {'loss': collections.OrderedDict(),  # loss function value
                    'iter_time': collections.OrderedDict(),  # time cost for each iteration
                    'n_per_img': collections.OrderedDict(),  # average summed probability channel per image
                    'recall': collections.OrderedDict(),  # TP/(TP+FN)
                    'precision': collections.OrderedDict(),  # TP/(TP+FP)
                    'jaccard': collections.OrderedDict(),  # TP/(TP+FP+FN)
                    'rmse_lat': collections.OrderedDict(),  # root of mean squared error
                    'rmse_ax': collections.OrderedDict(),
                    'rmse_vol': collections.OrderedDict(),
                    'jor': collections.OrderedDict(),  # 100*jaccard/rmse_lat
                    'eff_lat': collections.OrderedDict(),  # 100-np.sqrt((100-100*jaccard)**2+1**2*rmse_lat**2)
                    'eff_ax': collections.OrderedDict(),  # 100-np.sqrt((100-100*jaccard)**2+0.5**2*rmse_ax**2)
                    'eff_3d': collections.OrderedDict()  # (eff_lat+eff_ax)/2
                    }

        return recorder

    @property
    def network(self):
        return self._network

    @property
    def data_simulator(self):
        return self._data_simulator

    def compute_loss(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, p_gt, xyzph_array_gt, mask_array_gt, bg_gt):
        """
        Loss function.
        """

        count_loss = torch.mean(ailoc.deeploc.count_loss(p_pred, p_gt))
        loc_loss = torch.mean(ailoc.deeploc.loc_loss(p_pred, xyzph_pred, xyzph_sig_pred, xyzph_array_gt, mask_array_gt))
        sample_loss = torch.mean(ailoc.deeploc.sample_loss(p_pred, p_gt))
        bg_loss = torch.mean(ailoc.deeploc.bg_loss(bg_pred, bg_gt))

        total_loss = count_loss + loc_loss + sample_loss + bg_loss

        return total_loss

    def online_train(self, batch_size=1, max_iterations=50000, eval_freq=500, file_name=None):
        """
        Train the network with training data generated online.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt' if file_name is None else file_name

        print('Start training...')

        if self._iter_train > 0:
            print('training from checkpoint, the recent performance is:')
            self.print_recorder(max_iterations)

        while self._iter_train < max_iterations:
            t0 = time.time()
            total_loss = []
            for i in range(eval_freq):
                train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
                    self.data_simulator.sample_training_data(batch_size=batch_size,
                                                             context_size=self.context_size,
                                                             iter_train=self._iter_train)
                p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = self.unfold_target(p_map_gt,
                                                                                        xyzph_array_gt,
                                                                                        mask_array_gt,
                                                                                        bg_map_gt)
                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data,
                                                                             self.data_simulator.camera)
                loss = self.compute_loss(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred,
                                         p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.03, norm_type=2)
                self.optimizer.step()
                self.scheduler.step()
                self._iter_train += 1

                total_loss.append(ailoc.common.cpu(loss))

            avg_iter_time = 1000 * (time.time() - t0) / eval_freq
            avg_loss = np.mean(total_loss)
            self.evaluation_recorder['loss'][self._iter_train] = avg_loss
            self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time

            if self._iter_train > 1000:
                print('-' * 200)
                self.online_evaluate(batch_size=batch_size)

            self.print_recorder(max_iterations)
            self.save(file_name)

        print('training finished!')

    # def online_train_test_speed(self, batch_size=1, max_iterations=50000, eval_freq=500, file_name=None):
    #     """
    #     Train the network.
    #     #todo: this is temporary function to test the speed of each part of the training process
    #
    #     Args:
    #         batch_size (int): batch size
    #         max_iterations (int): maximum number of iterations
    #         eval_freq (int): every eval_freq iterations the network will be saved
    #             and evaluated on the evaluation dataset to check the current performance
    #         file_name (str): the name of the file to save the network
    #     """
    #
    #     file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepLoc.pt' if file_name is None else file_name
    #
    #     print('Start training...')
    #
    #     if self._iter_train > 0:
    #         print('training from checkpoint, the recent performance is:')
    #         self.print_recorder(max_iterations)
    #
    #     t_total_start = time.time()
    #     t_data_gen = 0
    #     t_forward_backward = 0
    #     t_evaluate = 0
    #
    #     while self._iter_train < max_iterations:
    #         t0 = time.time()
    #         total_loss = []
    #         for i in range(eval_freq):
    #             t_tmp_data_gen = time.time()
    #             train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
    #                 self.data_simulator.sample_training_data(batch_size=batch_size,
    #                                                          context_size=self.context_size,
    #                                                          iter_train=self._iter_train)
    #             p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = self.unfold_target(p_map_gt,
    #                                                                                     xyzph_array_gt,
    #                                                                                     mask_array_gt,
    #                                                                                     bg_map_gt)
    #             torch.cuda.synchronize()
    #             t_data_gen += time.time() - t_tmp_data_gen
    #
    #             t_tmp_bf = time.time()
    #             p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data,
    #                                                                          self.data_simulator.camera)
    #             loss = self.compute_loss(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred,
    #                                      p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.03, norm_type=2)
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             torch.cuda.synchronize()
    #             t_forward_backward += time.time() - t_tmp_bf
    #
    #             self._iter_train += 1
    #
    #             total_loss.append(ailoc.common.cpu(loss))
    #
    #         avg_iter_time = 1000 * (time.time() - t0) / eval_freq
    #         avg_loss = np.mean(total_loss)
    #         self.evaluation_recorder['loss'][self._iter_train] = avg_loss
    #         self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time
    #
    #         t_tmp_eval = time.time()
    #         if self._iter_train > 1000:
    #             print('-' * 200)
    #             self.online_evaluate(batch_size=batch_size)
    #
    #         self.print_recorder(max_iterations)
    #         self.save(file_name)
    #         torch.cuda.synchronize()
    #         t_evaluate += time.time() - t_tmp_eval
    #
    #     print('training finished!')
    #
    #     print(f'total time cost: {time.time() - t_total_start:.2f}s')
    #     print(f'total data generation time cost: {t_data_gen:.2f}s')
    #     print(f'total forward and backward time cost: {t_forward_backward:.2f}s')
    #     print(f'total evaluation+save time cost: {t_evaluate:.2f}s')

    # def offline_train_test_speed(self, batch_size=1, max_iterations=50000, eval_freq=500, file_name=None):
    #     """
    #     Train the network in an offline manner. The training data are generated before the network training.
    #
    #     Args:
    #         batch_size (int): batch size
    #         max_iterations (int): maximum number of iterations
    #         eval_freq (int): every eval_freq iterations the network will be saved
    #             and evaluated on the evaluation dataset to check the current performance
    #         file_name (str): the name of the file to save the network
    #     """
    #
    #     file_name = datetime.datetime.now().strftime(
    #         '%Y-%m-%d-%H-%M') + 'DeepLoc.pt' if file_name is None else file_name
    #
    #     print('Start training...')
    #
    #     if self._iter_train > 0:
    #         print('training from checkpoint, the recent performance is:')
    #         self.print_recorder(max_iterations)
    #
    #     t_total_start = time.monotonic()
    #     t_data_gen = 0
    #     t_sample = 0
    #     t_gen_noiseless = 0
    #     t_gen_psfs = 0
    #     t_place_psfs = 0
    #     t_camera_forward = 0
    #     t_transf_data = 0
    #     t_forward_backward = 0
    #     t_append_loss = 0
    #     t_remain = 0
    #
    #     # generate all training data first
    #     train_data_all = []
    #     p_map_gt_all = []
    #     xyzph_array_gt_all = []
    #     mask_array_gt_all = []
    #     bg_map_gt_all = []
    #     for i in range(max_iterations):
    #         t_tmp_data_gen = time.monotonic()
    #
    #         # train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
    #         #     self.data_simulator.sample_training_data(batch_size=batch_size,
    #         #                                              context_size=self.context_size,
    #         #                                              iter_train=i)
    #
    #         t_tmp_sample = time.monotonic()
    #         p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
    #             p_map_gt, xyzph_array_gt, mask_array_gt = self.data_simulator.mol_sampler.sample_for_train(batch_size,
    #                                                                                         self.context_size,
    #                                                                                         self.data_simulator.psf_model,
    #                                                                                         i, )
    #         torch.cuda.synchronize()
    #         t_sample += time.monotonic() - t_tmp_sample
    #
    #         # generate psf patches
    #         t_tmp_genpsf = time.monotonic()
    #         batch_size, channels, height, width = p_map_sample.shape[0], p_map_sample.shape[1], p_map_sample.shape[2], \
    #         p_map_sample.shape[3]
    #         bg = bg_map_sample * self.data_simulator.mol_sampler.bg_scale
    #         x, y, z, photons = self.data_simulator._translate_maps(p_map_sample.reshape([-1, height, width]),
    #                                                 xyzph_map_sample.reshape([4, -1, height, width]))
    #         with torch.no_grad():
    #             if isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFCUDA):
    #                 psf_patches = self.data_simulator.psf_model.simulate(x, y, z, photons, zernike_coefs) \
    #                     if zernike_coefs is not None else self.data_simulator.psf_model.simulate(x, y, z, photons)
    #             elif isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFTorch):
    #                 psf_patches = self.data_simulator.psf_model.simulate(x, y, z, photons, zernike_coefs=zernike_coefs)
    #             else:
    #                 raise NotImplementedError('PSF model not supported.')
    #         torch.cuda.synchronize()
    #         t_gen_psfs += time.monotonic() - t_tmp_genpsf
    #
    #         t_tmp_placepsf = time.monotonic()
    #         # put psfs on the canvas
    #         data = (self.data_simulator.place_psfs_v2(p_map_sample.reshape([-1, height, width]), psf_patches) +
    #                 bg.reshape([-1, height, width]))
    #         # data_origin = self.data_simulator.place_psfs(p_map_sample.reshape([-1, height, width]), psf_patches) + bg.reshape(
    #         #     [-1, height, width])
    #         # # print if data and data_origin are the same
    #         # if torch.all(data == data_origin):
    #         #     print('data and data_origin are the same')
    #         torch.cuda.synchronize()
    #         t_place_psfs += time.monotonic() - t_tmp_placepsf
    #
    #         # t_tmp_gen_noiseless = time.monotonic()
    #         # data = self.data_simulator.gen_noiseless_data(self.data_simulator.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)
    #         # torch.cuda.synchronize()
    #         # t_gen_noiseless += time.monotonic() - t_tmp_gen_noiseless
    #
    #         t_tmp_camera_forward = time.monotonic()
    #         train_data = self.data_simulator.camera.forward(data, curr_sub_fov_xy) \
    #             if isinstance(self.data_simulator.camera, ailoc.simulation.SCMOS) else self.data_simulator.camera.forward(data)
    #         bg_map_gt = bg_map_sample
    #         torch.cuda.synchronize()
    #         t_camera_forward += time.monotonic() - t_tmp_camera_forward
    #
    #
    #
    #         p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = self.unfold_target(p_map_gt,
    #                                                                                 xyzph_array_gt,
    #                                                                                 mask_array_gt,
    #                                                                                 bg_map_gt)
    #         train_data_all.append(ailoc.common.cpu(train_data))
    #         p_map_gt_all.append(ailoc.common.cpu(p_map_gt))
    #         xyzph_array_gt_all.append(ailoc.common.cpu(xyzph_array_gt))
    #         mask_array_gt_all.append(ailoc.common.cpu(mask_array_gt))
    #         bg_map_gt_all.append(ailoc.common.cpu(bg_map_gt))
    #         torch.cuda.synchronize()
    #         t_data_gen += time.monotonic() - t_tmp_data_gen
    #
    #     while self._iter_train < max_iterations:
    #         t0 = time.monotonic()
    #         total_loss = []
    #         for i in range(eval_freq):
    #             t_tmp_transf_data = time.monotonic()
    #             train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = ailoc.common.gpu(train_data_all[self._iter_train]), \
    #                 ailoc.common.gpu(p_map_gt_all[self._iter_train]), \
    #                 ailoc.common.gpu(xyzph_array_gt_all[self._iter_train]), \
    #                 ailoc.common.gpu(mask_array_gt_all[self._iter_train]), \
    #                 ailoc.common.gpu(bg_map_gt_all[self._iter_train])
    #             torch.cuda.synchronize()
    #             t_transf_data += time.monotonic() - t_tmp_transf_data
    #
    #             t_tmp_for_back = time.monotonic()
    #             p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data,
    #                                                                          self.data_simulator.camera)
    #             loss = self.compute_loss(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred,
    #                                      p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.03, norm_type=2)
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             self._iter_train += 1
    #             torch.cuda.synchronize()
    #             t_forward_backward += time.monotonic() - t_tmp_for_back
    #
    #             t_tmp_append_loss = time.monotonic()
    #             total_loss.append(loss.detach())
    #             torch.cuda.synchronize()
    #             t_append_loss += time.monotonic() - t_tmp_append_loss
    #
    #         t_tmp_remain = time.monotonic()
    #         avg_iter_time = 1000 * (time.monotonic() - t0) / eval_freq
    #         avg_loss = ailoc.common.cpu(torch.mean(torch.tensor(total_loss)))
    #         self.evaluation_recorder['loss'][self._iter_train] = avg_loss
    #         self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time
    #
    #         if self._iter_train > 1000:
    #             print('-' * 200)
    #             self.online_evaluate(batch_size=batch_size)
    #
    #         self.print_recorder(max_iterations)
    #         self.save(file_name)
    #
    #         t_remain += time.monotonic() - t_tmp_remain
    #
    #     print('training finished!')
    #
    #     print(f'total time cost: {time.monotonic() - t_total_start:.2f}s')
    #     print(f'total data generation time cost: {t_data_gen:.2f}s')
    #     print(f'total sampling time cost: {t_sample:.2f}s')
    #     print(f'total noiseless data generation time cost: {t_gen_noiseless:.2f}s')
    #     print(f'total PSF generation time cost: {t_gen_psfs:.2f}s')
    #     print(f'total place PSFs time cost: {t_place_psfs:.2f}s')
    #     print(f'total camera forward time cost: {t_camera_forward:.2f}s')
    #     print(f'total data transfer to gpu time cost: {t_transf_data:.2f}s')
    #     print(f'total forward and backward time cost: {t_forward_backward:.2f}s')
    #     print(f'total append loss time cost: {t_append_loss:.2f}s')
    #     print(f'total remain time cost: {t_remain:.2f}s')

    def unfold_target(self, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt):
        if self.local_context:
            extra_length = self.attn_length // 2
            if extra_length > 0:
                p_map_gt = p_map_gt[:, extra_length:-extra_length]
                xyzph_array_gt = xyzph_array_gt[:, extra_length:-extra_length]
                mask_array_gt = mask_array_gt[:, extra_length:-extra_length]
                bg_map_gt = bg_map_gt[:, extra_length:-extra_length]
        else:
            pass
        return p_map_gt.flatten(start_dim=0, end_dim=1), \
               xyzph_array_gt.flatten(start_dim=0, end_dim=1), \
               mask_array_gt.flatten(start_dim=0, end_dim=1), \
               bg_map_gt.flatten(start_dim=0, end_dim=1)

    def inference(self, data, camera):
        """
        Inference with the network, the input data should be transformed into photon unit,
        output are prediction maps that can be directly used for loss computation.

        Args:
            data (torch.Tensor): input data, shape (batch_size, optional local context, H, W)
            camera (ailoc.simulation.Camera): camera object used to transform the adu data to photon data
        """

        data_photon = camera.backward(data)
        data_scaled = (data_photon - self.scale_ph_offset)/self.scale_ph_factor
        p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.network(data_scaled)

        return p_pred, xyzph_pred, xyzph_sig_pred, bg_pred

    def post_process(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, return_infer_map=False):
        """
        Postprocess a batch of inference output map, output is GMM maps and molecule array
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty, z uncertainty,
        photon uncertainty, x_offset, y_offset].
        """

        # # old version, slower
        # inference_dict = {'prob': [], 'x_offset': [], 'y_offset': [], 'z_offset': [], 'photon': [],
        #                   'bg': [], 'x_sig': [], 'y_sig': [], 'z_sig': [], 'photon_sig': []}
        #
        # inference_dict['prob'].append(ailoc.common.cpu(p_pred))
        # inference_dict['x_offset'].append(ailoc.common.cpu(xyzph_pred[:, 0, :, :]))
        # inference_dict['y_offset'].append(ailoc.common.cpu(xyzph_pred[:, 1, :, :]))
        # inference_dict['z_offset'].append(ailoc.common.cpu(xyzph_pred[:, 2, :, :]))
        # inference_dict['photon'].append(ailoc.common.cpu(xyzph_pred[:, 3, :, :]))
        # inference_dict['x_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 0, :, :]))
        # inference_dict['y_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 1, :, :]))
        # inference_dict['z_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 2, :, :]))
        # inference_dict['photon_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 3, :, :]))
        # inference_dict['bg'].append(ailoc.common.cpu(bg_pred))
        #
        # for k in inference_dict.keys():
        #     inference_dict[k] = np.vstack(inference_dict[k])
        #
        # inference_dict['prob_sampled'] = None
        # inference_dict['bg_sampled'] = None
        #
        # molecule_array, inference_dict = ailoc.common.gmm_to_localizations_old(inference_dict=inference_dict,
        #                                                                    thre_integrated=0.7,
        #                                                                    pixel_size_xy=self.data_simulator.psf_model.pixel_size_xy,
        #                                                                    z_scale=self.data_simulator.mol_sampler.z_scale,
        #                                                                    photon_scale=self.data_simulator.mol_sampler.photon_scale,
        #                                                                    bg_scale=self.data_simulator.mol_sampler.bg_scale,
        #                                                                    batch_size=p_pred.shape[0])

        # new version, faster
        molecule_array, inference_dict = ailoc.common.gmm_to_localizations(p_pred=p_pred,
                                                                           xyzph_pred=xyzph_pred,
                                                                           xyzph_sig_pred=xyzph_sig_pred,
                                                                           bg_pred=bg_pred,
                                                                           thre_integrated=0.7,
                                                                           pixel_size_xy=self.data_simulator.psf_model.pixel_size_xy,
                                                                           z_scale=self.data_simulator.mol_sampler.z_scale,
                                                                           photon_scale=self.data_simulator.mol_sampler.photon_scale,
                                                                           bg_scale=self.data_simulator.mol_sampler.bg_scale,
                                                                           batch_size=p_pred.shape[0],
                                                                           return_infer_map=return_infer_map)

        return molecule_array, inference_dict

    def analyze(self, data, camera, sub_fov_xy=None, return_infer_map=False):
        """
        Wrap the inference and post_process function, receive a batch of data and return the molecule list.

        Args:
            data (torch.Tensor): a batch of data to be analyzed.
            camera (ailoc.simulation.Camera): camera object used to transform the data to photon unit.
            sub_fov_xy (tuple of int): (x_start, x_end, y_start, y_end), start from 0, in pixel unit,
                the FOV indicator for these images
            return_infer_map (bool): whether to return the prediction maps, which may occupy some memory
                and take more time during data analysis

        Returns:
            (np.ndarray, dict): molecule array, [frame, x, y, z, photon, integrated prob, x uncertainty,
                y uncertainty, z uncertainty, photon uncertainty...], the xy position are relative
                to the current image size, may need to be translated outside this function, the second output
                is a dict that contains the inferred multichannel maps from the network.
        """

        self.network.eval()
        with torch.no_grad():
            p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(data, camera)
            molecule_array, inference_dict = self.post_process(p_pred,
                                                               xyzph_pred,
                                                               xyzph_sig_pred,
                                                               bg_pred,
                                                               return_infer_map)

        return molecule_array, inference_dict

    def online_evaluate(self, batch_size):
        """
        Evaluate the network during training using the validation dataset.
        """

        self.network.eval()
        with torch.no_grad():
            print('evaluating...')
            t0 = time.time()

            n_per_img = []
            molecule_list_pred = []
            for i in range(int(np.ceil(self.evaluation_dataset['data'].shape[0]/batch_size))):
                molecule_array_tmp, inference_dict_tmp = \
                    self.analyze(
                        ailoc.common.gpu(self.evaluation_dataset['data'][i * batch_size: (i + 1) * batch_size]),
                        self.data_simulator.camera,
                        return_infer_map=True)

                n_per_img.append(inference_dict_tmp['prob'].sum((-2, -1)).mean())

                if len(molecule_array_tmp) > 0:
                    molecule_array_tmp[:, 0] += i * batch_size * (self.context_size-2*(self.attn_length//2) if self.local_context else self.context_size)
                    molecule_list_pred += molecule_array_tmp.tolist()

            metric_dict, paired_array = ailoc.common.pair_localizations(prediction=np.array(molecule_list_pred),
                                                                        ground_truth=self.evaluation_dataset['molecule_list_gt'],
                                                                        frame_num=self.evaluation_dataset['data'].shape[0] * (self.context_size-2*(self.attn_length//2) if self.local_context else self.context_size),
                                                                        fov_xy_nm=(0, self.evaluation_dataset['data'].shape[-1]*self.data_simulator.psf_model.pixel_size_xy[0],
                                                                                   0, self.evaluation_dataset['data'].shape[-2]*self.data_simulator.psf_model.pixel_size_xy[1]))

            for k in self.evaluation_recorder.keys():
                if k in metric_dict.keys():
                    self.evaluation_recorder[k][self._iter_train] = metric_dict[k]

            self.evaluation_recorder['n_per_img'][self._iter_train] = np.mean(n_per_img)

            print(f'evaluating done! time cost: {time.time() - t0:.2f}s')

        self.network.train()

    def build_evaluation_dataset(self, napari_plot=False):
        """
        Build the evaluation dataset, sampled by the same way as training data.
        """

        print("building evaluation dataset, this may take a while...")
        t0 = time.time()
        eval_data, molecule_list_gt, sub_fov_xy_list = \
            self.data_simulator.sample_evaluation_data(batch_size=self.dict_sampler_params['eval_batch_size'],
                                                       context_size=self.context_size,)
        self.evaluation_dataset['data'] = ailoc.common.cpu(eval_data)

        molecule_list_gt = np.array(molecule_list_gt)
        if self.local_context:
            molecule_list_gt_corrected = []
            for i in range(self.dict_sampler_params['eval_batch_size']):
                curr_context_idx = np.where((molecule_list_gt[:, 0]>i*self.context_size+(self.attn_length//2))&
                                            (molecule_list_gt[:, 0]<(i+1)*self.context_size-(self.attn_length//2)+1))
                curr_molecule_list_gt = molecule_list_gt[curr_context_idx]
                curr_molecule_list_gt[:, 0] -= ((2*i+1) * (self.attn_length//2))
                molecule_list_gt_corrected.append(curr_molecule_list_gt)
            molecule_list_gt = np.concatenate(molecule_list_gt_corrected, axis=0)

        self.evaluation_dataset['molecule_list_gt'] = molecule_list_gt
        self.evaluation_dataset['sub_fov_xy_list'] = sub_fov_xy_list
        print(f"evaluation dataset with shape {eval_data.shape} building done! "
              f"contain {len(molecule_list_gt)} target molecules, "
              f"time cost: {time.time() - t0:.2f}s")

        if napari_plot:
            print('visually checking evaluation data...')
            ailoc.common.viewdata_napari(eval_data)

    def save(self, file_name):
        """
        Save the whole DeepLoc instance, including the network, optimizer, recorder, etc.
        """

        with open(file_name, 'wb') as f:
            torch.save(self, f)
        print(f"DeepLoc instance saved to {file_name}")

    def check_training_psf(self, num_z_step=21):
        """
        Check the PSF.
        """

        print(f"checking PSF...")
        x = ailoc.common.gpu(torch.zeros(num_z_step))
        y = ailoc.common.gpu(torch.zeros(num_z_step))
        z = ailoc.common.gpu(torch.linspace(*self.data_simulator.mol_sampler.z_range, num_z_step))
        photons = ailoc.common.gpu(torch.ones(num_z_step))

        psf = ailoc.common.cpu(self.data_simulator.psf_model.simulate(x, y, z, photons))

        plt.figure(constrained_layout=True)
        for j in range(num_z_step):
            plt.subplot(int(np.ceil(num_z_step/7)), 7, j + 1)
            plt.imshow(psf[j], cmap='gray')
            plt.title(f"{ailoc.common.cpu(z[j]):.0f} nm")
        plt.show()

    def check_training_data(self):
        """
        Check the training data ,randomly sample a batch of training data and visualize it.
        """

        print(f"checking training data...")
        data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy = \
            self.data_simulator.sample_training_data(batch_size=1,
                                                     context_size=self.context_size,
                                                     iter_train=0, )

        cmap = 'gray'

        im_num = self.context_size * 2
        n_col = 4
        n_row = int(np.ceil(im_num / n_col))

        fig, ax_arr = plt.subplots(n_row,
                                   n_col,
                                   figsize=(n_col * 3,
                                            n_row * 2),
                                   constrained_layout=True)
        ax = []
        plts = []
        for i in ax_arr:
            try:
                for j in i:
                    ax.append(j)
            except:
                ax.append(i)

        for i in range(self.context_size):
            plts.append(ax[i].imshow(ailoc.common.cpu(data_cam)[0, i], cmap=cmap))
            ax[i].set_title(f"frame {i}")
            plt.colorbar(mappable=plts[-1], ax=ax[i], fraction=0.046, pad=0.04)

        for i in range(self.context_size):
            ax_num = i + self.context_size
            plts.append(ax[ax_num].imshow(ailoc.common.cpu(data_cam)[0, i], cmap=cmap))
            ax[ax_num].set_title(f"GT frame {i}")
            pix_gt = ailoc.common.cpu(p_map_gt[0, i].nonzero())
            ax[ax_num].scatter(pix_gt[:, 1], pix_gt[:, 0], s=5, c='m', marker='x')
            plt.colorbar(mappable=plts[-1], ax=ax[ax_num], fraction=0.046, pad=0.04)

        plt.show()

    def print_recorder(self, max_iterations):
        try:
            print(f"Iterations: {self._iter_train}/{max_iterations} || "
                  f"Loss: {self.evaluation_recorder['loss'][self._iter_train]:.2f} || "
                  f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                  f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train] * (max_iterations - self._iter_train) / 3600000:.2f} h || ",
                  end='')

            print(f"SumProb: {self.evaluation_recorder['n_per_img'][self._iter_train]:.2f} || "
                  f"Eff_3D: {self.evaluation_recorder['eff_3d'][self._iter_train]:.2f} || "
                  f"Jaccard: {self.evaluation_recorder['jaccard'][self._iter_train]:.2f} || "
                  f"Recall: {self.evaluation_recorder['recall'][self._iter_train]:.2f} || "
                  f"Precision: {self.evaluation_recorder['precision'][self._iter_train]:.2f} || "
                  f"RMSE_lat: {self.evaluation_recorder['rmse_lat'][self._iter_train]:.2f} || "
                  f"RMSE_ax: {self.evaluation_recorder['rmse_ax'][self._iter_train]:.2f}")

        except KeyError:
            print('No recent performance record found')

    def remove_gpu_attribute(self):
        """
        Remove the gpu attribute of the loc model, so that can be shared between processes.
        """

        self._network.to('cpu')
        self.optimizer = None
        self.scheduler = None
        self.evaluation_dataset = None
        self._data_simulator = None
