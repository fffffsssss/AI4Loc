import torch
import numpy as np
import time
import collections
import matplotlib.pyplot as plt
import datetime

import ailoc.common
import ailoc.simulation
import ailoc.liteloc


class LiteLoc(ailoc.common.XXLoc):
    """
    LiteLoc class, can only process spatially invariant PSF, its performance should be similar to DECODE.
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
        self.context_size = sampler_params_dict['context_size'] + 2 * (self.attn_length // 2) if self.local_context else \
        sampler_params_dict['context_size']
        self._network = ailoc.liteloc.LiteLocNet(self.local_context,
                                                 self.attn_length,
                                                 self.context_size,).cuda()
        self.network.get_parameter_number()

        self.evaluation_dataset = {}
        self.evaluation_recorder = self._init_recorder()

        self._iter_train = 0

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.NAdam(self.network.parameters(),
                                           lr=8e-4,
                                           betas=(0.8, 0.8888),
                                           eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.85)

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

        count_loss = torch.mean(ailoc.liteloc.count_loss(p_pred, p_gt))
        loc_loss = torch.mean(ailoc.liteloc.loc_loss(p_pred, xyzph_pred, xyzph_sig_pred, xyzph_array_gt, mask_array_gt))
        sample_loss = torch.mean(ailoc.liteloc.sample_loss(p_pred, p_gt))
        bg_loss = 0

        total_loss = count_loss + loc_loss + sample_loss + bg_loss

        return total_loss

    def online_train(self, batch_size=1, max_iterations=50000, eval_freq=500, file_name=None):
        """
        Train the network.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LiteLoc.pt' if file_name is None else file_name

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
                print('----------------------------------------------------------------------------------------------')
                self.online_evaluate(batch_size=batch_size)

            self.print_recorder(max_iterations)
            self.save(file_name)

        print('training finished!')

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
        photon uncertainty, x_offset_pixel, y_offset_pixel].
        """

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
            for i in range(int(np.ceil(self.evaluation_dataset['data'].shape[0] / batch_size))):
                molecule_array_tmp, inference_dict_tmp = \
                    self.analyze(
                        ailoc.common.gpu(self.evaluation_dataset['data'][i * batch_size: (i + 1) * batch_size]),
                        self.data_simulator.camera,
                        return_infer_map=True)

                n_per_img.append(inference_dict_tmp['prob'].sum((-2, -1)).mean())

                if len(molecule_array_tmp) > 0:
                    molecule_array_tmp[:, 0] += i * batch_size * (
                        self.context_size - 2 * (self.attn_length // 2) if self.local_context else self.context_size)
                    molecule_list_pred += molecule_array_tmp.tolist()

            metric_dict, paired_array = ailoc.common.pair_localizations(prediction=np.array(molecule_list_pred),
                                                                        ground_truth=self.evaluation_dataset['molecule_list_gt'],
                                                                        frame_num=self.evaluation_dataset['data'].shape[0] * (self.context_size - 2 * (self.attn_length // 2) if self.local_context else self.context_size),
                                                                        fov_xy_nm=(0, self.evaluation_dataset['data'].shape[-1] *self.data_simulator.psf_model.pixel_size_xy[0],0, self.evaluation_dataset['data'].shape[-2] *self.data_simulator.psf_model.pixel_size_xy[1]))

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
                                                       context_size=self.context_size, )
        self.evaluation_dataset['data'] = ailoc.common.cpu(eval_data)

        molecule_list_gt = np.array(molecule_list_gt)
        if self.local_context:
            molecule_list_gt_corrected = []
            for i in range(self.dict_sampler_params['eval_batch_size']):
                curr_context_idx = np.where((molecule_list_gt[:, 0] > i * self.context_size + (self.attn_length // 2)) &
                                            (molecule_list_gt[:, 0] < (i + 1) * self.context_size - (self.attn_length // 2) + 1))
                curr_molecule_list_gt = molecule_list_gt[curr_context_idx]
                curr_molecule_list_gt[:, 0] -= ((2 * i + 1) * (self.attn_length // 2))
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
        Save the whole LiteLoc instance, including the network, optimizer, recorder, etc.
        """

        with open(file_name, 'wb') as f:
            torch.save(self, f)
        print(f"LiteLoc instance saved to {file_name}")

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

        fig, ax_arr = plt.subplots(int(np.ceil(self.context_size / 2)), 2,
                                   figsize=(3 * 2, 2 * int(np.ceil(self.context_size / 2))),
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
            pix_gt = ailoc.common.cpu(p_map_gt[0, i].nonzero())
            ax[i].scatter(pix_gt[:, 1], pix_gt[:, 0], s=5, c='m', marker='x')

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
