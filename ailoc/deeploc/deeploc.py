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

    def __init__(self, psf_params_dict, camera_params_dict, sampler_params_dict):
        self.dict_psf_params, self.dict_camera_params, self.dict_sampler_params = \
            psf_params_dict, camera_params_dict, sampler_params_dict

        self._data_simulator = ailoc.simulation.Simulator(psf_params_dict, camera_params_dict, sampler_params_dict)
        self.scale_ph_offset = np.mean(self.dict_sampler_params['bg_range'])
        self.scale_ph_factor = self.dict_sampler_params['photon_range'][1]/50

        self.local_context = self.dict_sampler_params['local_context']
        self._network = ailoc.deeploc.DeepLocNet(self.local_context)

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

    def online_train(self, batch_size=16, max_iterations=50000, eval_freq=500, file_name=None):
        """
        Train the network.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H') + 'DeepLoc.pt' if file_name is None else file_name

        print('Start training...')

        if self._iter_train > 0:
            print('training from checkpoint, the recent performance is:')
            try:
                print(f"Iterations: {self._iter_train}/{max_iterations} || "
                      f"Loss: {self.evaluation_recorder['loss'][self._iter_train]:.2f} || "
                      f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                      f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train]*(max_iterations-self._iter_train)/3600000:.2f} h || ", end='')

                print(f"SumProb: {self.evaluation_recorder['n_per_img'][self._iter_train]:.2f} || "
                      f"Eff_3D: {self.evaluation_recorder['eff_3d'][self._iter_train]:.2f} || "
                      f"Jaccard: {self.evaluation_recorder['jaccard'][self._iter_train]:.2f} || "
                      f"Recall: {self.evaluation_recorder['recall'][self._iter_train]:.2f} || "
                      f"Precision: {self.evaluation_recorder['precision'][self._iter_train]:.2f} || "
                      f"RMSE_lat: {self.evaluation_recorder['rmse_lat'][self._iter_train]:.2f} || "
                      f"RMSE_ax: {self.evaluation_recorder['rmse_ax'][self._iter_train]:.2f}")

            except KeyError:
                print('No recent performance record found')

        while self._iter_train < max_iterations:
            t0 = time.time()
            total_loss = []
            for i in range(eval_freq):
                train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
                    self.data_simulator.sample_training_data(batch_size=batch_size, iter_train=self._iter_train)
                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data, self.data_simulator.camera)
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
                print(f"Iterations: {self._iter_train}/{max_iterations} || "
                      f"Loss: {self.evaluation_recorder['loss'][self._iter_train]:.2f} || "
                      f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                      f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train]*(max_iterations-self._iter_train)/3600000:.2f} h || "
                      f"SumProb: {self.evaluation_recorder['n_per_img'][self._iter_train]:.2f} || "
                      f"Eff_3D: {self.evaluation_recorder['eff_3d'][self._iter_train]:.2f} || "
                      f"Jaccard: {self.evaluation_recorder['jaccard'][self._iter_train]:.2f} || "
                      f"Recall: {self.evaluation_recorder['recall'][self._iter_train]:.2f} || "
                      f"Precision: {self.evaluation_recorder['precision'][self._iter_train]:.2f} || "
                      f"RMSE_lat: {self.evaluation_recorder['rmse_lat'][self._iter_train]:.2f} || "
                      f"RMSE_ax: {self.evaluation_recorder['rmse_ax'][self._iter_train]:.2f}")
            else:
                print(f"Iterations: {self._iter_train}/{max_iterations} || "
                      f"Loss: {self.evaluation_recorder['loss'][self._iter_train]:.2f} || "
                      f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                      f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train]*(max_iterations-self._iter_train)/3600000:.2f} h")

            self.save(file_name)

        print('training finished!')

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

    def post_process(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred):
        """
        Postprocess a batch of inference output map, output is GMM maps and molecule array
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty, z uncertainty,
        photon uncertainty, x_offset_pixel, y_offset_pixel].
        """

        inference_dict = {'prob': [], 'x_offset': [], 'y_offset': [], 'z_offset': [], 'photon': [],
                          'bg': [], 'x_sig': [], 'y_sig': [], 'z_sig': [], 'photon_sig': []}

        inference_dict['prob'].append(ailoc.common.cpu(p_pred))
        inference_dict['x_offset'].append(ailoc.common.cpu(xyzph_pred[:, 0, :, :]))
        inference_dict['y_offset'].append(ailoc.common.cpu(xyzph_pred[:, 1, :, :]))
        inference_dict['z_offset'].append(ailoc.common.cpu(xyzph_pred[:, 2, :, :]))
        inference_dict['photon'].append(ailoc.common.cpu(xyzph_pred[:, 3, :, :]))
        inference_dict['x_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 0, :, :]))
        inference_dict['y_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 1, :, :]))
        inference_dict['z_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 2, :, :]))
        inference_dict['photon_sig'].append(ailoc.common.cpu(xyzph_sig_pred[:, 3, :, :]))
        inference_dict['bg'].append(ailoc.common.cpu(bg_pred))

        for k in inference_dict.keys():
            inference_dict[k] = np.vstack(inference_dict[k])

        inference_dict['prob_sampled'] = None
        inference_dict['bg_sampled'] = None

        molecule_array,inference_dict = ailoc.common.gmm_to_localizations(inference_dict=inference_dict,
                                                                          thre_integrated=0.7,
                                                                          pixel_size_xy=self.data_simulator.psf_model.pixel_size_xy,
                                                                          z_scale=self.data_simulator.mol_sampler.z_scale,
                                                                          photon_scale=self.data_simulator.mol_sampler.photon_scale,
                                                                          bg_scale=self.data_simulator.mol_sampler.bg_scale,
                                                                          batch_size=p_pred.shape[0])

        return molecule_array, inference_dict

    def analyze(self, data, camera, sub_fov_xy=None):
        """
        Wrap the inference and post_process function, receive a batch of data and return the molecule list.

        Args:
            data (torch.Tensor): a batch of data to be analyzed.
            camera (ailoc.simulation.Camera): camera object used to transform the data to photon unit.
            sub_fov_xy (tuple of int): (x_start, x_end, y_start, y_end), start from 0, in pixel unit,
                the FOV indicator for these images

        Returns:
            (np.ndarray, dict): molecule array, [frame, x, y, z, photon, integrated prob, x uncertainty,
                y uncertainty, z uncertainty, photon uncertainty...], the xy position are relative
                to the current image size, may need to be translated outside this function, the second output
                is a dict that contains the inferred multichannel maps from the network.
        """

        p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(data, camera)
        molecule_array, inference_dict = self.post_process(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred)

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
                    self.analyze(ailoc.common.gpu(self.evaluation_dataset['data'][i*batch_size: (i+1)*batch_size]),
                                 self.data_simulator.camera)

                n_per_img.append(inference_dict_tmp['prob'].sum((-2, -1)).mean())

                molecule_array_tmp[:, 0] += i*batch_size
                molecule_list_pred += molecule_array_tmp.tolist()

            metric_dict, paired_array = ailoc.common.pair_localizations(prediction=np.array(molecule_list_pred),
                                                                        ground_truth=self.evaluation_dataset['molecule_list_gt'],
                                                                        frame_num=self.evaluation_dataset['data'].shape[0],
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
            self.data_simulator.sample_evaluation_data(num_image=self.dict_sampler_params['num_evaluation_data'])
        self.evaluation_dataset['data'] = ailoc.common.cpu(eval_data)
        self.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt)
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
            self.data_simulator.sample_training_data(batch_size=1, iter_train=0)

        cmap = 'gray'

        if self.data_simulator.mol_sampler.local_context:
            fig, ax = plt.subplots(2, 2, constrained_layout=True)
            img_tmp = ax[0, 0].imshow(ailoc.common.cpu(data_cam)[0, 0], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[0, 0], fraction=0.046, pad=0.04)
            ax[0, 0].set_title('last frame')

            img_tmp = ax[0, 1].imshow(ailoc.common.cpu(data_cam)[0, 1], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[0, 1], fraction=0.046, pad=0.04)
            ax[0, 1].set_title('middle frame')

            img_tmp = ax[1, 0].imshow(ailoc.common.cpu(data_cam)[0, 2], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[1, 0], fraction=0.046, pad=0.04)
            ax[1, 0].set_title('next frame')

            img_tmp = ax[1, 1].imshow(ailoc.common.cpu(data_cam)[0, 1], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[1, 1], fraction=0.046, pad=0.04)
            pix_gt = ailoc.common.cpu(p_map_gt[0].nonzero())
            ax[1, 1].scatter(pix_gt[:, 1], pix_gt[:, 0], s=10, c='m', marker='x')
            ax[1, 1].set_title('ground truth \non middle frame')

            plt.show()
        else:
            fig, ax = plt.subplots(1, 2, constrained_layout=True)
            img_tmp = ax[0].imshow(ailoc.common.cpu(data_cam)[0, 0], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[0], fraction=0.046, pad=0.04)
            ax[0].set_title('frame')

            img_tmp = ax[1].imshow(ailoc.common.cpu(data_cam)[0, 0], cmap=cmap)
            plt.colorbar(mappable=img_tmp, ax=ax[1], fraction=0.046, pad=0.04)
            pix_gt = ailoc.common.cpu(p_map_gt[0].nonzero())
            ax[1].scatter(pix_gt[:, 1], pix_gt[:, 0], s=10, c='m', marker='x')
            ax[1].set_title('ground truth \non middle frame')
            plt.show()
