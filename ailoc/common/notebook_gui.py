import csv

import deprecated
import numpy as np
from ipywidgets import widgets
from tkinter import Tk, filedialog
import tifffile
import stackview
from IPython.display import display
import scipy.io as scio
import torch
import os
import copy

import ailoc.common


# gui for jupyter notebook
class SelectFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, nofile_description='Select files'):
        super(SelectFilesButton, self).__init__()
        # Add the selected_files trait
        # self.add_traits(files=traitlets.traitlets.List())
        self.files = None
        self.nofile_description = nofile_description

        # Create the button.
        self.description = self.nofile_description
        self.icon = "square-o"
        self.style.button_color = "orange"
        self.layout = widgets.Layout(width='100%', height='80px')
        # Set on click behavior.
        self.on_click(self.select_files)

    def select_files(self, b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        # self.files = filedialog.askopenfilename(multiple=True)
        self.files = filedialog.askopenfilename(multiple=False)

        if self.files == '':
            self.description = self.nofile_description
            self.icon = "square-o"
            self.style.button_color = "orange"
        else:
            self.description = "Files selected: " + str(self.files)
            self.icon = "check-square-o"
            b.style.button_color = "Salmon"


class SelectFolderButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""
    def __init__(self, nofile_description='Select folder'):
        super().__init__()
        # Add the selected_files trait
        # self.add_traits(files=traitlets.traitlets.List())
        self.files = None
        self.nofile_description = nofile_description

        # Create the button.
        self.description = self.nofile_description
        self.icon = "square-o"
        self.style.button_color = "orange"
        self.layout = widgets.Layout(width='100%', height='80px')
        # Set on click behavior.
        self.on_click(self.select_folder)

    def select_folder(self, b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        # self.files = filedialog.askopenfilename(multiple=True)
        self.files = filedialog.askdirectory(mustexist=True,)

        if self.files == '':
            self.description = self.nofile_description
            self.icon = "square-o"
            self.style.button_color = "orange"
        else:
            self.description = "Folder selected: " + str(self.files)
            self.icon = "check-square-o"
            b.style.button_color = "Salmon"


class SaveFilesButton(widgets.Button):
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self,
                 nofile_description='Save files',
                 initialdir="/",
                 initialfile="results",
                 # defaultextension='.csv',
                 filetypes=[('csv', '*.csv')]):
        super().__init__()
        self.files = None
        self.nofile_description = nofile_description
        self.initialdir = initialdir
        self.initialfile = initialfile
        # self.defaultextension = defaultextension
        self.filetypes = filetypes

        # Create the button.
        self.description = self.nofile_description
        self.icon = "square-o"
        self.style.button_color = "orange"
        self.layout = widgets.Layout(width='100%', height='80px')
        # Set on click behavior.
        self.on_click(self.save_files)

    def save_files(self, b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)

        self.files = filedialog.asksaveasfilename(initialdir=self.initialdir,
                                                  initialfile=self.initialfile,
                                                  # defaultextension=self.defaultextension,
                                                  filetypes=self.filetypes)

        if self.files == '':
            self.description = self.nofile_description
            self.icon = "square-o"
            self.style.button_color = "orange"
        else:
            self.description = "Save files: " + str(self.files)
            self.icon = "check-square-o"
            b.style.button_color = "Salmon"


class SelectFilesButtonShow:
    """A file widget that leverages tkinter.filedialog."""

    def __init__(self, nofile_description='Select files'):
        self.nofile_description = nofile_description

        self.button = widgets.Button(description=self.nofile_description,
                                     icon="square-o",
                                     layout=widgets.Layout(width='100%', height='80px'))
        self.button.style.button_color = "orange"
        # Set on click behavior.
        self.button.on_click(self.select_files)

        self.slice_show_widget = widgets.GridspecLayout(1, 1)

        self.files = ''

    def select_files(self, b):
        """Generate instance of tkinter.filedialog.

        Parameters
        ----------
        b : obj:
            An instance of ipywidgets.widgets.Button
        """
        # Create Tk root
        root = Tk()
        # Hide the main window
        root.withdraw()
        # Raise the root to the top of all windows.
        root.call('wm', 'attributes', '.', '-topmost', True)
        # List of selected fileswill be set to b.value
        # self.files = filedialog.askopenfilename(multiple=True)
        self.files = filedialog.askopenfilename(multiple=False)

        if self.files == '':
            self.button.description = self.nofile_description
            self.button.icon = "square-o"
            self.button.style.button_color = "orange"
            self.slice_show_widget[0, 0] = widgets.Label(value="No images to show")
        else:
            self.button.description = "Files selected: " + str(self.files)
            self.button.icon = "check-square-o"
            self.button.style.button_color = "Salmon"
            self.slice_show_widget[0, 0] = stackview.slice(tifffile.imread(self.files),
                                                           colormap='turbo',
                                                           continuous_update=True)

    def display_notebook_gui(self):
        display(self.button)
        display(self.slice_show_widget)


@deprecated.deprecated(reason="using BeadsCalibrationWidget instead")
class ZernikeFitParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(12, 2)
        # create widgets
        self.peak_finding_label = widgets.Label(value='Peak finding parameters', width='100%')
        self.filter_size_receiver = widgets.BoundedIntText(description='Peak filter:',
                                                           value=2,
                                                           step=1,
                                                           max=10,
                                                           min=1)
        self.minimum_distance_receiver = widgets.BoundedIntText(description='Min dist:',
                                                                value=25,
                                                                step=1,
                                                                max=100,
                                                                min=1,
                                                                width='100%')
        self.dz_receiver = widgets.BoundedFloatText(description='Z step:',
                                                    value=10,
                                                    step=10,
                                                    max=10000,
                                                    min=1)
        self.cutoff_receiver = widgets.BoundedFloatText(description='Cutoff:',
                                                        value=2,
                                                        step=0.1,
                                                        max=10,
                                                        min=0.1)
        self.roi_size_receiver = widgets.BoundedIntText(description='ROI size:',
                                                        value=27,
                                                        step=1,
                                                        max=100,
                                                        min=1)
        self.zernike_fit_label = widgets.Label(value='Zernike fitting parameters', width='100%')
        self.na_receiver = widgets.BoundedFloatText(description='NA:',
                                                    value=1.5,
                                                    step=0.1)
        self.wavelength_receiver = widgets.BoundedFloatText(description='Wavelength:',
                                                            value=660,
                                                            step=10,
                                                            max=1000,
                                                            min=400)
        self.refmed_receiver = widgets.BoundedFloatText(description='RI medium:',
                                                        value=1.518,
                                                        step=0.1)
        self.refcov_receiver = widgets.BoundedFloatText(description='RI coverslip:',
                                                        value=1.524,
                                                        step=0.1)
        self.refimm_receiver = widgets.BoundedFloatText(description='RI oil:', value=1.518,
                                                        step=0.1)
        self.otf_rescale_receiver = widgets.BoundedFloatText(description='OTF rescale:',
                                                             value=0.8,
                                                             step=0.1,
                                                             max=2,
                                                             min=0.01)
        self.pixelsizex_receiver = widgets.BoundedFloatText(description='PixelsizeX:',
                                                            value=110,
                                                            step=1,
                                                            max=1000,
                                                            min=1)
        self.pixelsizey_receiver = widgets.BoundedFloatText(description='PixelsizeY:',
                                                            value=110,
                                                            step=1,
                                                            max=1000,
                                                            min=1)
        self.iterations_receiver = widgets.BoundedIntText(description='Iterations:',
                                                          value=75,
                                                          step=1,
                                                          max=1000,
                                                          min=1)
        # Create a Button widget
        self.set_button = widgets.Button(description='Set parameters', width='100%', height='80px')
        self.set_button.on_click(self.set_zernike_calibration_params)
        self.output = widgets.Output()

        # layout
        self[0, :] = self.peak_finding_label
        self[1, 0] = self.filter_size_receiver
        self[1, 1] = self.minimum_distance_receiver
        self[2, 0] = self.dz_receiver
        self[2, 1] = self.cutoff_receiver
        self[3, 0] = self.roi_size_receiver
        self[4, :] = self.zernike_fit_label
        self[5, 0] = self.na_receiver
        self[5, 1] = self.wavelength_receiver
        self[6, 0] = self.refmed_receiver
        self[6, 1] = self.refcov_receiver
        self[7, 0] = self.refimm_receiver
        self[7, 1] = self.otf_rescale_receiver
        self[8, 0] = self.pixelsizex_receiver
        self[8, 1] = self.pixelsizey_receiver
        self[9, 0] = self.iterations_receiver
        self[10, :] = self.set_button

        self.params_dict = {}

    def set_zernike_calibration_params(self, b):
        self.params_dict['filtersize'] = self.filter_size_receiver.value
        self.params_dict['mindistance'] = self.minimum_distance_receiver.value
        self.params_dict['dz'] = self.dz_receiver.value
        self.params_dict['relative_cutoff'] = self.cutoff_receiver.value
        self.params_dict['ROIxy'] = self.roi_size_receiver.value

        self.params_dict['na'] = self.na_receiver.value
        self.params_dict['lambda'] = self.wavelength_receiver.value
        self.params_dict['refmed'] = self.refmed_receiver.value
        self.params_dict['refcov'] = self.refcov_receiver.value
        self.params_dict['refimm'] = self.refimm_receiver.value
        self.params_dict['otf_rescale'] = self.otf_rescale_receiver.value
        self.params_dict['pixelSizeX'] = self.pixelsizex_receiver.value
        self.params_dict['pixelSizeY'] = self.pixelsizey_receiver.value
        self.params_dict['iterations'] = self.iterations_receiver.value

        self.params_dict['imageRoi'] = np.zeros(2)
        self.params_dict['xrange'] = np.array([-np.inf, np.inf])
        self.params_dict['yrange'] = np.array([-np.inf, np.inf])
        self.params_dict['modality'] = 'arbitrary'
        self.params_dict['zcorr'] = 'cross-correlation'
        self.params_dict['zcorrframes'] = 50
        self.params_dict['smoothz'] = 1
        self.params_dict['smoothxy'] = 0
        self.params_dict['emgain'] = 0
        self.params_dict['use_zernike_fit'] = True

        self.output.clear_output()
        with self.output:
            print(self.params_dict)


class CalibParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(3, 2)
        # create widgets
        self.z_step_receiver = widgets.BoundedFloatText(description='Z step:',
                                                        value=10,
                                                        step=10,
                                                        max=10000,
                                                        min=1)
        self.filter_sigma_receiver = widgets.BoundedIntText(description='Peak filter:',
                                                            value=3,
                                                            step=1,
                                                            max=10,
                                                            min=1)
        self.threshold_receiver = widgets.BoundedFloatText(description='Cutoff:',
                                                           value=20,
                                                           step=1,
                                                           max=100,
                                                           min=0)
        self.fit_brightest_receiver = widgets.Checkbox(description='Fit brightest',
                                                       value=True)
        # layout
        self[0, :] = widgets.Label(value='Calibration parameters')
        self[1, 0] = self.z_step_receiver
        self[1, 1] = self.filter_sigma_receiver
        self[2, 0] = self.threshold_receiver
        self[2, 1] = self.fit_brightest_receiver

        self.calib_params_dict = {}

    def get_calib_params(self):
        self.calib_params_dict['z_step'] = self.z_step_receiver.value
        self.calib_params_dict['filter_sigma'] = self.filter_sigma_receiver.value
        self.calib_params_dict['threshold_abs'] = self.threshold_receiver.value
        self.calib_params_dict['fit_brightest'] = self.fit_brightest_receiver.value

        return self.calib_params_dict

    def display_notebook_gui(self):
        display(self)


class BeadsCalibrationWidget:
    def __init__(self):
        self.select_file_widget = SelectFilesButtonShow(nofile_description='Select the tiff file to calibrate')
        self.psf_param_widget = SetPSFParamWidget()
        self.cam_param_widget = SetCamParamWidget()
        self.calib_param_widget = CalibParamWidget()
        self.ok_button = widgets.Button(description='Set Parameters')
        self.ok_button.on_click(self.set_beads_calib_params)
        self.output_widget = widgets.Output()

        self.beads_calib_params_dict = {}

    def set_beads_calib_params(self, b):
        psf_params_dict = self.psf_param_widget.get_psf_params()
        camera_params_dict = self.cam_param_widget.get_camera_params()
        calib_params_dict = self.calib_param_widget.get_calib_params()

        self.beads_calib_params_dict = {'beads_file_name': self.select_file_widget.files,
                                        'psf_params_dict': psf_params_dict,
                                        'camera_params_dict': camera_params_dict,
                                        'calib_params_dict': calib_params_dict}
        self.output_widget.clear_output()
        with self.output_widget:
            display(self.beads_calib_params_dict)

    def run(self):
        beads_calib_params_dict = copy.deepcopy(self.beads_calib_params_dict)
        ailoc.common.beads_psf_calibrate(beads_calib_params_dict, napari_plot=False)

    def display_notebook_gui(self):
        self.select_file_widget.display_notebook_gui()
        self.psf_param_widget.display_notebook_gui(file_receiver=False)
        self.cam_param_widget.display_notebook_gui()
        self.calib_param_widget.display_notebook_gui()
        display(self.ok_button)
        display(self.output_widget)


class SetPSFParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(11, 2)

        # create widgets
        self.calibration_file_receiver = SelectFilesButton(nofile_description='Select the calibration file')
        self.load_calibration_button = widgets.Button(description='Load from calib', width='100%', height='80px')
        self.load_calibration_button.on_click(self.load_calibration_params)
        self.psf_param_label = widgets.Label(value='PSF parameters', width='100%')
        self.na_receiver = widgets.BoundedFloatText(description='NA:',
                                                    value=1.5,
                                                    step=0.1,
                                                    max=2,
                                                    min=0.1)
        self.wavelength_receiver = widgets.BoundedFloatText(description='Wavelength:',
                                                            value=680,
                                                            step=10,
                                                            max=10000,
                                                            min=0)
        self.refmed_receiver = widgets.BoundedFloatText(description='RI medium:',
                                                        value=1.518,
                                                        step=0.1)
        self.refcov_receiver = widgets.BoundedFloatText(description='RI coverslip:',
                                                        value=1.518,
                                                        step=0.1)
        self.refimm_receiver = widgets.BoundedFloatText(description='RI oil:',
                                                        value=1.518,
                                                        step=0.1)
        self.otf_rescale_receiver = widgets.BoundedFloatText(description='OTF rescale:',
                                                             value=0.5,
                                                             step=0.1,
                                                             max=2,
                                                             min=0)
        self.pixelsizex_receiver = widgets.BoundedFloatText(description='PixelsizeX:',
                                                            value=108,
                                                            step=1,
                                                            max=1000,
                                                            min=1)
        self.pixelsizey_receiver = widgets.BoundedFloatText(description='PixelsizeY:',
                                                            value=108,
                                                            step=1,
                                                            max=1000,
                                                            min=1)
        self.npupil_receiver = widgets.BoundedIntText(description='Npupil:',
                                                      value=64,
                                                      step=1,
                                                      max=1000,
                                                      min=1)
        self.psf_size_receiver = widgets.BoundedIntText(description='PSF size:',
                                                        value=51,
                                                        step=1,
                                                        max=100,
                                                        min=17)
        self.objstage0_receiver = widgets.HBox([widgets.Label(value='Nominal focus position (close to sample is minus, '
                                                                    'focus on coverslip is 0):'),
                                                widgets.BoundedFloatText(value=0, step=1, max=10000, min=-10000)])
        # self.zemit0_receiver = widgets.BoundedFloatText(description='zemit0:',
        #                                                 value=0,
        #                                                 step=1,
        #                                                 max=10000,
        #                                                 min=-10000)
        self.zernike_coef_label = widgets.Label('Zernike coefficients: ')
        self.zernike_mode = widgets.GridspecLayout(2, 21)
        self.zernike_mode[0, 0] = widgets.Label(value='2,-2')
        self.zernike_mode[0, 1] = widgets.Label(value='2, 2')
        self.zernike_mode[0, 2] = widgets.Label(value='3,-1')
        self.zernike_mode[0, 3] = widgets.Label(value='3, 1')
        self.zernike_mode[0, 4] = widgets.Label(value='4, 0')
        self.zernike_mode[0, 5] = widgets.Label(value='3,-3')
        self.zernike_mode[0, 6] = widgets.Label(value='3, 3')
        self.zernike_mode[0, 7] = widgets.Label(value='4,-2')
        self.zernike_mode[0, 8] = widgets.Label(value='4, 2')
        self.zernike_mode[0, 9] = widgets.Label(value='5,-1')
        self.zernike_mode[0, 10] = widgets.Label(value='5, 1')
        self.zernike_mode[0, 11] = widgets.Label(value='6, 0')
        self.zernike_mode[0, 12] = widgets.Label(value='4,-4')
        self.zernike_mode[0, 13] = widgets.Label(value='4, 4')
        self.zernike_mode[0, 14] = widgets.Label(value='5,-3')
        self.zernike_mode[0, 15] = widgets.Label(value='5, 3')
        self.zernike_mode[0, 16] = widgets.Label(value='6,-2')
        self.zernike_mode[0, 17] = widgets.Label(value='6, 2')
        self.zernike_mode[0, 18] = widgets.Label(value='7, 1')
        self.zernike_mode[0, 19] = widgets.Label(value='7,-1')
        self.zernike_mode[0, 20] = widgets.Label(value='8, 0')
        for i in range(21):
            self.zernike_mode[1, i] = widgets.Text('0')
        # self.ok_button = widgets.Button(description='OK', width='100%', height='80px')
        # self.ok_button.on_click(self.set_psf_params)
        # self.output = widgets.Output()

        # layout
        # self[0, :] = self.load_calibration_button
        self[1, :] = self.psf_param_label
        self[2, 0] = self.na_receiver
        self[2, 1] = self.wavelength_receiver
        self[3, 0] = self.refmed_receiver
        self[3, 1] = self.refcov_receiver
        self[4, 0] = self.refimm_receiver
        self[4, 1] = self.otf_rescale_receiver
        self[5, 0] = self.pixelsizex_receiver
        self[5, 1] = self.pixelsizey_receiver
        self[6, 0] = self.npupil_receiver
        self[6, 1] = self.psf_size_receiver
        self[7, :] = self.objstage0_receiver
        # self[7, 1] = self.zemit0_receiver
        self[8, :] = self.zernike_coef_label
        self[9:11, :] = self.zernike_mode
        # self[11, :] = self.ok_button

        self.psf_params_dict = {}

    def load_calibration_params(self, b):
        try:
            zernike_fit_results = scio.loadmat(self.calibration_file_receiver.files)['zernike_fit_results']
        except:
            raise ValueError('Please select a valid calibration file')
        self.na_receiver.value = zernike_fit_results['na']
        self.wavelength_receiver.value = zernike_fit_results['wavelength']
        self.refmed_receiver.value = zernike_fit_results['refmed']
        self.refcov_receiver.value = zernike_fit_results['refcov']
        self.refimm_receiver.value = zernike_fit_results['refimm']
        self.otf_rescale_receiver.value = zernike_fit_results['otf_rescale']
        self.pixelsizex_receiver.value = zernike_fit_results['pixelSizeX']
        self.pixelsizey_receiver.value = zernike_fit_results['pixelSizeY']

        zernike_mode = zernike_fit_results['zernike_coefficients'][0, 0][0:2, :]
        zernike_coef = zernike_fit_results['zernike_coefficients'][0, 0][2, :]

        for i in range(21):
            tmp = self.zernike_mode[0, i].value
            radial_order, angular_freq = tmp.split(',')
            radial_order = int(radial_order)
            angular_freq = int(angular_freq)
            assert radial_order == zernike_mode[0, i] and angular_freq == zernike_mode[1, i], \
                'calibration file does not match'
            self.zernike_mode[1, i].value = str(zernike_coef[i])

    def get_psf_params(self):
        self.psf_params_dict['na'] = self.na_receiver.value
        self.psf_params_dict['wavelength'] = self.wavelength_receiver.value
        self.psf_params_dict['refmed'] = self.refmed_receiver.value
        self.psf_params_dict['refcov'] = self.refcov_receiver.value
        self.psf_params_dict['refimm'] = self.refimm_receiver.value
        self.psf_params_dict['objstage0'] = self.objstage0_receiver.children[1].value
        # self.psf_params_dict['zemit0'] = self.zemit0_receiver.value
        self.psf_params_dict['zemit0'] = -self.psf_params_dict['objstage0']/self.psf_params_dict['refimm']*self.psf_params_dict['refmed']
        self.psf_params_dict['pixel_size_xy'] = (self.pixelsizex_receiver.value,
                                                 self.pixelsizey_receiver.value)
        self.psf_params_dict['otf_rescale_xy'] = (self.otf_rescale_receiver.value,
                                                  self.otf_rescale_receiver.value)
        self.psf_params_dict['npupil'] = self.npupil_receiver.value
        self.psf_params_dict['psf_size'] = self.psf_size_receiver.value

        zernike_mode = []
        zernike_coef = []
        for i in range(21):
            tmp = self.zernike_mode[0, i].value
            radial_order, angular_freq = tmp.split(',')
            radial_order = int(radial_order)
            angular_freq = int(angular_freq)
            zernike_mode.append([radial_order, angular_freq])
            zernike_coef.append(float(self.zernike_mode[1, i].value))

        self.psf_params_dict['zernike_mode'] = np.array(zernike_mode, dtype=np.float32)
        self.psf_params_dict['zernike_coef'] = np.array(zernike_coef, dtype=np.float32)
        # self.psf_params_dict['zernike_coef_map'] = None

        # with self.output:
        #     self.output.clear_output()
        #     print(self.psf_params_dict)

        return self.psf_params_dict

    def display_notebook_gui(self, file_receiver=True):
        if file_receiver:
            display(self.calibration_file_receiver)
            display(self.load_calibration_button)
        display(self)


class sCMOSParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(3, 2)

        self.qe_receiver = widgets.BoundedFloatText(description='QE:',
                                                    value=0.9,
                                                    min=0.1,
                                                    max=1,
                                                    step=0.1,)
        self[0, 0] = self.qe_receiver
        self.spurious_charge_receiver = widgets.BoundedFloatText(description='Spur charge:',
                                                                 value=0.002,
                                                                 min=0,
                                                                 max=1,
                                                                 step=0.001,)
        self[0, 1] = self.spurious_charge_receiver
        self.readout_noise_receiver = widgets.BoundedFloatText(description='Read noise:',
                                                               value=1.6,
                                                               min=0,
                                                               max=1000,
                                                               step=0.1,)
        self[1, 0] = self.readout_noise_receiver
        self.eperadu_receiver = widgets.BoundedFloatText(description='e-/ADU:',
                                                            value=0.5,
                                                            min=0.001,
                                                            max=1000,
                                                            step=0.1,)
        self[1, 1] = self.eperadu_receiver
        self.baseline_receiver = widgets.BoundedFloatText(description='Baseline:',
                                                            value=100,
                                                            min=0,
                                                            max=1000,
                                                            step=1,)
        self[2, 0] = self.baseline_receiver


class EMCCDParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(3, 2)

        self.qe_receiver = widgets.BoundedFloatText(description='QE:',
                                                    value=0.9,
                                                    min=0.1,
                                                    max=1,
                                                    step=0.1,)
        self[0, 0] = self.qe_receiver
        self.spurious_charge_receiver = widgets.BoundedFloatText(description='Spur charge:',
                                                                 value=0.002,
                                                                 min=0,
                                                                 max=1,
                                                                 step=0.001,)
        self[0, 1] = self.spurious_charge_receiver
        self.emgain_receiver = widgets.BoundedFloatText(description='EM gain:',
                                                        value=300,
                                                        min=1,
                                                        max=1000,
                                                        step=1,)
        self[1, 0] = self.emgain_receiver
        self.readout_noise_receiver = widgets.BoundedFloatText(description='Read noise:',
                                                               value=74.4,
                                                               min=0,
                                                               max=1000,
                                                               step=0.1,)
        self[1, 1] = self.readout_noise_receiver
        self.eperadu_receiver = widgets.BoundedFloatText(description='e-/ADU:',
                                                            value=45,
                                                            min=0.001,
                                                            max=1000,
                                                            step=0.1,)
        self[2, 0] = self.eperadu_receiver
        self.baseline_receiver = widgets.BoundedFloatText(description='Baseline:',
                                                            value=100,
                                                            min=0,
                                                            max=1000,
                                                            step=1,)
        self[2, 1] = self.baseline_receiver


class IdeaCamParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super().__init__(1, 1)
        self[0, 0] = widgets.Label(value='No parameters to set for Idea camera')


class SetCamParamWidget:
    def __init__(self):
        self.camera_params_label = widgets.Label(value='Camera parameters')
        self.select_cam_dropdown = widgets.Dropdown(options=['Idea Camera', 'sCMOS', 'EMCCD'],
                                                    value='sCMOS',
                                                    description='Camera type:',
                                                    )
        self.select_cam_dropdown.observe(self.select_cam, names='value')
        self.idea_param_receiver = IdeaCamParamWidget()
        self.scmos_param_receiver = sCMOSParamWidget()
        self.emccd_param_receiver = EMCCDParamWidget()
        # self.ok_button = widgets.Button(description='OK')
        # self.ok_button.on_click(self.set_camera_params)

        # layout
        self.camera_param_receiver = widgets.GridspecLayout(1, 1)
        self.camera_param_receiver[0, 0] = self.scmos_param_receiver

        self.camera_params_dict = {}

    def select_cam(self, change):
        if change['new'] == 'Idea Camera':
            self.camera_param_receiver[0, 0] = self.idea_param_receiver
        elif change['new'] == 'sCMOS':
            self.camera_param_receiver[0, 0] = self.scmos_param_receiver
        elif change['new'] == 'EMCCD':
            self.camera_param_receiver[0, 0] = self.emccd_param_receiver

    def get_camera_params(self):
        if self.select_cam_dropdown.value == 'Idea Camera':
            self.camera_params_dict['camera_type'] = 'idea'
        elif self.select_cam_dropdown.value == 'sCMOS':
            self.camera_params_dict['camera_type'] = 'scmos'
            self.camera_params_dict['qe'] = self.scmos_param_receiver.qe_receiver.value
            self.camera_params_dict['spurious_charge'] = self.scmos_param_receiver.spurious_charge_receiver.value
            self.camera_params_dict['read_noise_sigma'] = self.scmos_param_receiver.readout_noise_receiver.value
            # self.camera_params_dict['read_noise_map'] = None
            self.camera_params_dict['e_per_adu'] = self.scmos_param_receiver.eperadu_receiver.value
            self.camera_params_dict['baseline'] = self.scmos_param_receiver.baseline_receiver.value
        elif self.select_cam_dropdown.value == 'EMCCD':
            self.camera_params_dict['camera_type'] = 'emccd'
            self.camera_params_dict['qe'] = self.emccd_param_receiver.qe_receiver.value
            self.camera_params_dict['spurious_charge'] = self.emccd_param_receiver.spurious_charge_receiver.value
            self.camera_params_dict['em_gain'] = self.emccd_param_receiver.emgain_receiver.value
            self.camera_params_dict['read_noise_sigma'] = self.emccd_param_receiver.readout_noise_receiver.value
            self.camera_params_dict['e_per_adu'] = self.emccd_param_receiver.eperadu_receiver.value
            self.camera_params_dict['baseline'] = self.emccd_param_receiver.baseline_receiver.value

        return self.camera_params_dict

    def display_notebook_gui(self):
        display(self.camera_params_label)
        display(self.select_cam_dropdown)
        display(self.camera_param_receiver)


class DeepLocSetSamplerParamWidget(widgets.GridspecLayout):
    def __init__(self):
        super(DeepLocSetSamplerParamWidget, self).__init__(7, 2)

        self.local_context_receiver = widgets.Checkbox(description='Local context',
                                                       value=True,
                                                       indent=False,)
        self.robust_training_receiver = widgets.Checkbox(description='Robust training',
                                                         value=True,
                                                         indent=False,)
        self.train_size_receiver = widgets.BoundedIntText(description='Train size:',
                                                          value=128,
                                                          min=32,
                                                          max=1024,
                                                          step=4,)
        self.num_em_avg_receiver = widgets.BoundedIntText(description='Density:',
                                                          value=10,
                                                          min=1,
                                                          max=1000,
                                                          step=1,)
        self.num_eval_data_receiver = widgets.BoundedIntText(description='Eval images:',
                                                             value=1000,
                                                             min=1,
                                                             max=10000,
                                                             step=1,)
        self.photon_range_receiver = widgets.HBox([widgets.Label(value='Photon range:'),
                                                   widgets.IntRangeSlider(
                                                                          value=[1000, 10000],
                                                                          min=0,
                                                                          max=20000,
                                                                          step=100,
                                                                          orientation='horizontal',
                                                                          readout=True,
                                                                          readout_format='d',)
                                                   ])
        self.z_range_receiver = widgets.IntRangeSlider(description='Z range:',
                                                       value=[-700, 700],
                                                       min=-3000,
                                                       max=3000,
                                                       step=50,
                                                       orientation='horizontal',
                                                       readout=True,
                                                       readout_format='d',)
        self.bg_range_receiver = widgets.IntRangeSlider(description='BG range:',
                                                        value=[50, 100],
                                                        min=0,
                                                        max=300,
                                                        step=10,
                                                        orientation='horizontal',
                                                        readout=True,
                                                        readout_format='d',)
        # self.ok_button = widgets.Button(description='OK')
        # self.ok_button.on_click(self.set_sampler_params)

        # layout
        self[0, :] = widgets.Label(value='Sampler parameters')
        self[1, 0] = self.local_context_receiver
        self[1, 1] = self.robust_training_receiver
        self[2, 0] = self.train_size_receiver
        self[2, 1] = self.num_em_avg_receiver
        self[3, :] = self.num_eval_data_receiver
        self[4, :] = self.photon_range_receiver
        self[5, 0] = self.z_range_receiver
        self[5, 1] = self.bg_range_receiver
        # self[6, :] = self.ok_button

        self.sampler_params_dict = {}

    def set_sampler_params(self, b):
        self.sampler_params_dict['local_context'] = self.local_context_receiver.value
        self.sampler_params_dict['robust_training'] = self.robust_training_receiver.value
        self.sampler_params_dict['train_size'] = self.train_size_receiver.value
        self.sampler_params_dict['num_em_avg'] = self.num_em_avg_receiver.value
        self.sampler_params_dict['num_evaluation_data'] = self.num_eval_data_receiver.value
        self.sampler_params_dict['photon_range'] = self.photon_range_receiver.children[1].value
        self.sampler_params_dict['z_range'] = self.z_range_receiver.value
        self.sampler_params_dict['bg_range'] = self.bg_range_receiver.value

    def display_notebook_gui(self):
        display(self)


class DeepLocSetLearnParamWidget:
    def __init__(self):
        self.psf_param_widget = SetPSFParamWidget()
        self.cam_param_widget = SetCamParamWidget()
        self.sampler_param_widget = DeepLocSetSamplerParamWidget()
        self.ok_button = widgets.Button(description='OK')
        self.ok_button.on_click(self.set_learn_params)
        self.output_widget = widgets.Output()

        self.deeploc_params_dict = {}

    def set_learn_params(self, b):
        self.psf_param_widget.set_psf_params(None)
        self.cam_param_widget.set_camera_params(None)
        self.sampler_param_widget.set_sampler_params(None)

        self.deeploc_params_dict['psf_params_dict'] = self.psf_param_widget.psf_params_dict
        self.deeploc_params_dict['camera_params_dict'] = self.cam_param_widget.camera_params_dict
        self.deeploc_params_dict['sampler_params_dict'] = self.sampler_param_widget.sampler_params_dict

        self.output_widget.clear_output()
        with self.output_widget:
            print('DeepLoc learning parameters: ')
            print(self.deeploc_params_dict)

        # self.output_widget.clear_output()
        # self.output_widget.append_stdout('DeepLoc learning parameters: ')
        # self.output_widget.append_stdout(str(self.deeploc_params_dict))

    def display_notebook_gui(self):
        self.psf_param_widget.display_notebook_gui()
        self.cam_param_widget.display_notebook_gui()
        self.sampler_param_widget.display_notebook_gui()
        display(self.ok_button)
        display(self.output_widget)


class DeepLocSetAnalyzerParamWidget:
    def __init__(self):
        self.select_deeploc_button = SelectFilesButton(nofile_description='Select DeepLoc model')
        self.select_deeploc_button.observe(self.set_save_csv_button_initialdir, names='description')

        self.select_folder_receiver = widgets.Checkbox(value=True,
                                                       description='Select all tiff files under the folder',
                                                       disabled=False,
                                                       indent=False)

        # self.select_data_button = SelectFolderButton(nofile_description='Select data folder')
        self.select_data_button = SelectFilesButton(nofile_description='Select tiff file')

        self.save_csv_button = SaveFilesButton(nofile_description='Save predicted CSV',
                                               initialdir='../../results/',
                                               initialfile='predictions.csv')

        self.time_block_gb_receiver = widgets.BoundedIntText(description='Block(GB)',
                                                             value=1,
                                                             step=0.1,
                                                             min=0.01,
                                                             max=100,)
        self.batch_size_receiver = widgets.BoundedIntText(description='Batch size',
                                                          value=16,
                                                          step=1,
                                                          min=1,
                                                          max=1024)
        self.sub_fov_size_receiver = widgets.BoundedIntText(description='Sub-FOV',
                                                            value=128,
                                                            step=4,
                                                            min=32,
                                                            max=2048)
        self.over_cut_receiver = widgets.BoundedIntText(description='Over-cut',
                                                        value=8,
                                                        step=4,
                                                        min=4,
                                                        max=32)
        self.num_workers_receiver = widgets.BoundedIntText(description='Workers',
                                                            value=0,
                                                            step=1,
                                                            min=0,
                                                            max=16)
        self.ok_button = widgets.Button(description='OK')
        self.ok_button.on_click(self.set_analyzer_params)
        self.output_widget = widgets.Output()

        # layout
        self.analyzer_param_widget = widgets.GridspecLayout(3, 2)
        self.analyzer_param_widget[0, 0] = self.time_block_gb_receiver
        self.analyzer_param_widget[0, 1] = self.batch_size_receiver
        self.analyzer_param_widget[1, 0] = self.sub_fov_size_receiver
        self.analyzer_param_widget[1, 1] = self.over_cut_receiver
        self.analyzer_param_widget[2, 0] = self.num_workers_receiver

        self.analyzer_param = {}

    def set_analyzer_params(self, b):
        loc_model_path = self.select_deeploc_button.files
        # load the completely trained model
        with open(loc_model_path, 'rb') as f:
            deeploc_model = torch.load(f)
        self.analyzer_param['deeploc_model'] = deeploc_model
        if self.select_folder_receiver.value:
            self.analyzer_param['tiff_path'] = os.path.dirname(self.select_data_button.files)
        else:
            self.analyzer_param['tiff_path'] = self.select_data_button.files
        self.analyzer_param['output_path'] = self.save_csv_button.files
        self.analyzer_param['time_block_gb'] = self.time_block_gb_receiver.value
        self.analyzer_param['batch_size'] = self.batch_size_receiver.value
        self.analyzer_param['sub_fov_size'] = self.sub_fov_size_receiver.value
        self.analyzer_param['over_cut'] = self.over_cut_receiver.value
        self.analyzer_param['num_workers'] = self.num_workers_receiver.value

        self.output_widget.clear_output()
        with self.output_widget:
            print('DeepLoc analysis parameters: ')
            print(self.analyzer_param)
            print('Plot training process: ')
            ailoc.common.plot_train_record(deeploc_model)

    def set_save_csv_button_initialdir(self, change):
        if change['new'] == self.select_deeploc_button.nofile_description:
            self.save_csv_button.initialdir = '../../results/'
            self.save_csv_button.initialfile = 'predictions.csv'
        else:
            self.save_csv_button.initialdir = os.path.dirname(self.select_deeploc_button.files)
            self.save_csv_button.initialfile = os.path.basename(self.select_deeploc_button.files).split('.')[0] + \
                                               '_predictions.csv'

    def display_notebook_gui(self):
        display(self.select_deeploc_button)
        display(self.select_folder_receiver)
        display(self.select_data_button)
        display(self.save_csv_button)
        display(self.analyzer_param_widget)
        display(self.ok_button)
        display(self.output_widget)
