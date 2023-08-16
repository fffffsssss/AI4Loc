# Calibration widget
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import scipy.io as scio
import sys

from ailoc.common.calibrationwidget.ui_files.calibrationform import Ui_CalibrationForm
from ailoc.common.calibrationwidget.tabplotwidget import TabPlotWidget
from ailoc.common.calibrationwidget.calibration_funcs.calibrate3D import Calibrate3DProcess


class CalibrationWidget(QtWidgets.QWidget, Ui_CalibrationForm):
    """
    Calibration widget.
    """

    def __init__(self, parent=None):
        """
        Initialization
        """
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_CalibrationForm()
        self.ui.setupUi(self)
        self._init_ui()

        self.plot_widget = TabPlotWidget()
        self.plot_widget.resize(1024, 1024)
        self.plot_widget.setWindowTitle("Calibration result")

        self.calibrate_process = Calibrate3DProcess()
        self.calibrate_process_thread = QtCore.QThread()
        self.calibrate_process.moveToThread(self.calibrate_process_thread)
        self.calibrate_process.raw_data_recv.connect(self.calibrate_process.calibrate_data)
        self.calibrate_process.figure_show.connect(self.display_figure)
        self.calibrate_process.message_send.connect(self.print_message)
        self.calibrate_process.data_calibrated.connect(self.save_calibrated_data)
        self.calibrate_process_thread.start()
        self.process_idle = True

    def __del__(self):
        """
        Destructor.
        """
        self.calibrate_process_thread.exit()
        self.calibrate_process_thread.wait()

    def _init_ui(self):
        """
        Initialize UI.
        """
        self.ui.distance_btw_frames_le.setValidator(QtGui.QIntValidator(0, 10000))
        self.ui.frames_le.setValidator(QtGui.QIntValidator(1, 10000))
        self.ui.filter_size_le.setValidator(QtGui.QIntValidator(1, 10000))
        self.ui.min_distance_le.setValidator(QtGui.QIntValidator(1, 2048))
        self.ui.roi_size_xy_le.setValidator(QtGui.QIntValidator(1, 2048))
        self.ui.smoothing_param_le.setValidator(QtGui.QDoubleValidator())
        self.ui.gauss_fit_range_min_le.setValidator(QtGui.QIntValidator())
        self.ui.gauss_fit_range_max_le.setValidator(QtGui.QIntValidator())
        self.ui.gauss_fit_roi_size_le.setValidator(QtGui.QIntValidator(1, 2048))

        self.ui.image_lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.ui.correct_bead_z_cbb.currentIndexChanged.connect(self.on_correct_bead_z_current_index_changed)
        self.ui.modality_cbb.currentIndexChanged.connect(self.on_modality_current_index_changed)
        self.ui.gauss_fit_param_gb.setVisible(False)

        self.ui.open_files_btn.clicked.connect(self.open_files)
        self.ui.open_dir_btn.clicked.connect(self.open_dir)
        self.ui.delete_btn.clicked.connect(self.delete_images)
        self.ui.clear_btn.clicked.connect(self.clear_images)
        self.ui.select_btn.clicked.connect(self.select_output_file)
        self.ui.calibrate_btn.clicked.connect(self.do_calibration)

        self.ui.use_zernike_fit_cb.stateChanged.connect(self.ui.zernike_fit_gb.setEnabled)

    @QtCore.pyqtSlot(int)
    def on_correct_bead_z_current_index_changed(self, index):
        """
        Correct bead Z-position combobox index changed.
        """
        self.ui.frames_lbl.setVisible(index == 1)
        self.ui.frames_le.setVisible(index == 1)

    @QtCore.pyqtSlot(int)
    def on_modality_current_index_changed(self, index):
        """
        3D Modality combobox index changed.
        """
        self.ui.gauss_fit_param_gb.setVisible(index == 0)

    @QtCore.pyqtSlot()
    def select_output_file(self):
        """
        Select output file
        """
        ret = QtWidgets.QFileDialog.getSaveFileUrl(self, self.tr("Select output file"), QtCore.QUrl("C:\\bead_3dcal.mat"),
                                         self.tr("Mat file (*.mat)"))
        self.ui.output_file_le.setText(ret[0].toLocalFile())

    @QtCore.pyqtSlot()
    def open_files(self):
        """
        Open files
        """
        ret = QtWidgets.QFileDialog.getOpenFileNames(self, self.tr("Select images"), "C:\\",
                                           self.tr("TIFF Images(*.tif *.TIF *.tiff *.TIFF)"))
        for file in ret[0]:
            self.ui.image_lw.addItem(file)

    @QtCore.pyqtSlot()
    def open_dir(self):
        """
        Open directory.
        """
        ret = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Select images directory"), "C:\\",
                                               QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
        dir = QtCore.QDir(ret)
        for file_info in dir.entryInfoList(QtCore.QDir.Files | QtCore.QDir.Readable):
            if file_info.completeSuffix().lower() == "tif" or file_info.completeSuffix().lower() == "tiff":
                self.ui.image_lw.addItem(file_info.absoluteFilePath())

    @QtCore.pyqtSlot()
    def delete_images(self):
        """
        Delete selected images
        """
        images = self.ui.image_lw.selectedItems()
        for image in images:
            self.ui.image_lw.takeItem(self.ui.image_lw.row(image))

    @QtCore.pyqtSlot()
    def clear_images(self):
        """
        Clear images.
        """
        self.ui.image_lw.clear()

    @QtCore.pyqtSlot(str)
    def print_message(self, msg: str):
        """
        Print messsage
        """
        self.ui.status_lbl.setText(msg)
        self.update()

    @QtCore.pyqtSlot(list)
    def display_figure(self, figure_list: list):
        """
        Display figure
        """
        for i in range(0, len(figure_list)):
            self.plot_widget.add_plot(figure_list[i][0], figure_list[i][1]) # 0 is title, 1 is figure
        self.plot_widget.update()

    @QtCore.pyqtSlot(dict)
    def save_calibrated_data(self, data: dict):
        """
        Save calibrated data
        """
        self.process_idle = True
        scio.savemat(data['parameters']['outputfile'], data)

    @QtCore.pyqtSlot()
    def do_calibration(self):
        """
        Calculate calibration.
        """
        if not self.process_idle:
            self.print_message("calibration is busy!")
            return
        self.plot_widget.clear()
        self.plot_widget.show()

        parameters = dict()
        parameters['filelist'] = []
        for i in range(0, self.ui.image_lw.count()):
            parameters['filelist'].append(self.ui.image_lw.item(i).text())
        parameters['outputfile'] = self.ui.output_file_le.text()

        # general parameters
        parameters['dz'] = int(self.ui.distance_btw_frames_le.text())
        parameters['modality'] = self.ui.modality_cbb.currentText()
        parameters['zcorr'] = self.ui.correct_bead_z_cbb.currentText()
        parameters['zcorrframes'] = int(self.ui.frames_le.text())
        parameters['filtersize'] = int(self.ui.filter_size_le.text())
        parameters['mindistance'] = int(self.ui.min_distance_le.text())
        parameters['relative_cutoff'] = float(self.ui.relative_cutoff_le.text())

        # Cspline parameters
        parameters['ROIxy'] = int(self.ui.roi_size_xy_le.text())
        parameters['ROIz'] = np.NaN  # FIXME
        parameters['smoothxy'] = 0  # FIXME
        parameters['smoothz'] = int(self.ui.smoothing_param_le.text())

        # Gauss fit parameters
        parameters['gaussrange'] = (int(self.ui.gauss_fit_range_min_le.text()),
                                    int(self.ui.gauss_fit_range_max_le.text()))
        parameters['gaussroi'] = int(self.ui.gauss_fit_roi_size_le.text())

        # smap
        parameters['emgain'] = 0  # FIXME
        parameters['smap'] = 1  # FIXME
        parameters['xrange'] = np.array([-np.inf, np.inf])
        parameters['yrange'] = np.array([-np.inf, np.inf])
        parameters['imageRoi'] = np.zeros(2)

        if 'smap' not in parameters:
            parameters['smap'] = 0
            parameters['imageRoi'] = np.zeros(2)
        if 'xrange' not in parameters:
            parameters['xrange'] = np.array([-np.inf, np.inf])
            parameters['yrange'] = np.array([-np.inf, np.inf])
        if 'emgain' not in parameters:
            parameters['emgain'] = 0
        if 'smoothxy' not in parameters:
            parameters['smoothxy'] = 0

        # for zernike fit
        parameters['use_zernike_fit'] = self.ui.use_zernike_fit_cb.isChecked()
        if parameters['use_zernike_fit']:
            parameters['na'] = float(self.ui.na_le.text())
            parameters['refmed'] = float(self.ui.refmed_le.text())
            parameters['refcov'] = float(self.ui.refcov_le.text())
            parameters['refimm'] = float(self.ui.refimm_le.text())
            parameters['lambda'] = float(self.ui.lambda_le.text())
            parameters['pixelSizeX'] = float(self.ui.pixelsizex_le.text())
            parameters['pixelSizeY'] = float(self.ui.pixelsizey_le.text())
            parameters['otf_rescale'] = float(self.ui.otfrescale_le.text())
            parameters['iterations'] = int(self.ui.iterations_le.text())

        self.calibrate_process.raw_data_recv.emit(parameters)
        self.process_idle = False


if __name__ == '__main__':
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    window = CalibrationWidget()
    window.show()
    sys.exit(app.exec_())