import gc
import sys
import time
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
import os

import ailoc.common
from ailoc.deeploc.ui_files.ui_deeploc_analyzer import Ui_DeepLoc_Analyzer
from ailoc.deeploc.ui_files.ui_deeploc_learning import Ui_DeepLoc_learning
from ailoc.common.calibrationwidget.calibrationwidget import CalibrationWidget


class DeepLoc_MainWindow(QtWidgets.QMainWindow):
    #todo learning gui
    """
    Main window
    """

    def __init__(self, parent=None):
        """
        Initialization
        :param parent: Parent widget
        """
        super().__init__()

        self.setMinimumSize(1000, 800)
        self.setWindowTitle("Analysis")

        # menu
        # self._add_menu()

        # statusbar
        self.statusBar()

        # center widget
        self.tab_widget = QtWidgets.QTabWidget()

        self.calibration_widget = CalibrationWidget()
        self.tab_widget.addTab(self.calibration_widget, self.tr("Calibration"))

        self.learning_widget = DeepLoc_Learning_GUI()
        self.tab_widget.addTab(self.learning_widget, self.tr("Learning"))

        self.analyzer_widget = DeepLoc_Analyzer_GUI()
        self.tab_widget.addTab(self.analyzer_widget, self.tr("Analyzing"))

        self.setCentralWidget(self.tab_widget)

    def _add_menu(self):
        """
        Add menu
        """
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu(self.tr("&File"))
        open_action = QtWidgets.QAction(self.tr("&Open..."), self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip(self.tr("Open configuration file"))
        file_menu.addAction(open_action)
        file_menu.addSeparator()

        setting_action = QtWidgets.QAction(self.tr("Setting..."), self)
        setting_action.setStatusTip(self.tr("Setting"))
        file_menu.addAction(setting_action)

        help_menu = menu_bar.addMenu(self.tr("&Help"))
        about_action = QtWidgets.QAction(self.tr("About..."), self)
        help_menu.addAction(about_action)


class DeepLoc_Learning_GUI(QtWidgets.QWidget, Ui_DeepLoc_learning):
    # todo learning gui, i feel it is not comfortable to use a learning gui for deep learning
    def __init__(self):
        super().__init__()
        self.ui = Ui_DeepLoc_learning()
        self.ui.setupUi(self)

        self._init_ui()
        self.learning_thread = DeepLoc_Learning_Thread()
        self.learning_thread.finished.connect(self.enable_all)
        self.learning_thread.ui_print_signal.connect(self.display_text)

    def _init_ui(self):
        self.ui.emccdparameter_gb.setVisible(False)
        self.ui.cameratype_cb.currentIndexChanged.connect(self.on_cameratype_cb_currentIndexChanged)

    def on_cameratype_cb_currentIndexChanged(self):
        if self.ui.cameratype_cb.currentText() == 'EMCCD':
            self.ui.emccdparameter_gb.setVisible(True)
            self.ui.scmosparameter_gb.setVisible(False)
        elif self.ui.cameratype_cb.currentText() == 'sCMOS':
            self.ui.emccdparameter_gb.setVisible(False)
            self.ui.scmosparameter_gb.setVisible(True)
        elif self.ui.cameratype_cb.currentText() == 'Idea Camera':
            self.ui.emccdparameter_gb.setVisible(False)
            self.ui.scmosparameter_gb.setVisible(False)

    @QtCore.pyqtSlot()
    def on_select_model_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load DeepLoc model",
                                                          "../../",
                                                          "Pytorch files (*.pt*);;All files (*)")

        self.ui.model_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_image_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load image",
                                                          "../../",
                                                          "Image files (*.tif*);;All files (*)")

        self.ui.image_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_directory_bt_clicked(self):
        # directory = QtWidgets.QFileDialog.getExistingDirectory(self,
        #                                                        "Select directory",
        #                                                        "../../",
        #                                                        QtWidgets.QFileDialog.ShowDirsOnly)
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load image",
                                                          "../../",
                                                          "Image files (*.tif*)")
        directory = os.path.split(directory[0])[0]

        self.ui.image_path_le.setText(directory)

    @QtCore.pyqtSlot()
    def on_select_output_bt_clicked(self):
        model_path = self.ui.model_path_le.text()
        image_path = self.ui.image_path_le.text()
        if model_path == "" or image_path == "":
            QtWidgets.QMessageBox.warning(self,
                                          "Warning",
                                          "Please select model and image first!")
            return

        if os.path.isfile(image_path):
            save_path = os.path.split(image_path)[-2] + '/' + \
                        os.path.split(model_path)[-1].split('.')[0] + '_' + \
                        os.path.split(image_path)[-2].split('/')[-1] + '_predictions.csv'
        else:
            save_path = image_path + '/' + \
                        os.path.split(model_path)[-1].split('.')[0] + '_' + \
                        os.path.split(image_path)[-1].split('/')[-1] + '_predictions.csv'

        directory = QtWidgets.QFileDialog.getSaveFileName(self,
                                                          "Select output file",
                                                          save_path,
                                                          "Molecule list files (*.csv*);;All files (*)")
        self.ui.output_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_start_anlz_bt_clicked(self):
        self.disable_all()
        loc_model_path, image_path, output_path, time_block_gb, batch_size, sub_fov_size, \
        over_cut, num_workers, fov_xy_start = self.get_ui_parameters()
        self.analyzer_thread.set_parameters(loc_model_path, image_path, output_path, time_block_gb, batch_size,
                                            sub_fov_size, over_cut, num_workers, fov_xy_start)

        self.analyzer_thread.start()

        # self.display_text("test")

    @QtCore.pyqtSlot()
    def on_stop_anlz_bt_clicked(self):
        # todo, terminate() is dangerous, need to be replaced
        self.analyzer_thread.terminate()
        # self.analyzer_thread.wait()
        while self.analyzer_thread.isRunning():
            self.display_text("\nWaiting algorithm to be terminated!")
        self.display_text("\nAnalysis terminated!")
        self.enable_all()

    def disable_all(self):
        self.ui.select_model_bt.setEnabled(False)
        self.ui.model_path_le.setEnabled(False)
        self.ui.select_image_bt.setEnabled(False)
        self.ui.image_path_le.setEnabled(False)
        self.ui.select_directory_bt.setEnabled(False)
        self.ui.select_output_bt.setEnabled(False)
        self.ui.output_path_le.setEnabled(False)
        self.ui.time_block_gb_sb.setEnabled(False)
        self.ui.batch_size_sb.setEnabled(False)
        self.ui.sub_fov_size_sb.setEnabled(False)
        self.ui.over_cut_sb.setEnabled(False)
        self.ui.num_workers_sb.setEnabled(False)
        self.ui.fov_start_x_sb.setEnabled(False)
        self.ui.fov_start_y_sb.setEnabled(False)
        self.ui.start_anlz_bt.setEnabled(False)

        self.ui.stop_anlz_bt.setEnabled(True)

    def enable_all(self):
        self.ui.select_model_bt.setEnabled(True)
        self.ui.model_path_le.setEnabled(True)
        self.ui.select_image_bt.setEnabled(True)
        self.ui.image_path_le.setEnabled(True)
        self.ui.select_directory_bt.setEnabled(True)
        self.ui.select_output_bt.setEnabled(True)
        self.ui.output_path_le.setEnabled(True)
        self.ui.time_block_gb_sb.setEnabled(True)
        self.ui.batch_size_sb.setEnabled(True)
        self.ui.sub_fov_size_sb.setEnabled(True)
        self.ui.over_cut_sb.setEnabled(True)
        self.ui.num_workers_sb.setEnabled(True)
        self.ui.fov_start_x_sb.setEnabled(True)
        self.ui.fov_start_y_sb.setEnabled(True)
        self.ui.start_anlz_bt.setEnabled(True)

        self.ui.stop_anlz_bt.setEnabled(False)

    def display_text(self, print_message):
        self.ui.print_info_tb.moveCursor(QtGui.QTextCursor.End)  # move cursor to the end of the text

        if print_message.startswith('\r'):  # should perform like print('\r print_message', end='')
            print_message = print_message.strip("\r")
            lastLine = self.ui.print_info_tb.textCursor()  # instantiate the cursor
            lastLine.select(QtGui.QTextCursor.LineUnderCursor)  # select the entire line
            lastLine.removeSelectedText()  # remove the selected text
            self.ui.print_info_tb.insertPlainText(print_message)  # replace the end line with the new text

        else:  # should perform like print
            # self.ui.print_info_tb.append(print_message+'\n')  # append will automatically start a new line
            self.ui.print_info_tb.insertPlainText(print_message+'\n')

        # # test
        # for i in range(10):
        #     self.ui.print_info_tb.append('block:' + str(i))  # write new text at the next line
        #     self.ui.print_info_tb.append(' ')
        #     for j in range(3):
        #         print_message = '\r analyzing'+str(i) + '_' + str(j)
        #
        #         print_message = print_message.strip("\r")
        #
        #         self.ui.print_info_tb.moveCursor(QtGui.QTextCursor.End)  # move to the end of the text
        #         lastLine = self.ui.print_info_tb.textCursor()
        #         lastLine.select(QtGui.QTextCursor.LineUnderCursor)
        #         lastLine.removeSelectedText()  # remove this line of text
        #         self.ui.print_info_tb.insertPlainText(print_message)  # write new text at the same line

    def get_ui_parameters(self):
        loc_model_path = self.ui.model_path_le.text()
        image_path = self.ui.image_path_le.text()
        output_path = self.ui.output_path_le.text()
        time_block_gb = self.ui.time_block_gb_sb.value()
        batch_size = self.ui.batch_size_sb.value()
        sub_fov_size = self.ui.sub_fov_size_sb.value()
        over_cut = self.ui.over_cut_sb.value()
        num_workers = self.ui.num_workers_sb.value()
        fov_xy_start = [self.ui.fov_start_x_sb.value(), self.ui.fov_start_y_sb.value()]

        return loc_model_path, image_path, output_path, time_block_gb, batch_size, \
               sub_fov_size, over_cut, num_workers, fov_xy_start

    def reset(self):
        """
        Reset parameters.
        """
        pass


class DeepLoc_Learning_Thread(QtCore.QThread):
    # todo
    ui_print_signal = QtCore.pyqtSignal(str)
    def __init__(self):
        super().__init__()


    def run(self):
        pass

    def set_parameters(self, parameters: dict):
        pass


class DeepLoc_Analyzer_GUI(QtWidgets.QWidget, Ui_DeepLoc_Analyzer):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DeepLoc_Analyzer()
        self.ui.setupUi(self)

        self.analyzer_thread = DeepLoc_Analyzer_Thread()
        self.analyzer_thread.finished.connect(self.enable_all)
        self.analyzer_thread.ui_print_signal.connect(self.display_text)

    @QtCore.pyqtSlot()
    def on_select_model_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load DeepLoc model",
                                                          "../../",
                                                          "Pytorch files (*.pt*);;All files (*)")

        self.ui.model_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_image_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load image",
                                                          "../../",
                                                          "Image files (*.tif*);;All files (*)")

        self.ui.image_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_directory_bt_clicked(self):
        # directory = QtWidgets.QFileDialog.getExistingDirectory(self,
        #                                                        "Select directory",
        #                                                        "../../",
        #                                                        QtWidgets.QFileDialog.ShowDirsOnly)
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load image",
                                                          "../../",
                                                          "Image files (*.tif*)")
        directory = os.path.split(directory[0])[0]

        self.ui.image_path_le.setText(directory)

    @QtCore.pyqtSlot()
    def on_select_output_bt_clicked(self):
        model_path = self.ui.model_path_le.text()
        image_path = self.ui.image_path_le.text()
        if model_path == "" or image_path == "":
            QtWidgets.QMessageBox.warning(self,
                                          "Warning",
                                          "Please select model and image first!")
            return

        if os.path.isfile(image_path):
            save_path = os.path.split(image_path)[-2] + '/' + \
                        os.path.split(model_path)[-1].split('.')[0] + '_' + \
                        os.path.split(image_path)[-2].split('/')[-1] + '_predictions.csv'
        else:
            save_path = image_path + '/' + \
                        os.path.split(model_path)[-1].split('.')[0] + '_' + \
                        os.path.split(image_path)[-1].split('/')[-1] + '_predictions.csv'

        directory = QtWidgets.QFileDialog.getSaveFileName(self,
                                                          "Select output file",
                                                          save_path,
                                                          "Molecule list files (*.csv*);;All files (*)")
        self.ui.output_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_start_anlz_bt_clicked(self):
        self.disable_all()
        loc_model_path, image_path, output_path, time_block_gb, batch_size, sub_fov_size, \
        over_cut, num_workers, fov_xy_start = self.get_ui_parameters()
        self.analyzer_thread.set_parameters(loc_model_path, image_path, output_path, time_block_gb, batch_size,
                                            sub_fov_size, over_cut, num_workers, fov_xy_start)

        self.analyzer_thread.start()

        # self.display_text("test")

    @QtCore.pyqtSlot()
    def on_stop_anlz_bt_clicked(self):
        # todo, terminate() is dangerous, need to be replaced
        self.analyzer_thread.terminate()
        # self.analyzer_thread.wait()
        while self.analyzer_thread.isRunning():
            self.display_text("\nWaiting algorithm to be terminated!")
        self.display_text("\nAnalysis terminated!")
        self.enable_all()

    def disable_all(self):
        self.ui.select_model_bt.setEnabled(False)
        self.ui.model_path_le.setEnabled(False)
        self.ui.select_image_bt.setEnabled(False)
        self.ui.image_path_le.setEnabled(False)
        self.ui.select_directory_bt.setEnabled(False)
        self.ui.select_output_bt.setEnabled(False)
        self.ui.output_path_le.setEnabled(False)
        self.ui.time_block_gb_sb.setEnabled(False)
        self.ui.batch_size_sb.setEnabled(False)
        self.ui.sub_fov_size_sb.setEnabled(False)
        self.ui.over_cut_sb.setEnabled(False)
        self.ui.num_workers_sb.setEnabled(False)
        self.ui.fov_start_x_sb.setEnabled(False)
        self.ui.fov_start_y_sb.setEnabled(False)
        self.ui.start_anlz_bt.setEnabled(False)

        self.ui.stop_anlz_bt.setEnabled(True)

    def enable_all(self):
        self.ui.select_model_bt.setEnabled(True)
        self.ui.model_path_le.setEnabled(True)
        self.ui.select_image_bt.setEnabled(True)
        self.ui.image_path_le.setEnabled(True)
        self.ui.select_directory_bt.setEnabled(True)
        self.ui.select_output_bt.setEnabled(True)
        self.ui.output_path_le.setEnabled(True)
        self.ui.time_block_gb_sb.setEnabled(True)
        self.ui.batch_size_sb.setEnabled(True)
        self.ui.sub_fov_size_sb.setEnabled(True)
        self.ui.over_cut_sb.setEnabled(True)
        self.ui.num_workers_sb.setEnabled(True)
        self.ui.fov_start_x_sb.setEnabled(True)
        self.ui.fov_start_y_sb.setEnabled(True)
        self.ui.start_anlz_bt.setEnabled(True)

        self.ui.stop_anlz_bt.setEnabled(False)

    def display_text(self, print_message):
        self.ui.print_info_tb.moveCursor(QtGui.QTextCursor.End)  # move cursor to the end of the text

        if print_message.startswith('\r'):  # should perform like print('\r print_message', end='')
            print_message = print_message.strip("\r")
            lastLine = self.ui.print_info_tb.textCursor()  # instantiate the cursor
            lastLine.select(QtGui.QTextCursor.LineUnderCursor)  # select the entire line
            lastLine.removeSelectedText()  # remove the selected text
            self.ui.print_info_tb.insertPlainText(print_message)  # replace the end line with the new text

        else:  # should perform like print
            # self.ui.print_info_tb.append(print_message+'\n')  # append will automatically start a new line
            self.ui.print_info_tb.insertPlainText(print_message+'\n')

        # # test
        # for i in range(10):
        #     self.ui.print_info_tb.append('block:' + str(i))  # write new text at the next line
        #     self.ui.print_info_tb.append(' ')
        #     for j in range(3):
        #         print_message = '\r analyzing'+str(i) + '_' + str(j)
        #
        #         print_message = print_message.strip("\r")
        #
        #         self.ui.print_info_tb.moveCursor(QtGui.QTextCursor.End)  # move to the end of the text
        #         lastLine = self.ui.print_info_tb.textCursor()
        #         lastLine.select(QtGui.QTextCursor.LineUnderCursor)
        #         lastLine.removeSelectedText()  # remove this line of text
        #         self.ui.print_info_tb.insertPlainText(print_message)  # write new text at the same line

    def get_ui_parameters(self):
        loc_model_path = self.ui.model_path_le.text()
        image_path = self.ui.image_path_le.text()
        output_path = self.ui.output_path_le.text()
        time_block_gb = self.ui.time_block_gb_sb.value()
        batch_size = self.ui.batch_size_sb.value()
        sub_fov_size = self.ui.sub_fov_size_sb.value()
        over_cut = self.ui.over_cut_sb.value()
        num_workers = self.ui.num_workers_sb.value()
        fov_xy_start = [self.ui.fov_start_x_sb.value(), self.ui.fov_start_y_sb.value()]

        return loc_model_path, image_path, output_path, time_block_gb, batch_size, \
               sub_fov_size, over_cut, num_workers, fov_xy_start

    def reset(self):
        """
        Reset parameters.
        """
        pass


class DeepLoc_Analyzer_Thread(QtCore.QThread):

    ui_print_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.loc_model_path = None
        self.image_path = None
        self.output_path = None
        self.time_block_gb = None
        self.batch_size = None
        self.sub_fov_size = None
        self.over_cut = None
        self.num_workers = None
        self.fov_xy_start = None

    def run(self):
        ailoc.deeploc.app_inference(self.loc_model_path,
                                    self.image_path,
                                    self.output_path,
                                    self.time_block_gb,
                                    self.batch_size,
                                    self.sub_fov_size,
                                    self.over_cut,
                                    self.num_workers,
                                    self.fov_xy_start,
                                    self.ui_print_signal)

    def set_parameters(self, loc_model_path, image_path, output_path, time_block_gb, batch_size, sub_fov_size, over_cut,
                       num_workers, fov_xy_start):
        self.loc_model_path = loc_model_path
        self.image_path = image_path
        self.output_path = output_path
        self.time_block_gb = time_block_gb
        self.batch_size = batch_size
        self.sub_fov_size = sub_fov_size
        self.over_cut = over_cut
        self.num_workers = num_workers
        self.fov_xy_start = fov_xy_start

def app_inference(loc_model_path,
                  image_path,
                  save_path,
                  time_block_gb,
                  batch_size,
                  sub_fov_size,
                  over_cut,
                  num_workers,
                  fov_xy_start,
                  ui_print_signal=None):

    # load the completely trained model
    with open(loc_model_path, 'rb') as f:
        deeploc_model = torch.load(f)

    # plot evaluation performance during the training
    ailoc.common.plot_train_record(deeploc_model)

    deeploc_analyzer = ailoc.common.SmlmDataAnalyzer(loc_model=deeploc_model,
                                                     tiff_path=image_path,
                                                     output_path=save_path,
                                                     time_block_gb=time_block_gb,
                                                     batch_size=batch_size,
                                                     sub_fov_size=sub_fov_size,
                                                     over_cut=over_cut,
                                                     num_workers=num_workers,
                                                     fov_xy_start=fov_xy_start,
                                                     ui_print_signal=ui_print_signal)

    # check the output of a single frame
    deeploc_analyzer.check_single_frame_output(frame_num=11)

    # start the analysis
    image_shape, fov_xy_nm, preds_rescale_array = deeploc_analyzer.divide_and_conquer()

    print('analysis finished ! the file containing results is:', save_path)


def app_inference_gui():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    window = DeepLoc_Analyzer_GUI()
    window.show()
    app.exec_()


def app_learning():
    # todo
    pass


def app_learning_gui():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    window = DeepLoc_Learning_GUI()
    window.show()
    app.exec_()


def app_main_window():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    window = DeepLoc_MainWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    # app_inference_gui()
    # app_learning_gui()
    app_main_window()
