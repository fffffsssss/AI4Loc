import torch
from PyQt5 import QtCore, QtGui, QtWidgets
import os

import ailoc.common
from ailoc.deeploc.ui_files.ui_deeploc_analyzer import Ui_DeepLoc_Analyzer


class DeepLoc_Analyzer_GUI(QtWidgets.QWidget, Ui_DeepLoc_Analyzer):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DeepLoc_Analyzer()
        self.ui.setupUi(self)
        self.show()

    @QtCore.pyqtSlot()
    def on_select_model_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load DeepLoc model",
                                                          "./",
                                                          "Pytorch files (*.pt*);;All files (*)")

        self.ui.model_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_image_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getOpenFileName(self,
                                                          "Load image",
                                                          "./",
                                                          "Image files (*.tif*);;All files (*)")

        self.ui.image_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_select_directory_bt_clicked(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                               "Select directory",
                                                               "./",
                                                               QtWidgets.QFileDialog.ShowDirsOnly)

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
                        os.path.split(image_path)[-2].split('/')[-1] + '.csv'
        else:
            save_path = image_path + '/' + \
                        os.path.split(model_path)[-1].split('.')[0] + '_' + \
                        os.path.split(image_path)[-1].split('/')[-1] + '.csv'

        directory = QtWidgets.QFileDialog.getSaveFileName(self,
                                                          "Select output file",
                                                          save_path,
                                                          "Molecule list files (*.csv*);;All files (*)")
        self.ui.output_path_le.setText(directory[0])

    @QtCore.pyqtSlot()
    def on_start_anlz_bt_clicked(self):
        for j in range(100):
            self.ui.print_info_tb.append(f'analyze {j} block')
            for i in range(10):
                current_info = f'\ranalyzing{i}' if i != 0 else f'analyzing{i}'
                if current_info.startswith('\r'):
                    lastLine = self.ui.print_info_tb.textCursor()
                    lastLine.select(QtGui.QTextCursor.LineUnderCursor)
                    lastLine.removeSelectedText()
                    self.ui.print_info_tb.moveCursor(QtGui.QTextCursor.StartOfLine, QtGui.QTextCursor.MoveAnchor)
                    current_info = current_info.strip("\r")

                # self.ui.print_info_tb.append(current_info)
                self.ui.print_info_tb.insertPlainText(current_info)


def deeploc_analyze(loc_model_path,
                    image_path,
                    save_path,
                    time_block_gb,
                    batch_size,
                    sub_fov_size,
                    over_cut,
                    num_workers,
                    fov_xy_start):

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
                                                     fov_xy_start=fov_xy_start)

    # check the output of a single frame
    deeploc_analyzer.check_single_frame_output(frame_num=12345)

    # start the analysis
    image_shape, fov_xy_nm, preds_rescale_array = deeploc_analyzer.divide_and_conquer()


def deeploc_analyze_gui():
    app = QtWidgets.QApplication([])
    window = DeepLoc_Analyzer_GUI()
    app.exec_()


def deeploc_train():
    # todo
    pass


def deeploc_train_gui():
    # todo
    pass


if __name__ == "__main__":
    deeploc_analyze_gui()
