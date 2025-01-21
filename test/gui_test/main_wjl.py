# -*- coding: utf-8 -*-
import random

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot
from QtUi_wjl import Ui_MainWindow
import sys
import time
import numpy as np


class PI_Thread(QThread):
    # 主线程接受信号，线程处理

    # signal_set_piz = pyqtSignal(float)
    signal_ui = pyqtSignal(float)

    def __init__(self):
        super(PI_Thread, self).__init__()
        # self.signal_set_piz.connect(self.set_piz)

    # @pyqtSlot(float)
    def run(self):
        """
       set_piz 设置z位置
       """
        time.sleep(3)
        z = random.randint(1, 50)
        print("z====", z)
        self.signal_ui.emit(float(z))


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_Start.clicked.connect(self.workStart)
        self.pushButton_Stop.clicked.connect(self.workStop)

    def flush(self, count):
        self.label.setText(str(count))
        self.pushButton_Start.setEnabled(True)

    def workStart(self):
        print('button start.')

        self.pushButton_Start.setEnabled(False)

        self.pi_thread = PI_Thread()
        self.pi_thread.signal_ui.connect(self.flush)
        self.pi_thread.start()  # create a new thread and put the run function in it to run
        # self.pi_thread.run()  # simply call the run function in this main thread

    def workStop(self):
        print('button stop.')
        try:
            self.pi_thread.terminate()
            self.pushButton_Start.setEnabled(True)
        except:
            print("no thread to terminate")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())
