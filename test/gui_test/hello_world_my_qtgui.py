from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
import pathlib
from ui_file.mygui import Ui_MainWindow


# class MyGUI(QtWidgets.QMainWindow):
#     def __init__(self):
#         super(MyGUI, self).__init__()
#         uic.loadUi('./ui_file/mygui.ui', self)
#         self.show()
#
#         self.pushButton.clicked.connect(self.login)
#
#     def login(self):
#         if self.username.text() == "admin" and self.password.text() == "admin":
#             self.textEdit.setEnabled(True)
#             self.pushButton_2.setEnabled(True)
#         else:
#             message = QtWidgets.QMessageBox()
#             message.setText("Invalid username or password")
#             message.exec_()


class MyGUI(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        # super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.show()

        # self.ui.loginbutton.clicked.connect(self.login)
        self.ui.actionclose.triggered.connect(self.close)
        self.ui.sayitbutton.clicked.connect(lambda: self.sayit(self.ui.textEdit.toPlainText()))

    @QtCore.pyqtSlot()
    def on_loginbutton_clicked(self):
        if self.ui.username.text() == "admin" and self.ui.password.text() == "admin":
            self.ui.textEdit.setEnabled(True)
            self.ui.sayitbutton.setEnabled(True)
        else:
            message = QtWidgets.QMessageBox()
            message.setText("Invalid username or password")
            message.exec_()
            self.ui.textEdit.setEnabled(False)
            self.ui.sayitbutton.setEnabled(False)

    def sayit(self, text):
        message = QtWidgets.QMessageBox()
        message.setText(text)
        message.exec_()

def main():
    app = QtWidgets.QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == "__main__":
    main()
