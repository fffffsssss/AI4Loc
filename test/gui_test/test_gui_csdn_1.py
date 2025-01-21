import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QWaitCondition, QMutex
from PyQt5.QtWidgets import QWidget, QApplication, QDialog, QHBoxLayout, QListWidget


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(30)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.start_btn = QtWidgets.QPushButton(Form)
        self.start_btn.setObjectName("start_btn")
        self.verticalLayout.addWidget(self.start_btn)
        self.pause_btn = QtWidgets.QPushButton(Form)
        self.pause_btn.setObjectName("pause_btn")
        self.verticalLayout.addWidget(self.pause_btn)
        self.resume_btn = QtWidgets.QPushButton(Form)
        self.resume_btn.setObjectName("resume_btn")
        self.verticalLayout.addWidget(self.resume_btn)
        self.stop_btn = QtWidgets.QPushButton(Form)
        self.stop_btn.setObjectName("stop_btn")
        self.verticalLayout.addWidget(self.stop_btn)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.listWidget = QtWidgets.QListWidget(Form)
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout.addWidget(self.listWidget)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.start_btn.setText(_translate("Form", "开始"))
        self.pause_btn.setText(_translate("Form", "暂停"))
        self.resume_btn.setText(_translate("Form", "唤醒"))
        self.stop_btn.setText(_translate("Form", "停止"))


# 主程序点击后，会有弹窗
class Show_msg(QDialog):
    quit_trig = pyqtSignal()

    def __init__(self, parent=None):
        super(Show_msg, self).__init__(parent)
        self.initUi()

    def initUi(self):
        hLayout = QHBoxLayout()
        self.list_msg = QListWidget()
        hLayout.addWidget(self.list_msg)
        self.setLayout(hLayout)
        self.setWindowTitle('弹窗显示')

    def msg_setValue(self, msg):
        self.list_msg.addItem(msg)

    def closeEvent(self, event):
        self.quit_trig.emit()  # 关闭弹窗时，发出关闭信号，让主程序的进程关闭
        event.accept()
        pass


class ThreadStopTest(QWidget, Ui_Form):
    def __init__(self):
        super(ThreadStopTest, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('主程序窗口')
        self.handle()

    def handle(self):
        self.start_btn.clicked.connect(self.start_thread)
        self.stop_btn.clicked.connect(self.stop_thread)
        self.pause_btn.clicked.connect(self.pause_thread)
        self.resume_btn.clicked.connect(self.resume_thread)

    def start_thread(self):

        self.msg_dialog = Show_msg(self)  # 点击开始按钮后加载弹窗
        self.msg_dialog.show()  # 显示弹窗
        self.msg_dialog.quit_trig.connect(self.stop_thread)  # 将弹窗中的退出信号绑定到退出线程的方法上

        self.t = My_thread()  # 实例化一个线程，i并启动
        self.t.num_trig.connect(self.setValue)  # 线程信号绑定到主程序listWidget上
        self.t.num_trig.connect(self.msg_dialog.msg_setValue)  # 线程信号绑定到弹窗中的listWidget上
        self.t.start()

    # 暂停线程
    def pause_thread(self):
        if self.t:
            self.t.pause()

    # 唤醒线程
    def resume_thread(self):
        if self.t:
            self.t.resume()

    # 停止线程
    def stop_thread(self):
        if self.t:
            self.t.terminate()
            self.t = None
            self.msg_dialog.close()  # 停止线程时，关闭弹窗
        else:
            self.listWidget.addItem('线程不存在')

    def setValue(self, v):
        self.listWidget.addItem(v)


class My_thread(QThread):
    num_trig = pyqtSignal(str)

    def __init__(self):
        super(My_thread, self).__init__()
        '''
        QWaitCondition()用于多线程同步，一个线程调用QWaitCondition.wait()阻塞等待，
        直到另外一个线程调用QWaitCondition.wake()唤醒才继续往下执行
        QMutex()：是锁对象
        '''
        self._isPause = False
        self.cond = QWaitCondition()
        self.mutex = QMutex()

    def run(self) -> None:
        a = 0
        while True:
            self.mutex.lock()  # 上锁
            if self._isPause:
                self.cond.wait(self.mutex)
            self.num_trig.emit(f'item{a}')
            a += 1
            QThread.sleep(2)
            self.mutex.unlock()  # 解锁

    # 线程暂停
    def pause(self):
        self._isPause = True

    # 线程恢复
    def resume(self):
        self._isPause = False
        self.cond.wakeAll()


if __name__ == '__main__':
    # PyQt5高清屏幕自适应设置,以及让添加的高清图标显示清晰，不然designer导入的图标在程序加载时会特别模糊
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    main_win = ThreadStopTest()

    main_win.show()
    sys.exit(app.exec_())
