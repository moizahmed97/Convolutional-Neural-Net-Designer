#!/usr/bin/env python


from PyQt5 import QtCore, QtGui, QtWidgets
from LeNetChooseParameter import Ui_MainWindow as LNCP
from AlexNetChooseParameter import Ui_MainWindow as ANCP
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 250)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 60, 113, 81))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setMouseTracking(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(282, 60, 113, 81))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 10, 151, 20))
        self.label.setMinimumSize(QtCore.QSize(60, 0))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 22))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.openWin)
        self.pushButton_2.clicked.connect(self.openWin1)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "AlexNet"))
        self.pushButton_2.setText(_translate("MainWindow", "Lenet"))
        self.label.setText(_translate("MainWindow", "Choose an architecture"))


    def openWin1(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = LNCP()
        self.ui.setupUi(self.window)
        MainWindow.hide()
        from PIL import Image
        # creating a object
        im = Image.open(r"LeNet.png")
        im.show()
        self.window.show()


    def openWin(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = ANCP()
        self.ui.setupUi(self.window)
        MainWindow.hide()
        from PIL import Image
        # creating a object
        im = Image.open(r"AlexNet.png")
        im.show()
        self.window.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    #ui.pushButton.clicked.connect(AlexNet)
    #ui.pushButton_2.clicked.connect(openWin)
    MainWindow.show()
    sys.exit(app.exec_())
