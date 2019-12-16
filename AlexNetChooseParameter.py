# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ChooseParameters.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QCoreApplication
import AlexNet

class Ui_MainWindow(object):

    def dialog(self):
        msg = QMessageBox()
        msg.setText("Training\nPlease Wait.")
        msg.setModal(False)
        msg.show()
        QCoreApplication.processEvents()

    def openWin(self):
        from progress import Ui_MainWindow as CP
        self.window = QtWidgets.QMainWindow()
        self.ui = CP()
        self.ui.setupUi(self.window,self.model2,AlexNet.data)
        self.MainWindow.hide()
        self.window.show()

    def train(self):
        strides = self.spinBox.value()
        kernal_size = self.spinBox_2.value()
        self.model2 = AlexNet.AlexNetBuild(kernal_size,strides)
        self.dialog()
        self.openWin()

    def loadModel(self):
        from tensorflow.python.keras.models import load_model
        from visual import Ui_MainWindow as CP
        model = load_model("AlexNet-model.h5")
        self.window = QtWidgets.QMainWindow()
        self.ui = CP()
        self.ui.setupUi(self.window,model,AlexNet.data)
        self.MainWindow.hide()
        self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 250)
        self.MainWindow = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(182, 10, 121, 16))
        self.label.setObjectName("label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(160, 40, 171, 161))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(0, 170, 141, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.spinBox.setWrapping(True)
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(3)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_2.addWidget(self.spinBox)
        self.spinBox_2 = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.spinBox_2.setWrapping(True)
        self.spinBox_2.setMinimum(2)
        self.spinBox_2.setMaximum(5)
        self.spinBox_2.setObjectName("spinBox_2")
        self.verticalLayout_2.addWidget(self.spinBox_2)
        self.spinBox_3 = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.spinBox_3.setWrapping(True)
        self.spinBox_3.setMinimum(16)
        self.spinBox_3.setMaximum(36)
        self.spinBox_3.setObjectName("spinBox_3")
        self.verticalLayout_2.addWidget(self.spinBox_3)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(380, 170, 113, 32))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton.clicked.connect(self.train)
        self.pushButton_2.clicked.connect(self.loadModel)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Choose Parameters"))
        self.label_2.setText(_translate("MainWindow", "Strides:"))
        self.label_3.setText(_translate("MainWindow", "Kernel Size:"))
        self.label_4.setText(_translate("MainWindow", "Filters:"))
        self.pushButton.setText(_translate("MainWindow", "Start training"))
        self.pushButton_2.setText(_translate("MainWindow", "Pre-Trained Model"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
