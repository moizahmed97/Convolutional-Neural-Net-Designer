# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'visual.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtWidgets
from  lenet import plot_example_errors, plot_conv1_output, plot_conv2_output, plot_confusion_matrix, plot_image

class Ui_MainWindow(object):



    def setupUi(self, MainWindow, model2, data):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 250)
        self.model2 = model2
        self.data = data
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(210, 10, 91, 16))
        self.label.setObjectName("label")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(30, 70, 451, 80))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 0, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 1, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 500, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton_4.clicked.connect(self.pltConv2)
        self.pushButton_2.clicked.connect(self.pltConv1)
        self.pushButton_6.clicked.connect(self.pltError)
        self.pushButton.clicked.connect(self.pltConf)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Visualizations"))
        self.pushButton_2.setText(_translate("MainWindow", "convolutional layer 1"))
        self.pushButton.setText(_translate("MainWindow", "Confusion Matrix"))
        self.pushButton_4.setText(_translate("MainWindow", "convolutional layer 2"))
        self.pushButton_6.setText(_translate("MainWindow", "Plot example error"))

    def pltConv2(self):
        plot_image(self.data)
        plot_conv2_output(self.data,self.model2)

    def pltConv1(self):
        plot_image(self.data)
        plot_conv1_output(self.data,self.model2)

    def pltError(self):
        plot_example_errors(self.data,self.model2)

    def pltConf(self):
        plot_confusion_matrix(self.data,self.model2)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
