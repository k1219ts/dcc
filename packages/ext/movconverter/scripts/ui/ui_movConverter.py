# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_movConverter.ui',
# licensing of 'ui_movConverter.ui' applies.
#
# Created: Fri Jan 15 17:09:01 2021
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(858, 282)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input_frame = QtWidgets.QFrame(self.centralwidget)
        self.input_frame.setGeometry(QtCore.QRect(270, 10, 581, 51))
        self.input_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input_frame.setObjectName("input_frame")
        self.input_lineEdit = QtWidgets.QLineEdit(self.input_frame)
        self.input_lineEdit.setGeometry(QtCore.QRect(70, 10, 471, 30))
        self.input_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.input_lineEdit.setObjectName("input_lineEdit")
        self.input_label = QtWidgets.QLabel(self.input_frame)
        self.input_label.setGeometry(QtCore.QRect(10, 17, 41, 21))
        self.input_label.setObjectName("input_label")
        self.findInput_pushButton = QtWidgets.QToolButton(self.input_frame)
        self.findInput_pushButton.setGeometry(QtCore.QRect(550, 10, 23, 26))
        self.findInput_pushButton.setObjectName("findInput_pushButton")
        self.output_frame = QtWidgets.QFrame(self.centralwidget)
        self.output_frame.setGeometry(QtCore.QRect(270, 70, 581, 131))
        self.output_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.output_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.output_frame.setObjectName("output_frame")
        self.output_lineEdit = QtWidgets.QLineEdit(self.output_frame)
        self.output_lineEdit.setGeometry(QtCore.QRect(70, 10, 471, 30))
        self.output_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.output_lineEdit.setObjectName("output_lineEdit")
        self.output_label = QtWidgets.QLabel(self.output_frame)
        self.output_label.setGeometry(QtCore.QRect(10, 17, 51, 21))
        self.output_label.setObjectName("output_label")
        self.codec_comboBox = QtWidgets.QComboBox(self.output_frame)
        self.codec_comboBox.setGeometry(QtCore.QRect(70, 45, 151, 28))
        self.codec_comboBox.setObjectName("codec_comboBox")
        self.codec_label = QtWidgets.QLabel(self.output_frame)
        self.codec_label.setGeometry(QtCore.QRect(10, 50, 51, 21))
        self.codec_label.setObjectName("codec_label")
        self.convert_pushButton = QtWidgets.QPushButton(self.output_frame)
        self.convert_pushButton.setGeometry(QtCore.QRect(450, 80, 121, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.convert_pushButton.sizePolicy().hasHeightForWidth())
        self.convert_pushButton.setSizePolicy(sizePolicy)
        self.convert_pushButton.setObjectName("convert_pushButton")
        self.findOutput_pushButton = QtWidgets.QToolButton(self.output_frame)
        self.findOutput_pushButton.setGeometry(QtCore.QRect(550, 10, 23, 26))
        self.findOutput_pushButton.setObjectName("findOutput_pushButton")
        self.machineType_comboBox = QtWidgets.QComboBox(self.output_frame)
        self.machineType_comboBox.setGeometry(QtCore.QRect(320, 80, 120, 41))
        self.machineType_comboBox.setObjectName("machineType_comboBox")
        self.machineType_comboBox.addItem("")
        self.machineType_comboBox.addItem("")
        self.remakeMov_checkBox = QtWidgets.QCheckBox(self.output_frame)
        self.remakeMov_checkBox.setGeometry(QtCore.QRect(10, 90, 111, 25))
        self.remakeMov_checkBox.setObjectName("remakeMov_checkBox")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 250, 841, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.info_label = QtWidgets.QLabel(self.centralwidget)
        self.info_label.setGeometry(QtCore.QRect(10, 220, 841, 21))
        self.info_label.setObjectName("info_label")
        self.logo_label = QtWidgets.QLabel(self.centralwidget)
        self.logo_label.setGeometry(QtCore.QRect(14, 10, 241, 191))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logo_label.sizePolicy().hasHeightForWidth())
        self.logo_label.setSizePolicy(sizePolicy)
        self.logo_label.setObjectName("logo_label")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.input_label.setText(QtWidgets.QApplication.translate("MainWindow", "input", None, -1))
        self.findInput_pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "...", None, -1))
        self.output_label.setText(QtWidgets.QApplication.translate("MainWindow", "output", None, -1))
        self.codec_label.setText(QtWidgets.QApplication.translate("MainWindow", "codec", None, -1))
        self.convert_pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "Convert", None, -1))
        self.findOutput_pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "...", None, -1))
        self.machineType_comboBox.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "LOCAL", None, -1))
        self.machineType_comboBox.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "TRACTOR", None, -1))
        self.remakeMov_checkBox.setText(QtWidgets.QApplication.translate("MainWindow", "remake Mov", None, -1))
        self.info_label.setText(QtWidgets.QApplication.translate("MainWindow", "Info", None, -1))
        self.logo_label.setText(QtWidgets.QApplication.translate("MainWindow", "label", None, -1))
