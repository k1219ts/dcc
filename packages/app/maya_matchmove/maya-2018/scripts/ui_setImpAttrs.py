# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_setImpAttrs.ui',
# licensing of 'ui_setImpAttrs.ui' applies.
#
# Created: Mon Sep 14 16:47:55 2020
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("setAttr imageplanes")
        MainWindow.resize(748, 140)
        MainWindow.setMinimumSize(QtCore.QSize(748, 140))
        MainWindow.setStyleSheet("background-color: rgb(80, 80, 80);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageplane_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.imageplane_textEdit.setGeometry(QtCore.QRect(20, 40, 681, 31))
        self.imageplane_textEdit.setObjectName("imageplane_textEdit")
        self.imageplane_label = QtWidgets.QLabel(self.centralwidget)
        self.imageplane_label.setGeometry(QtCore.QRect(20, 10, 250, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imageplane_label.sizePolicy().hasHeightForWidth())
        self.imageplane_label.setSizePolicy(sizePolicy)
        self.imageplane_label.setMaximumSize(QtCore.QSize(250, 30))
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.imageplane_label.setFont(font)
        self.imageplane_label.setStyleSheet("color : white;")
        self.imageplane_label.setObjectName("imageplane_label")
        self.imageplane_find_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.imageplane_find_pushButton.setGeometry(QtCore.QRect(710, 40, 30, 30))
        self.imageplane_find_pushButton.setMinimumSize(QtCore.QSize(30, 30))
        self.imageplane_find_pushButton.setMaximumSize(QtCore.QSize(30, 30))
        self.imageplane_find_pushButton.setStyleSheet("background-color: rgb(100, 100, 100);\n"
"color: rgb(255, 255, 255);")
        self.imageplane_find_pushButton.setIconSize(QtCore.QSize(20, 20))
        self.imageplane_find_pushButton.setObjectName("imageplane_find_pushButton")
        self.ok_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.ok_pushButton.setGeometry(QtCore.QRect(610, 80, 130, 40))
        self.ok_pushButton.setMinimumSize(QtCore.QSize(130, 40))
        self.ok_pushButton.setMaximumSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.ok_pushButton.setFont(font)
        self.ok_pushButton.setStyleSheet("color : white;\n"
"background-color: rgb(62, 109, 186);")
        self.ok_pushButton.setObjectName("ok_pushButton")
#        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.imageplane_label.setText(QtWidgets.QApplication.translate("MainWindow", "Imageplane image path", None, -1))
        self.imageplane_find_pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "...", None, -1))
        self.ok_pushButton.setText(QtWidgets.QApplication.translate("MainWindow", "OK", None, -1))

