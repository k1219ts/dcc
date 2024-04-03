# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'allPlatesImport.ui'
#
# Created: Wed Jan 11 18:43:15 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtWidgets, QtCore



class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(650, 320)


        self.searchLine = QtWidgets.QLineEdit(Form)
        self.searchLine.setGeometry(QtCore.QRect(150, 10, 110, 20))
        self.searchLine.setObjectName("searchLine")
        #self.searchLine.setText(_fromUtf8(""))

        self.searchButton = QtWidgets.QPushButton(Form)
        self.searchButton.setGeometry(QtCore.QRect(270, 10, 200, 20))
        self.searchButton.setObjectName("searchButton")


        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(14, 10, 150, 21))
        self.label.setObjectName("label")
        #self.listWidget = QtWidgets.QListWidget(Form)
        #self.listWidget.setGeometry(QtCore.QRect(10, 30, 631, 241))
        #self.listWidget.setObjectName("listWidget")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(420, 280, 121, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(540, 280, 99, 31))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Plates Arrange v02 by Kim Giuk, Dexter Digital", None))
        self.pushButton.setText(QtWidgets.QApplication.translate("Form", "Load Sequence", None))
        self.pushButton_2.setText(QtWidgets.QApplication.translate("Form", "Close", None))
        self.label.setText(QtWidgets.QApplication.translate("Form", "PROJECT NAME Typing", None))
        self.searchButton.setText(QtWidgets.QApplication.translate("Form", "SET Project Directory", None))
