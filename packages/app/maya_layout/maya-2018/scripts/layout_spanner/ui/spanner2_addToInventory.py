# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spanner2_addToInventory.ui'
#
# Created: Tue Oct 17 19:16:08 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# import Qt
# from Qt import QtGui
# from Qt import QtWidgets
# import Qt.QtWidgets as QtGui
# from Qt import QtCore

from PySide2 import QtWidgets, QtCore, QtGui


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(330, 456)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.insertInventory_pushButton = QtGui.QPushButton(Form)
        self.insertInventory_pushButton.setMinimumSize(QtCore.QSize(0, 30))
        self.insertInventory_pushButton.setMaximumSize(QtCore.QSize(16777215, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.insertInventory_pushButton.setFont(font)
        self.insertInventory_pushButton.setObjectName(_fromUtf8("insertInventory_pushButton"))
        self.gridLayout.addWidget(self.insertInventory_pushButton, 2, 1, 1, 3)
        self.del_pushButton = QtGui.QPushButton(Form)
        self.del_pushButton.setMinimumSize(QtCore.QSize(25, 25))
        self.del_pushButton.setMaximumSize(QtCore.QSize(25, 25))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.del_pushButton.setFont(font)
        self.del_pushButton.setText(_fromUtf8(""))
        self.del_pushButton.setObjectName(_fromUtf8("del_pushButton"))
        self.gridLayout.addWidget(self.del_pushButton, 2, 0, 1, 1)
        self.fileList_listWidget = QtGui.QListWidget(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.fileList_listWidget.setFont(font)
        self.fileList_listWidget.setObjectName(_fromUtf8("fileList_listWidget"))
        self.gridLayout.addWidget(self.fileList_listWidget, 1, 0, 1, 4)
        self.add_pushButton = QtGui.QPushButton(Form)
        self.add_pushButton.setMinimumSize(QtCore.QSize(25, 25))
        self.add_pushButton.setMaximumSize(QtCore.QSize(25, 25))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.add_pushButton.setFont(font)
        self.add_pushButton.setObjectName(_fromUtf8("add_pushButton"))
        self.gridLayout.addWidget(self.add_pushButton, 0, 3, 1, 1)
        self.filePath_lineEdit = QtGui.QLineEdit(Form)
        self.filePath_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.filePath_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.filePath_lineEdit.setFont(font)
        self.filePath_lineEdit.setObjectName(_fromUtf8("filePath_lineEdit"))
        self.gridLayout.addWidget(self.filePath_lineEdit, 0, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.insertInventory_pushButton.setText(_translate("Form", "ADD TO INVENTORY", None))
        self.add_pushButton.setText(_translate("Form", "+", None))
