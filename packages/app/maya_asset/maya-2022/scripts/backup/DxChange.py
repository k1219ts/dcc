# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DxChange.ui'
#
# Created: Tue Jun  5 12:02:03 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtGui # original

# new
from PySide2 import QtWidgets as QtGui
from PySide2 import QtCore
import PySide2

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

class Ui_DxTextureChange_bymoonseok(object):
    def setupUi(self, DxTextureChange_bymoonseok):
        DxTextureChange_bymoonseok.setObjectName(_fromUtf8("DxTextureChange_bymoonseok"))
        DxTextureChange_bymoonseok.resize(213, 115)
        self.Dev = QtGui.QPushButton(DxTextureChange_bymoonseok)
        self.Dev.setGeometry(QtCore.QRect(10, 40, 97, 31))
        self.Dev.setObjectName(_fromUtf8("Dev"))
        self.line_3 = QtGui.QFrame(DxTextureChange_bymoonseok)
        self.line_3.setGeometry(QtCore.QRect(10, 20, 191, 16))
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.Version = QtGui.QLineEdit(DxTextureChange_bymoonseok)
        self.Version.setGeometry(QtCore.QRect(90, 80, 41, 31))
        self.Version.setObjectName(_fromUtf8("Version"))
        self.Pub = QtGui.QPushButton(DxTextureChange_bymoonseok)
        self.Pub.setGeometry(QtCore.QRect(110, 40, 97, 31))
        self.Pub.setObjectName(_fromUtf8("Pub"))
        self.Change = QtGui.QPushButton(DxTextureChange_bymoonseok)
        self.Change.setGeometry(QtCore.QRect(140, 80, 61, 31))
        self.Change.setObjectName(_fromUtf8("Change"))
        self.label = QtGui.QLabel(DxTextureChange_bymoonseok)
        self.label.setGeometry(QtCore.QRect(10, 80, 101, 31))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_5 = QtGui.QLabel(DxTextureChange_bymoonseok)
        self.label_5.setGeometry(QtCore.QRect(10, 0, 211, 21))
        font = PySide2.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))

        self.retranslateUi(DxTextureChange_bymoonseok)
        QtCore.QMetaObject.connectSlotsByName(DxTextureChange_bymoonseok)

    def retranslateUi(self, DxTextureChange_bymoonseok):
        DxTextureChange_bymoonseok.setWindowTitle(_translate("DxTextureChange_bymoonseok", "Form", None))
        self.Dev.setText(_translate("DxTextureChange_bymoonseok", "Dev", None))
        self.Pub.setText(_translate("DxTextureChange_bymoonseok", "Pub", None))
        self.Change.setText(_translate("DxTextureChange_bymoonseok", "change", None))
        self.label.setText(_translate("DxTextureChange_bymoonseok", "txVersion :", None))
        self.label_5.setText(_translate("DxTextureChange_bymoonseok", "DxChange_by.moonseok", None))

