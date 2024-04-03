# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'createActionSetGUI.ui'
#
# Created: Tue Sep 22 16:41:43 2015
#      by: PyQt4 UI code generator 4.10.3
# 
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtGui
from Qt import QtCore, QtGui

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
        Form.resize(237, 253)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.btnImport = QtGui.QPushButton(Form)
        self.btnImport.setObjectName(_fromUtf8("btnImport"))
        self.verticalLayout.addWidget(self.btnImport)
        self.btnTime = QtGui.QPushButton(Form)
        self.btnTime.setObjectName(_fromUtf8("btnTime"))
        self.verticalLayout.addWidget(self.btnTime)
        self.btnOrigin = QtGui.QPushButton(Form)
        self.btnOrigin.setObjectName(_fromUtf8("btnOrigin"))
        self.verticalLayout.addWidget(self.btnOrigin)
        self.btnCreateAction = QtGui.QPushButton(Form)
        self.btnCreateAction.setObjectName(_fromUtf8("btnCreateAction"))
        self.verticalLayout.addWidget(self.btnCreateAction)
        
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.btnImport.setText(_translate("Form", "Import File", None))
        self.btnTime.setText(_translate("Form", "Timeline Set", None))
        self.btnOrigin.setText(_translate("Form", "Origin Asix", None))
        self.btnCreateAction.setText(_translate("Form", "Create Action", None))
