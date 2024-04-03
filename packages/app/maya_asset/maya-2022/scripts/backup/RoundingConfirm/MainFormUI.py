# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainFormUI.ui'
#
# Created: Thu Jul 26 16:45:47 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# from PyQt4 import QtCore, QtGui
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets as QtGui

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
        Form.resize(400, 300)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.baseModelEdit = QtGui.QLineEdit(Form)
        self.baseModelEdit.setObjectName(_fromUtf8("baseModelEdit"))
        self.gridLayout.addWidget(self.baseModelEdit, 0, 1, 1, 1)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.childTargetList = QtGui.QListWidget(Form)
        self.childTargetList.setObjectName(_fromUtf8("childTargetEdit"))
        self.gridLayout.addWidget(self.childTargetList, 1, 0, 1, 2)
        self.execBtn = QtGui.QPushButton(Form)
        self.execBtn.setObjectName(_fromUtf8("execBtn"))
        self.gridLayout.addWidget(self.execBtn, 2, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "BaseModel", None))
        self.execBtn.setText(_translate("Form", "Execute", None))

