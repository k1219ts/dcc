# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'txAttrToolsUI.ui'
#
# Created: Fri Jan  4 13:13:42 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(1200, 818)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.gridLayout.addWidget(self.pushButton, 1, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.treeWidget = QtWidgets.QTreeWidget(Form)
        self.treeWidget.setRootIsDecorated(True)
        self.treeWidget.setObjectName(_fromUtf8("treeWidget"))
        self.treeWidget.headerItem().setText(0, _fromUtf8("attribute Name"))
        self.treeWidget.headerItem().setText(1, _fromUtf8("value"))
        self.treeWidget.headerItem().setText(2, _fromUtf8("reset"))
        self.treeWidget.header().setVisible(True)
        self.treeWidget.setStyleSheet("background-color:#333333")
        self.gridLayout.addWidget(self.treeWidget, 0, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Attribute Controller", None))
        self.pushButton.setText(_translate("Form", "OK", None))

