# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mmvSearch.ui'
#
# Created: Tue Apr 25 17:21:23 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtWidgets, QtCore

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
        Form.resize(900, 500)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.prjComboBox = QtWidgets.QComboBox(Form)
        self.prjComboBox.setMinimumSize(QtCore.QSize(120, 0))
        self.prjComboBox.setObjectName(_fromUtf8("prjComboBox"))
        self.gridLayout_3.addWidget(self.prjComboBox, 0, 0, 1, 1)
        self.seqComboBox = QtWidgets.QComboBox(Form)
        self.seqComboBox.setMinimumSize(QtCore.QSize(120, 0))
        self.seqComboBox.setObjectName(_fromUtf8("seqComboBox"))
        self.gridLayout_3.addWidget(self.seqComboBox, 0, 1, 1, 1)
        self.shotLineEdit = QtWidgets.QLineEdit(Form)
        self.shotLineEdit.setObjectName(_fromUtf8("shotLineEdit"))
        self.gridLayout_3.addWidget(self.shotLineEdit, 0, 2, 1, 1)
        self.shotPushButton = QtWidgets.QPushButton(Form)
        self.shotPushButton.setObjectName(_fromUtf8("shotPushButton"))
        self.gridLayout_3.addWidget(self.shotPushButton, 0, 3, 1, 1)
        self.mmvGroup = QtWidgets.QGroupBox(Form)
        self.mmvGroup.setObjectName(_fromUtf8("mmvGroup"))
        self.gridLayout = QtWidgets.QGridLayout(self.mmvGroup)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.cameraTree = QtWidgets.QTreeWidget(self.mmvGroup)
        self.cameraTree.setObjectName(_fromUtf8("cameraTree"))
        self.cameraTree.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.cameraTree, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.mmvGroup)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)
        self.distortTree = QtWidgets.QTreeWidget(self.mmvGroup)
        self.distortTree.setObjectName(_fromUtf8("distortTree"))
        self.distortTree.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.distortTree, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.mmvGroup, 1, 0, 1, 4)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.shotPushButton.setText(_translate("Form", "Search", None))
        self.mmvGroup.setTitle(_translate("Form", "Matchmove", None))
