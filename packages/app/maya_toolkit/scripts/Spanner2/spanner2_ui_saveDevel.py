# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spanner_dev_ui.ui'
#
# Created: Thu Feb 16 19:32:02 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

import sys
import site

# import Qt
from PySide2 import QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore


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

class saveDev_Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(379, 512)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(379, 512))
        Form.setMaximumSize(QtCore.QSize(379, 544))
        self.gridLayout_3 = QtGui.QGridLayout(Form)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout_3.addWidget(self.buttonBox, 2, 1, 1, 1)
        self.snapshot_checkBox = QtGui.QCheckBox(Form)
        self.snapshot_checkBox.setTristate(False)
        self.snapshot_checkBox.setObjectName(_fromUtf8("snapshot_checkBox"))
        self.gridLayout_3.addWidget(self.snapshot_checkBox, 2, 0, 1, 1)
        self.groupBox = QtGui.QGroupBox(Form)
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(60, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.label_5 = QtGui.QLabel(self.groupBox)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 1)
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.saveDevComment_textEdit = QtGui.QTextEdit(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveDevComment_textEdit.sizePolicy().hasHeightForWidth())
        self.saveDevComment_textEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.saveDevComment_textEdit.setFont(font)
        self.saveDevComment_textEdit.setObjectName(_fromUtf8("saveDevComment_textEdit"))
        self.gridLayout.addWidget(self.saveDevComment_textEdit, 5, 0, 1, 3)
        self.dsc_lineEdit = QtGui.QLineEdit(self.groupBox)
        self.dsc_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.dsc_lineEdit.setObjectName(_fromUtf8("dsc_lineEdit"))
        self.gridLayout.addWidget(self.dsc_lineEdit, 3, 0, 1, 3)
        self.version_spinBox = QtGui.QSpinBox(self.groupBox)
        self.version_spinBox.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.version_spinBox.setFont(font)
        self.version_spinBox.setObjectName(_fromUtf8("version_spinBox"))
        self.gridLayout.addWidget(self.version_spinBox, 1, 0, 1, 1)
        self.wipVersion_spinBox = QtGui.QSpinBox(self.groupBox)
        self.wipVersion_spinBox.setMinimumSize(QtCore.QSize(0, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.wipVersion_spinBox.setFont(font)
        self.wipVersion_spinBox.setObjectName(_fromUtf8("wipVersion_spinBox"))
        self.gridLayout.addWidget(self.wipVersion_spinBox, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 1, 0, 1, 2)
        self.groupBox1 = QtGui.QGroupBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox1.sizePolicy().hasHeightForWidth())
        self.groupBox1.setSizePolicy(sizePolicy)
        self.groupBox1.setObjectName(_fromUtf8("groupBox1"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox1)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_2 = QtGui.QLabel(self.groupBox1)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.fileName_lineEdit = QtGui.QLineEdit(self.groupBox1)
        self.fileName_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.fileName_lineEdit.setObjectName(_fromUtf8("fileName_lineEdit"))
        self.gridLayout_2.addWidget(self.fileName_lineEdit, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox1, 0, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.snapshot_checkBox.setText(_translate("Form", "Take Snapshot", None))
        self.label_3.setText(_translate("Form", "Version:", None))
        self.label_4.setText(_translate("Form", "Work Version:", None))
        self.label_5.setText(_translate("Form", "Description:", None))
        self.label.setText(_translate("Form", "memo", None))
        self.groupBox1.setTitle(_translate("Form", "Save As", None))
        self.label_2.setText(_translate("Form", "File Name:", None))
