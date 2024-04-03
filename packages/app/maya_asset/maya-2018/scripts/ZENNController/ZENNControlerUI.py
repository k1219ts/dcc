# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ZENNControlerUI.ui'
#
# Created: Thu Jan  3 14:23:28 2019
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
        Form.resize(382, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.znImportUpdateMeshBtn = QtWidgets.QPushButton(self.groupBox)
        self.znImportUpdateMeshBtn.setObjectName(_fromUtf8("znImportUpdateMeshBtn"))
        self.gridLayout_2.addWidget(self.znImportUpdateMeshBtn, 0, 0, 1, 1)
        self.znImportUpdateCurveBtn = QtWidgets.QPushButton(self.groupBox)
        self.znImportUpdateCurveBtn.setObjectName(_fromUtf8("znImportUpdateCurveBtn"))
        self.gridLayout_2.addWidget(self.znImportUpdateCurveBtn, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.znGenerateUpdateBtn = QtWidgets.QPushButton(self.groupBox_2)
        self.znGenerateUpdateBtn.setObjectName(_fromUtf8("znGenerateUpdateBtn"))
        self.gridLayout.addWidget(self.znGenerateUpdateBtn, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.ratioLabel = QtWidgets.QLabel(self.groupBox_3)
        self.ratioLabel.setObjectName(_fromUtf8("ratioLabel"))
        self.horizontalLayout.addWidget(self.ratioLabel)
        self.ratioSlider = QtWidgets.QSlider(self.groupBox_3)
        self.ratioSlider.setMaximum(1000)
        self.ratioSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ratioSlider.setObjectName(_fromUtf8("ratioSlider"))
        self.horizontalLayout.addWidget(self.ratioSlider)
        self.ratioEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.ratioEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.ratioEdit.setMaxLength(5)
        self.ratioEdit.setObjectName(_fromUtf8("ratioEdit"))
        self.horizontalLayout.addWidget(self.ratioEdit)
        self.verticalLayout.addWidget(self.groupBox_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "ZENN Controller", None))
        self.groupBox.setTitle(_translate("Form", "ZN_Import", None))
        self.znImportUpdateMeshBtn.setText(_translate("Form", "Update Mesh", None))
        self.znImportUpdateCurveBtn.setText(_translate("Form", "Update Curve", None))
        self.groupBox_2.setTitle(_translate("Form", "ZN_Generate", None))
        self.znGenerateUpdateBtn.setText(_translate("Form", "Update", None))
        self.groupBox_3.setTitle(_translate("Form", "ZN_StrandsViewer", None))
        self.ratioLabel.setText(_translate("Form", "display ratio", None))
        self.ratioEdit.setText(_translate("Form", "1.000", None))

