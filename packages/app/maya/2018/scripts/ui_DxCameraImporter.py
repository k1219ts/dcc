# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DxCameraImporter.ui'
#
# Created: Wed May 24 10:59:42 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtWidgets

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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(650, 600)
        self.gridLayout_3 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.showLabel = QtWidgets.QLabel(Dialog)
        self.showLabel.setObjectName(_fromUtf8("showLabel"))
        self.horizontalLayout.addWidget(self.showLabel)
        self.showCombo = QtWidgets.QComboBox(Dialog)
        self.showCombo.setObjectName(_fromUtf8("showCombo"))
        self.horizontalLayout.addWidget(self.showCombo)
        self.seqLabel = QtWidgets.QLabel(Dialog)
        self.seqLabel.setObjectName(_fromUtf8("seqLabel"))
        self.horizontalLayout.addWidget(self.seqLabel)
        self.seqCombo = QtWidgets.QComboBox(Dialog)
        self.seqCombo.setObjectName(_fromUtf8("seqCombo"))
        self.horizontalLayout.addWidget(self.seqCombo)
        self.shotLabel = QtWidgets.QLabel(Dialog)
        self.shotLabel.setObjectName(_fromUtf8("shotLabel"))
        self.horizontalLayout.addWidget(self.shotLabel)
        self.shotCombo = QtWidgets.QComboBox(Dialog)
        self.shotCombo.setObjectName(_fromUtf8("shotCombo"))
        self.horizontalLayout.addWidget(self.shotCombo)
        self.gridLayout_3.addLayout(self.horizontalLayout, 0, 0, 1, 3)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.cameraTree = QtWidgets.QTreeWidget(self.groupBox)
        self.cameraTree.setObjectName(_fromUtf8("cameraTree"))
        self.cameraTree.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.cameraTree, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 1, 0, 1, 3)
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.fileTree = QtWidgets.QTreeWidget(self.groupBox_2)
        self.fileTree.setObjectName(_fromUtf8("fileTree"))
        self.fileTree.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout_2.addWidget(self.fileTree, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_2, 2, 0, 1, 3)
        spacerItem = QtWidgets.QSpacerItem(359, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 3, 0, 1, 1)
        self.importButton = QtWidgets.QPushButton(Dialog)
        self.importButton.setObjectName(_fromUtf8("importButton"))
        self.gridLayout_3.addWidget(self.importButton, 3, 1, 1, 1)
        self.cancelButton = QtWidgets.QPushButton(Dialog)
        self.cancelButton.setObjectName(_fromUtf8("cancelButton"))
        self.gridLayout_3.addWidget(self.cancelButton, 3, 2, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "DxCamera Importer", None))
        self.showLabel.setText(_translate("Dialog", "Show", None))
        self.seqLabel.setText(_translate("Dialog", "Seq", None))
        self.shotLabel.setText(_translate("Dialog", "Shot", None))
        self.groupBox.setTitle(_translate("Dialog", "Cameras", None))
        self.groupBox_2.setTitle(_translate("Dialog", "Files", None))
        self.importButton.setText(_translate("Dialog", "Import", None))
        self.cancelButton.setText(_translate("Dialog", "Cancel", None))
