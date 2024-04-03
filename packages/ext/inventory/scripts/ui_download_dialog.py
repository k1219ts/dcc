# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'download_dialog.ui'
#
# Created: Thu Jul 12 18:11:13 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

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
        Dialog.resize(809, 639)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.treeWidget = QtWidgets.QTreeWidget(Dialog)
        self.treeWidget.setObjectName(_fromUtf8("treeWidget"))
        self.treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.treeWidget, 2, 0, 1, 2)
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.gridLayout.addWidget(self.progressBar, 1, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(697, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.locationLabel = QtWidgets.QLabel(Dialog)
        self.locationLabel.setObjectName(_fromUtf8("locationLabel"))
        self.horizontalLayout.addWidget(self.locationLabel)
        self.locationLineEdit = QtWidgets.QLineEdit(Dialog)
        self.locationLineEdit.setObjectName(_fromUtf8("locationLineEdit"))
        self.horizontalLayout.addWidget(self.locationLineEdit)
        self.selectDirButton = QtWidgets.QPushButton(Dialog)
        self.selectDirButton.setObjectName(_fromUtf8("selectDirButton"))
        self.horizontalLayout.addWidget(self.selectDirButton)
        self.downloadButton = QtWidgets.QPushButton(Dialog)
        self.downloadButton.setObjectName(_fromUtf8("downloadButton"))
        self.horizontalLayout.addWidget(self.downloadButton)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.closeButton = QtWidgets.QPushButton(Dialog)
        self.closeButton.setObjectName(_fromUtf8("closeButton"))
        self.gridLayout.addWidget(self.closeButton, 3, 1, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.locationLabel.setText(_translate("Dialog", "Location", None))
        self.selectDirButton.setText(_translate("Dialog", "...", None))
        self.downloadButton.setText(_translate("Dialog", "Download", None))
        self.closeButton.setText(_translate("Dialog", "Close", None))

