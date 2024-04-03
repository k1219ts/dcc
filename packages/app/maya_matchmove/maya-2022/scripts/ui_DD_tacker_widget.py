# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DD_tacker.ui'
#
# Created: Tue Aug  1 14:59:19 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtGui
from PySide2 import QtCore, QtGui, QtWidgets

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

class ui_DD_tacker_Dialog(object):
    def setupUi(self, DD_tacker_Dialog):
        DD_tacker_Dialog.setObjectName(_fromUtf8("DD_tacker_Dialog"))
        DD_tacker_Dialog.setWindowModality(QtCore.Qt.NonModal)
        DD_tacker_Dialog.setEnabled(True)
        DD_tacker_Dialog.resize(406, 266)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DD_tacker_Dialog.sizePolicy().hasHeightForWidth())
        DD_tacker_Dialog.setSizePolicy(sizePolicy)
        DD_tacker_Dialog.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)
        DD_tacker_Dialog.setAutoFillBackground(False)
        DD_tacker_Dialog.setSizeGripEnabled(False)
        #self.centralwidget.addWidget(DD_tacker_Dialog)
        #self.centralwidget.setGeometry(QtCore.QRect(0, 0, 411, 261))
        #self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.groupCam = QtWidgets.QGroupBox(DD_tacker_Dialog)
        self.groupCam.setGeometry(QtCore.QRect(10, 10, 391, 60))
        self.groupCam.setMouseTracking(False)
        self.groupCam.setAutoFillBackground(False)
        self.groupCam.setFlat(False)
        self.groupCam.setCheckable(False)
        self.groupCam.setObjectName(_fromUtf8("groupCam"))
        self.listCam = QtWidgets.QComboBox(self.groupCam)
        self.listCam.setGeometry(QtCore.QRect(10, 22, 271, 23))
        self.listCam.setObjectName(_fromUtf8("listCam"))
        self.groupStat = QtWidgets.QGroupBox(DD_tacker_Dialog)
        self.groupStat.setGeometry(QtCore.QRect(10, 70, 391, 171))
        self.groupStat.setObjectName(_fromUtf8("groupStat"))
        self.listLoc = QtWidgets.QListWidget(self.groupStat)
        self.listLoc.setGeometry(QtCore.QRect(10, 20, 261, 141))
        self.listLoc.setObjectName(_fromUtf8("listLoc"))
        self.enbLoc = QtWidgets.QLabel(DD_tacker_Dialog)
        self.enbLoc.setGeometry(QtCore.QRect(13, 243, 381, 21))
        self.enbLoc.setObjectName(_fromUtf8("enbLoc"))
        self.btnDisable = QtWidgets.QPushButton(DD_tacker_Dialog)
        self.btnDisable.setGeometry(QtCore.QRect(303, 202, 91, 27))
        self.btnDisable.setObjectName(_fromUtf8("btnDisable"))
        self.btnEnable = QtWidgets.QPushButton(DD_tacker_Dialog)
        self.btnEnable.setGeometry(QtCore.QRect(303, 172, 91, 27))
        self.btnEnable.setObjectName(_fromUtf8("btnEnable"))
        self.btnReload = QtWidgets.QPushButton(DD_tacker_Dialog)
        self.btnReload.setGeometry(QtCore.QRect(301, 30, 91, 27))
        self.btnReload.setObjectName(_fromUtf8("btnReload"))

        self.retranslateUi(DD_tacker_Dialog)
        QtCore.QMetaObject.connectSlotsByName(DD_tacker_Dialog)

    def retranslateUi(self, DD_tacker_Dialog):
        DD_tacker_Dialog.setWindowTitle(_translate("DD_tacker_Dialog", "DD_Tacker - matchmove", None))
        self.groupCam.setTitle(_translate("DD_tacker_Dialog", "Camera", None))
        self.groupStat.setTitle(_translate("DD_tacker_Dialog", "Status", None))
        self.enbLoc.setText(_translate("DD_tacker_Dialog", "select Camera and one Locator", None))
        self.btnDisable.setText(_translate("DD_tacker_Dialog", "Disable", None))
        self.btnEnable.setText(_translate("DD_tacker_Dialog", "Enable", None))
        self.btnReload.setText(_translate("DD_tacker_Dialog", "Reload", None))

