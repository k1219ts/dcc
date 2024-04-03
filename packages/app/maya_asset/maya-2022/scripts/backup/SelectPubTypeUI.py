# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'selectExportTypeUI.ui'
#
# Created: Tue May 30 12:20:45 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

import Qt
import Qt.QtWidgets as QtGui
from Qt import QtCore

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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(399, 234)
        self.groupBox = QtGui.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 381, 181))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.verticalLayoutWidget = QtGui.QWidget(self.groupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 30, 381, 161))
        self.verticalLayoutWidget.setObjectName(_fromUtf8("verticalLayoutWidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.meshRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.meshRadioButton.setChecked(True)
        self.meshRadioButton.setObjectName(_fromUtf8("meshRadioButton"))
        self.verticalLayout.addWidget(self.meshRadioButton)
        self.envRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.envRadioButton.setObjectName(_fromUtf8("envRadioButton"))
        self.verticalLayout.addWidget(self.envRadioButton)
        self.zenvRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.zenvRadioButton.setObjectName(_fromUtf8("zenvRadioButton"))
        self.verticalLayout.addWidget(self.zenvRadioButton)
        self.shotRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.shotRadioButton.setObjectName(_fromUtf8("shotRadioButton"))
        self.verticalLayout.addWidget(self.shotRadioButton)
        self.zennRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.zennRadioButton.setObjectName(_fromUtf8("zennRadioButton"))
        self.verticalLayout.addWidget(self.zennRadioButton)
        self.zennShotRadioButton = QtGui.QRadioButton(self.verticalLayoutWidget)
        self.zennShotRadioButton.setObjectName(_fromUtf8("zennShotRadioButton"))
        self.verticalLayout.addWidget(self.zennShotRadioButton)
        self.okButton = QtGui.QPushButton(Dialog)
        self.okButton.setGeometry(QtCore.QRect(290, 190, 97, 31))
        self.okButton.setObjectName(_fromUtf8("okButton"))
        self.cancelButton = QtGui.QPushButton(Dialog)
        self.cancelButton.setGeometry(QtCore.QRect(190, 190, 97, 31))
        self.cancelButton.setObjectName(_fromUtf8("cancelButton"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "SelectExportType", None))
        self.groupBox.setTitle(_translate("Dialog", "Select Publish Type [default : model/pub/]", None))
        self.meshRadioButton.setText(_translate("Dialog", "Basic Mesh [scenes/asset_model_v01]", None))
        self.envRadioButton.setText(_translate("Dialog", "ENV Data [data/abc/name/nameA/nameA_model_v01]", None))
        self.zenvRadioButton.setText(_translate("Dialog", "ZEnv [zenv/abc/nameA/model/nameA_model_v01]", None))
        self.shotRadioButton.setText(_translate("Dialog", "SHOT [/show/project/shot/....../SEQ_model_v01]", None))
        self.zennRadioButton.setText(_translate("Dialog", "ZENN Scene [assetName/hair/pub/scenes/assetName_hair_v01]", None))
        self.zennShotRadioButton.setText(_translate("Dialog", "ZENN By Shot [shotPath/model/hair/pub/scenes/SHOT_hair_v01]", None))
        self.okButton.setText(_translate("Dialog", "OK", None))
        self.cancelButton.setText(_translate("Dialog", "Cancel", None))

