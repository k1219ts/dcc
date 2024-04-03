# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainForm.ui'
#
# Created: Tue Jan  8 22:06:57 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui

import os
currentDir = os.path.dirname(__file__)

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
        Form.resize(447, 184)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.gridLayout_3.addWidget(self.line, 1, 0, 1, 2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label = QtWidgets.QLabel(Form)
        self.label.setText(_fromUtf8(""))
        self.label.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/company_logoB.png" % currentDir)))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setMaximumSize(QtCore.QSize(40, 40))
        self.label_3.setText(_fromUtf8(""))
        self.label_3.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/hsMaya.png" % currentDir)))
        self.label_3.setScaledContents(True)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setMaximumSize(QtCore.QSize(40, 40))
        self.label_4.setSizeIncrement(QtCore.QSize(0, 0))
        self.label_4.setText(_fromUtf8(""))
        self.label_4.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/A57-Arrow-Yellow-Left.png" % currentDir)))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_4.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setMaximumSize(QtCore.QSize(40, 40))
        self.label_5.setText(_fromUtf8(""))
        self.label_5.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/Mari.png" % currentDir)))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_4.addWidget(self.label_5)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.statusLabel = QtWidgets.QLabel(Form)
        self.statusLabel.setMaximumSize(QtCore.QSize(30, 30))
        self.statusLabel.setText(_fromUtf8(""))
        self.statusLabel.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/Circle04-DarkRed.png" % currentDir)))
        self.statusLabel.setScaledContents(True)
        self.statusLabel.setObjectName(_fromUtf8("statusLabel"))
        self.horizontalLayout_4.addWidget(self.statusLabel)
        # self.connectBtn = QtWidgets.QPushButton(Form)
        # font = QtGui.QFont()
        # font.setPointSize(13)
        # font.setBold(True)
        # font.setWeight(75)
        # self.connectBtn.setFont(font)
        # self.connectBtn.setObjectName(_fromUtf8("connectBtn"))
        # self.horizontalLayout_4.addWidget(self.connectBtn)
        self.gridLayout_3.addLayout(self.horizontalLayout_4, 0, 0, 1, 2)
        self.projectGrpBox = QtWidgets.QGroupBox(Form)
        self.projectGrpBox.setObjectName(_fromUtf8("projectGrpBox"))
        self.gridLayout = QtWidgets.QGridLayout(self.projectGrpBox)
        self.gridLayout.setContentsMargins(2, -1, 2, 0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtWidgets.QLabel(self.projectGrpBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.assetNamePath = QtWidgets.QLineEdit(self.projectGrpBox)
        self.assetNamePath.setText(_fromUtf8(""))
        self.assetNamePath.setObjectName(_fromUtf8("assetNamePath"))
        self.horizontalLayout_2.addWidget(self.assetNamePath)
        self.newRadio = QtWidgets.QRadioButton(self.projectGrpBox)
        self.newRadio.setObjectName(_fromUtf8("newRadio"))
        self.horizontalLayout_2.addWidget(self.newRadio)
        # self.addObjRadio = QtWidgets.QRadioButton(self.projectGrpBox)
        # self.addObjRadio.setObjectName(_fromUtf8("addObjRadio"))
        # self.horizontalLayout_2.addWidget(self.addObjRadio)
        self.addVerRadio = QtWidgets.QRadioButton(self.projectGrpBox)
        self.addVerRadio.setObjectName(_fromUtf8("addVerRadio"))
        self.horizontalLayout_2.addWidget(self.addVerRadio)
        self.versionEdit = QtWidgets.QLineEdit(self.projectGrpBox)
        self.versionEdit.setObjectName(_fromUtf8("versionEdit"))
        self.versionEdit.setPlaceholderText("001")
        self.horizontalLayout_2.addWidget(self.versionEdit)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.sendObjBtn = QtWidgets.QPushButton(self.projectGrpBox)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.sendObjBtn.setFont(font)
        self.sendObjBtn.setObjectName(_fromUtf8("sendObjBtn"))
        self.gridLayout.addWidget(self.sendObjBtn, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.projectGrpBox, 2, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Maya To Mari", None))
        # self.connectBtn.setText(_translate("Form", "Connect", None))
        self.projectGrpBox.setTitle(_translate("Form", "Project", None))
        self.label_2.setText(_translate("Form", "Asset", None))
        self.newRadio.setText(_translate("Form", "New", None))
        # self.addObjRadio.setText(_translate("Form", "Add Obj ", None))
        self.addVerRadio.setText(_translate("Form", "Add Ver", None))
        self.sendObjBtn.setText(_translate("Form", "Send Object", None))
