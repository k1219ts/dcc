# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, os
import Qt
from Qt import QtGui
from Qt import QtWidgets as QtGui
from Qt import QtCore
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)

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
        Form.resize(303, 248)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setSpacing(20)
        self.gridLayout.setContentsMargins(-1, 5, -1, 5)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.firsttxt = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.firsttxt.setFont(font)
        self.firsttxt.setObjectName(_fromUtf8("firsttxt"))
        self.horizontalLayout.addWidget(self.firsttxt)
        self.firstbtn = QtGui.QPushButton(Form)
        self.firstbtn.setMinimumSize(QtCore.QSize(170, 0))
        self.firstbtn.setMaximumSize(QtCore.QSize(150, 16777215))
        self.firstbtn.setObjectName(_fromUtf8("firstbtn"))
        self.horizontalLayout.addWidget(self.firstbtn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.centertxt = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.centertxt.setFont(font)
        self.centertxt.setObjectName(_fromUtf8("centertxt"))
        self.horizontalLayout_2.addWidget(self.centertxt)
        self.centerbtn = QtGui.QPushButton(Form)
        self.centerbtn.setMinimumSize(QtCore.QSize(170, 0))
        self.centerbtn.setObjectName(_fromUtf8("centerbtn"))
        self.horizontalLayout_2.addWidget(self.centerbtn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.endtxt = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.endtxt.setFont(font)
        self.endtxt.setObjectName(_fromUtf8("endtxt"))
        self.horizontalLayout_3.addWidget(self.endtxt)
        self.endbtn = QtGui.QPushButton(Form)
        self.endbtn.setMinimumSize(QtCore.QSize(170, 0))
        self.endbtn.setObjectName(_fromUtf8("endbtn"))
        self.horizontalLayout_3.addWidget(self.endbtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.helpbtn = QtGui.QPushButton(Form)
        self.helpbtn.setMinimumSize(QtCore.QSize(25, 25))
        self.helpbtn.setText(_fromUtf8(""))
        self.horizontalLayout_4.addWidget(self.helpbtn)
        self.importbtn = QtGui.QPushButton(Form)
        self.importbtn.setMinimumSize(QtCore.QSize(70, 0))
        self.importbtn.setObjectName(_fromUtf8("importbtn"))
        self.horizontalLayout_4.addWidget(self.importbtn)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.runbtn = QtGui.QPushButton(Form)
        self.runbtn.setMinimumSize(QtCore.QSize(70, 0))
        self.runbtn.setObjectName(_fromUtf8("runbtn"))
        self.horizontalLayout_4.addWidget(self.runbtn)
        self.stopbtn = QtGui.QPushButton(Form)
        self.stopbtn.setMinimumSize(QtCore.QSize(70, 0))
        self.stopbtn.setObjectName(_fromUtf8("stopbtn"))
        self.horizontalLayout_4.addWidget(self.stopbtn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Angle360_TopView", None))
        self.firsttxt.setText(_translate("Form", "First Point", None))
        self.firstbtn.setText(_translate("Form", "Select standard locator", None))
        self.centertxt.setText(_translate("Form", "Center Point", None))
        self.centerbtn.setText(_translate("Form", "Select center locator", None))
        self.endtxt.setText(_translate("Form", "End Point", None))
        self.endbtn.setText(_translate("Form", "Select another locator", None))
        self.importbtn.setText(_translate("Form", "Import", None))
        self.runbtn.setText(_translate("Form", "Run", None))
        self.stopbtn.setText(_translate("Form", "Stop", None))

