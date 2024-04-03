# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'EnvSourceLoad.ui'
#
# Created: Wed Mar 22 13:06:52 2017
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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(702, 480)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.fileListTreeWidget = QtGui.QTreeWidget(Form)
        self.fileListTreeWidget.setRootIsDecorated(False)
        self.fileListTreeWidget.setObjectName(_fromUtf8("fileListTreeWidget"))
        self.fileListTreeWidget.header().setDefaultSectionSize(20)

        if not Form.isZenv:
            self.gridLayout.addWidget(self.fileListTreeWidget, 3, 0, 1, 6)
        else:
            self.gridLayout.addWidget(self.fileListTreeWidget, 3, 0, 1, 4)

        self.label = QtGui.QLabel(Form)
        self.label.setMaximumSize(QtCore.QSize(50, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.showComboBox = QtGui.QComboBox(Form)
        self.showComboBox.setObjectName(_fromUtf8("showComboBox"))
        self.gridLayout.addWidget(self.showComboBox, 0, 1, 1, 1)
        self.AssetComboBox = QtGui.QComboBox(Form)
        self.AssetComboBox.setObjectName(_fromUtf8("AssetComboBox"))
        self.gridLayout.addWidget(self.AssetComboBox, 0, 3, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setMaximumSize(QtCore.QSize(60, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        if not Form.isZenv:
            self.gridLayout.addWidget(self.showComboBox, 0, 1, 1, 1)
            self.dataComoboBox = QtGui.QComboBox(Form)
            self.dataComoboBox.setObjectName("dataComoboBox")
            self.gridLayout.addWidget(self.dataComoboBox, 0, 5, 1, 1)
            self.label_3 = QtGui.QLabel(Form)
            self.label_3.setMaximumSize(QtCore.QSize(60, 16777215))
            font = Qt.QtGui.QFont()
            font.setPointSize(13)
            font.setBold(True)
            font.setWeight(75)
            self.label_3.setFont(font)
            self.label_3.setAlignment(QtCore.Qt.AlignCenter)
            self.label_3.setObjectName("label_3")
            self.label_3.setText("data")
            self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1)
        self.pushButton = QtGui.QPushButton(Form)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        if not Form.isZenv:
            self.gridLayout.addWidget(self.pushButton, 4, 5, 1, 1)
        else:
            self.gridLayout.addWidget(self.pushButton, 4, 3, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "EnvSourceLoad", None))
        self.fileListTreeWidget.headerItem().setText(1, _translate("Form", "source Name", None))
        self.label.setText(_translate("Form", "Show", None))
        self.label_2.setText(_translate("Form", "Asset", None))
        self.pushButton.setText(_translate("Form", "Add Asset", None))

