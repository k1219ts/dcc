# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ContentItem.ui'
#
# Created: Fri Jan  5 11:06:10 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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
        Form.resize(282, 237)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.maxFrameEdit = QtGui.QLineEdit(Form)
        self.maxFrameEdit.setMaximumSize(QtCore.QSize(45, 16))
        self.maxFrameEdit.setObjectName(_fromUtf8("maxFrameEdit"))
        self.horizontalLayout.addWidget(self.maxFrameEdit)
        self.gifSlider = QtGui.QSlider(Form)
        self.gifSlider.setOrientation(QtCore.Qt.Horizontal)
        self.gifSlider.setObjectName(_fromUtf8("gifSlider"))
        self.horizontalLayout.addWidget(self.gifSlider)
        self.curFrameEdit = QtGui.QLineEdit(Form)
        self.curFrameEdit.setMaximumSize(QtCore.QSize(45, 16))
        self.curFrameEdit.setObjectName(_fromUtf8("curFrameEdit"))
        self.horizontalLayout.addWidget(self.curFrameEdit)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.assetLabel = QtGui.QLabel(Form)
        self.assetLabel.setMinimumSize(QtCore.QSize(282, 16))
        self.assetLabel.setMaximumSize(QtCore.QSize(282, 16))
        self.assetLabel.setObjectName(_fromUtf8("assetLabel"))
        self.gridLayout.addWidget(self.assetLabel, 2, 0, 1, 1)
        self.previewLabel = QtGui.QLabel(Form)
        self.previewLabel.setMinimumSize(QtCore.QSize(282, 200))
        self.previewLabel.setMaximumSize(QtCore.QSize(282, 200))
        self.previewLabel.setObjectName(_fromUtf8("previewLabel"))
        self.gridLayout.addWidget(self.previewLabel, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.maxFrameEdit.setText(_translate("Form", "1111", None))
        self.curFrameEdit.setText(_translate("Form", "1111", None))
        self.assetLabel.setText(_translate("Form", "TextLabel", None))
        self.previewLabel.setText(_translate("Form", "TextLabel", None))

