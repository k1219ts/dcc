# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LookdevMainUI.ui'
#
# Created: Tue Mar 14 17:50:34 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

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
        Form.setObjectName(_fromUtf8("LookdevTool"))
        Form.resize(800, 783)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.scopeBtn = QtGui.QPushButton(Form)
        self.scopeBtn.setObjectName(_fromUtf8("scopeBtn"))
        self.gridLayout.addWidget(self.scopeBtn, 3, 0, 1, 1)
        
        self.bindingTreeWidget = QtGui.QTreeWidget(Form)
        self.bindingTreeWidget.setRootIsDecorated(True)
        self.bindingTreeWidget.setObjectName(_fromUtf8("bindingTreeWidget"))
        self.bindingTreeWidget.headerItem().setTextAlignment(0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.bindingTreeWidget.headerItem().setTextAlignment(1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.bindingTreeWidget.header().setDefaultSectionSize(400)
        self.gridLayout.addWidget(self.bindingTreeWidget, 2, 0, 1, 1)
        
        self.loadBindingBtn = QtGui.QPushButton(Form)
        self.loadBindingBtn.setObjectName(_fromUtf8("loadBindingBtn"))
        self.gridLayout.addWidget(self.loadBindingBtn, 1, 0, 1, 1)
        
        self.assetTreeWidget = QtGui.QTreeWidget(Form)
        self.assetTreeWidget.setObjectName(_fromUtf8("assetTreeWidget"))
        self.assetTreeWidget.setRootIsDecorated(False)
        self.assetTreeWidget.headerItem().setTextAlignment(0, QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.assetTreeWidget.headerItem().setTextAlignment(1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.assetTreeWidget.headerItem().setTextAlignment(2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.assetTreeWidget.headerItem().setTextAlignment(3, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.assetTreeWidget.headerItem().setTextAlignment(4, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.assetTreeWidget.header().setDefaultSectionSize(195)
        self.assetTreeWidget.setColumnWidth(0, 20)
        self.assetTreeWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.gridLayout.addWidget(self.assetTreeWidget, 0, 0, 1, 1)

        self.turntableBtn = QtGui.QPushButton(Form)
        self.turntableBtn.setObjectName(_fromUtf8("turntableBtn"))
        self.gridLayout.addWidget(self.turntableBtn, 4, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("LookdevTool", "LookdevTool by Dexter RND daeseok.chae & Dexter ASSET byungwhee.kim", None))
        self.scopeBtn.setText(_translate("Form", "Scope", None))
        self.turntableBtn.setText(_translate("Form", "turnTable Rendering", None))
        self.bindingTreeWidget.headerItem().setText(0, _translate("Form", "xPath", None))
        self.bindingTreeWidget.headerItem().setText(1, _translate("Form", "shader", None))
        self.loadBindingBtn.setText(_translate("Form", "Load Binding Info ", None))
        self.assetTreeWidget.headerItem().setText(0, _translate("Form", "", None))
        self.assetTreeWidget.headerItem().setText(1, _translate("Form", "AssetName", None))
        self.assetTreeWidget.headerItem().setText(2, _translate("Form", "Alembic Version", None))
        self.assetTreeWidget.headerItem().setText(3, _translate("Form", "Texture Version", None))
        self.assetTreeWidget.headerItem().setText(4, _translate("Form", "Shader Channel", None))

