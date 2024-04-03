# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/MainUI.ui'
#
# Created: Thu Jan 25 18:19:41 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

import pymodule.Qt.QtWidgets as QtGui
from pymodule.Qt import QtCore

from Item.RelativeView import RelativeGraphicsView
from Item.CategoryTreeWidget import CategoryTreeWidget

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
        Form.resize(1200, 800)
        Form.setMinimumSize(QtCore.QSize(1200, 800))
        Form.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(Form)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.aniCheckBox = QtGui.QCheckBox(Form)
        self.aniCheckBox.setChecked(True)
        self.aniCheckBox.setObjectName(_fromUtf8("aniCheckBox"))
        self.horizontalLayout_2.addWidget(self.aniCheckBox)
        self.mcpCheckBox = QtGui.QCheckBox(Form)
        self.mcpCheckBox.setChecked(True)
        self.mcpCheckBox.setObjectName(_fromUtf8("mcpCheckBox"))
        self.horizontalLayout_2.addWidget(self.mcpCheckBox)
        self.crdCheckBox = QtGui.QCheckBox(Form)
        self.crdCheckBox.setChecked(True)
        self.crdCheckBox.setObjectName(_fromUtf8("crdCheckBox"))
        self.horizontalLayout_2.addWidget(self.crdCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.tagSearchEdit = QtGui.QLineEdit(Form)
        self.tagSearchEdit.setMaximumSize(QtCore.QSize(200, 16777215))
        self.tagSearchEdit.setText(_fromUtf8(""))
        self.tagSearchEdit.setObjectName(_fromUtf8("tagSearchEdit"))
        self.verticalLayout.addWidget(self.tagSearchEdit)
        self.aniTreeWidget = CategoryTreeWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.aniTreeWidget.sizePolicy().hasHeightForWidth())
        self.aniTreeWidget.setSizePolicy(sizePolicy)
        self.aniTreeWidget.setMinimumSize(QtCore.QSize(200, 0))
        self.aniTreeWidget.setMaximumSize(QtCore.QSize(200, 16777215))
        self.aniTreeWidget.setHeaderHidden(True)
        self.aniTreeWidget.setObjectName(_fromUtf8("aniTreeWidget"))
        self.aniTreeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout.addWidget(self.aniTreeWidget)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.splitter = QtGui.QSplitter(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(10)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.layoutWidget = QtGui.QWidget(self.splitter)
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setMargin(0)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.ContentTabWidget = QtGui.QTabWidget(self.layoutWidget)
        self.ContentTabWidget.setMinimumSize(QtCore.QSize(670, 0))
        self.ContentTabWidget.setObjectName(_fromUtf8("ContentTabWidget"))
        self.verticalLayout_2.addWidget(self.ContentTabWidget)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.ContentScaleSlider = QtGui.QSlider(self.layoutWidget)
        self.ContentScaleSlider.setMaximumSize(QtCore.QSize(250, 16777215))
        self.ContentScaleSlider.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.ContentScaleSlider.setPageStep(100)
        self.ContentScaleSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ContentScaleSlider.setObjectName(_fromUtf8("ContentScaleSlider"))
        self.horizontalLayout.addWidget(self.ContentScaleSlider)
        spacerItem = QtGui.QSpacerItem(208, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        # self.moreBtn = QtGui.QPushButton(self.layoutWidget)
        # self.moreBtn.setObjectName(_fromUtf8("moreBtn"))
        # self.horizontalLayout.addWidget(self.moreBtn)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.layoutWidget1 = QtGui.QWidget(self.splitter)
        self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
        self.gridLayout_2 = QtGui.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.relativeView = RelativeGraphicsView(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.relativeView.sizePolicy().hasHeightForWidth())
        self.relativeView.setSizePolicy(sizePolicy)
        self.relativeView.setMinimumSize(QtCore.QSize(300, 0))
        self.relativeView.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.relativeView.setObjectName(_fromUtf8("relativeView"))
        self.gridLayout_2.addWidget(self.relativeView, 0, 0, 1, 1)
        self.previewWidget = QtGui.QLabel(self.layoutWidget1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previewWidget.sizePolicy().hasHeightForWidth())
        self.previewWidget.setSizePolicy(sizePolicy)
        self.previewWidget.setMinimumSize(QtCore.QSize(300, 300))
        self.previewWidget.setMaximumSize(QtCore.QSize(300, 300))
        self.previewWidget.setObjectName(_fromUtf8("previewWidget"))
        self.gridLayout_2.addWidget(self.previewWidget, 1, 0, 1, 1)
        self.horizontalLayout_3.addWidget(self.splitter)

        self.retranslateUi(Form)
        self.ContentTabWidget.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "AnimBrowser by daeseok.chae in Dexter RND", None))
        self.aniCheckBox.setText(_translate("Form", "ANI", None))
        self.mcpCheckBox.setText(_translate("Form", "MCP", None))
        self.crdCheckBox.setText(_translate("Form", "CRD", None))
        # self.moreBtn.setText(_translate("Form", "More (0 / 0)", None))

