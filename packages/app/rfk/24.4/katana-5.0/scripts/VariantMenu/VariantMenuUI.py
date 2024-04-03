# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'VariantMenuUI.ui'
#
# Created: Tue Aug 28 15:06:02 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os

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

currentDir = os.path.dirname(__file__)

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(650, 322)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtWidgets.QSpacerItem(107, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setMinimumSize(QtCore.QSize(400, 100))
        self.pushButton.setMaximumSize(QtCore.QSize(400, 100))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton.setIcon(QtGui.QIcon("%s/resources/Refresh.png" % currentDir))
        self.pushButton.setIconSize(QtCore.QSize(36, 36))
        self.pushButton.setStyleSheet('''
        font-size:20px;
        ''')
        self.gridLayout.addWidget(self.pushButton, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(107, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        self.treeWidget = QtWidgets.QTreeWidget(Form)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.treeWidget.setRootIsDecorated(False)
        self.treeWidget.header().resizeSection(0, 200)
        self.treeWidget.setObjectName(_fromUtf8("treeWidget"))
        self.treeWidget.setStyleSheet('''
            QTreeWidget::item {font-size:20px; padding-top: 5px;}
            QComboBox { font-size:20px; }
            QComboBox QAbstractItemView { border : 2px; padding:1px 1px 1px 1px; }
            QComboBox QAbstractItemView:item{ min-height:30px; }
        ''')
        self.gridLayout.addWidget(self.treeWidget, 0, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.pushButton.setText(_translate("Form", "Variant Refresh", None))
        self.treeWidget.headerItem().setText(0, _translate("Form", "variant Name", None))
        self.treeWidget.headerItem().setText(1, _translate("Form", "variant Value", None))
