# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dxrunner.ui',
# licensing of 'dxrunner.ui' applies.
#
# Created: Thu Aug 20 10:22:56 2020
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(690, 273)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.showLabel = QtWidgets.QLabel(Form)
        self.showLabel.setObjectName("showLabel")
        self.horizontalLayout.addWidget(self.showLabel)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.seqLabel = QtWidgets.QLabel(Form)
        self.seqLabel.setObjectName("seqLabel")
        self.horizontalLayout.addWidget(self.seqLabel)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.shotLabel = QtWidgets.QLabel(Form)
        self.shotLabel.setObjectName("shotLabel")
        self.horizontalLayout.addWidget(self.shotLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.listWidget = QtWidgets.QListWidget(Form)
        self.listWidget.setIconSize(QtCore.QSize(48, 48))
        self.listWidget.setViewMode(QtWidgets.QListView.IconMode)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Form", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "show", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("Form", "seq", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("Form", "shot", None, -1))

