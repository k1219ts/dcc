# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/NoneHIKDialog.ui'
#
# Created: Mon Jan  8 16:02:40 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# import Qt.QtWidgets as QtGui
# from Qt import QtCore

from PySide2 import QtGui, QtCore

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
        Form.resize(353, 156)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.Title2ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title2ComboBox.sizePolicy().hasHeightForWidth())
        self.Title2ComboBox.setSizePolicy(sizePolicy)
        self.Title2ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title2ComboBox.setObjectName(_fromUtf8("Title2ComboBox"))
        self.gridLayout.addWidget(self.Title2ComboBox, 1, 1, 1, 2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.tractorCheckBox = QtGui.QCheckBox(Form)
        self.tractorCheckBox.setChecked(True)
        self.tractorCheckBox.setObjectName(_fromUtf8("tractorCheckBox"))
        self.horizontalLayout.addWidget(self.tractorCheckBox)
        spacerItem = QtGui.QSpacerItem(118, 17, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.okBtn = QtGui.QPushButton(Form)
        self.okBtn.setObjectName(_fromUtf8("okBtn"))
        self.horizontalLayout.addWidget(self.okBtn)
        self.cancelBtn = QtGui.QPushButton(Form)
        self.cancelBtn.setObjectName(_fromUtf8("cancelBtn"))
        self.horizontalLayout.addWidget(self.cancelBtn)
        self.gridLayout.addLayout(self.horizontalLayout, 4, 0, 1, 3)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.Title1ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title1ComboBox.sizePolicy().hasHeightForWidth())
        self.Title1ComboBox.setSizePolicy(sizePolicy)
        self.Title1ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title1ComboBox.setObjectName(_fromUtf8("Title1ComboBox"))
        self.gridLayout.addWidget(self.Title1ComboBox, 0, 1, 1, 2)
        self.label = QtGui.QLabel(Form)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.Title3ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title3ComboBox.sizePolicy().hasHeightForWidth())
        self.Title3ComboBox.setSizePolicy(sizePolicy)
        self.Title3ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title3ComboBox.setObjectName(_fromUtf8("Title3ComboBox"))
        self.gridLayout.addWidget(self.Title3ComboBox, 3, 1, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Remap Dialog", None))
        self.label_3.setText(_translate("Form", "Part SubTitle", None))
        self.tractorCheckBox.setText(_translate("Form", "Tractor", None))
        self.okBtn.setText(_translate("Form", "Import", None))
        self.cancelBtn.setText(_translate("Form", "Cancel", None))
        self.label_2.setText(_translate("Form", "Sub Title", None))
        self.label.setText(_translate("Form", "Title", None))
