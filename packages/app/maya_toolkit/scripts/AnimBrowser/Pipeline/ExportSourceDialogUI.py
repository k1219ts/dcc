# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RetargetBVHExport.ui'
#
# Created: Thu Mar 15 11:27:23 2018
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
        Form.resize(423, 307)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(Form)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 4, 2, 1, 1)
        self.endFrameLineEdit = QtGui.QLineEdit(Form)
        self.endFrameLineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.endFrameLineEdit.setObjectName(_fromUtf8("endFrameLineEdit"))
        self.gridLayout.addWidget(self.endFrameLineEdit, 4, 3, 1, 1)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.startFrameLineEdit = QtGui.QLineEdit(Form)
        self.startFrameLineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.startFrameLineEdit.setObjectName(_fromUtf8("startFrameLineEdit"))
        self.gridLayout.addWidget(self.startFrameLineEdit, 4, 1, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.aniRadioBtn = QtGui.QRadioButton(Form)
        self.aniRadioBtn.setChecked(True)
        self.aniRadioBtn.setObjectName(_fromUtf8("aniRadioBtn"))
        self.horizontalLayout_2.addWidget(self.aniRadioBtn)
        self.mcpRadioBtn = QtGui.QRadioButton(Form)
        self.mcpRadioBtn.setObjectName(_fromUtf8("mcpRadioBtn"))
        self.horizontalLayout_2.addWidget(self.mcpRadioBtn)
        self.crdRadioBtn = QtGui.QRadioButton(Form)
        self.crdRadioBtn.setObjectName(_fromUtf8("crdRadioBtn"))
        self.horizontalLayout_2.addWidget(self.crdRadioBtn)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 1, 4)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.tractorCheckBox = QtGui.QCheckBox(Form)
        self.tractorCheckBox.setChecked(False)
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
        self.gridLayout.addLayout(self.horizontalLayout, 5, 0, 1, 4)
        self.Title2ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title2ComboBox.sizePolicy().hasHeightForWidth())
        self.Title2ComboBox.setSizePolicy(sizePolicy)
        self.Title2ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title2ComboBox.setObjectName(_fromUtf8("Title2ComboBox"))
        self.gridLayout.addWidget(self.Title2ComboBox, 2, 1, 1, 3)
        self.Title3ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title3ComboBox.sizePolicy().hasHeightForWidth())
        self.Title3ComboBox.setSizePolicy(sizePolicy)
        self.Title3ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title3ComboBox.setObjectName(_fromUtf8("Title3ComboBox"))
        self.gridLayout.addWidget(self.Title3ComboBox, 3, 1, 1, 3)
        self.Title1ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title1ComboBox.sizePolicy().hasHeightForWidth())
        self.Title1ComboBox.setSizePolicy(sizePolicy)
        self.Title1ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title1ComboBox.setObjectName(_fromUtf8("Title1ComboBox"))
        self.gridLayout.addWidget(self.Title1ComboBox, 1, 1, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Upload Source for Retarget", None))
        self.label.setText(_translate("Form", "Title", None))
        self.label_2.setText(_translate("Form", "Sub Title", None))
        self.label_3.setText(_translate("Form", "Part SubTitle", None))
        self.label_7.setText(_translate("Form", "Rest Frame", None))
        self.endFrameLineEdit.setText(_translate("Form", "", None))
        self.label_6.setText(_translate("Form", "Start Frame", None))
        self.startFrameLineEdit.setText(_translate("Form", "", None))
        self.aniRadioBtn.setText(_translate("Form", "Animation", None))
        self.mcpRadioBtn.setText(_translate("Form", "MotionCapture", None))
        self.crdRadioBtn.setText(_translate("Form", "Crowd", None))
        self.tractorCheckBox.setText(_translate("Form", "Tractor", None))
        self.okBtn.setText(_translate("Form", "Export", None))
        self.cancelBtn.setText(_translate("Form", "Cancel", None))
