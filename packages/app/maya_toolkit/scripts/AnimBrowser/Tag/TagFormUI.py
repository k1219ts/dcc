# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Tag/TagMainUI.ui'
#
# Created: Thu Jan 25 18:06:46 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

import pymodule.Qt as Qt
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets as QtGui

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
        Form.resize(657, 394)
        Form.setStyleSheet(_fromUtf8(""))
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_8 = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = Qt.QtGui.QFont()
        font.setFamily(_fromUtf8("DejaVu Sans"))
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet(_fromUtf8(""))
        self.label_8.setFrameShape(QtGui.QFrame.NoFrame)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 2)
        self.tagTreeWidget = QtGui.QTreeWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tagTreeWidget.sizePolicy().hasHeightForWidth())
        self.tagTreeWidget.setSizePolicy(sizePolicy)
        self.tagTreeWidget.setMinimumSize(QtCore.QSize(256, 300))
        self.tagTreeWidget.setStyleSheet(_fromUtf8(""))
        self.tagTreeWidget.setObjectName(_fromUtf8("tagTreeWidget"))
        self.gridLayout.addWidget(self.tagTreeWidget, 1, 0, 8, 1)
        self.animRadioBtn = QtGui.QRadioButton(Form)
        self.animRadioBtn.setChecked(True)
        self.animRadioBtn.setObjectName(_fromUtf8("animRadioBtn"))
        self.gridLayout.addWidget(self.animRadioBtn, 1, 1, 1, 1)
        self.mocapRadioBtn = QtGui.QRadioButton(Form)
        self.mocapRadioBtn.setObjectName(_fromUtf8("mocapRadioBtn"))
        self.gridLayout.addWidget(self.mocapRadioBtn, 1, 2, 1, 1)
        self.crowdRadioBtn = QtGui.QRadioButton(Form)
        self.crowdRadioBtn.setObjectName(_fromUtf8("crowdRadioBtn"))
        self.gridLayout.addWidget(self.crowdRadioBtn, 1, 3, 1, 2)
        self.label = QtGui.QLabel(Form)
        self.label.setStyleSheet(_fromUtf8(""))
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 2, 1, 1, 1)
        self.tagTier1ComboBox = QtGui.QComboBox(Form)
        self.tagTier1ComboBox.setStyleSheet(_fromUtf8(""))
        self.tagTier1ComboBox.setObjectName(_fromUtf8("tagTier1ComboBox"))
        self.gridLayout.addWidget(self.tagTier1ComboBox, 2, 2, 1, 1)
        self.label_7 = QtGui.QLabel(Form)
        self.label_7.setStyleSheet(_fromUtf8(""))
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout.addWidget(self.label_7, 3, 1, 1, 1)
        self.tagTier1lineEdit = QtGui.QLineEdit(Form)
        self.tagTier1lineEdit.setStyleSheet(_fromUtf8(""))
        self.tagTier1lineEdit.setObjectName(_fromUtf8("tagTier1lineEdit"))
        self.gridLayout.addWidget(self.tagTier1lineEdit, 3, 2, 1, 2)
        self.tier1PushBtn = QtGui.QPushButton(Form)
        self.tier1PushBtn.setStyleSheet(_fromUtf8(""))
        self.tier1PushBtn.setObjectName(_fromUtf8("tier1PushBtn"))
        self.gridLayout.addWidget(self.tier1PushBtn, 3, 4, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setStyleSheet(_fromUtf8(""))
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 4, 1, 1, 1)
        self.tagTier2ComboBox = QtGui.QComboBox(Form)
        self.tagTier2ComboBox.setStyleSheet(_fromUtf8(""))
        self.tagTier2ComboBox.setObjectName(_fromUtf8("tagTier2ComboBox"))
        self.gridLayout.addWidget(self.tagTier2ComboBox, 4, 2, 1, 1)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setStyleSheet(_fromUtf8(""))
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 5, 1, 1, 1)
        self.tagTier2lineEdit = QtGui.QLineEdit(Form)
        self.tagTier2lineEdit.setStyleSheet(_fromUtf8(""))
        self.tagTier2lineEdit.setObjectName(_fromUtf8("tagTier2lineEdit"))
        self.gridLayout.addWidget(self.tagTier2lineEdit, 5, 2, 1, 2)
        self.tier2PushBtn = QtGui.QPushButton(Form)
        self.tier2PushBtn.setStyleSheet(_fromUtf8(""))
        self.tier2PushBtn.setObjectName(_fromUtf8("tier2PushBtn"))
        self.gridLayout.addWidget(self.tier2PushBtn, 5, 4, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setStyleSheet(_fromUtf8(""))
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 6, 1, 1, 1)
        self.tagTier3ComboBox = QtGui.QComboBox(Form)
        self.tagTier3ComboBox.setStyleSheet(_fromUtf8(""))
        self.tagTier3ComboBox.setObjectName(_fromUtf8("tagTier3ComboBox"))
        self.gridLayout.addWidget(self.tagTier3ComboBox, 6, 2, 1, 1)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setStyleSheet(_fromUtf8(""))
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 7, 1, 1, 1)
        self.tagTier3lineEdit = QtGui.QLineEdit(Form)
        self.tagTier3lineEdit.setStyleSheet(_fromUtf8(""))
        self.tagTier3lineEdit.setObjectName(_fromUtf8("tagTier3lineEdit"))
        self.gridLayout.addWidget(self.tagTier3lineEdit, 7, 2, 1, 2)
        self.tier3PushBtn = QtGui.QPushButton(Form)
        self.tier3PushBtn.setStyleSheet(_fromUtf8(""))
        self.tier3PushBtn.setObjectName(_fromUtf8("tier3PushBtn"))
        self.gridLayout.addWidget(self.tier3PushBtn, 7, 4, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 8, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "AnimTagManager by daeseok.chae", None))
        self.label_8.setText(_translate("Form", "Tag Management Tool", None))
        self.tagTreeWidget.headerItem().setText(0, _translate("Form", "Tags", None))
        self.animRadioBtn.setText(_translate("Form", "Animation", None))
        self.mocapRadioBtn.setText(_translate("Form", "Motion Capture", None))
        self.crowdRadioBtn.setText(_translate("Form", "Crowd", None))
        self.label.setText(_translate("Form", "Tag Tier1", None))
        self.label_7.setText(_translate("Form", "Tag Tier1 Input", None))
        self.tier1PushBtn.setText(_translate("Form", "Tier 1 Add", None))
        self.label_2.setText(_translate("Form", "Tag Tier2", None))
        self.label_5.setText(_translate("Form", "Tag Tier2 Input", None))
        self.tier2PushBtn.setText(_translate("Form", "Tier 2 Add", None))
        self.label_3.setText(_translate("Form", "Tag Tier3", None))
        self.label_6.setText(_translate("Form", "Tag Tier3 Input", None))
        self.tier3PushBtn.setText(_translate("Form", "Tier 3 Add", None))

