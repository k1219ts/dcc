# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/MocapDialog.ui'
#
# Created: Mon Jan 29 11:52:08 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

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
        Form.resize(423, 273)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(Form)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.Title1ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title1ComboBox.sizePolicy().hasHeightForWidth())
        self.Title1ComboBox.setSizePolicy(sizePolicy)
        self.Title1ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title1ComboBox.setObjectName(_fromUtf8("Title1ComboBox"))
        self.gridLayout.addWidget(self.Title1ComboBox, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.Title2ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title2ComboBox.sizePolicy().hasHeightForWidth())
        self.Title2ComboBox.setSizePolicy(sizePolicy)
        self.Title2ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title2ComboBox.setObjectName(_fromUtf8("Title2ComboBox"))
        self.gridLayout.addWidget(self.Title2ComboBox, 1, 1, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.Title3ComboBox = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title3ComboBox.sizePolicy().hasHeightForWidth())
        self.Title3ComboBox.setSizePolicy(sizePolicy)
        self.Title3ComboBox.setMinimumSize(QtCore.QSize(80, 0))
        self.Title3ComboBox.setObjectName(_fromUtf8("Title3ComboBox"))
        self.gridLayout.addWidget(self.Title3ComboBox, 2, 1, 1, 1)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.animPathEdit = QtGui.QLineEdit(Form)
        self.animPathEdit.setReadOnly(True)
        self.animPathEdit.setObjectName(_fromUtf8("animPathEdit"))
        self.gridLayout.addWidget(self.animPathEdit, 3, 1, 1, 1)
        self.loadAnimFileBtn = QtGui.QPushButton(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadAnimFileBtn.sizePolicy().hasHeightForWidth())
        self.loadAnimFileBtn.setSizePolicy(sizePolicy)
        self.loadAnimFileBtn.setMinimumSize(QtCore.QSize(80, 0))
        self.loadAnimFileBtn.setObjectName(_fromUtf8("loadAnimFileBtn"))
        self.gridLayout.addWidget(self.loadAnimFileBtn, 3, 2, 1, 1)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 4, 0, 1, 1)
        self.movPathEdit = QtGui.QLineEdit(Form)
        self.movPathEdit.setReadOnly(True)
        self.movPathEdit.setObjectName(_fromUtf8("movPathEdit"))
        self.gridLayout.addWidget(self.movPathEdit, 4, 1, 1, 1)
        self.loadMovFileBtn = QtGui.QPushButton(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loadMovFileBtn.sizePolicy().hasHeightForWidth())
        self.loadMovFileBtn.setSizePolicy(sizePolicy)
        self.loadMovFileBtn.setMinimumSize(QtCore.QSize(80, 0))
        self.loadMovFileBtn.setObjectName(_fromUtf8("loadMovFileBtn"))
        self.gridLayout.addWidget(self.loadMovFileBtn, 4, 2, 1, 1)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 5, 0, 1, 1)
        self.hasTagEdit = QtGui.QLineEdit(Form)
        self.hasTagEdit.setObjectName(_fromUtf8("hasTagEdit"))
        self.gridLayout.addWidget(self.hasTagEdit, 5, 1, 1, 1)
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
        self.gridLayout.addLayout(self.horizontalLayout, 6, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Mocap 데이터 업로드", None))
        self.label.setText(_translate("Form", "Tier", None))
        self.label_2.setText(_translate("Form", "Sub Tier", None))
        self.label_3.setText(_translate("Form", "Part SubTier", None))
        self.label_5.setText(_translate("Form", "Anim File Path", None))
        self.loadAnimFileBtn.setText(_translate("Form", "...", None))
        self.label_6.setText(_translate("Form", "Mov Path", None))
        self.loadMovFileBtn.setText(_translate("Form", "...", None))
        self.label_4.setText(_translate("Form", "Hash Tag", None))
        self.hasTagEdit.setPlaceholderText(_translate("Form", "ex) god sword jahong ...", None))
        self.tractorCheckBox.setText(_translate("Form", "Tractor", None))
        self.okBtn.setText(_translate("Form", "Import", None))
        self.cancelBtn.setText(_translate("Form", "Cancel", None))

