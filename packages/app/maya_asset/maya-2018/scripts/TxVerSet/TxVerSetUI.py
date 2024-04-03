# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TxVerSetUI.ui'
#
# Created: Tue Jan 15 15:18:52 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui

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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(520, 272)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtWidgets.QLabel(Form)
        self.label.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.showDirEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.showDirEdit.setFont(font)
        self.showDirEdit.setObjectName(_fromUtf8("showDirEdit"))
        self.horizontalLayout.addWidget(self.showDirEdit)
        self.elementCheckBox = QtWidgets.QCheckBox(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.elementCheckBox.setFont(font)
        self.elementCheckBox.setObjectName(_fromUtf8("elementCheckBox"))
        self.horizontalLayout.addWidget(self.elementCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.assetNameEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.assetNameEdit.setFont(font)
        self.assetNameEdit.setObjectName(_fromUtf8("assetNameEdit"))
        self.horizontalLayout_2.addWidget(self.assetNameEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.elementLabel = QtWidgets.QLabel(Form)
        self.elementLabel.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.elementLabel.setFont(font)
        self.elementLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.elementLabel.setObjectName(_fromUtf8("elementLabel"))
        self.horizontalLayout_5.addWidget(self.elementLabel)
        self.elementEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.elementEdit.setFont(font)
        self.elementEdit.setObjectName(_fromUtf8("elementEdit"))
        self.horizontalLayout_5.addWidget(self.elementEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_3.addWidget(self.label_3)
        self.modelVersionEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.modelVersionEdit.setFont(font)
        self.modelVersionEdit.setObjectName(_fromUtf8("modelVersionEdit"))
        self.horizontalLayout_3.addWidget(self.modelVersionEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_4.addWidget(self.label_4)
        self.textureVersionEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.textureVersionEdit.setFont(font)
        self.textureVersionEdit.setObjectName(_fromUtf8("textureVersionEdit"))
        self.horizontalLayout_4.addWidget(self.textureVersionEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.execBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.execBtn.setFont(font)
        self.execBtn.setObjectName(_fromUtf8("execBtn"))
        self.verticalLayout.addWidget(self.execBtn)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Texture Version Setup", None))
        self.label.setText(_translate("Form", "showDir", None))
        self.elementCheckBox.setText(_translate("Form", "element", None))
        self.label_2.setText(_translate("Form", "assetName", None))
        self.elementLabel.setText(_translate("Form", "elementName", None))
        self.label_3.setText(_translate("Form", "modelVersion", None))
        self.label_4.setText(_translate("Form", "textureVersion", None))
        self.execBtn.setText(_translate("Form", "Execute", None))

