# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################

import Qt
from Qt import QtGui
from Qt import QtWidgets as QtGui
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
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(471, 499)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.start_txt = QtGui.QLineEdit(Form)
        self.start_txt.setMinimumSize(QtCore.QSize(50, 30))
        self.start_txt.setMaximumSize(QtCore.QSize(123233, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.start_txt.setFont(font)
        self.start_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.start_txt.setObjectName(_fromUtf8("start_txt"))
        self.horizontalLayout.addWidget(self.start_txt)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.end_txt = QtGui.QLineEdit(Form)
        self.end_txt.setMinimumSize(QtCore.QSize(50, 30))
        self.end_txt.setMaximumSize(QtCore.QSize(121222, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.end_txt.setFont(font)
        self.end_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.end_txt.setObjectName(_fromUtf8("end_txt"))
        self.horizontalLayout_2.addWidget(self.end_txt)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_3 = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_3.addWidget(self.label_3)
        self.change_txt = QtGui.QLineEdit(Form)
        self.change_txt.setMinimumSize(QtCore.QSize(50, 30))
        self.change_txt.setMaximumSize(QtCore.QSize(112222, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.change_txt.setFont(font)
        self.change_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.change_txt.setObjectName(_fromUtf8("change_txt"))
        self.horizontalLayout_3.addWidget(self.change_txt)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.ok_btn = QtGui.QPushButton(Form)
        self.ok_btn.setMinimumSize(QtCore.QSize(60, 0))
        self.ok_btn.setMaximumSize(QtCore.QSize(60, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ok_btn.setFont(font)
        self.ok_btn.setObjectName(_fromUtf8("ok_btn"))
        self.horizontalLayout_4.addWidget(self.ok_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.list_txt = QtGui.QListWidget(Form)
        self.list_txt.setMinimumSize(QtCore.QSize(400, 400))
        self.list_txt.setMaximumSize(QtCore.QSize(16777215, 133333))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.list_txt.setFont(font)
        self.list_txt.setObjectName(_fromUtf8("list_txt"))
        self.verticalLayout.addWidget(self.list_txt)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem1)
        self.selbtn = QtGui.QPushButton(Form)
        self.selbtn.setMinimumSize(QtCore.QSize(200, 0))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.selbtn.setFont(font)
        self.selbtn.setObjectName(_fromUtf8("selbtn"))
        self.horizontalLayout_6.addWidget(self.selbtn)
        self.helpbtn = QtGui.QPushButton(Form)
        self.helpbtn.setMinimumSize(QtCore.QSize(25, 25))
        self.helpbtn.setMaximumSize(QtCore.QSize(25, 25))
        self.helpbtn.setText(_fromUtf8(""))
        self.helpbtn.setObjectName(_fromUtf8("helpbtn"))
        self.horizontalLayout_6.addWidget(self.helpbtn)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "KeyMove by RND youkyoung.kim", None))
        self.label.setText(_translate("Form", "| Strat", None))
        self.label_2.setText(_translate("Form", " | End", None))
        self.label_3.setText(_translate("Form", " | Change ", None))
        self.ok_btn.setText(_translate("Form", "Ok", None))
        self.selbtn.setText(_translate("Form", "Select Animation Key", None))

