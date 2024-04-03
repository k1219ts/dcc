# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os
import Qt
from Qt import QtGui
from Qt import QtWidgets
from Qt import QtCore
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

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
        Form.setGeometry(QtCore.QRect(500, 400, 611, 414))
        Form.resize(611, 414)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setMargin(20)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtWidgets.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.file_txt = QtWidgets.QLineEdit(Form)
        self.file_txt.setReadOnly(True)
        self.file_txt.setObjectName(_fromUtf8("file_txt"))
        self.horizontalLayout.addWidget(self.file_txt)
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.open_btn = QtWidgets.QPushButton(Form)
        self.open_btn.setMaximumSize(QtCore.QSize(30, 30))
        self.open_btn.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(CURRENT_DIR+"/img/open.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.open_btn.setIcon(icon)
        self.open_btn.setIconSize(QtCore.QSize(30, 30))
        self.open_btn.setObjectName(_fromUtf8("open_btn"))
        self.horizontalLayout.addWidget(self.open_btn)
        self.minus_btn = QtWidgets.QPushButton(Form)
        self.minus_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.minus_btn.setMaximumSize(QtCore.QSize(30, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.minus_btn.setFont(font)
        self.minus_btn.setObjectName(_fromUtf8("minus_btn"))
        self.horizontalLayout.addWidget(self.minus_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem1)
        self.file_list = QtWidgets.QListWidget(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.file_list.setFont(font)
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)  #### #multi selection ctrl+
        self.file_list.setObjectName(_fromUtf8("file_list"))
        self.verticalLayout.addWidget(self.file_list)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(50, 0))
        self.label_2.setMaximumSize(QtCore.QSize(80, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.user_txt = QtWidgets.QLabel(Form)
        self.user_txt.setObjectName(_fromUtf8("user_txt"))
        self.user_txt.setFont(font)
        self.horizontalLayout_2.addWidget(self.user_txt)
        self.user_txt.setFont(font)
        spacerItem3 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.send_btn = QtWidgets.QPushButton(Form)
        self.send_btn.setMinimumSize(QtCore.QSize(150, 0))
        self.send_btn.setMaximumSize(QtCore.QSize(150, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.send_btn.setFont(font)
        self.send_btn.setObjectName(_fromUtf8("send_btn"))
        self.horizontalLayout_2.addWidget(self.send_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Lay Inventory Add Mov", None))
        self.label.setText(_translate("Form", "File Path : ", None))
        self.minus_btn.setText(_translate("Form", "-", None))
        self.label_2.setText(_translate("Form", "Total :", None))
        self.user_txt.setText(_translate("Form", "0", None))
        self.send_btn.setText(_translate("Form", "Send", None))

