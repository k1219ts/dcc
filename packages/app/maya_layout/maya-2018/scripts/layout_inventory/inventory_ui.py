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
        Form.resize(800, 357)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setSpacing(20)
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setSpacing(15)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        spacerItem = QtGui.QSpacerItem(50, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.title_txt = QtGui.QLineEdit(Form)
        self.title_txt.setMinimumSize(QtCore.QSize(0, 25))
        self.title_txt.setMaximumSize(QtCore.QSize(16777215, 25))
        self.title_txt.setObjectName(_fromUtf8("title_txt"))
        self.horizontalLayout.addWidget(self.title_txt)
        spacerItem1 = QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(90, 0))
        self.label_2.setMaximumSize(QtCore.QSize(90, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.type_combo = QtGui.QComboBox(Form)
        self.type_combo.setMinimumSize(QtCore.QSize(130, 25))
        self.type_combo.setMaximumSize(QtCore.QSize(130, 25))
        self.type_combo.setObjectName(_fromUtf8("type_combo"))
        self.horizontalLayout_2.addWidget(self.type_combo)
        spacerItem2 = QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setMinimumSize(QtCore.QSize(70, 0))
        self.label_3.setMaximumSize(QtCore.QSize(70, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.category_combo = QtGui.QComboBox(Form)
        self.category_combo.setMinimumSize(QtCore.QSize(130, 25))
        self.category_combo.setMaximumSize(QtCore.QSize(130, 25))
        self.category_combo.setEditable(True)
        self.category_combo.setObjectName(_fromUtf8("category_combo"))
        self.horizontalLayout_2.addWidget(self.category_combo)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_3)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setMinimumSize(QtCore.QSize(110, 0))
        self.label_4.setMaximumSize(QtCore.QSize(110, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_4.addWidget(self.label_4)
        self.source_txt = QtGui.QLineEdit(Form)
        self.source_txt.setEnabled(False)
        self.source_txt.setMinimumSize(QtCore.QSize(0, 25))
        self.source_txt.setMaximumSize(QtCore.QSize(16777215, 25))
        self.source_txt.setObjectName(_fromUtf8("source_txt"))
        self.horizontalLayout_4.addWidget(self.source_txt)
        self.source_btn = QtGui.QPushButton(Form)
        self.source_btn.setMinimumSize(QtCore.QSize(25, 25))
        self.source_btn.setMaximumSize(QtCore.QSize(25, 25))
        self.source_btn.setText(_fromUtf8(""))
        self.source_btn.setObjectName(_fromUtf8("source_btn"))
        self.horizontalLayout_4.addWidget(self.source_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(10)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setMinimumSize(QtCore.QSize(110, 0))
        self.label_5.setMaximumSize(QtCore.QSize(110, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_5.addWidget(self.label_5)
        self.file_txt = QtGui.QLineEdit(Form)
        self.file_txt.setEnabled(False)
        self.file_txt.setMinimumSize(QtCore.QSize(0, 25))
        self.file_txt.setMaximumSize(QtCore.QSize(16777215, 25))
        self.file_txt.setObjectName(_fromUtf8("file_txt"))
        self.horizontalLayout_5.addWidget(self.file_txt)
        self.reset_btn = QtGui.QPushButton(Form)
        self.reset_btn.setMinimumSize(QtCore.QSize(25, 25))
        self.reset_btn.setMaximumSize(QtCore.QSize(25, 25))
        self.reset_btn.setText(_fromUtf8(""))
        self.reset_btn.setObjectName(_fromUtf8("reset_btn"))
        self.horizontalLayout_5.addWidget(self.reset_btn)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)
        spacerItem3 = QtGui.QSpacerItem(20, 10, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.verticalLayout_5.addItem(spacerItem3)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_7 = QtGui.QLabel(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.verticalLayout.addWidget(self.label_7)
        self.tag_txt = QtGui.QTextEdit(Form)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.tag_txt.setFont(font)
        self.tag_txt.setObjectName(_fromUtf8("tag_txt"))
        self.verticalLayout.addWidget(self.tag_txt)
        self.verticalLayout_5.addLayout(self.verticalLayout)
        self.horizontalLayout_8.addLayout(self.verticalLayout_5)
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setSpacing(20)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setSpacing(20)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.snapshot_img = QtGui.QLabel(Form)
        self.snapshot_img.setMinimumSize(QtCore.QSize(300, 250))
        self.snapshot_img.setMaximumSize(QtCore.QSize(355, 300))
        self.snapshot_img.setMidLineWidth(250)
        self.snapshot_img.setText(_fromUtf8(""))
        self.snapshot_img.setAlignment(QtCore.Qt.AlignCenter)
        self.snapshot_img.setObjectName(_fromUtf8("snapshot_img"))
        self.verticalLayout_6.addWidget(self.snapshot_img)
        self.snapshot_btn = QtGui.QPushButton(Form)
        self.snapshot_btn.setMinimumSize(QtCore.QSize(200, 30))
        self.snapshot_btn.setMaximumSize(QtCore.QSize(1999999, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.snapshot_btn.setFont(font)
        self.snapshot_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.snapshot_btn.setObjectName(_fromUtf8("snapshot_btn"))
        self.verticalLayout_6.addWidget(self.snapshot_btn)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        self.horizontalLayout_8.addLayout(self.verticalLayout_8)
        self.gridLayout.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(10)
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem4)
        self.inventory_btn = QtGui.QPushButton(Form)
        self.inventory_btn.setMinimumSize(QtCore.QSize(30, 25))
        self.inventory_btn.setMaximumSize(QtCore.QSize(30, 25))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.inventory_btn.setFont(font)
        self.inventory_btn.setText(_fromUtf8(""))
        self.inventory_btn.setObjectName(_fromUtf8("inventory_btn"))
        self.horizontalLayout_7.addWidget(self.inventory_btn)
        self.send_btn = QtGui.QPushButton(Form)
        self.send_btn.setMinimumSize(QtCore.QSize(80, 25))
        self.send_btn.setMaximumSize(QtCore.QSize(80, 25))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.send_btn.setFont(font)
        self.send_btn.setObjectName(_fromUtf8("send_btn"))
        self.horizontalLayout_7.addWidget(self.send_btn)
        self.close_btn = QtGui.QPushButton(Form)
        self.close_btn.setMinimumSize(QtCore.QSize(80, 25))
        self.close_btn.setMaximumSize(QtCore.QSize(80, 25))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.close_btn.setFont(font)
        self.close_btn.setObjectName(_fromUtf8("close_btn"))
        self.horizontalLayout_7.addWidget(self.close_btn)
        self.gridLayout.addLayout(self.horizontalLayout_7, 1, 0, 1, 1)

        self.retranslateUi(Form)
        self.type_combo.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Layout Inventory by RND youkyoung.kim", None))
        self.label.setText(_translate("Form", "Title", None))
        self.label_2.setText(_translate("Form", "Asset Type", None))
        self.label_3.setText(_translate("Form", "Category", None))
        self.label_4.setText(_translate("Form", "Scene Path", None))
        self.label_5.setText(_translate("Form", "File Name", None))
        self.label_7.setText(_translate("Form", "Tags", None))
        self.snapshot_btn.setText(_translate("Form", "Take SnapShot", None))
        self.send_btn.setText(_translate("Form", "Send", None))
        self.close_btn.setText(_translate("Form", "Close", None))
