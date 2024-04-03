# -*- coding: utf-8 -*-
####################################################
#          coding by RND youkyoung.kim             #
####################################################

import Qt
import Qt.QtWidgets as QtGui
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

class CameraCreate_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(500, 400)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setMinimumSize(QtCore.QSize(150, 30))
        self.label_6.setMaximumSize(QtCore.QSize(150, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_5.addWidget(self.label_6)
        self.show_txt = QtGui.QLabel(Form)
        self.show_txt.setMinimumSize(QtCore.QSize(0, 30))
        
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)
        self.show_txt.setFont(font)
        self.show_txt.setObjectName(_fromUtf8("show_txt"))
        self.horizontalLayout_5.addWidget(self.show_txt)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_8 = QtGui.QLabel(Form)
        self.label_8.setMinimumSize(QtCore.QSize(150, 30))
        self.label_8.setMaximumSize(QtCore.QSize(150, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_6.addWidget(self.label_8)
        
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)        
        self.model_txt = QtGui.QLabel(Form)
        self.model_txt.setFont(font)
        self.model_txt.setMinimumSize(QtCore.QSize(0, 30))
        self.model_txt.setObjectName(_fromUtf8("model_txt"))
        self.horizontalLayout_6.addWidget(self.model_txt)

        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.line_2 = QtGui.QFrame(Form)
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout_4.addWidget(self.line_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(60, 0))
        self.label.setMaximumSize(QtCore.QSize(60, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.cam_spin = QtGui.QSpinBox(Form)
        self.cam_spin.setMinimumSize(QtCore.QSize(100, 30))
        self.cam_spin.setMaximumSize(QtCore.QSize(100, 30))
        self.cam_spin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.cam_spin.setObjectName(_fromUtf8("cam_spin"))
        self.horizontalLayout.addWidget(self.cam_spin)
        self.create_btn = QtGui.QPushButton(Form)
        self.create_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.create_btn.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.create_btn.setFont(font)
        self.create_btn.setObjectName(_fromUtf8("create_btn"))
        self.horizontalLayout.addWidget(self.create_btn)
        spacerItem = QtGui.QSpacerItem(100, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.plus_btn = QtGui.QPushButton(Form)
        self.plus_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.plus_btn.setMaximumSize(QtCore.QSize(30, 30))

        self.plus_btn.setFont(font)
        self.plus_btn.setObjectName(_fromUtf8("plus_btn"))
        self.horizontalLayout.addWidget(self.plus_btn)
        self.minus_btn = QtGui.QPushButton(Form)
        self.minus_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.minus_btn.setMaximumSize(QtCore.QSize(30, 30))

        self.minus_btn.setFont(font)
        self.minus_btn.setObjectName(_fromUtf8("minus_btn"))
        self.horizontalLayout.addWidget(self.minus_btn)
        self.verticalLayout.addLayout(self.horizontalLayout)
        spacerItem1 = QtGui.QSpacerItem(20, 10, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem1)
        self.list_tree = QtGui.QTreeWidget(Form)
        self.list_tree.setObjectName(_fromUtf8("list_tree"))
        self.list_tree.headerItem().setTextAlignment(0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)
        self.list_tree.headerItem().setFont(0, font)
        self.list_tree.headerItem().setFont(1, font)
        self.list_tree.headerItem().setTextAlignment(1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)
        self.list_tree.headerItem().setTextAlignment(2, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)

        self.list_tree.headerItem().setFont(2, font)
        self.list_tree.headerItem().setTextAlignment(3, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)

        self.list_tree.headerItem().setFont(3, font)
        self.verticalLayout.addWidget(self.list_tree)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem2 = QtGui.QSpacerItem(153, 27, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.jsonwrite_btn = QtGui.QPushButton(Form)
        self.jsonwrite_btn.setMaximumSize(QtCore.QSize(30, 30))
        self.jsonwrite_btn.setMinimumSize(QtCore.QSize(30, 30))

        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.jsonwrite_btn.setFont(font)
        self.jsonwrite_btn.setObjectName(_fromUtf8("jsonwrite_btn"))
        self.horizontalLayout_2.addWidget(self.jsonwrite_btn)

        self.cancel_btn = QtGui.QPushButton(Form)
        self.cancel_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.cancel_btn.setMaximumSize(QtCore.QSize(100, 30))
        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName(_fromUtf8("cancel_btn"))
        self.horizontalLayout_2.addWidget(self.cancel_btn)

        self.ok_btn = QtGui.QPushButton(Form)
        self.ok_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.ok_btn.setFont(font)
        self.ok_btn.setObjectName(_fromUtf8("ok_btn"))
        self.horizontalLayout_2.addWidget(self.ok_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Layout Camera Modify", None))
        self.label_6.setText(_translate("Form", "Show :", None))
        self.show_txt.setText(_translate("Form", "", None))
        self.label_8.setText(_translate("Form", "Camera Model :", None))
        self.model_txt.setText(_translate("Form", "", None))
        self.label.setText(_translate("Form", "Camera", None))
        self.create_btn.setText(_translate("Form", "Create", None))
        self.plus_btn.setText(_translate("Form", "+", None))
        self.minus_btn.setText(_translate("Form", "ㅡ", None))
        self.list_tree.headerItem().setText(0, _translate("Form", "No", None))
        self.list_tree.headerItem().setText(1, _translate("Form", "Camera Name", None))
        self.list_tree.headerItem().setText(2, _translate("Form", "Shape Name", None))
        self.list_tree.headerItem().setText(3, _translate("Form", "FocalLength", None))
        self.jsonwrite_btn.setToolTip(_translate("Form", "Camera Model Setting", None))
        self.jsonwrite_btn.setText(_translate("Form", "C", None))
        self.cancel_btn.setText(_translate("Form", "Cancel", None))
        self.ok_btn.setText(_translate("Form", "Ok", None))
