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

class CameraSet_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(424, 150)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        spacerItem = QtGui.QSpacerItem(20, 7, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.help_btn = QtGui.QPushButton(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.help_btn.sizePolicy().hasHeightForWidth())
        self.help_btn.setSizePolicy(sizePolicy)
        self.help_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.help_btn.setMaximumSize(QtCore.QSize(30, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.help_btn.setFont(font)
        self.help_btn.setIconSize(QtCore.QSize(30, 30))
        self.help_btn.setObjectName(_fromUtf8("help_btn"))
        self.horizontalLayout_3.addWidget(self.help_btn)
        self.cancel_btn = QtGui.QPushButton(Form)
        self.cancel_btn.setMinimumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName(_fromUtf8("cancel_btn"))
        self.horizontalLayout_3.addWidget(self.cancel_btn)
        self.ok_btn = QtGui.QPushButton(Form)
        self.ok_btn.setMinimumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ok_btn.setFont(font)
        self.ok_btn.setObjectName(_fromUtf8("ok_btn"))
        self.horizontalLayout_3.addWidget(self.ok_btn)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_17 = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setMinimumSize(QtCore.QSize(120, 30))
        self.label_17.setMaximumSize(QtCore.QSize(50, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.horizontalLayout.addWidget(self.label_17)
        spacerItem2 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.show_combo = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_combo.sizePolicy().hasHeightForWidth())
        self.show_combo.setSizePolicy(sizePolicy)
        self.show_combo.setMinimumSize(QtCore.QSize(250, 30))
        self.show_combo.setMaximumSize(QtCore.QSize(150, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.show_combo.setFont(font)
        self.show_combo.setStyleSheet(_fromUtf8("QComboBox QAbstractItemView::item{margin-top: 5px;}"))
        self.show_combo.setEditable(False)
        self.show_combo.setMaxVisibleItems(10)
        self.show_combo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.show_combo.setMinimumContentsLength(0)
        self.show_combo.setIconSize(QtCore.QSize(25, 30))
        self.show_combo.setFrame(True)
        self.show_combo.setObjectName(_fromUtf8("show_combo"))
        self.horizontalLayout.addWidget(self.show_combo)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtGui.QFrame(Form)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_18 = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.label_18.setMinimumSize(QtCore.QSize(120, 30))
        self.label_18.setMaximumSize(QtCore.QSize(50, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.horizontalLayout_2.addWidget(self.label_18)
        spacerItem3 = QtGui.QSpacerItem(20, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.model_combo = QtGui.QComboBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.model_combo.sizePolicy().hasHeightForWidth())
        self.model_combo.setSizePolicy(sizePolicy)
        self.model_combo.setMinimumSize(QtCore.QSize(250, 30))
        self.model_combo.setMaximumSize(QtCore.QSize(150, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.model_combo.setFont(font)
        self.model_combo.setStyleSheet(_fromUtf8("QComboBox QAbstractItemView::item{margin-top: 5px;}"))
        self.model_combo.setEditable(False)
        self.model_combo.setMaxVisibleItems(10)
        self.model_combo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(0)
        self.model_combo.setIconSize(QtCore.QSize(25, 30))
        self.model_combo.setFrame(True)
        self.model_combo.setObjectName(_fromUtf8("model_combo"))
        self.horizontalLayout_2.addWidget(self.model_combo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.show_combo.setCurrentIndex(-1)
        self.model_combo.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Camera Setting", None))
        self.help_btn.setText(_translate("Form", "?", None))
        self.cancel_btn.setText(_translate("Form", "Cancel", None))
        self.ok_btn.setText(_translate("Form", "Ok", None))
        self.label_17.setText(_translate("Form", "Show", None))
        self.label_18.setText(_translate("Form", "Camera Model", None))
