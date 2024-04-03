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

class ShotCreate_Form(object):
    def setupUi(self, shotform):
        shotform.setObjectName(_fromUtf8("shotform"))
        shotform.resize(1200, 506)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(shotform.sizePolicy().hasHeightForWidth())
        shotform.setSizePolicy(sizePolicy)
        self.gridLayout = QtGui.QGridLayout(shotform)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label2 = QtGui.QLabel(shotform)
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label2.setFont(font)
        self.label2.setObjectName(_fromUtf8("label2"))
        self.horizontalLayout_2.addWidget(self.label2)
        self.line = QtGui.QFrame(shotform)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.horizontalLayout_2.addWidget(self.line)
        self.shot_spin = QtGui.QSpinBox(shotform)
        self.shot_spin.setMinimumSize(QtCore.QSize(100, 30))
        self.shot_spin.setMaximumSize(QtCore.QSize(100, 30))
        self.shot_spin.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.shot_spin.setObjectName(_fromUtf8("shot_spin"))
        self.horizontalLayout_2.addWidget(self.shot_spin)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.create_btn = QtGui.QPushButton(shotform)
        self.create_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.create_btn.setMaximumSize(QtCore.QSize(100, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.create_btn.setFont(font)
        self.create_btn.setObjectName(_fromUtf8("create_btn"))
        self.horizontalLayout_4.addWidget(self.create_btn)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.add_btn = QtGui.QPushButton(shotform)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_btn.sizePolicy().hasHeightForWidth())
        self.add_btn.setSizePolicy(sizePolicy)
        self.add_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.add_btn.setMaximumSize(QtCore.QSize(30, 30))

        self.add_btn.setFont(font)
        self.add_btn.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.add_btn.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.add_btn.setObjectName(_fromUtf8("add_btn"))
        self.horizontalLayout.addWidget(self.add_btn)
        self.minus_btn = QtGui.QPushButton(shotform)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.minus_btn.sizePolicy().hasHeightForWidth())
        self.minus_btn.setSizePolicy(sizePolicy)
        self.minus_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.minus_btn.setMaximumSize(QtCore.QSize(30, 30))

        self.minus_btn.setFont(font)
        self.minus_btn.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.minus_btn.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.minus_btn.setObjectName(_fromUtf8("minus_btn"))
        self.horizontalLayout.addWidget(self.minus_btn)
        self.horizontalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.shot_tree = QtGui.QTreeWidget(shotform)

        self.shot_tree.setFont(font)
        self.shot_tree.setObjectName(_fromUtf8("shot_tree"))
        font = Qt.QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(75)
        for i in range(9):
            self.shot_tree.headerItem().setFont(i, font)
            self.shot_tree.headerItem().setTextAlignment(i, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignCenter)

        self.shot_tree.header().setHighlightSections(False)
        self.verticalLayout.addWidget(self.shot_tree)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)

        font = Qt.QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)

        self.shot_btn = QtGui.QPushButton(shotform)
        self.shot_btn.setMaximumSize(QtCore.QSize(30, 30))
        self.shot_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.shot_btn.setFont(font)
        self.shot_btn.setObjectName(_fromUtf8("shot_btn"))
        self.horizontalLayout_3.addWidget(self.shot_btn)

        self.fbx_btn = QtGui.QPushButton(shotform)
        self.fbx_btn.setMaximumSize(QtCore.QSize(30, 30))
        self.fbx_btn.setMinimumSize(QtCore.QSize(30, 30))
        self.fbx_btn.setFont(font)
        self.fbx_btn.setObjectName(_fromUtf8("fbx_btn"))
        self.horizontalLayout_3.addWidget(self.fbx_btn)

        self.cancel_btn = QtGui.QPushButton(shotform)
        self.cancel_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.cancel_btn.setMaximumSize(QtCore.QSize(100, 30))
        self.cancel_btn.setFont(font)
        self.cancel_btn.setObjectName(_fromUtf8("cancel_btn"))
        self.horizontalLayout_3.addWidget(self.cancel_btn)

        self.ok_btn = QtGui.QPushButton(shotform)
        self.ok_btn.setMinimumSize(QtCore.QSize(100, 30))
        self.ok_btn.setMaximumSize(QtCore.QSize(100, 30))
        self.ok_btn.setFont(font)
        self.ok_btn.setObjectName(_fromUtf8("ok_btn"))
        self.horizontalLayout_3.addWidget(self.ok_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(shotform)
        QtCore.QMetaObject.connectSlotsByName(shotform)

    def retranslateUi(self, shotform):
        shotform.setWindowTitle(_translate("shotform", "Shot Sequencer Edit", None))
        self.label2.setText(_translate("shotform", "Shot Count", None))
        self.create_btn.setText(_translate("shotform", "Create", None))
        self.add_btn.setText(_translate("shotform", "+", None))
        self.minus_btn.setText(_translate("shotform", "ㅡ", None))
        self.shot_tree.headerItem().setText(0, _translate("shotform", "No.", None))
        self.shot_tree.headerItem().setText(1, _translate("shotform", "Shot", None))
        self.shot_tree.headerItem().setText(2, _translate("shotform", "Camera", None))
        self.shot_tree.headerItem().setText(3, _translate("shotform", "T_Start", None))
        self.shot_tree.headerItem().setText(4, _translate("shotform", "T_End ", None))
        self.shot_tree.headerItem().setText(5, _translate("shotform", "S_Start", None))
        self.shot_tree.headerItem().setText(6, _translate("shotform", "S_End", None))
        self.shot_tree.headerItem().setText(7, _translate("shotform", "Scale", None))
        self.shot_tree.headerItem().setText(8, _translate("shotform", "Track", None))
        self.fbx_btn.setText(_translate("shotform", "U", None))
        self.shot_btn.setText(_translate("shotform", "S", None))
        self.cancel_btn.setText(_translate("shotform", "Cancel", None))
        self.ok_btn.setText(_translate("shotform", "Ok", None))

