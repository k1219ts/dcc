# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stereo_option.ui'
#
# Created: Wed Nov 29 21:58:16 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

#from PyQt4 import QtCore, QtGui
import Qt
import Qt.QtWidgets as QtGui
import Qt.QtCore as QtCore

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

import os
scriptRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(644, 194)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.cameraFrame = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cameraFrame.sizePolicy().hasHeightForWidth())
        self.cameraFrame.setSizePolicy(sizePolicy)
        self.cameraFrame.setMaximumSize(QtCore.QSize(16777215, 64))
        self.cameraFrame.setFrameShape(QtGui.QFrame.Box)
        self.cameraFrame.setFrameShadow(QtGui.QFrame.Sunken)
        self.cameraFrame.setObjectName(_fromUtf8("cameraFrame"))
        self.horizontalLayout_14 = QtGui.QHBoxLayout(self.cameraFrame)
        self.horizontalLayout_14.setSpacing(2)
        self.horizontalLayout_14.setMargin(0)
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.horizontalLayout_13 = QtGui.QHBoxLayout()
        self.horizontalLayout_13.setSpacing(2)
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.sectionOuterFrame = QtGui.QFrame(self.cameraFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sectionOuterFrame.sizePolicy().hasHeightForWidth())
        self.sectionOuterFrame.setSizePolicy(sizePolicy)
        self.sectionOuterFrame.setMinimumSize(QtCore.QSize(100, 0))
        self.sectionOuterFrame.setMaximumSize(QtCore.QSize(100, 16777215))
        self.sectionOuterFrame.setAutoFillBackground(False)
        self.sectionOuterFrame.setStyleSheet(_fromUtf8("QFrame { background-color:rgb(150,150,150); }"))
        self.sectionOuterFrame.setFrameShape(QtGui.QFrame.NoFrame)
        self.sectionOuterFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.sectionOuterFrame.setLineWidth(1)
        self.sectionOuterFrame.setMidLineWidth(0)
        self.sectionOuterFrame.setObjectName(_fromUtf8("sectionOuterFrame"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.sectionOuterFrame)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.sectionBrief = QtGui.QLabel(self.sectionOuterFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sectionBrief.sizePolicy().hasHeightForWidth())
        self.sectionBrief.setSizePolicy(sizePolicy)
        self.sectionBrief.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = Qt.QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.sectionBrief.setFont(font)
        self.sectionBrief.setAcceptDrops(True)
        self.sectionBrief.setAutoFillBackground(False)
        self.sectionBrief.setStyleSheet(_fromUtf8("QLabel {      color: rgb(22,22,22); }"))
        self.sectionBrief.setAlignment(QtCore.Qt.AlignCenter)
        self.sectionBrief.setObjectName(_fromUtf8("sectionBrief"))
        self.verticalLayout_4.addWidget(self.sectionBrief)
        self.horizontalLayout_13.addWidget(self.sectionOuterFrame)
        self.sectionTypeFrame = QtGui.QFrame(self.cameraFrame)
        self.sectionTypeFrame.setFrameShape(QtGui.QFrame.Box)
        self.sectionTypeFrame.setFrameShadow(QtGui.QFrame.Raised)
        self.sectionTypeFrame.setObjectName(_fromUtf8("sectionTypeFrame"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.sectionTypeFrame)
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.sectionLeft = QtGui.QToolButton(self.sectionTypeFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sectionLeft.sizePolicy().hasHeightForWidth())
        self.sectionLeft.setSizePolicy(sizePolicy)
        self.sectionLeft.setMinimumSize(QtCore.QSize(0, 0))
        self.sectionLeft.setMaximumSize(QtCore.QSize(16777215, 36))
        self.sectionLeft.setCursor(Qt.QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.sectionLeft.setMouseTracking(False)
        self.sectionLeft.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sectionLeft.setStyleSheet(_fromUtf8("QToolButton { border-style: none;  }\n"
"QToolButton:hover {background-color: rgb(86,86,86);\n"
"                                color:rgb(244,244,244);\n"
"                                   border-radius:10px;\n"
"}    \n"
"\n"
"QToolButton:pressed {background-color: rgb(128, 128, 128);\n"
"                                 border-style: inset;\n"
"                                border-width: 1px; \n"
"                                   border-radius: 10px; \n"
"                                border-color:rgb(128,128,128);\n"
"}\n"
"\n"
"\n"
"QToolButton:checked {background-color:rgb(128, 77, 77);\n"
"                                 border-style: inset;\n"
"                                border-width: 1px; \n"
"                   border-radius: 10px; \n"
"                   border-color: beige;}\n"
""))
        icon = Qt.QtGui.QIcon()
        icon.addPixmap(Qt.QtGui.QPixmap(os.path.join(scriptRoot, 'icons', 'menuIconView.png')), Qt.QtGui.QIcon.Normal, Qt.QtGui.QIcon.Off)
        self.sectionLeft.setIcon(icon)
        self.sectionLeft.setIconSize(QtCore.QSize(48, 48))
        self.sectionLeft.setCheckable(True)
        self.sectionLeft.setChecked(True)
        self.sectionLeft.setPopupMode(QtGui.QToolButton.DelayedPopup)
        self.sectionLeft.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.sectionLeft.setArrowType(QtCore.Qt.NoArrow)
        self.sectionLeft.setObjectName(_fromUtf8("sectionLeft"))
        self.horizontalLayout_2.addWidget(self.sectionLeft)
        self.sectionRight = QtGui.QToolButton(self.sectionTypeFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sectionRight.sizePolicy().hasHeightForWidth())
        self.sectionRight.setSizePolicy(sizePolicy)
        self.sectionRight.setMaximumSize(QtCore.QSize(16777215, 36))
        self.sectionRight.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sectionRight.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sectionRight.setStyleSheet(_fromUtf8("QToolButton { border-style: none;  }\n"
"QToolButton:hover {background-color: rgb(86,86,86);\n"
"                                color:rgb(244,244,244);\n"
"                                   border-radius:10px;\n"
"}    \n"
"\n"
"QToolButton:pressed {background-color: rgb(128, 128, 128);\n"
"                                 border-style: inset;\n"
"                                border-width: 1px; \n"
"                                   border-radius: 10px; \n"
"                                border-color:rgb(128,128,128);\n"
"}\n"
"\n"
"\n"
"QToolButton:checked {background-color:rgb( 77,117 , 145);\n"
"                                 border-style: inset;\n"
"                                border-width: 1px; \n"
"                   border-radius: 10px; \n"
"                   border-color: beige;}\n"
""))
        self.sectionRight.setIcon(icon)
        self.sectionRight.setIconSize(QtCore.QSize(48, 48))
        self.sectionRight.setCheckable(True)
        self.sectionRight.setChecked(True)
        self.sectionRight.setAutoRepeat(False)
        self.sectionRight.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.sectionRight.setAutoRaise(False)
        self.sectionRight.setObjectName(_fromUtf8("sectionRight"))
        self.horizontalLayout_2.addWidget(self.sectionRight)
        self.sectionStereo = QtGui.QToolButton(self.sectionTypeFrame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sectionStereo.sizePolicy().hasHeightForWidth())
        self.sectionStereo.setSizePolicy(sizePolicy)
        self.sectionStereo.setMaximumSize(QtCore.QSize(16777215, 48))
        self.sectionStereo.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sectionStereo.setStyleSheet(_fromUtf8("QToolButton { border-style: none;  }\n"
"QToolButton:hover {background-color: rgb(86,86,86);\n"
"                                  color: rgb(244,244,244);\n"
"                                     border-radius:10px;\n"
"}    \n"
"\n"
"QToolButton:pressed {background-color: rgb(150, 77, 150);\n"
"                                 border-style: inset;\n"
"                                border-width: 1px; \n"
"                                   border-radius: 10px; \n"
"                                border-color:rgb(128,128,128);\n"
"}\n"
""))
        icon1 = Qt.QtGui.QIcon()
        icon1.addPixmap(Qt.QtGui.QPixmap(os.path.join(scriptRoot, 'icons', 'viewStereo.png')), Qt.QtGui.QIcon.Normal, Qt.QtGui.QIcon.Off)
        self.sectionStereo.setIcon(icon1)
        self.sectionStereo.setIconSize(QtCore.QSize(48, 48))
        self.sectionStereo.setCheckable(True)
        self.sectionStereo.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.sectionStereo.setObjectName(_fromUtf8("sectionStereo"))
        self.horizontalLayout_2.addWidget(self.sectionStereo)
        self.horizontalLayout_13.addWidget(self.sectionTypeFrame)
        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)
        self.verticalLayout.addWidget(self.cameraFrame)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.sectionBrief.setText(_translate("Form", "Camera", None))
        self.sectionLeft.setText(_translate("Form", "Left", None))
        self.sectionRight.setText(_translate("Form", "Right", None))
        self.sectionStereo.setText(_translate("Form", "Stereo", None))
