# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AssetExportUI.ui',
# licensing of 'AssetExportUI.ui' applies.
#
# Created: Fri Feb 26 18:34:12 2021
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets
import os

currentDir = os.path.dirname(__file__)

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1100, 834)
        Form.setStyleSheet("background-color: rgb(90,90,90)")
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setHorizontalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(Form)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.treeWidget = QtWidgets.QTreeWidget(self.frame)
        self.treeWidget.setStyleSheet("")
        self.treeWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.treeWidget.setRootIsDecorated(False)
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "1")
        self.treeWidget.header().setVisible(True)
        self.treeWidget.header().setDefaultSectionSize(800)
        self.gridLayout_2.addWidget(self.treeWidget, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 4, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(6, -1, 6, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color : white;background-color: rgb(90,90,90);padding : 0 5 0 5 px;")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.showEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.showEdit.setFont(font)
        self.showEdit.setStyleSheet("color : white")
        self.showEdit.setText("")
        self.showEdit.setObjectName("showEdit")
        self.horizontalLayout_2.addWidget(self.showEdit)
        self.label_5 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color : white;background-color: rgb(90,90,90);padding : 0 5 0 5 px;")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_2.addWidget(self.label_5)
        self.shotEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.shotEdit.setFont(font)
        self.shotEdit.setStatusTip("")
        self.shotEdit.setStyleSheet("color : white")
        self.shotEdit.setText("")
        self.shotEdit.setObjectName("shotEdit")
        self.horizontalLayout_2.addWidget(self.shotEdit)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setMaximumSize(QtCore.QSize(70, 16777215))
        self.label.setStyleSheet("background-color: rgb(62, 115, 186);padding:  0 0 0 20 px;")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(_fromUtf8("%s/resources/USDLogo.png" % currentDir)))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(0, 70))
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setWeight(75)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color: rgb(62, 115, 186);color: rgb(255, 255, 255);")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setMinimumSize(QtCore.QSize(0, 70))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setWeight(75)
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: rgb(62, 115, 186);color: rgb(255, 255, 255);")
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignBottom|QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 2, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setContentsMargins(6, -1, 6, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.clipCheckBox = QtWidgets.QCheckBox(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.clipCheckBox.setFont(font)
        self.clipCheckBox.setStyleSheet("color : white;")
        self.clipCheckBox.setObjectName("clipCheckBox")
        self.horizontalLayout_3.addWidget(self.clipCheckBox)
        self.glmMotionCheckBox = QtWidgets.QCheckBox(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.glmMotionCheckBox.setFont(font)
        self.glmMotionCheckBox.setStyleSheet("color : white;")
        self.glmMotionCheckBox.setObjectName("glmMotionCheckBox")
        self.horizontalLayout_3.addWidget(self.glmMotionCheckBox)
        self.stepLabel = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.stepLabel.setFont(font)
        self.stepLabel.setStyleSheet("color : white;")
        self.stepLabel.setObjectName("stepLabel")
        self.horizontalLayout_3.addWidget(self.stepLabel)
        self.stepEdit = QtWidgets.QLineEdit(Form)
        self.stepEdit.setMaximumSize(QtCore.QSize(75, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.stepEdit.setFont(font)
        self.stepEdit.setStyleSheet("color : white;")
        self.stepEdit.setText("")
        self.stepEdit.setObjectName("stepEdit")
        self.horizontalLayout_3.addWidget(self.stepEdit)
        self.loopRangeLabel = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.loopRangeLabel.setFont(font)
        self.loopRangeLabel.setStyleSheet("color : white;")
        self.loopRangeLabel.setObjectName("loopRangeLabel")
        self.horizontalLayout_3.addWidget(self.loopRangeLabel)
        self.loopStartEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.loopStartEdit.setFont(font)
        self.loopStartEdit.setStyleSheet("color : white;")
        self.loopStartEdit.setObjectName("loopStartEdit")
        self.horizontalLayout_3.addWidget(self.loopStartEdit)
        self.loopEndEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.loopEndEdit.setFont(font)
        self.loopEndEdit.setStyleSheet("color : white;")
        self.loopEndEdit.setObjectName("loopEndEdit")
        self.horizontalLayout_3.addWidget(self.loopEndEdit)
        self.loopScaleLabel = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.loopScaleLabel.setFont(font)
        self.loopScaleLabel.setStyleSheet("color : white;")
        self.loopScaleLabel.setObjectName("loopScaleLabel")
        self.horizontalLayout_3.addWidget(self.loopScaleLabel)
        self.loopScaleEdit = QtWidgets.QLineEdit(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.loopScaleEdit.setFont(font)
        self.loopScaleEdit.setStyleSheet("color : white;")
        self.loopScaleEdit.setObjectName("loopScaleEdit")
        self.horizontalLayout_3.addWidget(self.loopScaleEdit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.overWriteCheckBox = QtWidgets.QCheckBox(Form)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setWeight(75)
        font.setBold(True)
        self.overWriteCheckBox.setFont(font)
        self.overWriteCheckBox.setStyleSheet("color : white;")
        self.overWriteCheckBox.setObjectName("overWriteCheckBox")
        self.horizontalLayout_3.addWidget(self.overWriteCheckBox)
        self.gridLayout.addLayout(self.horizontalLayout_3, 3, 0, 1, 1)
        self.assetExportBtn = QtWidgets.QPushButton(Form)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setWeight(75)
        font.setBold(True)
        self.assetExportBtn.setFont(font)
        self.assetExportBtn.setStyleSheet("color : white;background-color: rgb(62, 109, 186);")
        self.assetExportBtn.setObjectName("assetExportBtn")
        self.gridLayout.addWidget(self.assetExportBtn, 5, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "DXUSD-3.0 Asset Exporter", None, -1))
        self.treeWidget.headerItem().setText(1, QtWidgets.QApplication.translate("Form", "2", None, -1))
        self.label_4.setText(QtWidgets.QApplication.translate("Form", "SHOW", None, -1))
        self.label_5.setText(QtWidgets.QApplication.translate("Form", "SHOT NAME", None, -1))
        self.shotEdit.setPlaceholderText(QtWidgets.QApplication.translate("Form", "if you enter a shot name, it will be published as shot.", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("Form", "ASSET EXPORT", None, -1))
        self.clipCheckBox.setText(QtWidgets.QApplication.translate("Form", "Clip", None, -1))
        self.glmMotionCheckBox.setText(QtWidgets.QApplication.translate("Form", "Motion", None, -1))
        self.stepLabel.setText(QtWidgets.QApplication.translate("Form", "STEP", None, -1))
        self.stepEdit.setPlaceholderText(QtWidgets.QApplication.translate("Form", "1.0", None, -1))
        self.loopRangeLabel.setText(QtWidgets.QApplication.translate("Form", "Range", None, -1))
        self.loopStartEdit.setPlaceholderText(QtWidgets.QApplication.translate("Form", "0", None, -1))
        self.loopEndEdit.setPlaceholderText(QtWidgets.QApplication.translate("Form", "1000", None, -1))
        self.loopScaleLabel.setText(QtWidgets.QApplication.translate("Form", "Time Scale", None, -1))
        self.loopScaleEdit.setPlaceholderText(QtWidgets.QApplication.translate("Form", "0.8, 1.0, 1.5", None, -1))
        self.overWriteCheckBox.setText(QtWidgets.QApplication.translate("Form", "OverWrite", None, -1))
        self.assetExportBtn.setText(QtWidgets.QApplication.translate("Form", "EXPORT", None, -1))
