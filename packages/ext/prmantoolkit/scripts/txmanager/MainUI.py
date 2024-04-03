# -*- coding: utf-8 -*-
from __future__ import print_function

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created: Wed Aug  1 14:47:19 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!


from PySide2 import QtWidgets,QtCore,QtGui
import os
currentDir = os.path.dirname(__file__)

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
        Form.resize(900, 358)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label = QtWidgets.QLabel(Form)
        self.label.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.directoryEdit = QtWidgets.QLineEdit(Form)
        self.directoryEdit.setObjectName(_fromUtf8("directoryEdit"))
        self.gridLayout_2.addWidget(self.directoryEdit, 0, 1, 1, 2)
        self.filenameList = QtWidgets.QListWidget(Form)
        self.filenameList.setMinimumSize(QtCore.QSize(200, 0))
        self.filenameList.setObjectName(_fromUtf8("filenameList"))
        self.gridLayout_2.addWidget(self.filenameList, 1, 0, 1, 2)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        spacerItem = QtWidgets.QSpacerItem(20, 169, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 4, 1, 1, 1)
        self.resizeProxyBtn = QtWidgets.QPushButton(self.tab_3)
        self.resizeProxyBtn.setObjectName(_fromUtf8("resizeProxyBtn"))
        self.gridLayout_3.addWidget(self.resizeProxyBtn, 3, 0, 1, 4)
        self.resize512Btn = QtWidgets.QPushButton(self.tab_3)
        self.resize512Btn.setObjectName(_fromUtf8("resize512Btn"))
        self.gridLayout_3.addWidget(self.resize512Btn, 1, 0, 1, 1)
        self.resize1KBtn = QtWidgets.QPushButton(self.tab_3)
        self.resize1KBtn.setObjectName(_fromUtf8("resize1KBtn"))
        self.gridLayout_3.addWidget(self.resize1KBtn, 1, 1, 1, 1)
        self.resize2KBtn = QtWidgets.QPushButton(self.tab_3)
        self.resize2KBtn.setObjectName(_fromUtf8("resize2KBtn"))
        self.gridLayout_3.addWidget(self.resize2KBtn, 1, 2, 1, 1)
        self.resize4KBtn = QtWidgets.QPushButton(self.tab_3)
        self.resize4KBtn.setObjectName(_fromUtf8("resize4KBtn"))
        self.gridLayout_3.addWidget(self.resize4KBtn, 1, 3, 1, 1)
        self.resize8KBtn = QtWidgets.QPushButton(self.tab_3)
        self.resize8KBtn.setObjectName(_fromUtf8("resize8KBtn"))
        self.gridLayout_3.addWidget(self.resize8KBtn, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, _fromUtf8(""))
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.insertBtn = QtWidgets.QRadioButton(self.tab)
        self.insertBtn.setObjectName(_fromUtf8("insertBtn"))
        self.gridLayout.addWidget(self.insertBtn, 0, 0, 1, 1)
        self.changeBtn = QtWidgets.QRadioButton(self.tab)
        self.changeBtn.setObjectName(_fromUtf8("changeBtn"))
        self.gridLayout.addWidget(self.changeBtn, 0, 1, 1, 1)
        self.linkBtn = QtWidgets.QRadioButton(self.tab)
        self.linkBtn.setObjectName(_fromUtf8("linkBtn"))
        self.gridLayout.addWidget(self.linkBtn, 0, 2, 1, 1)
        self.removeBtn = QtWidgets.QRadioButton(self.tab)
        self.removeBtn.setObjectName(_fromUtf8("removeBtn"))
        self.gridLayout.addWidget(self.removeBtn, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 151, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 5, 2, 1, 1)
        self.orgVariantEdit = QtWidgets.QLineEdit(self.tab)
        self.orgVariantEdit.setObjectName(_fromUtf8("orgVariantEdit"))
        self.gridLayout.addWidget(self.orgVariantEdit, 1, 1, 1, 3)
        self.newVariantEdit = QtWidgets.QLineEdit(self.tab)
        self.newVariantEdit.setObjectName(_fromUtf8("newVariantEdit"))
        self.gridLayout.addWidget(self.newVariantEdit, 2, 1, 1, 3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.link1 = QtWidgets.QCheckBox(self.tab)
        self.link1.setObjectName(_fromUtf8("link1"))
        self.horizontalLayout.addWidget(self.link1)
        self.link2 = QtWidgets.QCheckBox(self.tab)
        self.link2.setObjectName(_fromUtf8("link2"))
        self.horizontalLayout.addWidget(self.link2)
        self.link3 = QtWidgets.QCheckBox(self.tab)
        self.link3.setObjectName(_fromUtf8("link3"))
        self.horizontalLayout.addWidget(self.link3)
        self.link4 = QtWidgets.QCheckBox(self.tab)
        self.link4.setObjectName(_fromUtf8("link4"))
        self.horizontalLayout.addWidget(self.link4)
        self.link5 = QtWidgets.QCheckBox(self.tab)
        self.link5.setObjectName(_fromUtf8("link5"))
        self.horizontalLayout.addWidget(self.link5)
        self.link6 = QtWidgets.QCheckBox(self.tab)
        self.link6.setObjectName(_fromUtf8("link6"))
        self.horizontalLayout.addWidget(self.link6)
        self.link7 = QtWidgets.QCheckBox(self.tab)
        self.link7.setObjectName(_fromUtf8("link7"))
        self.horizontalLayout.addWidget(self.link7)
        self.link8 = QtWidgets.QCheckBox(self.tab)
        self.link8.setObjectName(_fromUtf8("link8"))
        self.horizontalLayout.addWidget(self.link8)
        self.link9 = QtWidgets.QCheckBox(self.tab)
        self.link9.setObjectName(_fromUtf8("link9"))
        self.horizontalLayout.addWidget(self.link9)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 4)
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.gridLayout.addWidget(self.pushButton, 4, 0, 1, 4)
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        spacerItem2 = QtWidgets.QSpacerItem(20, 165, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem2, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_4.addWidget(self.label_4, 0, 0, 1, 1)
        self.renameOriginEdit = QtWidgets.QLineEdit(self.tab_2)
        self.renameOriginEdit.setObjectName(_fromUtf8("renameOriginEdit"))
        self.gridLayout_4.addWidget(self.renameOriginEdit, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_4.addWidget(self.label_5, 1, 0, 1, 1)
        self.renameReplaceEdit = QtWidgets.QLineEdit(self.tab_2)
        self.renameReplaceEdit.setObjectName(_fromUtf8("renameReplaceEdit"))
        self.gridLayout_4.addWidget(self.renameReplaceEdit, 1, 1, 1, 1)
        self.renameExecBtn = QtWidgets.QPushButton(self.tab_2)
        self.renameExecBtn.setObjectName(_fromUtf8("renameExecBtn"))
        self.gridLayout_4.addWidget(self.renameExecBtn, 2, 0, 1, 2)
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_3"))
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        spacerItem = QtWidgets.QSpacerItem(20, 169, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_4.addItem(spacerItem, 4, 1, 1, 1)
        self.txMakeBtn = QtWidgets.QPushButton(self.tab_4)
        self.txMakeBtn.setObjectName(_fromUtf8("txMakeBtn"))
        self.gridLayout_4.addWidget(self.txMakeBtn, 0, 0, 1, 4)
        self.texToTiffBtn = QtWidgets.QPushButton(self.tab_4)
        self.texToTiffBtn.setObjectName(_fromUtf8("texToTiffBtn"))
        self.gridLayout_4.addWidget(self.texToTiffBtn, 1, 0, 1, 4)
        self.texToJpgBtn = QtWidgets.QPushButton(self.tab_4)
        self.texToJpgBtn.setObjectName(_fromUtf8("texToJpgBtn"))
        self.gridLayout_4.addWidget(self.texToJpgBtn, 2, 0, 1, 4)
        self.imageToJpgBtn = QtWidgets.QPushButton(self.tab_4)
        self.imageToJpgBtn.setObjectName(_fromUtf8("imageToJpgBtn"))
        self.gridLayout_4.addWidget(self.imageToJpgBtn, 3, 0, 1, 4)
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))

        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName(_fromUtf8("tab_5"))
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.sampleExecBtn = QtWidgets.QPushButton(self.tab_5)
        self.sampleExecBtn.setObjectName(_fromUtf8("sampleExecBtn"))
        self.sampleExecBtn.setText("execute")
        self.gridLayout_5.addWidget(self.sampleExecBtn, 2, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 169, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem, 3, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.specG = QtWidgets.QCheckBox(self.tab_5)
        self.specG.setObjectName(_fromUtf8("specG"))
        self.horizontalLayout_2.addWidget(self.specG)
        self.specR = QtWidgets.QCheckBox(self.tab_5)
        self.specR.setObjectName(_fromUtf8("specR "))
        self.horizontalLayout_2.addWidget(self.specR )
        self.norm = QtWidgets.QCheckBox(self.tab_5)
        self.norm.setObjectName(_fromUtf8("norm"))
        self.horizontalLayout_2.addWidget(self.norm)
        self.bump = QtWidgets.QCheckBox(self.tab_5)
        self.bump.setObjectName(_fromUtf8("bump"))
        self.horizontalLayout_2.addWidget(self.bump)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.whiteImg = QtWidgets.QRadioButton(self.tab_5)
        self.whiteImg.setIcon(QtGui.QIcon("%s/Resources/icon/512/white.jpg" % currentDir))
        self.whiteImg.setObjectName(_fromUtf8("whiteImg"))
        self.whiteImg.setText("whiteImg")
        self.horizontalLayout_3.addWidget(self.whiteImg)
        self.normImg = QtWidgets.QRadioButton(self.tab_5)
        self.normImg.setIcon(QtGui.QIcon("%s/Resources/icon/512/norm.jpg" % currentDir))
        self.normImg.setObjectName(_fromUtf8("normImg"))
        self.normImg.setText("norm")
        self.horizontalLayout_3.addWidget(self.normImg)
        self.gray0_8 = QtWidgets.QRadioButton(self.tab_5)
        self.gray0_8.setIcon(QtGui.QIcon("%s/Resources/icon/512/gray0.8.jpg" % currentDir))
        self.gray0_8.setObjectName(_fromUtf8("gray0_8"))
        self.gray0_8.setText("gray0.8")
        self.horizontalLayout_3.addWidget(self.gray0_8)
        self.gray0_5 = QtWidgets.QRadioButton(self.tab_5)
        self.gray0_5.setIcon(QtGui.QIcon("%s/Resources/icon/512/gray0.5.jpg" % currentDir))
        self.gray0_5.setObjectName(_fromUtf8("gray0_5"))
        self.gray0_5.setText("gray0.5")
        self.horizontalLayout_3.addWidget(self.gray0_5)
        self.gray0_2 = QtWidgets.QRadioButton(self.tab_5)
        self.gray0_2.setIcon(QtGui.QIcon("%s/Resources/icon/512/gray0.2.jpg" % currentDir))
        self.gray0_2.setObjectName(_fromUtf8("gray0_2"))
        self.gray0_2.setText("gray0.2")
        self.horizontalLayout_3.addWidget(self.gray0_2)
        self.blackImg = QtWidgets.QRadioButton(self.tab_5)
        self.blackImg.setIcon(QtGui.QIcon("%s/Resources/icon/512/black.jpg" % currentDir))
        self.blackImg.setText("black")
        self.blackImg.setObjectName(_fromUtf8("blackImg"))
        self.horizontalLayout_3.addWidget(self.blackImg)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_5, _fromUtf8(""))

        self.gridLayout_2.addWidget(self.tabWidget, 1, 2, 1, 1)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "Path", None))
        self.txMakeBtn.setText(_translate("Form", "TxMake", None))
        self.texToTiffBtn.setText(_translate("Form", "TEX -> TIF", None))
        self.texToJpgBtn.setText(_translate("Form", "TEX -> JPG", None))
        self.imageToJpgBtn.setText(_translate("Form", "IMAGE (PNG, TGA, TIFF etc..) -> JPG", None))
        self.resizeProxyBtn.setText(_translate("Form", "Proxy", None))
        self.resize512Btn.setText(_translate("Form", "512", None))
        self.resize1KBtn.setText(_translate("Form", "1K", None))
        self.resize2KBtn.setText(_translate("Form", "2K", None))
        self.resize4KBtn.setText(_translate("Form", "4K", None))
        self.resize8KBtn.setText(_translate("Form", "8K", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "Resize", None))
        self.insertBtn.setText(_translate("Form", "Insert Variant", None))
        self.changeBtn.setText(_translate("Form", "Change Variant", None))
        self.linkBtn.setText(_translate("Form", "Link Variant", None))
        self.removeBtn.setText(_translate("Form", "RemoveVariant", None))
        self.label_2.setText(_translate("Form", "Original variant", None))
        self.label_3.setText(_translate("Form", "New Variant", None))
        self.link1.setText(_translate("Form", "1", None))
        self.link2.setText(_translate("Form", "2", None))
        self.link3.setText(_translate("Form", "3", None))
        self.link4.setText(_translate("Form", "4", None))
        self.link5.setText(_translate("Form", "5", None))
        self.link6.setText(_translate("Form", "6", None))
        self.link7.setText(_translate("Form", "7", None))
        self.link8.setText(_translate("Form", "8", None))
        self.link9.setText(_translate("Form", "9", None))
        self.pushButton.setText(_translate("Form", "Execute", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Variant", None))
        self.label_4.setText(_translate("Form", "originalName", None))
        self.label_5.setText(_translate("Form", "replace Name", None))
        self.renameExecBtn.setText(_translate("Form", "rename", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Rename", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("Form", "Metamong", None))
        self.specG.setText(_translate("Form", "specG", None))
        self.specR.setText(_translate("Form", "specR", None))
        self.norm.setText(_translate("Form", "norm", None))
        self.bump.setText(_translate("Form", "bump", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("Form", "Sample", None))
