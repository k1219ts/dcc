# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'textureExport2.ui'
#
# Created: Wed Jan  2 10:26:58 2019
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtWidgets, QtCore, QtGui

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
        Form.resize(872, 500)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.showLabel = QtWidgets.QLabel(Form)
        self.showLabel.setObjectName(_fromUtf8("showLabel"))
        self.horizontalLayout.addWidget(self.showLabel)
        self.showEdit = QtWidgets.QLineEdit(Form)
        self.showEdit.setMinimumSize(QtCore.QSize(120, 32))
        self.showEdit.setMaximumSize(QtCore.QSize(120, 32))
        self.showEdit.setObjectName(_fromUtf8("showEdit"))
        self.horizontalLayout.addWidget(self.showEdit)
        self.shotLabel = QtWidgets.QLabel(Form)
        self.shotLabel.setObjectName(_fromUtf8("shotLabel"))
        self.horizontalLayout.addWidget(self.shotLabel)
        self.shotEdit = QtWidgets.QLineEdit(Form)
        self.shotEdit.setMinimumSize(QtCore.QSize(160, 0))
        self.shotEdit.setMaximumSize(QtCore.QSize(160, 16777215))
        self.shotEdit.setObjectName(_fromUtf8("shotEdit"))
        self.horizontalLayout.addWidget(self.shotEdit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.shotCheckBox = QtWidgets.QCheckBox(Form)
        self.shotCheckBox.setObjectName(_fromUtf8("shotCheckBox"))
        self.horizontalLayout.addWidget(self.shotCheckBox)
        self.elementCheckBox = QtWidgets.QCheckBox(Form)
        self.elementCheckBox.setObjectName(_fromUtf8("elementCheckBox"))
        self.horizontalLayout.addWidget(self.elementCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.assetNameLabel = QtWidgets.QLabel(self.groupBox)
        self.assetNameLabel.setObjectName(_fromUtf8("assetNameLabel"))
        self.horizontalLayout_2.addWidget(self.assetNameLabel)
        self.assetNameEdit = QtWidgets.QLineEdit(self.groupBox)
        self.assetNameEdit.setMinimumSize(QtCore.QSize(160, 0))
        self.assetNameEdit.setMaximumSize(QtCore.QSize(160, 16777215))
        self.assetNameEdit.setObjectName(_fromUtf8("assetNameEdit"))
        self.horizontalLayout_2.addWidget(self.assetNameEdit)
        self.elementLabel = QtWidgets.QLabel(self.groupBox)
        self.elementLabel.setObjectName(_fromUtf8("elementLabel"))
        self.horizontalLayout_2.addWidget(self.elementLabel)
        self.elementEdit = QtWidgets.QLineEdit(self.groupBox)
        self.elementEdit.setMinimumSize(QtCore.QSize(160, 0))
        self.elementEdit.setMaximumSize(QtCore.QSize(160, 16777215))
        self.elementEdit.setObjectName(_fromUtf8("elementEdit"))
        self.horizontalLayout_2.addWidget(self.elementEdit)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.dataVersionEdit = QtWidgets.QLineEdit(self.groupBox)
        self.dataVersionEdit.setMinimumSize(QtCore.QSize(160, 0))
        self.dataVersionEdit.setMaximumSize(QtCore.QSize(160, 16777215))
        self.dataVersionEdit.setObjectName(_fromUtf8("dataVersionEdit"))
        self.horizontalLayout_2.addWidget(self.dataVersionEdit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.groupBox)
        self.line_2 = QtWidgets.QFrame(Form)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout.addWidget(self.line_2)
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(110, 0))
        self.label.setMaximumSize(QtCore.QSize(110, 16777215))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.textureVersionEdit = QtWidgets.QLineEdit(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textureVersionEdit.sizePolicy().hasHeightForWidth())
        self.textureVersionEdit.setSizePolicy(sizePolicy)
        self.textureVersionEdit.setMinimumSize(QtCore.QSize(120, 0))
        self.textureVersionEdit.setMaximumSize(QtCore.QSize(120, 16777215))
        self.textureVersionEdit.setObjectName(_fromUtf8("textureVersionEdit"))
        self.horizontalLayout_4.addWidget(self.textureVersionEdit)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_5.addWidget(self.label_4)
        self.outDirEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.outDirEdit.setObjectName(_fromUtf8("outDirEdit"))
        self.outDirEdit.setReadOnly(True)
        self.horizontalLayout_5.addWidget(self.outDirEdit)
        self.gridLayout.addLayout(self.horizontalLayout_5, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.channelLayout = QtWidgets.QGridLayout()
        self.channelLayout.setObjectName(_fromUtf8("channelLayout"))
        self.verticalLayout.addLayout(self.channelLayout)
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 460, 620))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.uvpatchLayout = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.uvpatchLayout.setObjectName(_fromUtf8("uvpatchLayout"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.line = QtWidgets.QFrame(Form)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.versionCopyCheckBox = QtWidgets.QCheckBox(Form)
        self.versionCopyCheckBox.setObjectName(_fromUtf8("versionCopyCheckBox"))
        self.horizontalLayout_3.addWidget(self.versionCopyCheckBox)
        spacerItem3 = QtWidgets.QSpacerItem(448, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.exportBtn = QtWidgets.QPushButton(Form)
        self.exportBtn.setMinimumSize(QtCore.QSize(300, 40))
        self.exportBtn.setObjectName(_fromUtf8("exportBtn"))
        self.horizontalLayout_3.addWidget(self.exportBtn)
        self.saverezBtn = QtWidgets.QPushButton(Form)
        self.saverezBtn.setMinimumSize(QtCore.QSize(100, 40))
        self.saverezBtn.setObjectName(_fromUtf8("saverezBtn"))
        self.horizontalLayout_3.addWidget(self.saverezBtn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_5 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout.addWidget(self.label_5)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "TextureName Window", None))
        self.showLabel.setText(_translate("Form", "Show Dir", None))
        self.shotCheckBox.setText(_translate("Form", "Shot", None))
        self.elementCheckBox.setText(_translate("Form", "Element", None))
        self.groupBox.setTitle(_translate("Form", "model Infomation", None))
        self.assetNameLabel.setText(_translate("Form", "asset Name", None))
        self.elementLabel.setText(_translate("Form", "element Name", None))
        self.shotLabel.setText(_translate("Form", "shot", None))
        self.label_3.setText(_translate("Form", "data Version", None))
        self.groupBox_2.setTitle(_translate("Form", "texture Information", None))
        self.label.setText(_translate("Form", "Texture Version", None))
        self.label_4.setText(_translate("Form", "Out Dir", None))
        self.versionCopyCheckBox.setText(_translate("Form", "versionCopy", None))
        self.exportBtn.setText(_translate("Form", "Export Texture", None))
        self.saverezBtn.setText(_translate("Form", "Save Rez", None))
        self.label_5.setText(_translate("Form", "@DexterDigital Texture", None))
