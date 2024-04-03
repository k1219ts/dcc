# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spanner2_ui_savePub.ui'
#
# Created: Fri Oct 13 18:26:22 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# import Qt
# from Qt import QtGui
# from Qt import QtWidgets
# import Qt.QtWidgets as QtGui
# from Qt import QtCore

from PySide2 import QtWidgets, QtCore, QtGui

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

class savePub_Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(400, 530)
        Form.setMinimumSize(QtCore.QSize(400, 513))
        Form.setMaximumSize(QtCore.QSize(400, 530))
        self.gridLayout_5 = QtGui.QGridLayout(Form)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.groupBox = QtGui.QGroupBox(Form)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.fileName_lineEdit = QtGui.QLineEdit(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fileName_lineEdit.sizePolicy().hasHeightForWidth())
        self.fileName_lineEdit.setSizePolicy(sizePolicy)
        self.fileName_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.fileName_lineEdit.setFont(font)
        self.fileName_lineEdit.setObjectName(_fromUtf8("fileName_lineEdit"))
        self.gridLayout_2.addWidget(self.fileName_lineEdit, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox, 0, 0, 1, 2)
        self.model_groupBox = QtGui.QGroupBox(Form)
        self.model_groupBox.setObjectName(_fromUtf8("model_groupBox"))
        self.gridLayout_4 = QtGui.QGridLayout(self.model_groupBox)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.abc_checkBox = QtGui.QCheckBox(self.model_groupBox)
        self.abc_checkBox.setChecked(True)
        self.abc_checkBox.setObjectName(_fromUtf8("abc_checkBox"))
        self.gridLayout_4.addWidget(self.abc_checkBox, 0, 1, 1, 1)
        self.tex_checkBox = QtGui.QCheckBox(self.model_groupBox)
        self.tex_checkBox.setChecked(True)
        self.tex_checkBox.setObjectName(_fromUtf8("tex_checkBox"))
        self.gridLayout_4.addWidget(self.tex_checkBox, 0, 2, 1, 1)
        self.mb_checkBox = QtGui.QCheckBox(self.model_groupBox)
        self.mb_checkBox.setMinimumSize(QtCore.QSize(130, 0))
        self.mb_checkBox.setObjectName(_fromUtf8("mb_checkBox"))
        self.gridLayout_4.addWidget(self.mb_checkBox, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.model_groupBox, 2, 0, 1, 2)
        spacerItem = QtGui.QSpacerItem(125, 20, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem, 5, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(Form)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout_5.addWidget(self.buttonBox, 5, 1, 1, 1)
        self.saveDev_groupBox = QtGui.QGroupBox(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveDev_groupBox.sizePolicy().hasHeightForWidth())
        self.saveDev_groupBox.setSizePolicy(sizePolicy)
        self.saveDev_groupBox.setCheckable(True)
        self.saveDev_groupBox.setObjectName(_fromUtf8("saveDev_groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.saveDev_groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.dsc_lineEdit = QtGui.QLineEdit(self.saveDev_groupBox)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.dsc_lineEdit.setFont(font)
        self.dsc_lineEdit.setObjectName(_fromUtf8("dsc_lineEdit"))
        self.gridLayout.addWidget(self.dsc_lineEdit, 5, 0, 1, 1)
        self.label_3 = QtGui.QLabel(self.saveDev_groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.nextDev_lineEdit = QtGui.QLineEdit(self.saveDev_groupBox)
        self.nextDev_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.nextDev_lineEdit.setFont(font)
        self.nextDev_lineEdit.setObjectName(_fromUtf8("nextDev_lineEdit"))
        self.gridLayout.addWidget(self.nextDev_lineEdit, 3, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.saveDev_groupBox)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1)
        self.gridLayout_5.addWidget(self.saveDev_groupBox, 1, 0, 1, 2)
        self.frame = QtGui.QFrame(Form)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.gridLayout_3 = QtGui.QGridLayout(self.frame)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.savePubComment_textEdit = QtGui.QTextEdit(self.frame)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.savePubComment_textEdit.setFont(font)
        self.savePubComment_textEdit.setObjectName(_fromUtf8("savePubComment_textEdit"))
        self.gridLayout_3.addWidget(self.savePubComment_textEdit, 2, 0, 1, 1)
        self.label = QtGui.QLabel(self.frame)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.frame, 3, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.groupBox.setTitle(_translate("Form", "Publish As", None))
        self.label_2.setText(_translate("Form", "File Name:", None))
        self.model_groupBox.setTitle(_translate("Form", "Publish Model", None))
        self.abc_checkBox.setText(_translate("Form", "Alembic", None))
        self.tex_checkBox.setText(_translate("Form", "Texture", None))
        self.mb_checkBox.setText(_translate("Form", "Maya Binary", None))
        self.saveDev_groupBox.setTitle(_translate("Form", "Save Devel", None))
        self.label_3.setText(_translate("Form", "File Name:", None))
        self.label_4.setText(_translate("Form", "Description:", None))
        self.label.setText(_translate("Form", "memo", None))
