# -*- coding: utf-8 -*-

from PySide2 import QtWidgets, QtCore

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
        Form.resize(550, 400)


        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.prjLabel = QtWidgets.QLabel(Form)
        self.prjLabel.setObjectName(_fromUtf8("Project"))
        self.gridLayout.addWidget(self.prjLabel, 0, 0, 1, 1)
        self.prjComboBox = QtWidgets.QComboBox(Form)
        self.prjComboBox.setObjectName(_fromUtf8("prjComboBox"))
        self.gridLayout.addWidget(self.prjComboBox, 0, 1, 1, 1)
        self.seqLabel = QtWidgets.QLabel(Form)
        self.seqLabel.setObjectName(_fromUtf8("sequence"))
        self.gridLayout.addWidget(self.seqLabel, 0, 2, 1, 1)
        self.seqComboBox = QtWidgets.QComboBox(Form)
        self.seqComboBox.setObjectName(_fromUtf8("seqComboBox"))
        self.gridLayout.addWidget(self.seqComboBox, 0, 3, 1, 1)
        self.browserGroup = QtWidgets.QGroupBox(Form)
        self.browserGroup.setObjectName("browserGroup")
        self.gridLayout_1 = QtWidgets.QGridLayout(self.browserGroup)
        self.gridLayout_1.setObjectName("gridLayout_1")
        self.browserTree = QtWidgets.QTreeWidget(self.browserGroup)
        self.browserTree.setObjectName("browserTree")
        self.gridLayout_1.addWidget(self.browserTree, 0, 0, 1, 4)
        self.closeButton = QtWidgets.QPushButton(Form)
        self.closeButton.setObjectName("close")
        self.gridLayout_1.addWidget(self.closeButton, 1, 2, 1, 1)
        self.pasteButton = QtWidgets.QPushButton(Form)
        self.pasteButton.setObjectName("PasteNode")
        self.gridLayout_1.addWidget(self.pasteButton, 1, 3, 1, 1)
        self.gridLayout.addWidget(self.browserGroup, 1, 0, 1, 4)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(QtWidgets.QApplication.translate("Form", "Comp Asset Tool by Giuk Kim, Dexter Digital", None))

        self.pasteButton.setText(
            QtWidgets.QApplication.translate("Form", "Paste Node", None))
        self.closeButton.setText(
            QtWidgets.QApplication.translate("Form", "close", None))
        self.browserGroup.setTitle(QtWidgets.QApplication.translate("Form", "Comp Asset Node List", None))
