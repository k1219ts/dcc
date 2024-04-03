# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'feedTopic_Nuke.ui'
#
# Created: Tue Jul 21 12:39:09 2015
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(923, 657)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        #self.gridLayout.setMargin(0)
        #self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.prjLabel = QtWidgets.QLabel(Dialog)
        self.prjLabel.setObjectName(_fromUtf8("prjLabel"))
        self.horizontalLayout.addWidget(self.prjLabel)
        self.prjCombo = QtWidgets.QComboBox(Dialog)
        self.prjCombo.setObjectName(_fromUtf8("prjCombo"))
        self.horizontalLayout.addWidget(self.prjCombo)
        self.teamLabel = QtWidgets.QLabel(Dialog)
        self.teamLabel.setObjectName(_fromUtf8("teamLabel"))
        self.horizontalLayout.addWidget(self.teamLabel)
        self.teamCombo = QtWidgets.QComboBox(Dialog)
        self.teamCombo.setObjectName(_fromUtf8("teamCombo"))
        self.horizontalLayout.addWidget(self.teamCombo)
        spacerItem = QtWidgets.QSpacerItem(268, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.snapshotTree = QtWidgets.QTreeWidget(Dialog)
        self.snapshotTree.setObjectName(_fromUtf8("snapshotTree"))
        self.snapshotTree.headerItem().setText(0, _fromUtf8("1"))
        self.gridLayout.addWidget(self.snapshotTree, 1, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.loopRadio = QtWidgets.QRadioButton(Dialog)
        self.loopRadio.setMinimumSize(QtCore.QSize(80, 0))
        self.loopRadio.setObjectName(_fromUtf8("loopRadio"))
        self.horizontalLayout_2.addWidget(self.loopRadio)
        self.holdRadio = QtWidgets.QRadioButton(Dialog)
        self.holdRadio.setMinimumSize(QtCore.QSize(80, 0))
        self.holdRadio.setObjectName(_fromUtf8("holdRadio"))
        self.horizontalLayout_2.addWidget(self.holdRadio)
        self.noneRadio = QtWidgets.QRadioButton(Dialog)
        self.noneRadio.setMinimumSize(QtCore.QSize(80, 0))
        self.noneRadio.setObjectName(_fromUtf8("noneRadio"))
        self.horizontalLayout_2.addWidget(self.noneRadio)
        self.importButton = QtWidgets.QPushButton(Dialog)
        self.importButton.setObjectName(_fromUtf8("importButton"))
        self.horizontalLayout_2.addWidget(self.importButton)
        self.pdplayerButton = QtWidgets.QPushButton(Dialog)
        self.pdplayerButton.setObjectName(_fromUtf8("pdplayerButton"))
        self.horizontalLayout_2.addWidget(self.pdplayerButton)
        self.closeButton = QtWidgets.QPushButton(Dialog)
        self.closeButton.setObjectName(_fromUtf8("closeButton"))
        self.horizontalLayout_2.addWidget(self.closeButton)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 1, 1, 1)
        self.topicList = QtWidgets.QListWidget(Dialog)
        self.topicList.setMinimumSize(QtCore.QSize(300, 0))
        self.topicList.setMaximumSize(QtCore.QSize(350, 16777215))
        self.topicList.setObjectName(_fromUtf8("topicList"))
        self.gridLayout.addWidget(self.topicList, 1, 0, 2, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Feedback Topic for Nuke By Tae Hyung Lee, Dexter Digital", None))
        self.prjLabel.setText(_translate("Dialog", "Project", None))
        self.teamLabel.setText(_translate("Dialog", "Team", None))
        self.loopRadio.setText(_translate("Dialog", "Loop", None))
        self.holdRadio.setText(_translate("Dialog", "Hold", None))
        self.noneRadio.setText(_translate("Dialog", "None", None))
        self.importButton.setText(_translate("Dialog", "Import to Nuke", None))
        self.pdplayerButton.setText(_translate("Dialog", "Open Pdplayer", None))
        self.closeButton.setText(_translate("Dialog", "Close", None))
