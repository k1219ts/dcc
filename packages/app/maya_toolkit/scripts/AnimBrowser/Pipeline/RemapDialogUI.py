# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RemapDialogUI.ui'
#
# Created: Wed May  2 12:23:13 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

# import Qt.QtWidgets as QtGui
# from Qt import QtCore
from PySide2 import QtGui, QtCore

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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(733, 878)
        Form.setMinimumSize(QtCore.QSize(547, 678))
        self.gridLayout_4 = QtGui.QGridLayout(Form)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.preset_label = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preset_label.sizePolicy().hasHeightForWidth())
        self.preset_label.setSizePolicy(sizePolicy)
        self.preset_label.setMinimumSize(QtCore.QSize(0, 30))
        self.preset_label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.preset_label.setObjectName(_fromUtf8("preset_label"))
        self.gridLayout_3.addWidget(self.preset_label, 0, 0, 1, 1)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.biped_radioButton = QtGui.QRadioButton(Form)
        self.biped_radioButton = QtGui.QRadioButton(Form)
        self.biped_radioButton.setObjectName(_fromUtf8("biped_radioButton"))
        self.verticalLayout.addWidget(self.biped_radioButton)
        self.quad_radioButton = QtGui.QRadioButton(Form)
        self.quad_radioButton.setObjectName(_fromUtf8("quad_radioButton"))
        self.verticalLayout.addWidget(self.quad_radioButton)
        self.gridLayout_3.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.preset_listWidget = QtGui.QListWidget(Form)
        self.preset_listWidget.setMaximumSize(QtCore.QSize(120, 16777215))
        self.preset_listWidget.setObjectName(_fromUtf8("preset_listWidget"))
        self.gridLayout_3.addWidget(self.preset_listWidget, 2, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 2, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label = QtGui.QLabel(Form)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.findSource_lineEdit = QtGui.QLineEdit(Form)
        self.findSource_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.findSource_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.findSource_lineEdit.setObjectName(_fromUtf8("findSource_lineEdit"))
        self.gridLayout_2.addWidget(self.findSource_lineEdit, 0, 1, 1, 1)
        self.sourceNs_label = QtGui.QLabel(Form)
        self.sourceNs_label.setMinimumSize(QtCore.QSize(0, 30))
        self.sourceNs_label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.sourceNs_label.setObjectName(_fromUtf8("sourceNs_label"))
        self.gridLayout_2.addWidget(self.sourceNs_label, 1, 0, 1, 1)
        self.sourceNs_lineEdit = QtGui.QLineEdit(Form)
        self.sourceNs_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.sourceNs_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.sourceNs_lineEdit.setObjectName(_fromUtf8("sourceNs_lineEdit"))
        self.gridLayout_2.addWidget(self.sourceNs_lineEdit, 1, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setMinimumSize(QtCore.QSize(0, 30))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 0, 1, 1)
        self.findTarget_lineEdit = QtGui.QLineEdit(Form)
        self.findTarget_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.findTarget_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.findTarget_lineEdit.setObjectName(_fromUtf8("findTarget_lineEdit"))
        self.gridLayout.addWidget(self.findTarget_lineEdit, 0, 1, 1, 1)
        self.targetNs_label = QtGui.QLabel(Form)
        self.targetNs_label.setMinimumSize(QtCore.QSize(0, 30))
        self.targetNs_label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.targetNs_label.setObjectName(_fromUtf8("targetNs_label"))
        self.gridLayout.addWidget(self.targetNs_label, 1, 0, 1, 1)
        self.targetNs_lineEdit = QtGui.QLineEdit(Form)
        self.targetNs_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.targetNs_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.targetNs_lineEdit.setObjectName(_fromUtf8("targetNs_lineEdit"))
        self.gridLayout.addWidget(self.targetNs_lineEdit, 1, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 2, 1, 1)
        self.matching_treeWidget = QtGui.QTreeWidget(Form)
        self.matching_treeWidget.setProperty("showDropIndicator", False)
        self.matching_treeWidget.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.matching_treeWidget.setSelectionBehavior(QtGui.QAbstractItemView.SelectItems)
        self.matching_treeWidget.setExpandsOnDoubleClick(False)
        self.matching_treeWidget.setColumnCount(5)
        self.matching_treeWidget.setObjectName(_fromUtf8("matching_treeWidget"))
        self.matching_treeWidget.headerItem().setText(0, _fromUtf8("Source"))
        self.matching_treeWidget.headerItem().setText(1, _fromUtf8(">"))
        self.matching_treeWidget.headerItem().setText(2, _fromUtf8("Target"))
        self.matching_treeWidget.header().setCascadingSectionResizes(False)
        self.matching_treeWidget.header().setDefaultSectionSize(200)
        self.matching_treeWidget.header().setHighlightSections(False)
        self.matching_treeWidget.header().setMinimumSectionSize(200)
        self.matching_treeWidget.header().setSortIndicatorShown(False)
        self.matching_treeWidget.header().setStretchLastSection(True)
        self.gridLayout_4.addWidget(self.matching_treeWidget, 1, 1, 1, 2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.ok_pushButton = QtGui.QPushButton(Form)
        self.ok_pushButton.setObjectName(_fromUtf8("ok_pushButton"))
        self.horizontalLayout.addWidget(self.ok_pushButton)
        self.cancel_pushButton = QtGui.QPushButton(Form)
        self.cancel_pushButton.setObjectName(_fromUtf8("cancel_pushButton"))
        self.horizontalLayout.addWidget(self.cancel_pushButton)
        self.gridLayout_4.addLayout(self.horizontalLayout, 2, 0, 1, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "HIK ImportDialog", None))
        self.preset_label.setText(_translate("Form", "PRESET", None))
        self.biped_radioButton.setText(_translate("Form", "biped", None))
        self.quad_radioButton.setText(_translate("Form", "quadruped", None))
        self.label.setText(_translate("Form", "Source", None))
        self.sourceNs_label.setText(_translate("Form", "namespace", None))
        self.label_2.setText(_translate("Form", "Target", None))
        self.targetNs_label.setText(_translate("Form", "namespace", None))
        self.ok_pushButton.setText(_translate("Form", "apply", None))
        self.cancel_pushButton.setText(_translate("Form", "cancel", None))
