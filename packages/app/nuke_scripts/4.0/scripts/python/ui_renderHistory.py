# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'renderHistory.ui'
#
# Created: Thu Aug 14 17:00:14 2014
#      by: PyQt4 UI code generator 4.7.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtWidgets, QtCore, QtGui

class NkTree(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        super(NkTree, self).__init__(parent)
        self.setDragEnabled(True)

    def startDrag(self, action):
        items = self.selectedItems()
        mimedata = QtCore.QMimeData()

        mimeText = ''
        for item in self.selectedItems():
            mimeText += 'file://' + item.data(0, QtCore.Qt.UserRole) + '\r\n'
        mimedata.setText(mimeText)
        drag = QtGui.QDrag(self)
        drag.setMimeData(mimedata)
        drag.exec_()

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(795, 717)
        self.gridLayout_3 = QtWidgets.QGridLayout(Form)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox.setMinimumSize(QtCore.QSize(150, 0))
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_2.addWidget(self.comboBox, 0, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_2.setMinimumSize(QtCore.QSize(150, 0))
        self.comboBox_2.setObjectName("comboBox_2")
        self.gridLayout_2.addWidget(self.comboBox_2, 0, 1, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 0, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_2.addWidget(self.pushButton, 0, 3, 1, 1)
        self.treeWidget = NkTree(self.groupBox_2)
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "1")
        self.gridLayout_2.addWidget(self.treeWidget, 1, 0, 1, 4)
        self.gridLayout_3.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_3.setMaximumSize(QtCore.QSize(100, 16777215))
        self.comboBox_3.setObjectName("comboBox_3")
        self.gridLayout.addWidget(self.comboBox_3, 0, 1, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 2, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(168, 20,
                                           QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 1, 2, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setMaximumSize(QtCore.QSize(150, 16777215))
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 3, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        self.groupBox_2.setTitle(QtWidgets.QApplication.translate("Form", "Search", None, -1))
        self.pushButton.setText(QtWidgets.QApplication.translate("Form", "Search", None, -1))
        self.groupBox.setTitle(QtWidgets.QApplication.translate("Form", "Import", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("Form", "Process", None, -1))
        self.pushButton_2.setText(QtWidgets.QApplication.translate("Form", "Import&&Open", None, -1))
