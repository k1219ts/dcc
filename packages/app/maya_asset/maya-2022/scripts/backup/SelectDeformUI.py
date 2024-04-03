# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SelectDeformUI.ui'
#
# Created: Wed Dec 27 15:15:33 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets as QtGui


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

class Ui_Dialog(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)

        self.setObjectName(_fromUtf8("Dialog"))
        self.resize(400, 300)
        self.gridLayout = QtGui.QGridLayout(self)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.okBtn = QtGui.QPushButton(self)
        self.okBtn.setObjectName(_fromUtf8("okBtn"))
        self.gridLayout.addWidget(self.okBtn, 2, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 2, 0, 1, 1)
        self.deformList = QtGui.QListWidget(self)
        self.deformList.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.deformList.setObjectName(_fromUtf8("deformList"))
        self.gridLayout.addWidget(self.deformList, 1, 0, 1, 2)
        self.label = QtGui.QLabel(self)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.okBtn.setText(_translate("Dialog", "OK", None))
        self.label.setText(_translate("Dialog", "Select Deform for Bake Cache", None))

