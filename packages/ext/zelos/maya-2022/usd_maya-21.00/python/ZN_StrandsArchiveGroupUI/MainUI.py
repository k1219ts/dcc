# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainUI.ui'
#
# Created: Thu Jul 20 11:08:39 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

MY_PYPATH_MODUEL = "/netapp/backstage/pub/apps/maya2/versions/2017/global/linux/lib/site-packages"
import site
site.addsitedir(MY_PYPATH_MODUEL)

import Qt.QtWidgets as QtGui
from Qt import QtCore

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
        Form.setObjectName(_fromUtf8("Import Zenn Strands Cache for Rigid Binding Transfer"))
        Form.resize(537, 120)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.abcPathLineEdit = QtGui.QLineEdit(Form)
        self.abcPathLineEdit.setObjectName(_fromUtf8("abcPathLineEdit"))
        self.gridLayout.addWidget(self.abcPathLineEdit, 1, 0, 1, 1)
        self.abcPushBtn = QtGui.QPushButton(Form)
        self.abcPushBtn.setMinimumSize(QtCore.QSize(36, 36))
        self.abcPushBtn.setMaximumSize(QtCore.QSize(36, 36))
        self.abcPushBtn.setObjectName(_fromUtf8("abcPushBtn"))
        self.gridLayout.addWidget(self.abcPushBtn, 1, 1, 1, 1)
        self.zennPathLineEdit = QtGui.QLineEdit(Form)
        self.zennPathLineEdit.setObjectName(_fromUtf8("zennPathLineEdit"))
        self.gridLayout.addWidget(self.zennPathLineEdit, 1, 2, 1, 1)
        self.zennPushBtn = QtGui.QPushButton(Form)
        self.zennPushBtn.setMinimumSize(QtCore.QSize(36, 36))
        self.zennPushBtn.setMaximumSize(QtCore.QSize(36, 36))
        self.zennPushBtn.setObjectName(_fromUtf8("zennPushBtn"))
        self.gridLayout.addWidget(self.zennPushBtn, 1, 3, 1, 1)
        self.loadPushBtn = QtGui.QPushButton(Form)
        self.loadPushBtn.setObjectName(_fromUtf8("loadPushBtn"))
        self.gridLayout.addWidget(self.loadPushBtn, 2, 2, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "Import Zenn Strands Cache for Rigid Binding Transfer", None))
        self.label.setText(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "ABC File", None))
        self.label_2.setText(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "Zenn Strand Cache Directory", None))
        self.abcPushBtn.setText(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "B", None))
        self.zennPushBtn.setText(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "B", None))
        self.loadPushBtn.setText(_translate("Import Zenn Strands Cache for Rigid Binding Transfer", "Load", None))

