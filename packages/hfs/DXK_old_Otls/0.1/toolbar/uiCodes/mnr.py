# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mnr.ui'
#
# Created: Tue Jul 12 11:17:57 2016
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

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

class Ui_NetworkRender(object):
    def setupUi(self, NetworkRender):
        NetworkRender.setObjectName(_fromUtf8("NetworkRender"))
        NetworkRender.resize(418, 332)
        self.centralwidget = QtGui.QWidget(NetworkRender)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.labelHost1 = QtGui.QLabel(self.centralwidget)
        self.labelHost1.setObjectName(_fromUtf8("labelHost1"))
        self.horizontalLayout.addWidget(self.labelHost1)
        self.le_Host1 = QtGui.QLineEdit(self.centralwidget)
        self.le_Host1.setObjectName(_fromUtf8("le_Host1"))
        self.horizontalLayout.addWidget(self.le_Host1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.labelHost2 = QtGui.QLabel(self.centralwidget)
        self.labelHost2.setObjectName(_fromUtf8("labelHost2"))
        self.horizontalLayout_2.addWidget(self.labelHost2)
        self.le_Host2 = QtGui.QLineEdit(self.centralwidget)
        self.le_Host2.setObjectName(_fromUtf8("le_Host2"))
        self.horizontalLayout_2.addWidget(self.le_Host2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout_2.addWidget(self.line)
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_2.addWidget(self.label_3)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.tb_Enable = QtGui.QToolButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_Enable.sizePolicy().hasHeightForWidth())
        self.tb_Enable.setSizePolicy(sizePolicy)
        self.tb_Enable.setMinimumSize(QtCore.QSize(0, 36))
        self.tb_Enable.setObjectName(_fromUtf8("tb_Enable"))
        self.horizontalLayout_3.addWidget(self.tb_Enable)
        spacerItem1 = QtGui.QSpacerItem(10, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.tb_Disable = QtGui.QToolButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_Disable.sizePolicy().hasHeightForWidth())
        self.tb_Disable.setSizePolicy(sizePolicy)
        self.tb_Disable.setMinimumSize(QtCore.QSize(0, 36))
        self.tb_Disable.setObjectName(_fromUtf8("tb_Disable"))
        self.horizontalLayout_3.addWidget(self.tb_Disable)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        NetworkRender.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(NetworkRender)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 418, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        NetworkRender.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(NetworkRender)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        NetworkRender.setStatusBar(self.statusbar)

        self.retranslateUi(NetworkRender)
        QtCore.QMetaObject.connectSlotsByName(NetworkRender)

    def retranslateUi(self, NetworkRender):
        NetworkRender.setWindowTitle(_translate("NetworkRender", "Mantra Network Render", None))
        self.labelHost1.setText(_translate("NetworkRender", "Host1", None))
        self.labelHost2.setText(_translate("NetworkRender", "Host2", None))
        self.label_3.setText(_translate("NetworkRender", " Note : Host1 should be this machine", None))
        self.tb_Enable.setText(_translate("NetworkRender", "Enable", None))
        self.tb_Disable.setText(_translate("NetworkRender", "Disable", None))

