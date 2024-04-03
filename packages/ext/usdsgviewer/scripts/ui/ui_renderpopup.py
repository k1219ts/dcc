# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_renderpopup.ui',
# licensing of 'ui_renderpopup.ui' applies.
#
# Created: Thu Apr 22 14:10:23 2021
#      by: pyside2-uic  running on PySide2 5.12.6
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_FormRender(object):
    def setupUi(self, FormRender):
        FormRender.setObjectName("FormRender")
        FormRender.resize(501, 429)
        FormRender.setWindowTitle("")
        self.frame = QtWidgets.QFrame(FormRender)
        self.frame.setGeometry(QtCore.QRect(10, 10, 481, 411))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 261, 34))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.frameIn_lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.frameIn_lineEdit.setObjectName("frameIn_lineEdit")
        self.horizontalLayout.addWidget(self.frameIn_lineEdit)
        self.frameOut_lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.frameOut_lineEdit.setObjectName("frameOut_lineEdit")
        self.horizontalLayout.addWidget(self.frameOut_lineEdit)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 50, 171, 34))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.maxSam_lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.maxSam_lineEdit.setObjectName("maxSam_lineEdit")
        self.horizontalLayout_2.addWidget(self.maxSam_lineEdit)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(10, 90, 171, 34))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.pixelVal_lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_3)
        self.pixelVal_lineEdit.setObjectName("pixelVal_lineEdit")
        self.horizontalLayout_3.addWidget(self.pixelVal_lineEdit)
        self.render_pushButton = QtWidgets.QPushButton(self.frame)
        self.render_pushButton.setGeometry(QtCore.QRect(380, 10, 90, 28))
        self.render_pushButton.setObjectName("render_pushButton")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 130, 461, 271))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.selPrims_listWidget = QtWidgets.QListWidget(self.verticalLayoutWidget)
        self.selPrims_listWidget.setObjectName("selPrims_listWidget")
        self.verticalLayout.addWidget(self.selPrims_listWidget)

        self.retranslateUi(FormRender)
        QtCore.QMetaObject.connectSlotsByName(FormRender)

    def retranslateUi(self, FormRender):
        self.label.setText(QtWidgets.QApplication.translate("FormRender", "FrameRange", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("FormRender", "maxSamples", None, -1))
        self.maxSam_lineEdit.setText(QtWidgets.QApplication.translate("FormRender", "64", None, -1))
        self.label_3.setText(QtWidgets.QApplication.translate("FormRender", "pixelVariance", None, -1))
        self.pixelVal_lineEdit.setText(QtWidgets.QApplication.translate("FormRender", "0.1", None, -1))
        self.render_pushButton.setText(QtWidgets.QApplication.translate("FormRender", "RENDER", None, -1))
        self.label_4.setText(QtWidgets.QApplication.translate("FormRender", "selected Prims", None, -1))

