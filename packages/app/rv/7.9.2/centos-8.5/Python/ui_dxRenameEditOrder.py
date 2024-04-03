# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'aarYuaLE.ui'
##
## Created by: Qt User Interface Compiler version 5.15.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.setWindowModality(Qt.ApplicationModal)
        Dialog.resize(687, 328)
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(20, 100, 63, 20))
        self.directory_plainTextEdit = QPlainTextEdit(Dialog)
        self.directory_plainTextEdit.setObjectName(u"directory_plainTextEdit")
        self.directory_plainTextEdit.setGeometry(QRect(20, 170, 651, 61))
        self.directory_plainTextEdit.setStyleSheet(u"color: white")
        self.selectDirectory_pushButton = QPushButton(Dialog)
        self.selectDirectory_pushButton.setObjectName(u"selectDirectory_pushButton")
        self.selectDirectory_pushButton.setGeometry(QRect(20, 130, 101, 31))
        self.label_2 = QLabel(Dialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 20, 63, 20))
        self.show_comboBox = QComboBox(Dialog)
        self.show_comboBox.setObjectName(u"show_comboBox")
        self.show_comboBox.setGeometry(QRect(20, 50, 391, 31))
        self.apply_pushButton = QPushButton(Dialog)
        self.apply_pushButton.setObjectName(u"apply_pushButton")
        self.apply_pushButton.setGeometry(QRect(570, 270, 100, 36))
        self.label_3 = QLabel(Dialog)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 240, 301, 20))
        font = QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)

        self.retranslateUi(Dialog)

        QMetaObject.connectSlotsByName(Dialog)
    # setupUi

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"\uacbd\ub85c", None))
        self.selectDirectory_pushButton.setText(QCoreApplication.translate("Dialog", u"\uacbd\ub85c \ucc3e\uae30", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"\ud504\ub85c\uc81d\ud2b8", None))
        self.apply_pushButton.setText(QCoreApplication.translate("Dialog", u"\uc801\uc6a9", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"\ud3f4\ub354 \ub9e4\uce6d: (\ud504\ub85c\uc81d\ud2b8\uba85)_(\uc0f7)_(comp)_v(\ubc84\uc804)", None))
    # retranslateUi

