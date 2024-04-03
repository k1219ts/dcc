# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_tacticSubmit_3.ui'
#
# Created: Fri Feb 21 15:01:18 2020
#      by: PyQt4 UI code generator 4.10.1
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtWidgets, QtCore, QtGui

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

class Ui_Tacticsubmit(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.setWindowModality(QtCore.Qt.ApplicationModal)
        Form.resize(1171, 127)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.gridLayout_11 = QtWidgets.QGridLayout(Form)
        self.gridLayout_11.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        # self.gridLayout_11.setMargin(0)
        self.gridLayout_11.setSpacing(0)
        self.gridLayout_11.setObjectName(_fromUtf8("gridLayout_11"))
        self.main_frame = QtWidgets.QFrame(Form)
        self.main_frame.setMinimumSize(QtCore.QSize(900, 0))
        self.main_frame.setAutoFillBackground(False)
        self.main_frame.setStyleSheet(_fromUtf8("background-color: rgb(80, 80, 80);"))
        self.main_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.main_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.main_frame.setLineWidth(2)
        self.main_frame.setObjectName(_fromUtf8("main_frame"))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.main_frame)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.frame_2 = QtWidgets.QFrame(self.main_frame)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.layout_6 = QtWidgets.QHBoxLayout()
        self.layout_6.setSpacing(8)
        self.layout_6.setObjectName(_fromUtf8("layout_6"))
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setMinimumSize(QtCore.QSize(60, 20))
        self.label.setMaximumSize(QtCore.QSize(80, 35))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet(_fromUtf8("color : white;"))
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.layout_6.addWidget(self.label)
        self.info_label = QtWidgets.QLabel(self.frame_2)
        self.info_label.setMinimumSize(QtCore.QSize(300, 0))
        self.info_label.setFrameShape(QtWidgets.QFrame.Panel)
        self.info_label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.info_label.setAlignment(QtCore.Qt.AlignCenter|QtCore.Qt.AlignVCenter)
        self.info_label.setObjectName(_fromUtf8("info_label"))
        self.layout_6.addWidget(self.info_label)
        self.comment_label = QtWidgets.QLabel(self.frame_2)
        self.comment_label.setMinimumSize(QtCore.QSize(0, 20))
        self.comment_label.setMaximumSize(QtCore.QSize(80, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.comment_label.setFont(font)
        self.comment_label.setStyleSheet(_fromUtf8("color: rgb(255, 255, 255);"))
        self.comment_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.comment_label.setObjectName(_fromUtf8("comment_label"))
        self.layout_6.addWidget(self.comment_label)
        self.comment_textEdit = QtWidgets.QTextEdit(self.frame_2)
        self.comment_textEdit.setMinimumSize(QtCore.QSize(200, 0))
        self.comment_textEdit.setMaximumSize(QtCore.QSize(16777215, 100))
        self.comment_textEdit.setObjectName(_fromUtf8("comment_textEdit"))
        self.layout_6.addWidget(self.comment_textEdit)
        self.submit_pushButton = QtWidgets.QPushButton(self.frame_2)
        self.submit_pushButton.setMinimumSize(QtCore.QSize(130, 100))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.submit_pushButton.setFont(font)
        self.submit_pushButton.setStyleSheet(_fromUtf8("color : white;\n"
"background-color: rgb(62, 109, 186);"))
        self.submit_pushButton.setObjectName(_fromUtf8("submit_pushButton"))
        self.layout_6.addWidget(self.submit_pushButton)
        self.verticalLayout_8.addLayout(self.layout_6)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.gridLayout_11.addWidget(self.main_frame, 0, 0, 1, 2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "Info -", None))
        self.info_label.setText(_translate("Form", "", None))
        self.comment_label.setText(_translate("Form", "Comment -", None))
        self.submit_pushButton.setText(_translate("Form", "SUBMIT", None))
