# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nuke_inven_pub.ui'
#
# Created: Tue Apr 18 17:32:59 2017
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

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
        Form.resize(961, 476)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.titleLabel = QtWidgets.QLabel(Form)
        self.titleLabel.setObjectName(_fromUtf8("titleLabel"))
        self.gridLayout_2.addWidget(self.titleLabel, 0, 0, 1, 1)
        self.typeLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.typeLabel.sizePolicy().hasHeightForWidth())
        self.typeLabel.setSizePolicy(sizePolicy)
        self.typeLabel.setMinimumSize(QtCore.QSize(0, 0))
        self.typeLabel.setObjectName(_fromUtf8("typeLabel"))
        self.gridLayout_2.addWidget(self.typeLabel, 1, 0, 1, 1)
        self.typeComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.typeComboBox.sizePolicy().hasHeightForWidth())
        self.typeComboBox.setSizePolicy(sizePolicy)
        self.typeComboBox.setMinimumSize(QtCore.QSize(200, 0))
        self.typeComboBox.setObjectName(_fromUtf8("typeComboBox"))
        self.gridLayout_2.addWidget(self.typeComboBox, 1, 1, 1, 1)
        self.prjLabel = QtWidgets.QLabel(Form)
        self.prjLabel.setObjectName(_fromUtf8("prjLabel"))
        self.gridLayout_2.addWidget(self.prjLabel, 2, 0, 1, 1)
        self.prjComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prjComboBox.sizePolicy().hasHeightForWidth())
        self.prjComboBox.setSizePolicy(sizePolicy)
        self.prjComboBox.setMinimumSize(QtCore.QSize(200, 0))
        self.prjComboBox.setObjectName(_fromUtf8("prjComboBox"))
        self.gridLayout_2.addWidget(self.prjComboBox, 2, 1, 1, 1)
        self.prjLineEdit = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.prjLineEdit.sizePolicy().hasHeightForWidth())
        self.prjLineEdit.setSizePolicy(sizePolicy)
        self.prjLineEdit.setObjectName(_fromUtf8("prjLineEdit"))
        self.gridLayout_2.addWidget(self.prjLineEdit, 2, 2, 1, 2)
        self.categoryLabel = QtWidgets.QLabel(Form)
        self.categoryLabel.setObjectName(_fromUtf8("categoryLabel"))
        self.gridLayout_2.addWidget(self.categoryLabel, 3, 0, 1, 1)
        self.categoryComboBox = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.categoryComboBox.sizePolicy().hasHeightForWidth())
        self.categoryComboBox.setSizePolicy(sizePolicy)
        self.categoryComboBox.setMinimumSize(QtCore.QSize(200, 0))
        self.categoryComboBox.setObjectName(_fromUtf8("categoryComboBox"))
        self.gridLayout_2.addWidget(self.categoryComboBox, 3, 1, 1, 1)
        self.categoryLineEdit = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.categoryLineEdit.sizePolicy().hasHeightForWidth())
        self.categoryLineEdit.setSizePolicy(sizePolicy)
        self.categoryLineEdit.setObjectName(_fromUtf8("categoryLineEdit"))
        self.gridLayout_2.addWidget(self.categoryLineEdit, 3, 2, 1, 2)
        self.hipLabel = QtWidgets.QLabel(Form)
        self.hipLabel.setObjectName(_fromUtf8("hipLabel"))
        self.gridLayout_2.addWidget(self.hipLabel, 4, 0, 1, 1)
        self.hipLine = QtWidgets.QLineEdit(Form)
        self.hipLine.setObjectName(_fromUtf8("hipLine"))
        self.gridLayout_2.addWidget(self.hipLine, 4, 1, 1, 2)
        self.hipSearchButton = QtWidgets.QPushButton(Form)
        self.hipSearchButton.setMaximumSize(QtCore.QSize(100, 16777215))
        self.hipSearchButton.setObjectName(_fromUtf8("hipSearchButton"))
        self.gridLayout_2.addWidget(self.hipSearchButton, 4, 3, 1, 1)
        self.tagLabel = QtWidgets.QLabel(Form)
        self.tagLabel.setObjectName(_fromUtf8("tagLabel"))
        self.gridLayout_2.addWidget(self.tagLabel, 6, 0, 1, 1)
        self.tagTextEdit = QtWidgets.QTextEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tagTextEdit.sizePolicy().hasHeightForWidth())
        self.tagTextEdit.setSizePolicy(sizePolicy)
        self.tagTextEdit.setObjectName(_fromUtf8("tagTextEdit"))
        self.gridLayout_2.addWidget(self.tagTextEdit, 7, 0, 1, 5)
        self.titleLineEdit = QtWidgets.QLineEdit(Form)
        self.titleLineEdit.setObjectName(_fromUtf8("titleLineEdit"))
        self.gridLayout_2.addWidget(self.titleLineEdit, 0, 1, 1, 3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.readCombo = QtWidgets.QComboBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.readCombo.sizePolicy().hasHeightForWidth())
        self.readCombo.setSizePolicy(sizePolicy)
        self.readCombo.setObjectName(_fromUtf8("readCombo"))
        self.verticalLayout.addWidget(self.readCombo)
        self.thumbLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thumbLabel.sizePolicy().hasHeightForWidth())
        self.thumbLabel.setSizePolicy(sizePolicy)
        self.thumbLabel.setMinimumSize(QtCore.QSize(300, 0))
        self.thumbLabel.setObjectName(_fromUtf8("thumbLabel"))
        self.verticalLayout.addWidget(self.thumbLabel)
        self.frameSlider = QtWidgets.QSlider(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frameSlider.sizePolicy().hasHeightForWidth())
        self.frameSlider.setSizePolicy(sizePolicy)
        self.frameSlider.setOrientation(QtCore.Qt.Horizontal)
        self.frameSlider.setObjectName(_fromUtf8("frameSlider"))
        self.verticalLayout.addWidget(self.frameSlider)
        self.frameLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frameLabel.sizePolicy().hasHeightForWidth())
        self.frameLabel.setSizePolicy(sizePolicy)
        self.frameLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.frameLabel.setObjectName(_fromUtf8("frameLabel"))
        self.verticalLayout.addWidget(self.frameLabel)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 4, 6, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.closeButton = QtWidgets.QPushButton(Form)
        self.closeButton.setObjectName(_fromUtf8("closeButton"))
        self.horizontalLayout.addWidget(self.closeButton)
        self.publishButton = QtWidgets.QPushButton(Form)
        self.publishButton.setObjectName(_fromUtf8("publishButton"))
        self.horizontalLayout.addWidget(self.publishButton)
        self.gridLayout_2.addLayout(self.horizontalLayout, 8, 0, 1, 5)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.selectedPub = QtWidgets.QRadioButton(self.groupBox)
        self.selectedPub.setObjectName(_fromUtf8("selectedPub"))
        self.gridLayout.addWidget(self.selectedPub, 0, 0, 1, 1)
        self.allPub = QtWidgets.QRadioButton(self.groupBox)
        self.allPub.setChecked(True)
        self.allPub.setObjectName(_fromUtf8("allPub"))
        self.gridLayout.addWidget(self.allPub, 0, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox, 5, 0, 1, 4)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.titleLabel.setText(_translate("Form", "Asset Title", None))
        self.typeLabel.setText(_translate("Form", "Type", None))
        self.prjLabel.setText(_translate("Form", "Project", None))
        self.categoryLabel.setText(_translate("Form", "Category", None))
        self.hipLabel.setText(_translate("Form", "hip(optional)", None))
        self.hipSearchButton.setText(_translate("Form", "...", None))
        self.tagLabel.setText(_translate("Form", "Tags", None))
        self.thumbLabel.setText(_translate("Form", "TextLabel", None))
        self.frameLabel.setText(_translate("Form", "Frame", None))
        self.closeButton.setText(_translate("Form", "Close", None))
        self.publishButton.setText(_translate("Form", "Publish", None))
        self.groupBox.setTitle(_translate("Form", "Nuke script publish", None))
        self.selectedPub.setText(_translate("Form", "only selected", None))
        self.allPub.setText(_translate("Form", "all", None))