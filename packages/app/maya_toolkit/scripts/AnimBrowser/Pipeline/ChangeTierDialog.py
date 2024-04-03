# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ChangeTierDialog.ui'
#
# Created: Fri Feb  2 11:20:54 2018
#      by: PyQt4 UI code generator 4.10.3
#
# WARNING! All changes made in this file will be lost!

from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets as QtGui

import dbConfig

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

class ChangeTierDialog(QtGui.QDialog):
    def __init__(self, parent, dbItem, tagDict):
        QtGui.QDialog.__init__(self, parent)
        self.setObjectName(_fromUtf8("Form"))
        self.resize(196, 162)

        self.tagDict = tagDict
        self.dbItem = dbItem
        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label = QtGui.QLabel(self)
        self.label.setMinimumSize(QtCore.QSize(40, 0))
        self.label.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_4.addWidget(self.label)
        self.tier1ComboBox = QtGui.QComboBox(self)
        self.tier1ComboBox.setObjectName(_fromUtf8("tier1ComboBox"))
        self.horizontalLayout_4.addWidget(self.tier1ComboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_2 = QtGui.QLabel(self)
        self.label_2.setMinimumSize(QtCore.QSize(40, 0))
        self.label_2.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.tier2ComboBox = QtGui.QComboBox(self)
        self.tier2ComboBox.setObjectName(_fromUtf8("tier2ComboBox"))
        self.horizontalLayout_3.addWidget(self.tier2ComboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_3 = QtGui.QLabel(self)
        self.label_3.setMinimumSize(QtCore.QSize(40, 0))
        self.label_3.setMaximumSize(QtCore.QSize(40, 16777215))
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.tier3ComboBox = QtGui.QComboBox(self)
        self.tier3ComboBox.setObjectName(_fromUtf8("tier3ComboBox"))
        self.horizontalLayout_2.addWidget(self.tier3ComboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.okBtn = QtGui.QPushButton(self)
        self.okBtn.setObjectName(_fromUtf8("okBtn"))
        self.horizontalLayout.addWidget(self.okBtn)
        self.cancelBtn = QtGui.QPushButton(self)
        self.cancelBtn.setObjectName(_fromUtf8("cancelBtn"))
        self.horizontalLayout.addWidget(self.cancelBtn)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.setWindowTitle(_translate("Form", "Form", None))
        self.label.setText(_translate("Form", "Tier1", None))
        self.label_2.setText(_translate("Form", "Tier2", None))
        self.label_3.setText(_translate("Form", "Tier3", None))
        self.okBtn.setText(_translate("Form", "OK", None))
        self.cancelBtn.setText(_translate("Form", "CANCEL", None))

        print self.dbItem
        print self.tagDict

        tier1 = self.tagDict[self.dbItem['category']].keys()
        self.tier1ComboBox.currentIndexChanged.connect(self.currentTitle1IndexChange)
        self.tier2ComboBox.currentIndexChanged.connect(self.currentTitle2IndexChange)

        self.okBtn.clicked.connect(self.okBtnClicked)
        self.cancelBtn.clicked.connect(lambda : self.close())

        self.tier1ComboBox.addItems(tier1)

        tier1Index = self.tier1ComboBox.findText(self.dbItem['tag1tier'])
        self.tier1ComboBox.setCurrentIndex(tier1Index)

        print "1"
        tier2Index = self.tier2ComboBox.findText(self.dbItem['tag2tier'])
        self.tier2ComboBox.setCurrentIndex(tier2Index)

        print "2"
        tier3Index = self.tier3ComboBox.findText(self.dbItem['tag3tier'])
        self.tier3ComboBox.setCurrentIndex(tier3Index)

        print "3"

    def currentTitle1IndexChange(self, index):
        self.tier2ComboBox.clear()

        # print self.tagDict[self.dbItem['category']][self.tier1ComboBox.currentText()].keys()
        try:
            self.tier2ComboBox.addItems(self.tagDict[self.dbItem['category']][self.tier1ComboBox.currentText()].keys())
        except:
            pass

    def currentTitle2IndexChange(self, index):
        self.tier3ComboBox.clear()

        # print self.tagDict[self.dbItem['category']][self.tier1ComboBox.currentText()][self.tier2ComboBox.currentText()]
        try:
            tier2Item = self.tagDict[self.dbItem['category']][self.tier1ComboBox.currentText()][self.tier2ComboBox.currentText()]
            self.tier3ComboBox.addItems(tier2Item)
        except:
            pass

    def okBtnClicked(self):
        dbConfig.updateTier(self.dbItem['_id'],
                            self.tier1ComboBox.currentText(),
                            self.tier2ComboBox.currentText(),
                            self.tier3ComboBox.currentText())
        self.close()