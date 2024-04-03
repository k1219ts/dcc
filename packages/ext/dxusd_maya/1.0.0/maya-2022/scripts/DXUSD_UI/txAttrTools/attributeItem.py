#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2020.10.12'
##########################################

from PySide2 import QtWidgets
from PySide2 import QtGui
from PySide2 import QtCore

import maya.cmds as cmds


def labelColorSet(label, qcolor):
    palette = label.palette()
    palette.setColor(label.foregroundRole(), qcolor)
    label.setPalette(palette)

class ShapeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, shape, attributeList):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.shapeName = shape

        self.alreadySetupAttr = {}
        for attr in attributeList:
            self.alreadySetupAttr[attr] = None

        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        self.setFont(0, itemFont)
        self.setText(0, shape)

class AttritubeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, shape, attr, value, attributeInfo, error = False):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.shapeName = shape
        self.attrName = attr
        self.attrValue = value
        self.attributeInfo = attributeInfo

        self.defaultColor = QtGui.QColor(QtCore.Qt.white)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        self.setFont(0, itemFont)
        self.setText(0, attributeInfo['niceName'])

        if self.attrName == "USD_ATTR_subdivisionScheme":
            self.comboBoxAttrInfo = {"USD_ATTR_subdivisionScheme": {0: 'catmullClark', 1: 'loop', 100: 'none'}}
            self.attrValueComboBox = QtWidgets.QComboBox()
            itemFont = QtGui.QFont()
            itemFont.setPointSize(13)
            self.attrValueComboBox.setFont(itemFont)
            if type(parent) == QtWidgets.QTreeWidget:
                parent.setItemWidget(self, 1, self.attrValueComboBox)
            else:
                parent.treeWidget().setItemWidget(self, 1, self.attrValueComboBox)

            for attrInfo in self.comboBoxAttrInfo[self.attrName]:
                self.attrValueComboBox.addItem(self.comboBoxAttrInfo[self.attrName][attrInfo])
            index = self.attrValueComboBox.findText(self.attrValue)
            self.attrValueComboBox.setCurrentIndex(index)
            self.attrValueComboBox.currentIndexChanged.connect(self.attributeUpdate)

        elif self.attributeInfo.get("dataType"):
            if self.attributeInfo["dataType"] == "string":
                self.attrValueEdit = QtWidgets.QLineEdit()
                itemFont = QtGui.QFont()
                itemFont.setPointSize(13)
                self.attrValueEdit.setFont(itemFont)
                if type(parent) == QtWidgets.QTreeWidget:
                    parent.setItemWidget(self, 1, self.attrValueEdit)
                else:
                    parent.treeWidget().setItemWidget(self, 1, self.attrValueEdit)
                self.attrValueEdit.returnPressed.connect(self.attributeUpdate)
                self.attrValueEdit.setText(self.attrValue)
                if error:
                    labelColorSet(self.attrValueEdit, self.unavailableColor)

        elif self.attributeInfo.get("attrType"):
            if self.attributeInfo["attrType"] == "long":
                self.attrValueCheckBox = QtWidgets.QCheckBox()
                itemFont = QtGui.QFont()
                itemFont.setPointSize(13)
                self.attrValueCheckBox.setFont(itemFont)
                if type(parent) == QtWidgets.QTreeWidget:
                    parent.setItemWidget(self, 1, self.attrValueCheckBox)
                else:
                    parent.treeWidget().setItemWidget(self, 1, self.attrValueCheckBox)
                self.attrValueCheckBox.stateChanged.connect(self.attributeUpdate)
                self.attrValueCheckBox.setChecked(bool(self.attrValue))

        if self.attrName == "txBasePath":
            self.resetBtn = QtWidgets.QPushButton()
            itemFont = QtGui.QFont()
            itemFont.setPointSize(13)
            self.resetBtn.setFont(itemFont)
            if type(parent) == QtWidgets.QTreeWidget:
                parent.setItemWidget(self, 2, self.resetBtn)
            else:
                parent.treeWidget().setItemWidget(self, 2, self.resetBtn)
            self.resetBtn.clicked.connect(self.attributeReset)
            self.resetBtn.setText("reset")

    def attributeUpdate(self, reload = True):
        if self.attrName == "USD_ATTR_subdivisionScheme":
            if not reload:
                index = self.attrValueComboBox.findText(self.attrValue)
                self.attrValueComboBox.setCurrentIndex(index)
            self.attrValue = str(self.attrValueComboBox.currentText())
            cmds.setAttr("%s.%s" % (self.shapeName, self.attrName), self.attrValue, type="string")
        elif self.attributeInfo.get("dataType") and self.attributeInfo["dataType"] == "string":
            if not reload:
                self.attrValueEdit.setText(self.attrValue)
            self.attrValue = str(self.attrValueEdit.text())
            cmds.setAttr("%s.%s" % (self.shapeName, self.attrName), self.attrValue, type = "string")
        elif self.attributeInfo.get("attrType") and self.attributeInfo["attrType"] == "long":
            if not reload:
                self.attrValueCheckBox.setChecked(bool(self.attrValue))
            self.attrValue = self.attrValueCheckBox.isChecked()
            cmds.setAttr("%s.%s" % (self.shapeName, self.attrName), int(self.attrValue))

    def attributeReset(self):
        scenePath = cmds.file(q=True, sn=True)
        if scenePath == "":
            scenePath = cmds.workspace(q=True, rd=True)
        splitScenePath = scenePath.split("/")

        basePath = ""
        if "asset" in splitScenePath:
            assetName = splitScenePath[splitScenePath.index("asset") + 2]
            basePath = "asset/%s" % assetName
            if "element" in splitScenePath:
                elementName = splitScenePath[splitScenePath.index("element") + 1]
                basePath += "/element/%s" % elementName
        elif "shot" in splitScenePath:
            basePath = "/".join(splitScenePath[splitScenePath.index("shot"):splitScenePath.index("shot")+3])

        basePath += "/texture"

        self.attrValueEdit.setText(basePath)
        self.attributeUpdate()
