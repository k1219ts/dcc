#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2018.11.29'
##########################################

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import os
import maya.cmds as cmds

import Define

import dxsUsd

isDebug = False

class AssetNameItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, showDir, assetName, overWrite):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.showDir   = showDir
        self.assetName = assetName.split(' ')[-1]
        self.setText(0, assetName)
        self.overWrite = overWrite

        if isDebug:
            self.setText(1, self.showDir)

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        self.hasTypeDict = {}

    def getAssetType(self, assetType):
        if self.hasTypeDict.has_key(assetType):
            return self.hasTypeDict[assetType]
        else:
            return None

    def addAssetType(self, assetType, isElement = False, elementName = ""):
        self.hasTypeDict[assetType] = AssetTypeItem(self, self.showDir, self.assetName, assetType, isElement, elementName, self.overWrite)

        return self.hasTypeDict[assetType]

class AssetTypeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, showDir, assetName, assetType, isElement=False, elementName="", overWrite=False):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        itemFont.setBold(True)
        if assetType != Define.ANI_TYPE and assetType != Define.AGENT_TYPE and not isElement:
            self.versionEdit = QtWidgets.QLineEdit()
            self.versionEdit.setFont(itemFont)
            if type(parent) == QtWidgets.QTreeWidget:
                parent.setItemWidget(self, 1, self.versionEdit)
            else:
                parent.treeWidget().setItemWidget(self, 1, self.versionEdit)
            self.versionEdit.textChanged.connect(lambda: self.overwriteVersionCheck())

        self.isElement = isElement
        if assetType == Define.SET_TYPE:
            self.elementName = elementName.replace(assetName.replace("_set", "") + "_", "", 1)
        else:
            self.elementName = elementName.replace(assetName +"_", "", 1)

        self.overWrite = overWrite

        self.showDir   = showDir
        self.assetName = assetName.split(' ')[-1]

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        self.typeColor = QtGui.QBrush(QtGui.QColor(228, 133, 36))
        self.setFont(0, itemFont)
        self.setForeground(0, self.typeColor)

        self.assetType = assetType
        self.setText(0, assetType)
        self.checkPubPath()

        self.nodes = {Define.MODEL_TYPE:[],
                      Define.CLIP_TYPE:[],
                      Define.SET_TYPE:[],
                      Define.RIG_TYPE:[],
                      Define.CAM_TYPE:[],
                      Define.ZENN_TYPE:[],
                      Define.ANI_TYPE:[],
                      Define.AGENT_TYPE:[],
                      Define.LGT_TYPE: []}

        self.nodeDict = {}

        self.elementNodeDict = {}
        self.isVariant = False
        self.isLod = False
        self.isPurpose = False

    def checkPubPath(self):
        pubPath = os.path.join(self.showDir, 'asset', self.assetName)

        self.versionDir = None
        if self.assetType == Define.MODEL_TYPE:
            if self.isElement:
                self.versionDir = None
            else:
                self.versionDir = os.path.join(pubPath, "model")
        elif self.assetType == Define.SET_TYPE:
            if self.isElement:
                self.versionDir = None
            else:
                self.versionDir = os.path.join(pubPath, "model")
        elif self.assetType == Define.ZENN_TYPE:
            self.versionDir = os.path.join(pubPath, "zenn")
        elif self.assetType == Define.CAM_TYPE:
            self.versionDir = os.path.join(pubPath, "cam")
        elif self.assetType == Define.CLIP_TYPE:
            self.versionDir = os.path.join(pubPath, "clip")
        elif self.assetType == Define.LGT_TYPE:
            self.versionDir = os.path.join(pubPath, "lighting")
        else:
            if self.assetType == Define.RIG_TYPE:
                self.versionEdit.setText("RIG ASSET")
                self.versionEdit.setEnabled(False)
        self.updateLastVersion()

    def updateLastVersion(self):
        if self.versionDir:
            self.versionEdit.setText(dxsUsd.GetVersion(self.versionDir, overWrite=self.overWrite))

    def overwriteVersionCheck(self):
        versionPath = os.path.join(self.showDir, 'asset', self.assetName)

        if self.assetType == Define.MODEL_TYPE:
            # if self.isElement:
            #     versionPath = os.path.join(versionPath, "element", self.elementName, 'model', self.versionEdit.text())
            # else:
            versionPath = os.path.join(versionPath, 'model', self.versionEdit.text())
        elif self.assetType == Define.SET_TYPE:
            versionPath = os.path.join(versionPath, 'model', self.versionEdit.text())
        elif self.assetType == Define.CAM_TYPE:
            versionPath = os.path.join(versionPath, 'cam', self.versionEdit.text())
        elif self.assetType == Define.RIG_TYPE:
            sceneName = cmds.file(q=True, sn=True)
            if sceneName:
                sceneName = os.path.splitext(os.path.basename(sceneName))[0]
            else:
                sceneName = 'xxx'
            versionPath = os.path.join(versionPath, 'rig', 'usd', sceneName)
        elif self.assetType == Define.CLIP_TYPE:
            versionPath = os.path.join(versionPath, 'clip', self.versionEdit.text())
        elif self.assetType == Define.ANI_TYPE:
            versionPath = os.path.join(versionPath, 'ani', self.versionEdit.text())
        elif self.assetType == Define.ZENN_TYPE:
            versionPath = os.path.join(versionPath, 'zenn', self.versionEdit.text())
        elif self.assetType == Define.LGT_TYPE:
            versionPath = os.path.join(versionPath, "lighting", self.versionEdit.text())

        if os.path.exists(versionPath):
            self.labelColorSet(self.versionEdit, self.unavailableColor)
            self.overwriteVersion = True
        else:
            self.labelColorSet(self.versionEdit, self.availableColor)
            self.overwriteVersion = False


    def addVariant(self, variantName):
        item = QtWidgets.QTreeWidgetItem(self, ["modelVariant => %s" % variantName])

        self.nodeDict[variantName] = {"item": item, "high": [], "mid": [], "low": []}
        self.isVariant = True

    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)



class AssetAniItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, showDir, assetName, nsLayer, node):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.versionEdit = QtWidgets.QLineEdit()
        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        itemFont.setBold(True)
        self.versionEdit.setFont(itemFont)
        if type(parent) == QtWidgets.QTreeWidget:
            parent.setItemWidget(self, 1, self.versionEdit)
        else:
            parent.treeWidget().setItemWidget(self, 1, self.versionEdit)
        self.versionEdit.textChanged.connect(lambda : self.overwriteVersionCheck())

        self.showDir   = showDir
        self.assetName = assetName.split(' ')[-1]
        self.nsLayer   = nsLayer
        self.node = node
        self.overWrite = True

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        self.setFont(0, itemFont)

        self.setText(0, nsLayer)
        self.checkPubPath()

        self.nodeDict = {}
        self.isVariant = False
        self.isLod = False
        self.isPurpose = False

    def checkPubPath(self):
        pubPath = os.path.join(self.showDir, 'asset', self.assetName, "ani", self.nsLayer)
        self.updateLastVersion(pubPath)

    def updateLastVersion(self, pubPath):
        self.versionEdit.setText(dxsUsd.GetVersion(pubPath, overWrite=self.overWrite))


    def overwriteVersionCheck(self):
        versionPath = os.path.join(self.showDir, 'asset', self.assetName, "ani", self.nsLayer, self.versionEdit.text())

        if os.path.exists(versionPath):
            self.labelColorSet(self.versionEdit, self.unavailableColor)
            self.overwriteVersion = True
        else:
            self.labelColorSet(self.versionEdit, self.availableColor)
            self.overwriteVersion = False


    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)



class AssetAgentItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, showDir, assetName, agentType, node, overWrite):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.versionEdit = QtWidgets.QLineEdit()
        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        itemFont.setBold(True)
        self.versionEdit.setFont(itemFont)
        if type(parent) == QtWidgets.QTreeWidget:
            parent.setItemWidget(self, 1, self.versionEdit)
        else:
            parent.treeWidget().setItemWidget(self, 1, self.versionEdit)
        self.versionEdit.textChanged.connect(lambda : self.overwriteVersionCheck())

        self.showDir   = showDir
        self.assetName = assetName.split(' ')[-1]
        self.agentType = agentType
        self.node = node
        self.overWrite = overWrite

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        self.setFont(0, itemFont)

        self.setText(0, agentType)
        self.versionDir = os.path.join(self.showDir, 'asset', self.assetName, 'agent', self.agentType)
        self.updateLastVersion()

        self.nodeDict = {}
        self.isVariant = False
        self.isLod = False
        self.isPurpose = False

    def updateLastVersion(self):
        self.versionEdit.setText(dxsUsd.GetVersion(self.versionDir, overWrite=self.overWrite))

    def overwriteVersionCheck(self):
        versionPath = os.path.join(self.showDir, 'asset', self.assetName, "agent", self.agentType, self.versionEdit.text())

        if os.path.exists(versionPath):
            self.labelColorSet(self.versionEdit, self.unavailableColor)
            self.overwriteVersion = True
        else:
            self.labelColorSet(self.versionEdit, self.availableColor)
            self.overwriteVersion = False


    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)


class AssetElementItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, showDir, assetName, elementName, node, overWrite):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        self.versionEdit = QtWidgets.QLineEdit()
        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        itemFont.setBold(True)
        self.versionEdit.setFont(itemFont)
        if type(parent) == QtWidgets.QTreeWidget:
            parent.setItemWidget(self, 1, self.versionEdit)
        else:
            parent.treeWidget().setItemWidget(self, 1, self.versionEdit)
        self.versionEdit.textChanged.connect(lambda : self.overwriteVersionCheck())

        self.showDir   = showDir
        self.assetName = assetName.split(' ')[-1]
        self.elementName = elementName
        self.node = node
        self.overWrite = overWrite

        self.availableColor = QtGui.QColor(QtCore.Qt.green)
        self.unavailableColor = QtGui.QColor(QtCore.Qt.red)

        self.setFont(0, itemFont)

        self.setText(0, "element : %s" % elementName)
        self.versionDir = os.path.join(self.showDir, 'asset', self.assetName, 'element', self.elementName, 'model')
        self.updateLastVersion()

        self.nodeDict = {}
        self.isVariant = False
        self.isLod = False
        self.isPurpose = False

    def updateLastVersion(self):
        self.versionEdit.setText(dxsUsd.GetVersion(self.versionDir, overWrite=self.overWrite))

    def overwriteVersionCheck(self):
        versionPath = os.path.join(self.showDir, 'asset', self.assetName, 'element', self.elementName, 'model', self.versionEdit.text())

        if os.path.exists(versionPath):
            self.labelColorSet(self.versionEdit, self.unavailableColor)
            self.overwriteVersion = True
        else:
            self.labelColorSet(self.versionEdit, self.availableColor)
            self.overwriteVersion = False


    def labelColorSet(self, label, qcolor):
        palette = label.palette()
        palette.setColor(label.foregroundRole(), qcolor)
        label.setPalette(palette)
