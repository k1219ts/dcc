#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter CGSupervisor'
__date__ = '2019.01.30'
__comment__ = 'export asset for usd'
__windowName__ = "dxsUsdAssetExport"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds
import maya.OpenMaya as om
import os

from .AssetExportUI import Ui_Form
from .AssetItem import AssetNameItem, AssetAgentItem, AssetElementItem
import Define

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

import dxsUsd
from dxsUsd import DBQuery

def GetModelList():
    result = list()
    for s in cmds.ls('*_model_*GRP', sl=True, r=True):
        if cmds.nodeType(s) != 'dxAssembly':
            result.append(s)

    # sorting model file.
    high = []
    mid = []
    low = []
    for node in result:
        if "model_low" in node:
            low.append(node)
        elif "model_mid" in node:
            mid.append(node)
        else:
            high.append(node)

    result = high + mid + low
    return result

def GetSetList():
    result = list()
    for n in cmds.ls('*_set*', sl=True, r=True):
        if cmds.pluginInfo('TaneForMaya', q=True, l=True):
            if cmds.nodeType(n) == 'TN_TaneMPxTransform':
                result.append(n)
            if cmds.nodeType(n) == 'TN_TaneTransform':
                result.append(n)
        if cmds.pluginInfo('ZMayaTools', q=True, l=True):
            result.append(n)
    return result

def GetZennList():
    zennRoot = list()
    if not cmds.pluginInfo('ZENNForMaya', q=True, l=True):
        return zennRoot
    if not cmds.objExists('ZN_ExportSet'):
        return zennRoot
    for n in cmds.ls(sl=True, dag=True, type='ZN_Global', l=True):
        rootNode = n.split('|')[1]
        if not rootNode in zennRoot:
            zennRoot.append(rootNode)
    return zennRoot

def GetCrowdAssetList():
    result = list()
    if not cmds.pluginInfo('MiarmyProForMaya2017', q=True, l=True) and not cmds.pluginInfo('MiarmyProForMaya2018', q=True, l=True):
        return result
    
    for node in cmds.ls("OriginalAgent_*", sl = True):
        geometryNode = node.replace("OriginalAgent", "Geometry")
        if cmds.objExists(geometryNode):
            result.append(node)

    return result

def GetLightAssetList():
    result = list()
    for node in cmds.ls("*_lgt*", sl = True, type = "xBlock"):
        if cmds.getAttr("%s.type" % node) == 5 and cmds.getAttr("%s.action" % node) == 1:
            result.append(node)

    return result

def getMayaWindow():
    '''
    get Maya Window Process
    :return: Maya window Process
    '''
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QMainWindow)
    except:
        return None

class AssetExportTool(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):

        # Load dependency plugin
        plugins = ['backstageMenu', 'pxrUsd']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.modelVariantColor = QtGui.QBrush(QtGui.QColor(9, 212, 255))
        self.assetNameColor = QtGui.QBrush(QtGui.QColor(9, 255, 212))

        self.isExport = False

        self.itemFont = QtGui.QFont()
        self.itemFont.setPointSize(13)
        self.itemFont.setBold(True)

        # project auto setup
        scenePath = cmds.file(q=True, sn=True)
        if scenePath == "":
            scenePath = cmds.workspace(q = True, rd=True)
        self.showDir, self.showName = dxsUsd.GetProjectPath(maya=scenePath)
        self.ui.showEdit.setText(self.showDir)

        self.ui.overWriteCheckBox.setChecked(True)

        # make completer of show
        customNeedCompleter = []
        customNeedCompleter.append("/assetlib/3D")

        showCompleter = QtWidgets.QCompleter(customNeedCompleter)
        showCompleter.popup().setFont(self.itemFont)
        self.ui.showEdit.setCompleter(showCompleter)

        self.outlineSelectionChanged()

        # Signal connect
        self.ui.showEdit.textChanged.connect(lambda : self.makeTreeWidgetFromNodes())
        self.ui.elementCheckBox.stateChanged.connect(lambda : self.outlineSelectionChanged())
        self.ui.purposeCheckBox.stateChanged.connect(lambda : self.outlineSelectionChanged())
        self.ui.lodVariantCheckBox.stateChanged.connect(lambda: self.outlineSelectionChanged())
        self.ui.loopCheckBox.stateChanged.connect(lambda: self.outlineSelectionChanged())
        self.ui.clipCheckBox.stateChanged.connect(lambda : self.outlineSelectionChanged())
        self.ui.overWriteCheckBox.stateChanged.connect(lambda : self.outlineVersionChanged())
        self.ui.assetExportBtn.clicked.connect(self.exportAsset)
        self.ui.shotExportBtn.clicked.connect(self.exportShot)

        # Maya selection event callback
        self.callback = om.MEventMessage.addEventCallback("SelectionChanged", self.outlineSelectionChanged)

    def closeEvent(self, event):
        # clean up callback
        try:
            om.MMessage.removeCallback(self.callback)
        except:
            pass

    def outlineSelectionChanged(self, *args, **kwargs):
        if self.isExport:
            return

        # model
        modelList = GetModelList()
        # rig
        rigList = cmds.ls(sl = True, type = 'dxRig')
        # set
        if cmds.ls('*_set*', sl=True):
            setList = dxsUsd.GetSetNodes(cmds.ls('*_set*', sl=True))
        else:
            setList = []
        for index, set in enumerate(setList):
            print set
            if set.startswith("|"):
                setList[index] = set[1:]

        # light
        lgtList = GetLightAssetList()
        # camera
        camList = cmds.ls(sl=True, type='dxCamera')
        # zenn
        zennList = GetZennList()
        # crowd
        crowdList = GetCrowdAssetList()

        self.selectNode = modelList + rigList + setList + camList + zennList + crowdList + lgtList

        #  ui changed for export asset type
        # ui visibility
        self.ui.clipCheckBox.setVisible(False)
        self.ui.elementCheckBox.setVisible(False)
        self.ui.purposeCheckBox.setVisible(False)
        self.ui.lodVariantCheckBox.setVisible(False)
        self.ui.stepEdit.setVisible(False)
        self.ui.stepLabel.setVisible(False)
        self.ui.loopCheckBox.setVisible(False)
        self.ui.loopRangeLabel.setVisible(False)
        self.ui.loopStartEdit.setVisible(False)
        self.ui.loopEndEdit.setVisible(False)
        self.ui.loopScaleLabel.setVisible(False)
        self.ui.loopScaleEdit.setVisible(False)

        if self.selectNode:
            if modelList:
                self.ui.clipCheckBox.setVisible(True)
                self.ui.purposeCheckBox.setVisible(True)
                if self.ui.clipCheckBox.isChecked():
                    self.ui.purposeCheckBox.setVisible(True)
                    self.ui.lodVariantCheckBox.setVisible(True)
                    self.clipVisibilityByUI()
                else:
                    self.ui.purposeCheckBox.setVisible(True)
                    self.ui.lodVariantCheckBox.setVisible(True)
                    self.ui.elementCheckBox.setVisible(True)
                    self.clipVisibilityByUI()
            if rigList:
                self.ui.clipCheckBox.setVisible(True)
                self.clipVisibilityByUI()
            if zennList:
                self.ui.clipCheckBox.setVisible(True)
                self.clipVisibilityByUI()
            if setList:
                self.ui.elementCheckBox.setVisible(True)

            self.makeTreeWidgetFromNodes()
        else:
            self.cleanupTreeWidget()
            return

    def clipVisibilityByUI(self):
        if not self.ui.clipCheckBox.isChecked():
            self.ui.stepEdit.setVisible(False)
            self.ui.stepLabel.setVisible(False)
            self.ui.loopCheckBox.setVisible(False)
            self.ui.loopScaleLabel.setVisible(False)
            self.ui.loopScaleEdit.setVisible(False)
        else:
            self.ui.loopCheckBox.setVisible(True)
            self.ui.stepLabel.setVisible(True)
            self.ui.stepEdit.setVisible(True)
            if self.ui.loopCheckBox.isChecked():
                # it is loop animation
                self.ui.loopScaleLabel.setVisible(True)
                self.ui.loopScaleEdit.setVisible(True)
                self.ui.loopRangeLabel.setVisible(True)
                self.ui.loopStartEdit.setVisible(True)
                self.ui.loopEndEdit.setVisible(True)
            else:
                # it isn't loop animation
                self.ui.loopScaleLabel.setVisible(False)
                self.ui.loopScaleEdit.setVisible(False)
                self.ui.loopRangeLabel.setVisible(False)
                self.ui.loopStartEdit.setVisible(False)
                self.ui.loopEndEdit.setVisible(False)

    def cleanupTreeWidget(self):
        while self.ui.treeWidget.topLevelItemCount() > 0:
            self.ui.treeWidget.takeTopLevelItem(0)
            treeitem = self.ui.treeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()

    def getAssetName(self, nodeName, isElement = False):
        nodeType = cmds.nodeType(nodeName)
        name = nodeName.split('|')[-1]  # get modelName
        name = name.split(':')[-1]      # remove namespace
        name = name.replace('model_low', 'model') # remove '_low' Name
        name = name.replace('model_mid', 'model') # remove '_mid' Name
        name = name.replace('model_sim', 'model') # remove '_sim' Name

        # set
        # if nodeType == 'TN_TaneTransform' or nodeType == 'dxAssembly':
        #     name = nodeName.split('|')[-1]
        #     if name.find('_set') > -1:
        #         if isElement:
        #             name = name.split('_')[0]
        #         else:
        #             name = name.split('_set')[0]
        #     else:
        #         name = name.split('_model')[0]
        #     name += '_set'
        #     return name, Define.SET_TYPE
        if "_lgt" in name:
            name = name.split("_lgt")[0]
            return name, Define.LGT_TYPE

        if "_set" in name:
            name = nodeName.split('|')[-1]
            if nodeType == 'TN_TaneTransform' or nodeType == 'dxAssembly' or nodeType == 'xBlock':
                if isElement:
                    name = name.split('_')[0]
                else:
                    name = name.split('_set')[0]
                name += '_set'
                return name, Define.SET_TYPE
        # rig
        elif nodeType == 'dxRig':
            assetType = Define.RIG_TYPE
            if self.ui.clipCheckBox.isChecked():
                assetType = Define.CLIP_TYPE
            return cmds.getAttr("%s.assetName" % nodeName), assetType
        # model
        elif "_model_" in name:
            if not self.ui.clipCheckBox.isChecked():
                if isElement:
                    return name.split('_')[0], Define.MODEL_TYPE
                else:
                    return name.split('_model_')[0], Define.MODEL_TYPE
            else:
                return name.split('_model_')[0], Define.CLIP_TYPE
        # zenn
        elif "_ZN_" in name:
            return name.split('_ZN_')[0], Define.ZENN_TYPE
        # Agent
        elif "OriginalAgent_" in name:
            return name.replace("OriginalAgent_", "").split("_")[0], Define.AGENT_TYPE
        # camera
        elif nodeType == 'dxCamera':
            return name.split("_camera_GRP")[0], Define.CAM_TYPE

    def getGeomType(self, nodeName):
        if "_low" in nodeName:
            return "low"
        elif "_mid" in nodeName:
            return "mid"
        elif "_sim" in nodeName:
            return "sim"
        else:
            return "high"

    def makeTreeWidgetFromNodes(self):
        self.cleanupTreeWidget()

        treeItemDic = {}

        for node in self.selectNode:
            assetName, assetType = self.getAssetName(node, self.ui.elementCheckBox.isChecked())

            # first, has same asset
            if not treeItemDic.has_key(assetName):
                treeItemDic[assetName] = AssetNameItem(self.ui.treeWidget, self.ui.showEdit.text(), assetName, self.ui.overWriteCheckBox.isChecked())
                treeItemDic[assetName].setFont(0, self.itemFont)
                treeItemDic[assetName].setForeground(0, self.assetNameColor)

            assetTypeItem = treeItemDic[assetName].getAssetType(assetType)
            if not assetTypeItem:
                assetTypeItem = treeItemDic[assetName].addAssetType(assetType,
                                                                    self.ui.elementCheckBox.isChecked(),
                                                                    self.getAssetName(node, False)[0])
                assetTypeItem.isLod = self.ui.lodVariantCheckBox.isChecked()
                assetTypeItem.isPurpose = self.ui.purposeCheckBox.isChecked()

            assetTypeItem.nodes[assetType].append(node)

            if assetType == Define.MODEL_TYPE:
                # normal
                if not self.ui.lodVariantCheckBox.isChecked() and not self.ui.purposeCheckBox.isChecked():
                    if not assetTypeItem.childCount() == 0 and not self.ui.elementCheckBox.isChecked():
                        assetTypeItem.nodes[assetType].remove(node)
                    else:
                        if self.ui.elementCheckBox.isChecked():
                            componentName, componentType = self.getAssetName(node, False)
                            componentName = componentName.replace(assetName + "_", "", 1)
                            if not assetTypeItem.elementNodeDict.has_key(componentName):
                                assetTypeItem.elementNodeDict[componentName] = []
                            assetTypeItem.elementNodeDict[componentName].append(node)

                            if not assetTypeItem.nodeDict.has_key(componentName):
                                assetTypeItem.nodeDict[componentName] = {}
                                assetTypeItem.nodeDict[componentName]['item'] = AssetElementItem(assetTypeItem,
                                                                                               self.ui.showEdit.text(),
                                                                                               assetName, componentName,
                                                                                               node,
                                                                                               self.ui.overWriteCheckBox.isChecked())
                                assetTypeItem.nodeDict[componentName]['item'].setFont(0, self.itemFont)
                                assetTypeItem.nodeDict[componentName]['item'].setForeground(0, self.modelVariantColor)
                            nodeDict = assetTypeItem.nodeDict[componentName]
                            item = assetTypeItem.nodeDict[componentName]['item']
                            QtWidgets.QTreeWidgetItem(item, [node]).setFont(0, self.itemFont)
                        else:
                            QtWidgets.QTreeWidgetItem(assetTypeItem, [node]).setFont(0, self.itemFont)

                # has lodVariant, not purpose
                elif self.ui.lodVariantCheckBox.isChecked() and not self.ui.purposeCheckBox.isChecked():
                    if self.ui.elementCheckBox.isChecked():
                        componentName, componentType = self.getAssetName(node, False)
                        componentName = componentName.replace(assetName + "_", "", 1)
                        if not assetTypeItem.elementNodeDict.has_key(componentName):
                            assetTypeItem.elementNodeDict[componentName] = []
                        assetTypeItem.elementNodeDict[componentName].append(node)

                        if not assetTypeItem.nodeDict.has_key(componentName):
                            assetTypeItem.nodeDict[componentName] = {}
                            assetTypeItem.nodeDict[componentName]['item'] = AssetElementItem(assetTypeItem,
                                                                                             self.ui.showEdit.text(),
                                                                                             assetName, componentName,
                                                                                             node,
                                                                                             self.ui.overWriteCheckBox.isChecked())
                            assetTypeItem.nodeDict[componentName]['item'].setFont(0, self.itemFont)
                            assetTypeItem.nodeDict[componentName]['item'].setForeground(0, self.modelVariantColor)

                        nodeDict = assetTypeItem.nodeDict[componentName]
                        item = assetTypeItem.nodeDict[componentName]['item']

                    else:
                        nodeDict = assetTypeItem.nodeDict
                        item = assetTypeItem

                    geomType = self.getGeomType(node)
                    if geomType == "high":
                        if not nodeDict.has_key('high'):
                            lodItem = QtWidgets.QTreeWidgetItem(item, ["high"])
                            lodItem.setFont(0, self.itemFont)
                            nodeDict['high'] = lodItem
                        QtWidgets.QTreeWidgetItem(nodeDict['high'], [node]).setFont(0, self.itemFont)

                    elif geomType == "mid":
                        if not nodeDict.has_key('mid'):
                            lodItem = QtWidgets.QTreeWidgetItem(item, ["mid"])
                            lodItem.setFont(0, self.itemFont)
                            nodeDict['mid'] = lodItem
                        QtWidgets.QTreeWidgetItem(nodeDict['mid'], [node]).setFont(0, self.itemFont)

                    elif geomType == "low":
                        if not nodeDict.has_key('low'):
                            lodItem = QtWidgets.QTreeWidgetItem(item, ["low"])
                            lodItem.setFont(0, self.itemFont)
                            nodeDict['low'] = lodItem
                        QtWidgets.QTreeWidgetItem(nodeDict['low'], [node]).setFont(0, self.itemFont)


                # has purpose, not lod
                elif not self.ui.lodVariantCheckBox.isChecked() and self.ui.purposeCheckBox.isChecked():
                    if self.ui.elementCheckBox.isChecked():
                        componentName, componentType = self.getAssetName(node, False)
                        componentName = componentName.replace(assetName + "_", "", 1)
                        if not assetTypeItem.elementNodeDict.has_key(componentName):
                            assetTypeItem.elementNodeDict[componentName] = []
                        assetTypeItem.elementNodeDict[componentName].append(node)

                        if not assetTypeItem.nodeDict.has_key(componentName):
                            assetTypeItem.nodeDict[componentName] = {}
                            assetTypeItem.nodeDict[componentName]['item'] = AssetElementItem(assetTypeItem,
                                                                                             self.ui.showEdit.text(),
                                                                                             assetName, componentName,
                                                                                             node,
                                                                                             self.ui.overWriteCheckBox.isChecked())
                            assetTypeItem.nodeDict[componentName]['item'].setFont(0, self.itemFont)
                            assetTypeItem.nodeDict[componentName]['item'].setForeground(0, self.modelVariantColor)
                        nodeDict = assetTypeItem.nodeDict[componentName]
                        item = assetTypeItem.nodeDict[componentName]['item']

                        nodeType = self.getGeomType(node)
                        if nodeType == "high":
                            if not nodeDict.has_key("render"):
                                nodeDict['render'] = QtWidgets.QTreeWidgetItem(item, ["render"])
                                nodeDict['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['render'], [node]).setFont(0, self.itemFont)
                        else:
                            if not nodeDict.has_key("proxy"):
                                nodeDict['proxy'] = QtWidgets.QTreeWidgetItem(item, ["proxy"])
                                nodeDict['proxy'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['proxy'], [node]).setFont(0, self.itemFont)
                    else:
                        nodeDict = assetTypeItem.nodeDict
                        item = assetTypeItem

                        geomType = self.getGeomType(node)
                        if geomType == "high":
                            if not nodeDict.has_key("render"):
                                nodeDict['render'] = QtWidgets.QTreeWidgetItem(item, ["render"])
                                nodeDict['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['render'], [node]).setFont(0, self.itemFont)
                        else:
                            if not nodeDict.has_key("proxy"):
                                nodeDict['proxy'] = QtWidgets.QTreeWidgetItem(item, ["proxy"])
                                nodeDict['proxy'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['proxy'], [node]).setFont(0, self.itemFont)

                # has purpose, has lod
                elif self.ui.lodVariantCheckBox.isChecked() and self.ui.purposeCheckBox.isChecked():
                    if self.ui.elementCheckBox.isChecked():
                        componentName, componentType = self.getAssetName(node, False)
                        componentName = componentName.replace(assetName + "_", "", 1)
                        if not assetTypeItem.elementNodeDict.has_key(componentName):
                            assetTypeItem.elementNodeDict[componentName] = []
                        assetTypeItem.elementNodeDict[componentName].append(node)

                        if not assetTypeItem.nodeDict.has_key(componentName):
                            assetTypeItem.nodeDict[componentName] = {}
                            assetTypeItem.nodeDict[componentName]['item'] = AssetElementItem(assetTypeItem,
                                                                                             self.ui.showEdit.text(),
                                                                                             assetName, componentName,
                                                                                             node,
                                                                                             self.ui.overWriteCheckBox.isChecked())
                            assetTypeItem.nodeDict[componentName]['item'].setFont(0, self.itemFont)
                            assetTypeItem.nodeDict[componentName]['item'].setForeground(0, self.modelVariantColor)

                        nodeDict = assetTypeItem.nodeDict[componentName]
                        item = assetTypeItem.nodeDict[componentName]['item']

                        geomType = self.getGeomType(node)
                        if geomType == "high":
                            if not nodeDict.has_key('high'):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["high"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict['high'] = {"item": lodItem, 'render': None, 'proxy': None}
                            if not nodeDict['high']['render']:
                                nodeDict['high']['render'] = QtWidgets.QTreeWidgetItem(nodeDict['high']['item'],
                                                                                       ["render"])
                                nodeDict['high']['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['high']['render'], [node]).setFont(0, self.itemFont)

                        elif geomType == "mid":
                            if not nodeDict.has_key("mid"):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["mid"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict["mid"] = {"item": lodItem, 'render': None, 'proxy': None}
                            if not nodeDict["mid"]['render']:
                                nodeDict["mid"]['render'] = QtWidgets.QTreeWidgetItem(nodeDict['mid']['item'],
                                                                                      ["render"])
                                nodeDict["mid"]['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict["mid"]['render'], [node]).setFont(0, self.itemFont)

                        elif geomType == "low":
                            if not nodeDict.has_key("low"):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["low"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict["low"] = {'item': lodItem, 'render': None, 'proxy': None}
                            if not nodeDict["low"]['render']:
                                nodeDict["low"]['render'] = QtWidgets.QTreeWidgetItem(nodeDict['low']['item'],
                                                                                      ["render"])
                                nodeDict["low"]['render'].setFont(0, self.itemFont)
                            if not nodeDict["low"]['proxy']:
                                nodeDict["low"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['low']['item'], ["proxy"])
                                nodeDict["low"]['proxy'].setFont(0, self.itemFont)

                            QtWidgets.QTreeWidgetItem(nodeDict["low"]['render'], [node]).setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict["low"]['proxy'], [node]).setFont(0, self.itemFont)

                            # proxy setting
                            if "mid" in nodeDict.keys():
                                if not nodeDict["mid"]['proxy']:
                                    nodeDict["mid"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['mid']['item'],
                                                                                         ["proxy"])
                                    nodeDict["mid"]['proxy'].setFont(0, self.itemFont)
                                QtWidgets.QTreeWidgetItem(nodeDict["mid"]['proxy'], [node]).setFont(0,
                                                                                                    self.itemFont)
                            if "high" in nodeDict.keys():
                                if not nodeDict["high"]['proxy']:
                                    nodeDict["high"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['high']['item'],
                                                                                          ["proxy"])
                                    nodeDict["high"]['proxy'].setFont(0, self.itemFont)
                                QtWidgets.QTreeWidgetItem(nodeDict["high"]['proxy'], [node]).setFont(0,
                                                                                                     self.itemFont)
                    else:
                        nodeDict = assetTypeItem.nodeDict
                        item = assetTypeItem

                        geomType = self.getGeomType(node)
                        if geomType == "high":
                            if not nodeDict.has_key('high'):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["high"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict['high'] = {'item': lodItem, 'render': None, 'proxy': None}
                            if not nodeDict['high']['render']:
                                nodeDict['high']['render'] = QtWidgets.QTreeWidgetItem(nodeDict['high']['item'],
                                                                                       ["render"])
                                nodeDict['high']['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict['high']['render'], [node]).setFont(0, self.itemFont)

                        elif geomType == "mid":
                            if not nodeDict.has_key("mid"):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["mid"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict["mid"] = {'item': lodItem, 'render': None, 'proxy': None}
                            if not nodeDict["mid"]['render']:
                                nodeDict["mid"]['render'] = QtWidgets.QTreeWidgetItem(nodeDict['mid']['item'],
                                                                                      ["render"])
                                nodeDict["mid"]['render'].setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict["mid"]['render'], [node]).setFont(0, self.itemFont)

                        elif geomType == "low":
                            if not nodeDict.has_key("low"):
                                lodItem = QtWidgets.QTreeWidgetItem(item, ["low"])
                                lodItem.setFont(0, self.itemFont)
                                nodeDict["low"] = {'item': lodItem, 'render': None, 'proxy': None}
                            if not nodeDict["low"]['render']:
                                nodeDict["low"]['render'] = QtWidgets.QTreeWidgetItem(nodeDict['low']['item'],
                                                                                      ["render"])
                                nodeDict["low"]['render'].setFont(0, self.itemFont)
                            if not nodeDict["low"]['proxy']:
                                nodeDict["low"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['low']['item'], ["proxy"])
                                nodeDict["low"]['proxy'].setFont(0, self.itemFont)

                            QtWidgets.QTreeWidgetItem(nodeDict["low"]['render'], [node]).setFont(0, self.itemFont)
                            QtWidgets.QTreeWidgetItem(nodeDict["low"]['proxy'], [node]).setFont(0, self.itemFont)

                            # proxy setting
                            if "mid" in nodeDict.keys():
                                if not nodeDict["mid"]['proxy']:
                                    nodeDict["mid"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['mid']['item'],
                                                                                         ["proxy"])
                                    nodeDict["mid"]['proxy'].setFont(0, self.itemFont)
                                QtWidgets.QTreeWidgetItem(nodeDict["mid"]['proxy'], [node]).setFont(0, self.itemFont)
                            if "high" in nodeDict.keys():
                                if not nodeDict["high"]['proxy']:
                                    nodeDict["high"]['proxy'] = QtWidgets.QTreeWidgetItem(nodeDict['high']['item'],
                                                                                          ["proxy"])
                                    nodeDict["high"]['proxy'].setFont(0, self.itemFont)
                                QtWidgets.QTreeWidgetItem(nodeDict["high"]['proxy'], [node]).setFont(0, self.itemFont)

            elif assetType == Define.ZENN_TYPE:
                for c in cmds.sets('ZN_ExportSet', q=True):
                    QtWidgets.QTreeWidgetItem(assetTypeItem, [c]).setFont(0, self.itemFont)

            elif assetType == Define.SET_TYPE:
                setList = dxsUsd.GetSetNodes()
                for index, set in enumerate(setList):
                    if set.startswith("|"):
                        setList[index] = set[1:]

                if node in setList:
                    if self.ui.elementCheckBox.isChecked():
                        elementName, elementType = self.getAssetName(node, False)
                        elementName = elementName.replace(assetName.replace("_set", "") + "_", "", 1)
                        if not assetTypeItem.elementNodeDict.has_key(elementName):
                            assetTypeItem.elementNodeDict[elementName] = []
                        assetTypeItem.elementNodeDict[elementName].append(node)

                        if not assetTypeItem.nodeDict.has_key(elementName):
                            assetTypeItem.nodeDict[elementName] = {}
                            assetTypeItem.nodeDict[elementName]['item'] = AssetElementItem(assetTypeItem, self.ui.showEdit.text(), assetName, elementName, node, self.ui.overWriteCheckBox.isChecked())
                            assetTypeItem.nodeDict[elementName]['item'].setFont(0, self.itemFont)
                            assetTypeItem.nodeDict[elementName]['item'].setForeground(0, self.modelVariantColor)
                        item = assetTypeItem.nodeDict[elementName]['item']
                        layerName = node.split('|')[-1].split('_set_')[-1]
                        QtWidgets.QTreeWidgetItem(item, [layerName]).setFont(0, self.itemFont)
                    else:
                        layerName = node.split('|')[-1].split('_set_')[-1]
                        QtWidgets.QTreeWidgetItem(assetTypeItem, [layerName]).setFont(0, self.itemFont)

            elif assetType == Define.AGENT_TYPE:
                agentType = node.replace("OriginalAgent_", "")
                AssetAgentItem(assetTypeItem, self.ui.showEdit.text(), assetName, agentType, node, self.ui.overWriteCheckBox.isChecked())

            elif assetType == Define.CLIP_TYPE:
                # has purpose, not lod
                if self.ui.purposeCheckBox.isChecked():
                    nodeDict = assetTypeItem.nodeDict
                    item = assetTypeItem

                    geomType = self.getGeomType(node)
                    if geomType == "high":
                        if not nodeDict.has_key("render"):
                            nodeDict['render'] = QtWidgets.QTreeWidgetItem(item, ["render"])
                            nodeDict['render'].setFont(0, self.itemFont)
                        QtWidgets.QTreeWidgetItem(nodeDict['render'], [node]).setFont(0, self.itemFont)
                    else:
                        if not nodeDict.has_key("proxy"):
                            nodeDict['proxy'] = QtWidgets.QTreeWidgetItem(item, ["proxy"])
                            nodeDict['proxy'].setFont(0, self.itemFont)
                        QtWidgets.QTreeWidgetItem(nodeDict['proxy'], [node]).setFont(0, self.itemFont)
                else:
                    QtWidgets.QTreeWidgetItem(assetTypeItem, [node]).setFont(0, self.itemFont)

            elif assetType == Define.CAM_TYPE:
                cameras = dxsUsd.GetCameraNodes(node)
                assetTypeItem.nodes[assetType] = cameras
                for cam in cameras:
                    name = cam.split('|')[-1].split(':')[-1]
                    QtWidgets.QTreeWidgetItem(assetTypeItem, [name]).setFont(0, self.itemFont)

            elif assetType == Define.LGT_TYPE:
                name = node.split("_lgt_")[1]
                QtWidgets.QTreeWidgetItem(assetTypeItem, [name]).setFont(0, self.itemFont)

            else:
                QtWidgets.QTreeWidgetItem(assetTypeItem, [node]).setFont(0, self.itemFont)

        self.ui.treeWidget.expandAll()


    def outlineVersionChanged(self):
        overWrite = self.ui.overWriteCheckBox.isChecked()
        for topLevelIndex in range(self.ui.treeWidget.topLevelItemCount()):
            assetItem = self.ui.treeWidget.topLevelItem(topLevelIndex)
            assetName = assetItem.text(0)
            for childIndex in range(assetItem.childCount()):
                typeItem = assetItem.child(childIndex)
                if typeItem.text(0) == Define.AGENT_TYPE:
                    typeItem.child(childIndex).overWrite = overWrite
                    typeItem.child(childIndex).updateLastVersion()
                else:
                    typeItem.overWrite = overWrite
                    typeItem.updateLastVersion()


    def mConfirmDialog(self, assetName, version, isOverwrite):
        if isOverwrite:
            msg = cmds.confirmDialog(
                title='Overwrite?', message='%s of %s overwrite?' % (version, assetName),
                icon='warning', button=['OK', 'CANCEL']
            )
            assert msg != 'CANCEL', '# msg : stopped process!'

    def exportAsset(self):
        # exception
        if not self.selectNode:
            cmds.confirmDialog(title="Error",
                               message="not found publish nodes\ncheck please",
                               icon="warning",
                               button=["OK"])
            return

        self.isExport = True
        self.showDir  = str(self.ui.showEdit.text())
        cmds.waitCursor(state=True)
        # try:
        for topLevelIndex in range(self.ui.treeWidget.topLevelItemCount()):
            assetItem = self.ui.treeWidget.topLevelItem(topLevelIndex)
            for childIndex in range(assetItem.childCount()):
                typeItem = assetItem.child(childIndex)
                if typeItem.text(0) == Define.MODEL_TYPE:
                    self.exportModelExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.SET_TYPE:
                    self.exportSetExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.ZENN_TYPE:
                    self.exportZennAssetExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.CAM_TYPE:
                    self.exportCameraAssetExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.RIG_TYPE:
                    self.exportRigExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.AGENT_TYPE:
                    self.exportAgentAssetExec(assetItem, typeItem)

                elif typeItem.text(0) == Define.CLIP_TYPE:
                    self.exportClipExec(assetItem, typeItem)
                elif typeItem.text(0) == Define.LGT_TYPE:
                    self.exportLgtExec(assetItem, typeItem)
                else:
                    cmds.confirmDialog(title="error!",
                                       message="type not found",
                                       icon="warning",
                                       button=["OK"])
                    continue
        # except Exception as e:
        #     cmds.confirmDialog(title="Fail!!!",
        #                        message="Export Fail\n%s" % e.message,
        #                        icon="warning",
        #                        button=["OK"])
        cmds.waitCursor(state=False)
        cmds.select(cl=True)
        cmds.confirmDialog(title="Success!",
                           message="Export Success",
                           icon="information",
                           button=["OK"])

        self.isExport = False
        self.outlineSelectionChanged()

    def exportShot(self):
        result = cmds.promptDialog(title = "input shot",
                          message="insert shot name",
                          button=["OK", "CANCEL"])

        if result == "OK":
            shotName = cmds.promptDialog(query = True, text = True)
        else:
            return

        # exception
        if not self.selectNode:
            cmds.confirmDialog(title="Error",
                               message="not found publish nodes\ncheck please",
                               icon="warning",
                               button=["OK"])
            return

        self.isExport = True
        self.showDir = str(self.ui.showEdit.text())
        cmds.waitCursor(state=True)
        try:
            for topLevelIndex in range(self.ui.treeWidget.topLevelItemCount()):
                assetItem = self.ui.treeWidget.topLevelItem(topLevelIndex)
                for childIndex in range(assetItem.childCount()):
                    typeItem = assetItem.child(childIndex)
                    if typeItem.text(0) == Define.MODEL_TYPE:
                        self.exportModelExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.SET_TYPE:
                        self.exportSetExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.ZENN_TYPE:
                        self.exportZennAssetExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.CAM_TYPE:
                        self.exportCameraAssetExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.RIG_TYPE:
                        self.exportRigExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.AGENT_TYPE:
                        self.exportAgentAssetExec(assetItem, typeItem, shotName)

                    elif typeItem.text(0) == Define.CLIP_TYPE:
                        self.exportClipExec(assetItem, typeItem, shotName)
                    else:
                        cmds.confirmDialog(title="error!",
                                           message="type not found",
                                           icon="warning",
                                           button=["OK"])
                        continue
        except Exception as e:
            cmds.confirmDialog(title="Fail!!!",
                               message="Export Fail\n%s" % e.message,
                               icon="warning",
                               button=["OK"])
        cmds.waitCursor(state=False)
        cmds.select(cl=True)
        cmds.confirmDialog(title="Success!",
                           message="Export Success",
                           icon="information",
                           button=["OK"])

        self.isExport = False
        self.outlineSelectionChanged()

    def exportModelExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        if typeItem.isElement:
            for elementIndex in range(typeItem.childCount()):
                elementItem = typeItem.child(elementIndex)
                elementVersion = elementItem.versionEdit.text() # element Version
                elementName = elementItem.elementName # element Name
                elementNodes = typeItem.elementNodeDict[elementName] # element Node list
                outDirs = []
                for node in elementNodes:
                    if shotName:
                        mdExp = dxsUsd.ModelExport(
                            node=node, isElement=True, isPurpose=typeItem.isPurpose, isLod=typeItem.isLod,
                            assetDir=assetDir, overWrite=typeItem.overWrite, version=elementVersion
                        )
                        mdExp.doIt()
                    else:
                        mdExp = dxsUsd.ModelExport(
                            node=node, isElement=True, isPurpose=typeItem.isPurpose, isLod=typeItem.isLod,
                            showDir=self.showDir, asset=assetName, overWrite=typeItem.overWrite, version=elementVersion
                        )
                        mdExp.doIt()

                    if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                        outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
                dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, elementVersion, "element", outDirs,
                                             elementName=elementName, elementTask="model")
        else:
            version = typeItem.versionEdit.text()
            outDirs = []
            for node in typeItem.nodes[Define.MODEL_TYPE]:
                if shotName:
                    mdExp = dxsUsd.ModelExport(
                        node=node, isElement=typeItem.isElement, isPurpose=typeItem.isPurpose, isLod=typeItem.isLod,
                        assetDir=assetDir, overWrite=typeItem.overWrite, version=version
                    )
                    mdExp.doIt()
                else:
                    mdExp = dxsUsd.ModelExport(
                        node=node, isElement=typeItem.isElement, isPurpose=typeItem.isPurpose, isLod=typeItem.isLod,
                        showDir=self.showDir, asset=assetName, overWrite=typeItem.overWrite, version=version
                    )
                    mdExp.doIt()
                if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                    outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
            dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "model", outDirs)

    def exportSetExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        if typeItem.isElement:
            for elementIndex in range(typeItem.childCount()):
                elementItem = typeItem.child(elementIndex)
                elementVersion = elementItem.versionEdit.text()
                elementName    = elementItem.elementName
                elementNodes   = typeItem.elementNodeDict[elementName]
                outDirs = []
                for node in elementNodes:
                    if shotName:
                        mdExp = dxsUsd.SetAssetExport(node=node, isElement=True, assetDir=assetDir,
                                              version=elementVersion)
                        mdExp.doIt()
                    else:
                        mdExp = dxsUsd.SetAssetExport(node=node, isElement=True, showDir=self.showDir, asset=assetName, version=elementVersion)
                        mdExp.doIt()
                    if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                        outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
                dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, elementVersion, "element", outDirs,
                                             elementName=elementName, elementTask="set")
        else:
            version = typeItem.versionEdit.text()
            outDirs = []
            for node in typeItem.nodes[Define.SET_TYPE]:
                if shotName:
                    mdExp = dxsUsd.SetAssetExport(node=node, assetDir=assetDir, version=version)
                    mdExp.doIt()
                else:
                    mdExp = dxsUsd.SetAssetExport(node=node, showDir=self.showDir, asset=assetName, version=version)
                    mdExp.doIt()

                if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                    outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
            dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "set", outDirs)

    def exportRigExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version = typeItem.versionEdit.text()
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        self.mConfirmDialog(assetName, version, typeItem.overwriteVersion)
        outDirs = []
        for node in typeItem.nodes[Define.RIG_TYPE]:
            if shotName:
                mdExp = dxsUsd.RigAssetExport(node=node, assetDir=assetDir)
                mdExp.doIt()
            else:
                mdExp = dxsUsd.RigAssetExport(node=node, showDir=self.showDir, asset=assetName)
                mdExp.doIt()
            if not mdExp.outDir in outDirs:
                outDirs.append(mdExp.outDir)
                version = os.path.basename(mdExp.outDir)

        dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "rig", outDirs)

    def exportLgtExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version = typeItem.versionEdit.text()
        # if shotName:
        #     seq = shotName.split("_")[0]
        #     assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/lighting/{ASSETNAME}".format(SHOWDIR=self.showDir,
        #                                                                     SEQ=seq,
        #                                                                     SHOT=shotName,
        #                                                                     ASSETNAME=assetName)
        self.mConfirmDialog(assetName, version, typeItem.overwriteVersion)
        outDirs = []
        for node in typeItem.nodes[Define.LGT_TYPE]:
            # if shotName:
            #     mdExp = dxsUsd.RigAssetExport(node=node, assetDir=assetDir)
            #     mdExp.doIt()
            # else:
            mdExp = dxsUsd.LightInstanceExport(node=node, showDir=self.showDir, asset=assetName, version=version)
            mdExp.doIt()
            if not mdExp.outDir in outDirs:
                outDirs.append(mdExp.outDir)
                version = os.path.basename(mdExp.outDir)

        dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "lighting", outDirs)

    def exportClipExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version   = typeItem.versionEdit.text()
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        isLoop = self.ui.loopCheckBox.isChecked()
        # default
        step = 1.0
        loopScales = [0.8, 1.0, 1.5]
        loopRange  = (1001, 5000)

        stepText = str(self.ui.stepEdit.text())
        if stepText:
            step = float(stepText)
        loopScalesText = str(self.ui.loopScaleEdit.text())
        if loopScalesText:
            loopScales = list()
            for i in loopScalesText.split(','):
                loopScales.append(float(i))
        if not isLoop:
            loopScales = list()
        loopStartText = str(self.ui.loopStartEdit.text())
        loopEndText = str(self.ui.loopEndEdit.text())
        if loopStartText and loopEndText:
            loopRange = (int(loopStartText), int(loopEndText))

        outDirs = []
        for node in typeItem.nodes[Define.CLIP_TYPE]:
            if shotName:
                if cmds.nodeType(node) == 'dxRig':
                    mdExp = dxsUsd.RigClipExport(
                        node=node, step=step, loopScales=loopScales, loopRange=loopRange, overWrite=typeItem.overWrite,
                        assetDir=assetDir, version=version
                    )
                    mdExp.doIt()
                else:
                    mdExp = dxsUsd.ModelClipExport(
                        node=node, step=step, loopScales=loopScales, loopRange=loopRange,
                        isPurpose=typeItem.isPurpose, isLod=typeItem.isLod, overWrite=typeItem.overWrite,
                        assetDir=assetDir, version=version
                    )
                    mdExp.doIt()
                if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                    outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
                dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "clip", outDirs)
            else:
                if cmds.nodeType(node) == 'dxRig':
                    mdExp = dxsUsd.RigClipExport(
                        node=node, step=step, loopScales=loopScales, loopRange=loopRange, overWrite=typeItem.overWrite,
                        showDir=self.showDir, asset=assetName, version=version
                    )
                    mdExp.doIt()
                else:
                    mdExp = dxsUsd.ModelClipExport(
                        node=node, step=step, loopScales=loopScales, loopRange=loopRange,
                        isPurpose=typeItem.isPurpose, isLod=typeItem.isLod, overWrite=typeItem.overWrite,
                        showDir=self.showDir, asset=assetName, version=version
                    )
                    mdExp.doIt()
                if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
                    outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
                dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "clip", outDirs)

    def exportZennAssetExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version = typeItem.versionEdit.text()
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        self.mConfirmDialog(assetName, version, typeItem.overwriteVersion)
        if shotName:
            mdExp = dxsUsd.ZennAssetExport(assetDir=assetDir, version=version)
            mdExp.doIt()
        else:
            mdExp = dxsUsd.ZennAssetExport(showDir=self.showDir, asset=assetName, version=version)
            mdExp.doIt()

        ourDirs = [os.path.join(mdExp.outDir, mdExp.version)]
        files = [mdExp.zennPubSceneFile, mdExp.zennPubSceneFile.replace(".mb", ".json")]
        dxsUsd.DBQuery.assetInsertDB(self.showDir, assetName, version, "zenn", ourDirs, files=files)

    def exportCameraAssetExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version = typeItem.versionEdit.text()
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        self.mConfirmDialog(assetName, version, typeItem.overwriteVersion)
        for node in typeItem.nodes[Define.CAM_TYPE]:
            if shotName:
                dxsUsd.CameraAssetExport(node=node, assetDir=assetDir, version=version).doIt()
            else:
                dxsUsd.CameraAssetExport(node=node, showDir=self.showDir, asset=assetName, version=version).doIt()

    def exportAgentAssetExec(self, assetItem, typeItem, shotName = ""):
        assetName = assetItem.text(0)
        version   = typeItem.child(0).versionEdit.text()
        if shotName:
            seq = shotName.split("_")[0]
            assetDir = "{SHOWDIR}/shot/{SEQ}/{SHOT}/asset/{ASSETNAME}".format(SHOWDIR=self.showDir,
                                                                            SEQ=seq,
                                                                            SHOT=shotName,
                                                                            ASSETNAME=assetName)
        node = typeItem.nodes[Define.AGENT_TYPE][0]

        if shotName:
            mdExp = dxsUsd.AgentExport(node=node, assetDir=assetDir, version=version)
            mdExp.doIt()
        else:
            mdExp = dxsUsd.AgentExport(node=node, showDir=self.showDir, asset=assetName, version=version)
            mdExp.doIt()

        ourDirs = [os.path.join(mdExp.outDir, mdExp.version)]
        dxsUsd.DBQuery.assetInsertDB(self.showDir, mdExp.agentType, version, "agent", ourDirs)


def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = AssetExportTool()
    window.setObjectName(__windowName__)
    window.show()
