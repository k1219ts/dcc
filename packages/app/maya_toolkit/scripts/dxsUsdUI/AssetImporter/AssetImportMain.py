#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2018.12.06'
__comment__ = 'import asset for usd'
__windowName__ = "dxsUsdAssetImport"
##########################################

import os

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds

from .AssetImportUI import Ui_Form

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

import dxsUsd
import xbUtils

from pxr import Sdf

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

class AssetImportTool(QtWidgets.QWidget):
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

        self.assetNameColor = QtGui.QBrush(QtGui.QColor(248, 137, 7))
        self.elementColor = QtGui.QBrush(QtGui.QColor(9, 212, 255))
        self.versionColor = QtGui.QBrush(QtGui.QColor(QtCore.Qt.green))
        self.otherColor = QtGui.QBrush(QtGui.QColor(QtCore.Qt.white))

        self.itemFont = QtGui.QFont()
        self.itemFont.setPointSize(13)
        self.itemFont.setBold(True)

        # project auto setup
        scenePath = cmds.file(q=True, sn=True)
        if scenePath == "":
            scenePath = cmds.workspace(q = True, rd=True)
        self.showDir, self.showName = dxsUsd.GetProjectPath(maya=scenePath)
        self.ui.showEdit.setText(self.showDir)

        # make completer of show
        customNeedCompleter = []
        customNeedCompleter.append("/assetlib/3D")

        showCompleter = QtWidgets.QCompleter(customNeedCompleter)
        showCompleter.popup().setFont(self.itemFont)
        self.ui.showEdit.setCompleter(showCompleter)

        # Signal connect
        self.ui.showEdit.returnPressed.connect(lambda : self.changeAssetListCompleter())
        self.ui.assetEdit.returnPressed.connect(lambda: self.makeTreeWidgetFromAssetList())
        self.ui.treeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.treeWidget.customContextMenuRequested.connect(self.rmbClicked)
        self.ui.treeWidget.itemDoubleClicked.connect(self.assetItemDoublieClicked)
        self.ui.importBtn.clicked.connect(self.importAsset)

    def changeAssetListCompleter(self):
        self.assetList = {}
        elementList = []
        assetUsdFilePath = os.path.join(self.ui.showEdit.text(), "asset", "asset.usd")
        if os.path.exists(assetUsdFilePath):
            outLayer = Sdf.Layer.FindOrOpen(assetUsdFilePath)

            if outLayer:
                for subLayer in outLayer.subLayerPaths:
                    relPath = os.path.relpath(subLayer)
                    assetName = relPath.split("/")[0]
                    if os.path.exists(os.path.join(self.ui.showEdit.text(), "asset", assetName)):
                        self.assetList[assetName] = []
                        elementDir = os.path.join(self.ui.showEdit.text(), "asset", assetName, "element")
                        if os.path.exists(elementDir):
                            # elementUsdFilePath = os.path.join(assetNameDir, "element.usd")
                            # elementOutLayer = Sdf.Layer.FindOrOpen(elementUsdFilePath)
                            # if elementOutLayer:
                            #     for elSubLayer in elementOutLayer.subLayerPaths:
                            for element in os.listdir(elementDir):
                                if os.path.isdir(os.path.join(elementDir, element)):
                                    self.assetList[assetName].append(element)
                                    elementList.append(element)

                assetCompleter = QtWidgets.QCompleter(self.assetList.keys() + elementList)
                assetCompleter.popup().setFont(self.itemFont)

                self.ui.assetEdit.setCompleter(assetCompleter)
                self.ui.assetEdit.setDisabled(False)
                self.ui.assetEdit.setText("")
                self.makeTreeWidgetFromAssetList()
        else:
            self.ui.assetEdit.setDisabled(True)
            self.cleanupTreeWidget()

    def cleanupTreeWidget(self):
        while self.ui.treeWidget.topLevelItemCount() > 0:
            self.ui.treeWidget.takeTopLevelItem(0)
            treeitem = self.ui.treeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()

    def makeTreeWidgetFromAssetList(self):
        self.cleanupTreeWidget()

        for assetName in self.assetList.keys():
            QtWidgets.QApplication.processEvents()
            # make assetList
            if self.ui.assetEdit.text().lower() in assetName.lower():
                assetNameItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget, [assetName, "", "0"])
                assetNameItem.setFont(0, self.itemFont)
                assetNameItem.setForeground(0, self.assetNameColor)
            elif self.assetList.has_key(assetName) and self.assetList[assetName]:
                # has element
                isInsert = False
                for elementName in self.assetList[assetName]:
                    if self.ui.assetEdit.text() in elementName.lower():
                        isInsert = True
                        break
                if isInsert:
                    assetNameItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget, [assetName, "", "0"])
                    assetNameItem.setFont(0, self.itemFont)
                    assetNameItem.setForeground(0, self.assetNameColor)


    def assetItemDoublieClicked(self, assetNameItem, column):
        assetDir = os.path.join(self.ui.showEdit.text(), "asset", assetNameItem.text(0))
        if assetNameItem.text(2) == "1" or assetNameItem.parent() is not None:
            return

        assetNameItem.setText(2, "1")
        if os.path.exists(os.path.join(assetDir, "model")):
            modelItem = QtWidgets.QTreeWidgetItem(assetNameItem, ["model"])
            modelItem.setFont(0, self.itemFont)

            modelDir = os.path.join(assetDir, "model")
            for i in os.listdir(modelDir):
                if os.path.isdir(os.path.join(modelDir, i)):
                    if i[0] == 'v':
                        versionItem = QtWidgets.QTreeWidgetItem(modelItem, [i, os.path.join(modelDir, i)])
                        versionItem.setFont(0, self.itemFont)
                        versionItem.setForeground(0, self.versionColor)

                        versionDir = os.path.join(modelDir, i)
                        for j in os.listdir(versionDir):
                            if "_geom.usd" in j:
                                geomItem = QtWidgets.QTreeWidgetItem(versionItem, [j, os.path.join(versionDir, j), i])
                                geomItem.setFont(0, self.itemFont)

        if os.path.exists(os.path.join(assetDir, "rig", "usd")):
            rigItem = QtWidgets.QTreeWidgetItem(assetNameItem, ["rig"])
            rigItem.setFont(0, self.itemFont)

            rigDir = os.path.join(assetDir, "rig", "usd")
            for i in os.listdir(rigDir):
                if os.path.isdir(os.path.join(rigDir, i)):
                    versionItem = QtWidgets.QTreeWidgetItem(rigItem, [i, os.path.join(rigDir, i)])
                    versionItem.setFont(0, self.itemFont)
                    versionItem.setForeground(0, self.versionColor)

        if os.path.exists(os.path.join(assetDir, "clip")):
            clipItem = QtWidgets.QTreeWidgetItem(assetNameItem, ["clip"])
            clipItem.setFont(0, self.itemFont)

            clipDir = os.path.join(assetDir, "clip")
            for i in os.listdir(clipDir):
                if os.path.isdir(os.path.join(clipDir, i)):
                    if i[0] == 'v':
                        versionItem = QtWidgets.QTreeWidgetItem(clipItem, [i, os.path.join(clipDir, i)])
                        versionItem.setFont(0, self.itemFont)
                        versionItem.setForeground(0, self.versionColor)

                        versionDir = os.path.join(clipDir, i)
                        for j in os.listdir(versionDir):
                            if os.path.isdir(os.path.join(versionDir, j)):
                                geomItem = QtWidgets.QTreeWidgetItem(versionItem, [j, os.path.join(versionDir, j), i])
                                geomItem.setFont(0, self.itemFont)

        if os.path.exists(os.path.join(assetDir, "element")):
            elementItem = QtWidgets.QTreeWidgetItem(assetNameItem, ["element"])
            elementItem.setFont(0, self.itemFont)

            elementDir = os.path.join(assetDir, "element")
            elementUsdFile = os.path.join(elementDir, "element.usd")
            outLayer = Sdf.Layer.FindOrOpen(elementUsdFile)

            if outLayer:
                for subLayer in outLayer.subLayerPaths:
                    relPath = os.path.relpath(subLayer)
                    elementName = relPath.split("/")[0]
                    if os.path.exists(os.path.join(elementDir, elementName)):
                        elementNameItem = QtWidgets.QTreeWidgetItem(elementItem,
                                                                    [elementName, os.path.join(elementDir, elementName)])
                        elementNameItem.setFont(0, self.itemFont)
                        elementNameItem.setForeground(0, self.elementColor)

                        modelDir = os.path.join(elementDir, elementName, "model")
                        if os.path.exists(modelDir):
                            modelItem = QtWidgets.QTreeWidgetItem(elementNameItem, ["model"])
                            modelItem.setFont(0, self.itemFont)

                            for i in os.listdir(modelDir):
                                if os.path.isdir(os.path.join(modelDir, i)):
                                    if i[0] == 'v':
                                        versionItem = QtWidgets.QTreeWidgetItem(modelItem, [i, os.path.join(modelDir, i)])
                                        versionItem.setFont(0, self.itemFont)
                                        versionItem.setForeground(0, self.versionColor)

                                        versionDir = os.path.join(modelDir, i)
                                        for j in os.listdir(versionDir):
                                            if "_geom.usd" in j:
                                                geomItem = QtWidgets.QTreeWidgetItem(versionItem,
                                                                                     [j, os.path.join(versionDir, j), i])
                                                geomItem.setFont(0, self.itemFont)

                        clipDir = os.path.join(elementDir, elementName, "clip")
                        if os.path.exists(clipDir):
                            clipItem = QtWidgets.QTreeWidgetItem(elementNameItem, ["clip"])
                            clipItem.setFont(0, self.itemFont)

                            clipDir = os.path.join(assetDir, "clip")
                            for i in os.listdir(clipDir):
                                if os.path.isdir(os.path.join(clipDir, i)):
                                    if i[0] == 'v':
                                        versionItem = QtWidgets.QTreeWidgetItem(clipItem,
                                                                                [i, os.path.join(clipDir, i)])
                                        versionItem.setFont(0, self.itemFont)
                                        versionItem.setForeground(0, self.versionColor)

                                        versionDir = os.path.join(clipDir, i)
                                        for j in os.listdir(versionDir):
                                            if os.path.isdir(os.path.join(versionDir, j)):
                                                geomItem = QtWidgets.QTreeWidgetItem(versionItem, [j, os.path.join(
                                                    versionDir, j), i])
                                                geomItem.setFont(0, self.itemFont)


    def mConfirmDialog(self, assetName, version, isOverwrite):
        if isOverwrite:
            msg = cmds.confirmDialog(
                title='Overwrite?', message='%s of %s overwrite?' % (version, assetName),
                icon='warning', button=['OK', 'CANCEL']
            )
            assert msg != 'CANCEL', '# msg : stopped process!'

    def rmbClicked(self, pos):
        item = self.ui.treeWidget.currentItem()

        menuTitle = 'Reference {NAME}'
        if cmds.pluginInfo('TaneForMaya', q=True, l=True):
            selected = cmds.ls(sl=True, dag=True, type='TN_Tane')
            if selected:
                menuTitle += ' in Tane'

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet('''
                                        QMenu::item:selected {
                                        background-color: #81CF3E;
                                        color: #404040; }
                                       ''')

        if item.text(0) in self.assetList:
            menu.addAction(QtGui.QIcon(), menuTitle.format(NAME=item.text(0)), lambda : self.referenceImport(item))
            menu.addAction(QtGui.QIcon(), menuTitle.format(NAME=item.text(0)) + ' (SceneAssembly)', lambda : self.referenceAssemblyImport(item))
        elif item.parent() and "element" in item.parent().text(0):
            menu.addAction(QtGui.QIcon(), menuTitle.format(NAME=item.text(0)), lambda : self.referenceImport(item))
            menu.addAction(QtGui.QIcon(), menuTitle.format(NAME=item.text(0)) + ' (SceneAssembly)', lambda : self.referenceAssemblyImport(item))
        elif item.parent().parent() and "clip" in item.parent().parent().text(0) and "loop" in item.text(0):
            menu.addAction(QtGui.QIcon(), menuTitle.format(NAME=item.text(0)),
                           lambda: self.referenceImport(item))
        else:
            return
        menu.popup(QtGui.QCursor.pos())

    def referenceImport(self, item):
        assetName = item.text(0)
        rootDir   = item.text(1)
        print assetName, rootDir
        if not rootDir:
            rootDir = '{DIR}/asset/{NAME}'.format(DIR=self.ui.showEdit.text(), NAME=assetName)

        filename = '{DIR}/{NAME}.usd'.format(DIR=rootDir, NAME=assetName)

        if cmds.pluginInfo('TaneForMaya', q=True, l=True):
            selected = cmds.ls(sl=True, dag=True, type='TN_Tane')
            if selected:
                self.referenceTane(selected[0], assetName, filename)
            else:
                dxsUsd.dxsMayaUtils.UsdProxyImport(filename)
        else:
            dxsUsd.dxsMayaUtils.UsdProxyImport(filename)

    def referenceAssemblyImport(self, item):
        assetName = item.text(0)
        rootDir   = item.text(1)
        if not rootDir:
            rootDir = '{DIR}/asset/{NAME}'.format(DIR=self.ui.showEdit.text(), NAME=assetName)
        filename = '{DIR}/{NAME}.usd'.format(DIR=rootDir, NAME=assetName)
        dxsUsd.dxsMayaUtils.UsdAssemblyImport(filename)

    def referenceTane(self, taneShape, assetName, sourceFile):
        environmentNode = cmds.listConnections(taneShape, s=True, d=False, type='TN_Environment')
        assert environmentNode, '# msg : tane setup error'
        environmentNode = environmentNode[0]
        index = 0
        for idx in range(0, 100):
            if not cmds.connectionInfo("%s.inSource[%d]" % (environmentNode, idx), ied=True):
                index = idx
                break
        proxyShape = cmds.TN_CreateNode(nt='TN_UsdProxy')
        proxyTrans = cmds.listRelatives(proxyShape, p=True)[0]
        cmds.setAttr('%s.visibility' % proxyTrans, 0)
        cmds.setAttr('%s.renderFile' % proxyShape, sourceFile, type='string')
        cmds.connectAttr('%s.outSource' % proxyShape, '%s.inSource[%d]' % (environmentNode, index))

        proxyTrans = cmds.rename(proxyTrans, 'TN_%s' % assetName)

        taneTrans = cmds.listRelatives(taneShape, p=True)[0]
        cmds.parent(proxyTrans, taneTrans)
        cmds.select(taneTrans)


    def importAsset(self):
        item = self.ui.treeWidget.currentItem()
        if not item:
            cmds.confirmDialog(
                title='Warning', message='first, select item',icon='warning', button=['OK']
            )
            return

        selected = cmds.ls(sl=True)

        filePath = item.text(1)
        if "_rig_" in item.text(0):
            assetName = item.text(0).split("_rig_")[0]
            rigFilePath = os.path.join(item.text(1), "%s_rig_GRP.usd" % assetName)
            xbUtils.UsdImport(rigFilePath).doIt()
        else:
            if os.path.isdir(filePath): # version directory
                for fileName in os.listdir(filePath):
                    if "high_geom.usd" in fileName or "mid_geom.usd" in fileName or "low_geom.usd" in fileName:
                        nodeList = dxsUsd.dxsMayaUtils.UsdImport(os.path.join(filePath, fileName))
                        for node in nodeList:
                            self.postProcess(node, selected, version=item.text(0), usdfile=os.path.join(filePath, fileName))

            else:
                nodeList = dxsUsd.dxsMayaUtils.UsdImport(filePath)
                for node in nodeList:
                    self.postProcess(node, selected, version=item.text(2), usdfile=filePath)

    # Under Post Process
    def postProcess(self, node, selected, **kwargs):
        # curve post process
        if kwargs.has_key('usdfile'):
            xbUtils.common.ImportGeomPostProcess(kwargs['usdfile'], node).doIt()

        # model version postProcess
        for mesh in cmds.ls(node, dag=True, type='shape', l=True):
            self.setAttribute(mesh, attr="modelVersion", value=kwargs['version'])

        # parenting post process
        if selected:
            cmds.parent(node, selected[0])
        cmds.select(node)

    def setAttribute(self, mesh, attr, value):
        if not cmds.attributeQuery(attr, n=mesh, exists=True):
            cmds.addAttr(mesh, ln=attr, dt='string')
        cmds.setAttr("%s.%s" % (mesh, attr), value, type='string')

def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = AssetImportTool()
    window.setObjectName(__windowName__)
    window.show()
    QtWidgets.QApplication.processEvents()
    window.changeAssetListCompleter()
