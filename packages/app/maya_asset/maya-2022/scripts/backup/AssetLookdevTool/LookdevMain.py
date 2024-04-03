import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as mui

MAYA_VERSION = "2017"

if Qt.__qt_version__ > "5.0.0":
    import shiboken2 as shiboken

    MAYA_VERSION = "2017"
else:
    import shiboken as shiboken

    MAYA_VERSION = "2016.5"

from LookdevMainUI import Ui_Form
from LookdevModule.String import String
from LookdevModule.HDRISet import HDRISet
from LookdevModule.MessageBox import MessageBox
from LookdevModule.DynamicBinding import DynamicBinding
from AssetEnv.EnvSourceLoad import EnvSourceLoad

from LdvAssetTreeWidgetItem import LdvAssetTreeWidgetItem
from LdvBindingTreeWidgetItem import LdvBindingTreeWidgetItem
from LdvZEnvTreeWidgetItem import LdvZEnvTreeWidgetItem
from LdvEnvTreeWidgetItem import LdvEnvTreeWidgetItem
from LdvShotTreeWidgetItem import LdvShotTreeWidgetItem

import os
import json
import getpass

from dxstats import inc_tool_by_user

import crowd_grid

currentScriptPath = os.path.abspath(__file__)
srcPath = os.path.dirname(currentScriptPath)


def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)


class LookdevMain(QtWidgets.QWidget):
    def __init__(self, parent=getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # connect signal
        self.ui.loadBindingBtn.clicked.connect(self.clickBindingBtn)
        self.ui.scopeBtn.clicked.connect(self.clickscopeBtn)
        self.ui.turntableBtn.clicked.connect(self.clickedTurntableBtn)

        self.ui.assetTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.assetTreeWidget.customContextMenuRequested.connect(self.rmbAssetTreeContextMenu)

        self.ui.bindingTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.bindingTreeWidget.customContextMenuRequested.connect(self.rmbBindingTreeContextMenu)

        # plugin check
        pluginInfo = cmds.pluginInfo(query=True, listPlugins=True)

        if not "RenderMan_for_Maya" in pluginInfo:
            MessageBox(Message='unload renderman',
                       Button=["OK"])
            cmds.loadPlugin('RenderMan_for_Maya')

        if not 'AbcImport' in pluginInfo:
            MessageBox(Message='unload abcimport',
                       Button=["OK"])
            cmds.loadPlugin('AbcImport')

        print "MAYA_VERSION : ", MAYA_VERSION

        if not 'backstageLight' in pluginInfo:
            MessageBox(Message='unload backstageLight',
                       Button=["OK"])
            cmds.loadPlugin('backstageLight')

        if not 'backstageAsset' in pluginInfo:
            cmds.loadPlugin('backstageAsset')

        # initailize svariable
        self.hdriEnv = None
        self.binding = None
        self.AOVNode = None
        self.mainAsset = None

        ldvNode = cmds.ls(type = 'dxLdvNode')
        print "ldvNode :", ldvNode, len(ldvNode)
        if len(ldvNode) == 0:
            ldvNode = cmds.createNode('dxLdvNode')
            parentNode = cmds.listRelatives(ldvNode, parent=True)[0]
            # cmds.rename(parentNode, ldvNode + "Transform")
        else:

            # asset List
            ldvNode = ldvNode[0]
            size = cmds.getAttr('%s.ldvAssetPath' % ldvNode, size=True)
            print "ldvAssetSize :", size
            for index in range(size):
                filePath = cmds.getAttr('%s.ldvAssetPath[%d]' % (ldvNode, index))
                ldvItem = LdvAssetTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                                 filePath=filePath, attrIndex=index)

            size = cmds.getAttr('%s.ldvShotPath' % ldvNode, size=True)
            print "ldvShotSize :", size
            for index in range(size):
                filePath = cmds.getAttr('%s.ldvShotPath[%d]' % (ldvNode, index))
                ldvItem = LdvShotTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                                 filePath=filePath, attrIndex=index)

            size = cmds.getAttr('%s.ldvZEnvPath' % ldvNode, size=True)
            print "ldvZEnvSize :", size
            for index in range(size):
                filePath = cmds.getAttr('%s.ldvZEnvPath[%d]' % (ldvNode, index))
                ldvItem = LdvZEnvTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                                filePath=filePath, attrIndex=index)

            size = cmds.getAttr('%s.ldvEnvPath' % ldvNode, size=True)
            print "ldvEnvSize :", size
            for index in range(size):
                filePath = cmds.getAttr('%s.ldvEnvPath[%d]' % (ldvNode, index))
                ldvItem = LdvEnvTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                                filePath=filePath, attrIndex=index)

        inc_tool_by_user.run('action.LookdevTool.open', getpass.getuser())

    def clickBindingBtn(self):
        if self.binding != None:
            del self.binding

        self.binding = DynamicBinding()

        self.ui.bindingTreeWidget.clear()

        currentSetRule = self.binding.getCurrentRuleInfo()

        for item in currentSetRule:
            bindingItem = LdvBindingTreeWidgetItem(parent=self.ui.bindingTreeWidget,
                                                   xPath=item["XPath"],
                                                   shaderPay=item["shader"])

        for index in range(self.ui.assetTreeWidget.topLevelItemCount()):
            itemWidget = self.ui.assetTreeWidget.topLevelItem(index)
            for materialName in itemWidget.shaderInfo.shaderList:
                # if "Pxr" in cmds.nodeType(materialName):
                if "SHD" == materialName.split('_')[-1]:
                    tempObj = self.binding.setRule(assetName=itemWidget.getAssetName(),
                                                   material=materialName)

                    bindingItem = LdvBindingTreeWidgetItem(parent=self.ui.bindingTreeWidget,
                                                           xPath=tempObj["XPath"],
                                                           shaderPay=tempObj["shader"])

    def clickscopeBtn(self):
        for index in range(self.ui.assetTreeWidget.topLevelItemCount()):
            itemWidget = self.ui.assetTreeWidget.topLevelItem(index)

            self.setSaveScene(itemWidget=itemWidget)

        currentRule = []
        for index in range(self.ui.bindingTreeWidget.topLevelItemCount()):
            itemWidget = self.ui.bindingTreeWidget.topLevelItem(index)

            ruleItem = {}
            ruleItem["XPath"] = itemWidget.xPath
            ruleItem["shader"] = itemWidget.shaderPay

            if not ruleItem in currentRule:
                currentRule.append(ruleItem)

        self.binding.assignRule(currentRule)

        # render Setting
        cmds.setAttr("renderManRISGlobals.rman__riopt__Hider_maxsamples", 64)

        MessageBox(winTitle="Success",
                   Message="Success assign Rule",
                   Icon="information",
                   Button=["OK"])

        inc_tool_by_user.run('action.LookdevTool.Scope', getpass.getuser())

    def setSaveScene(self, itemWidget):
        saveFilePath = '%s/texture/pub/scenes/' % itemWidget.getAssetPath()
        print "saveFilePath :", saveFilePath
        if not os.path.isdir(saveFilePath):
            os.makedirs(saveFilePath)

        saveFileName = '%s_tex_lookdev_%s' % (itemWidget.getAssetName(), itemWidget.textureVersionComboBox.currentText())

        workCodeCount = 1
        for listdir in os.listdir(saveFilePath):
            if saveFileName in listdir:
                workCodeCount = workCodeCount + 1

        sceneFileName = '%s_%s.mb' % (saveFileName, "w" + str(workCodeCount).zfill(2))

        finalFileName = "{0}/{1}".format(saveFilePath, sceneFileName)

        mel.eval('file -rename "{0}"'.format(finalFileName))
        mel.eval('file -save;')

    def rmbAssetTreeContextMenu(self):
        menu = QtWidgets.QMenu(self)
        menu.addAction("== Add asset ==", self.assetContextAddAsset)
        menu.addAction("== Remove asset ==", self.assetContextRemoveAsset)
        menu.addAction("== Select main asset ==", self.assetContextSelectMainAsset)
        menu.addAction("== Add Zenv source ==", self.assetContextAddZenvSource)
        menu.addAction("== Add Env source data ==", self.assetContextAddEnvSource)
        menu.addAction("== Set Default Hdr ==", self.setEnvSetting)
        menu.addAction("== Set agent setting for select item ==", self.setAgentSetting)
        menu.exec_(QtGui.QCursor.pos())

    def assetContextAddZenvSource(self):
        sourceLoad = EnvSourceLoad(self, True)
        sourceLoad.exec_()

    def assetContextAddEnvSource(self):
        sourceLoad = EnvSourceLoad(self)
        sourceLoad.exec_()

    def setEnvSetting(self):
        if self.hdriEnv == None:
            self.hdriEnv = HDRISet()

        showNames = "\n".join(self.hdriEnv.showDataDic['show_information'].keys())
        showName, button = QtWidgets.QInputDialog.getText(self, "Select show name", showNames, QtWidgets.QLineEdit.Normal, "default")

        self.hdriEnv.getEnvData(showName)
        self.hdriEnv.setEnvLight(showName=showName)

        self.setAOV()

        self.createLookdevCamera()

    def loadShader(self):
        for index in range(self.ui.assetTreeWidget.topLevelItemCount()):
            self.ui.assetTreeWidget.topLevelItem(index).changeShader()

    def addZenvSourceFile(self, show, asset, assetSourceList):
        print "addZenvSourceFile", show, asset, assetSourceList

        for source in assetSourceList:
            projectName = "/show/{0}/asset/env/{1}/model/pub/zenv/abc/{2}".format(show, asset, source)

            ldvZEnvItem = LdvZEnvTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                              filePath=projectName)

    def addEnvSourceFile(self, show, asset, dataName, assetSourceList):
        print "addEnvSourceFile", show, asset, dataName, assetSourceList

        projectName = "/show/{0}/asset/env/{1}/model/pub/data/abc/{2}".format(show, asset, dataName)
        for source in assetSourceList:
            projectName = "/show/{0}/asset/env/{1}/model/pub/data/abc/{2}/{3}".format(show, asset, dataName, source)

            ldvEnvItem = LdvEnvTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                              filePath=projectName)

    def assetContextAddAsset(self):
        try:
            projectName = cmds.fileDialog2(fileMode=3,
                                           caption="select Asset",
                                           okCaption="Load Asset",
                                           startingDirectory=String.rootPath)[0]
        except:
            MessageBox(Message="not select asset",
                       Button=["OK"])
            return

        if projectName.startswith('/netapp/dexter/show'):
            projectName.replace('/netapp/dexter/show', '/show')

        if '/shot' in projectName:
            ldvItem = LdvShotTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                            filePath=projectName)
        else:
            ldvItem = LdvAssetTreeWidgetItem(parent=self.ui.assetTreeWidget,
                                             filePath=projectName)

        if ldvItem.isLoadFailed:
            # remove Item Widget
            pass

    def assetContextRemoveAsset(self):
        itemIndex = self.ui.assetTreeWidget.indexOfTopLevelItem(self.ui.assetTreeWidget.currentItem())
        removeItem = self.ui.assetTreeWidget.takeTopLevelItem(itemIndex)

        removeItem.deleteItemInfo()

    def assetContextSelectMainAsset(self):
        if self.mainAsset is not None:
            icon = QtGui.QIcon()
            self.mainAsset.setIcon(0, icon)

        icon = QtGui.QIcon()
        icon.addFile(os.path.join(srcPath, 'data/camera.png'))
        self.ui.assetTreeWidget.currentItem().setIcon(0, icon)

        self.mainAsset = self.ui.assetTreeWidget.currentItem()

        if self.hdriEnv == None:
            self.hdriEnv = HDRISet()

        self.hdriEnv.getEnvData(self.mainAsset.showName)
        self.createLookdevCamera(self.mainAsset.alembicFileName)
        self.lookdevHdriEnvSetting()
        self.hdriEnv.setEnvLight(showName=self.mainAsset.showName)

        self.setAOV()

        self.ui.assetTreeWidget.repaint()

    def lookdevHdriEnvSetting(self):
        projectSetPath = self.ui.assetTreeWidget.currentItem().rulebook.product["asset_path"]

        if not os.path.isdir(projectSetPath):
            projectSetPath = self.ui.assetTreeWidget.currentItem().rulebook.product["shot_path"]

        mel.eval('setProject "{0}"'.format(projectSetPath))

        # Create hdrChecker
        if not cmds.objExists("CHECKER_GRP"):
            self.createLookDevChecker()

    def getCenterPos(self, value1, value2):
        return (value2 - value1) * 0.5

    def max(self, value1, value2):
        if value1 > value2:
            return value1
        else:
            return value2

    def createLookdevCamera(self, assetName = ""):
        if cmds.objExists("camera_lookdev") == False:
            cmds.camera()
            cmds.rename("camera1", "camera_lookdev")

        f = open(os.path.join(os.path.dirname(__file__), "data", "camera.json"), "r")
        camData = json.load(f)
        for attr in camData.keys():
            try:
                cmds.setAttr("camera_lookdev.%s" % attr, camData[attr])
            except:
                print attr, camData[attr]
        f.close()

        if assetName:
            cmds.select(assetName)
            cmds.viewFit(['camera_lookdev'], f = 0.7)

        self.camX = cmds.getAttr('camera_lookdev.translateX')
        self.camY = cmds.getAttr('camera_lookdev.translateY')
        self.camZ = cmds.getAttr('camera_lookdev.translateZ')

        cmds.setAttr("camera_lookdev.farClipPlane", 100000)

        cmds.setAttr('perspShape.renderable', 0)
        cmds.setAttr('camera_lookdev.renderable', 1)

        cmds.setAttr("defaultResolution.width", 1920)
        cmds.setAttr("defaultResolution.height", 1080)
        cmds.setAttr("defaultResolution.deviceAspectRatio", 1.778)
        cmds.setAttr("defaultResolution.pixelAspect", 1.0)

    def createLookDevChecker(self):
        mel.eval('AbcImport -mode import "{0}"'.format(
            self.ui.assetTreeWidget.currentItem().rulebook.product['lookdev_path']))

        cmds.setAttr("LDV_CHECKER.translate", self.camX - 5, self.camY, self.camZ - 5)

    def createStupidAOVNode(self):
        aovNode = cmds.ls(type='dxAOV')
        if aovNode:
            cmds.select(aovNode)
            return aovNode[0]
        else:
            aovNode = cmds.createNode('dxAOV')
            if aovNode != "unknown1":
                return aovNode
            else:
                MessageBox(Message="please load Plugin StupidMan and restart Lookdev",
                           Button=["OK"])
                return None

    def setAOV(self):
        if self.AOVNode == None:
            self.AOVNode = self.createStupidAOVNode()

            if self.AOVNode == None:
                return

        # cmds.setAttr("{0}.diffuse".format(self.AOVNode), 1)
        # cmds.setAttr("{0}.indirectdiffuse".format(self.AOVNode), 1)
        # cmds.setAttr("{0}.specular".format(self.AOVNode), 1)

    def rmbBindingTreeContextMenu(self):
        pass

    def clickedTurntableBtn(self):
        msg = MessageBox(winTitle = "Confirm?",
                         Message='Do you setting turntable?\n[Recommend use only one mainAsset]',
                         Icon="information")

        if msg == "Cancel":
            return

        try:
            maxTime, button = QtWidgets.QInputDialog.getInt(self, "Frame Range", "input frame range [ex) 100 or 50]", 50)

            if button == False:
                return
            # maxTime = int(raw_input("input frame range [ex) 100 or 50]"))
        except:
            MessageBox(winTitle="frame range error",
                       Message='try input only number.',
                       Icon="critical",
                       Button=["OK"])
            return

        cmds.setAttr('defaultRenderGlobals.currentRenderer', 'renderManRIS', type='string')

        ### RFM SETTING
        mel.eval('rmanChangeRendererUpdate')

        renderAsset = []
        for index in range(self.ui.assetTreeWidget.topLevelItemCount()):
            itemWidget = self.ui.assetTreeWidget.topLevelItem(index)
            renderAsset.append(itemWidget.alembicFileName)
        # set frame range
        cmds.playbackOptions(minTime=1)
        cmds.playbackOptions(maxTime=maxTime)

        if not renderAsset:
            renderAsset = cmds.ls(sl = True)

        for asset in renderAsset:
            # set object Animation
            cmds.select(asset)

            cmds.currentTime(1)
            cmds.setAttr("%s.rotateY" % asset, 0)
            cmds.setKeyframe("%s.rotateY" % asset, time = 1)

            cmds.currentTime(maxTime / 2)
            cmds.setAttr("%s.rotateY" % asset, 360)
            cmds.setKeyframe("%s.rotateY" % asset, time = maxTime / 2)

        # set HDRI Animation
        hdriCurRotation = cmds.getAttr("%s.rotateY" % self.hdriEnv.hdriNode)
        cmds.setKeyframe("%s.rotateY" % self.hdriEnv.hdriNode, time=1)
        cmds.setKeyframe("%s.rotateY" % self.hdriEnv.hdriNode, time=(maxTime / 2) - 1)
        cmds.currentTime(maxTime / 2)
        cmds.setAttr("%s.rotateY" % self.hdriEnv.hdriNode, hdriCurRotation)
        cmds.setKeyframe("%s.rotateY" % self.hdriEnv.hdriNode, time = maxTime / 2)

        cmds.currentTime(maxTime)
        cmds.setAttr("%s.rotateY" % self.hdriEnv.hdriNode, hdriCurRotation + 360)
        cmds.setKeyframe("%s.rotateY" % self.hdriEnv.hdriNode, time = maxTime)

        cmds.keyTangent(itt='linear', ott='linear')

        self.setAOV()

        # create FloorPlane if boundingBox Y > 0:
        # if cmds.xform(self.mainAsset.alembicFileName, boundingBox=True, q=True)[1] < -5:
        #     print "not create Plane"
        # else:
        #     print "create Plane"
        #     if not cmds.objExists("pPlane1"):
        #         cmds.CreatePolygonPlane()
        #     nodeName = "pPlane1"
        #     cmds.setAttr('%s.scaleX' % nodeName, cmds.getAttr('%s.scaleX' % self.hdriEnv.hdriNode))
        #     cmds.setAttr('%s.scaleY' % nodeName, cmds.getAttr('%s.scaleY' % self.hdriEnv.hdriNode))
        #     cmds.setAttr('%s.scaleZ' % nodeName, cmds.getAttr('%s.scaleZ' % self.hdriEnv.hdriNode))
        #     print "nodeName :", nodeName
        #     cmds.select(nodeName)
        #     mel.eval('rmanChangeToRenderMan; rmanCreateHoldout;')

        cmds.setAttr("camera_lookdev.backgroundColor", 0.5, 0.5, 0.5, type='double3')

        msg = MessageBox(winTitle="Warning!?",
                         Message='Do you try turntable rendering?',
                         Icon="warning")

        if msg == "Cancel":
            return

        from RfM_Submitter import renderManRIS_script
        options = {
            'm_engine': "10.0.0.30",
            'm_user': getpass.getuser(),
            'm_maxactive': 1,
            'm_priority': 90,
            'm_minsamples' : 0,
            'm_maxsamples' : 64,
            'm_width': cmds.getAttr('defaultResolution.width'),
            'm_height': cmds.getAttr('defaultResolution.height'),
            'm_recovery' : 0
        }
        jobClass = renderManRIS_script.JobMain(options)
        jobClass.doIt()

    ############################################## AGENT LOOKDEV ##############################################

    def setAgentSetting(self):
        selectedItemList = self.ui.assetTreeWidget.selectedItems()

        # Agent Setting
        agentPoint, agentSourceGRP = crowd_grid.AgentLookDev()

        # move source list
        for item in selectedItemList:
            try:
                cmds.parent(item.alembicFileName, agentSourceGRP)
            except:
                pass

        dialog = QtWidgets.QInputDialog()

        dialog.setWindowTitle("Input Crowd Count")
        dialog.setLabelText("input crowd count [1~2000]")

        dialog.exec_()
        try:
            maxCount = int(dialog.textValue())
        except:
            return

        # Setup Agnet to points
        crowd_grid.AgentGridSetup(agentPoint, agentSourceGRP, maxCount)

        cmds.viewFit(['Agents'])
