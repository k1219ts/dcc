__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import os
import sys
import GH_RefGpuSwitchModules_dexcmd as GH_RefGpuSwitchModules
reload(GH_RefGpuSwitchModules)

import maya.OpenMayaUI as omu
import sip

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4 import uic

# ======================================================================================================================= #

UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GH_RefGpuSwitch.ui")
windowObject = "SwitchReference_v0.1"
dockMode = False

def get_maya_window():
    ptr = omu.MQtUtil.mainWindow()
    if ptr is not None:
        return sip.wrapinstance(long(ptr), QtCore.QObject)

form_class, base_class = uic.loadUiType(UIFILE)

class GH_RefGpuSwitchUI(form_class, base_class):
    def __init__(self, parent = get_maya_window()):
        super(GH_RefGpuSwitchUI, self).__init__(parent)
        self.setupUi(self)
        self.setObjectName(windowObject)
        self.initGUI()
        self.connectSlot()

        self.GPUCacheDIC = dict()
        self.ReferenceDIC = dict()

    def initGUI(self):
        pass

    def connectSlot(self):
        self.CreateCacheButton.clicked.connect(self.DoCreateCache)
        self.SwitchButton.clicked.connect(self.DoSwitch)

    def refreshData(self):
        #self.ReferencelistView.clear()
        #self.GPUcachelistView.clear()
        #self.isGPUcachelistView.clear()
        #self.isReferencelistView.clear()
        self.GPUCacheDIC = dict()
        self.ReferenceDIC = dict()

    def updateData(self):
        self.Stime = cmds.playbackOptions(q=1, min=1)
        self.Etime = cmds.playbackOptions(q=1, max=1)

        self.sceneFilePath = cmds.file(q=1, sceneName = 1)
        self.sceneFile = cmds.file(q=1, sceneName = 1, shortName=1).split(".")[0]
        self.gpuCacheDir = "%s/data/geoCache/%s" % ( os.sep.join( self.sceneFilePath.split(os.sep)[:-2] ), self.sceneFile )

    def AddItems(self):
        sel = cmds.ls(sl=1)
        if sel:
            for i in sel:
                objType = GH_RefGpuSwitchModules.findObjectType(objectName = i) # "gpuCache", "others"

                if objType == "gpuCache":
                    if i not in self.ReferenceDIC.keys():
                        gpuPath, attrQ = self.getAttrFromSel(i, "gpuCache")
                        self.GPUcachelistView.addItems([ i ])
                        self.isReferencelistView.addItems([attrQ])
                        self.ReferenceDIC[i] = attrQ
                else:
                    root = GH_RefGpuSwitchModules.findRoot(i)
                    if root not in self.GPUCacheDIC.keys():
                        refPath, refattrQ = self.getAttrFromSel(i, "RootNode")
                        self.ReferencelistView.addItems([ root ])
                        self.isGPUcachelistView.addItems([ refattrQ ])
                        self.GPUCacheDIC[root] = refattrQ
        else:
            print "select object first"

    def GetItems(self):
        sel = cmds.ls(sl=1)
        if sel:
            for i in sel:
                objType = GH_RefGpuSwitchModules.findObjectType(objectName = i) # "gpuCache", "others"

                if objType == "gpuCache":
                    if i not in self.ReferenceDIC.keys():
                        gpuPath, attrQ = self.getAttrFromSel(i, "gpuCache")
                        #self.GPUcachelistView.addItems([ i ])
                        #self.isReferencelistView.addItems([attrQ])
                        self.ReferenceDIC[i] = attrQ
                else:
                    root = GH_RefGpuSwitchModules.findRoot(i)
                    if root not in self.GPUCacheDIC.keys():
                        refPath, refattrQ = self.getAttrFromSel(i, "RootNode")
                        #self.ReferencelistView.addItems([ root ])
                        #self.isGPUcachelistView.addItems([ refattrQ ])
                        self.GPUCacheDIC[root] = refattrQ
        else:
            print "select object first"

    def RemoveItems(self):
        itemList = list()
        RefselectedItems = self.ReferencelistView.selectedItems()
        GPUselectedItems = self.GPUcachelistView.selectedItems()

        if RefselectedItems:
            itemList = RefselectedItems

            for item in itemList:
                self.GPUCacheDIC.pop(str(item.text()))

        elif GPUselectedItems:
            itemList = GPUselectedItems

            for item in itemList:
                self.ReferenceDIC.pop(str(item.text()))

        self.ReferencelistView.clear()
        self.ReferencelistView.addItems(self.GPUCacheDIC.keys())
        self.isGPUcachelistView.clear()
        self.isGPUcachelistView.addItems(self.GPUCacheDIC.values())

        self.GPUcachelistView.clear()
        self.GPUcachelistView.addItems(self.ReferenceDIC.keys())
        self.isReferencelistView.clear()
        self.isReferencelistView.addItems(self.ReferenceDIC.values())

    def getAttrFromSel(self, sel, selType):
        AttrPath = str()

        if selType == "RootNode" and cmds.attributeQuery("gpuCachePath", n = sel, ex=True):
                AttrPath = cmds.getAttr(sel + ".gpuCachePath").split("/")[-1] + ".abc"

        elif selType == "gpuCache" and cmds.attributeQuery("OriRefPath", n = sel, ex=True):
                AttrPath = cmds.getAttr(sel + ".OriRefPath")

        if not AttrPath:
            AttrQ = "X"
        else:
            AttrQ = "O"

        return AttrPath, AttrQ

    def DoCreateCache(self):
        self.GetItems()
        self.updateData()
        selList = self.GPUCacheDIC.keys()

        enableSwitchCondition = self.switchToGPUcheckBox.isChecked()

        GH_RefGpuSwitchModules.createCache(selList, Stime = self.Stime, Etime = self.Etime, gpuCacheDir = self.gpuCacheDir, enableSwitch = enableSwitchCondition)

        if enableSwitchCondition:
            GH_RefGpuSwitchModules.switchRefGpu(selList, "reference")

        self.refreshData()

    def DoSwitch(self):
        self.GetItems()
        #RefselectedItems = self.ReferencelistView.selectedItems()
        #GPUselectedItems = self.GPUcachelistView.selectedItems()
        RefselectedItems = self.GPUCacheDIC.keys()
        GPUselectedItems = self.ReferenceDIC.keys()

        #RefselectedItemsSTR = [str( i.text() ) for i in RefselectedItems]
        #GPUselectedItemsSTR = [str( i.text() ) for i in GPUselectedItems]

        if RefselectedItems:
            GH_RefGpuSwitchModules.switchRefGpu( RefselectedItems, "reference" )

        elif GPUselectedItems:
            GPUselectedItemWithParents = list()

            for gpu in GPUselectedItems:
                allParents = GH_RefGpuSwitchModules.selectParents(gpu)
                GPUselectedItemWithParents += allParents

            GH_RefGpuSwitchModules.switchRefGpu( GPUselectedItemWithParents, "gpuCache" )

        self.refreshData()

def Doit():
    global GHRGS
    try:
        GHRGS.close()
    except:
        pass

    GHRGS = GH_RefGpuSwitchUI()

    if sys.platform != "darwin":
        fontPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "OpenSans-Regular.ttf")
        fontId = QtGui.QFontDatabase.addApplicationFont(fontPath)
        if fontId is not -1:
            family = QtGui.QFontDatabase.applicationFontFamilies(fontId)
            font = QtGui.QFont(family[0])
            font.setPointSize(9)
            GHRGS.setFont(font)

    if dockMode:
        cmds.dockControl(label=windowObject, area="right", content = GHRGS, allowedArea=["left", "right"])
    else:
        GHRGS.show()
        GHRGS.resize(200, 100)
