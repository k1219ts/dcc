__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import os
import sys
import GH_RefGpuSwitchModules_dexcmd as GH_RefGpuSwitchModules
reload(GH_RefGpuSwitchModules)

import maya.OpenMayaUI as mui
#import sip

#from PyQt4 import QtCore
#from PyQt4 import QtGui
#from PyQt4 import uic

from PySide.QtCore import *
from PySide.QtGui import *
import pysideuic
import xml.etree.ElementTree as xml
from cStringIO import StringIO
import shiboken

# ======================================================================================================================= #

UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GH_RefGpuSwitch.ui")
windowObject = "SwitchReference_v0.1"
dockMode = False

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QWidget)

def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' %form_class]
        base_class = eval('%s' % widget_class)

    return form_class, base_class

formclass, baseclass = loadUiType(UIFILE)

class GH_RefGpuSwitchUI(formclass, baseclass):
    def __init__(self, parent = getMayaWindow()):
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
        self.GPUCacheDIC = dict()
        self.ReferenceDIC = dict()

    def updateData(self):
        self.Stime = cmds.playbackOptions(q=1, min=1)
        self.Etime = cmds.playbackOptions(q=1, max=1)

        self.sceneFilePath = cmds.file(q=1, sceneName = 1)
        self.sceneFile = cmds.file(q=1, sceneName = 1, shortName=1).split(".")[0]
        self.gpuCacheDir = "%s/data/geoCache/%s" % ( os.sep.join( self.sceneFilePath.split(os.sep)[:-2] ), self.sceneFile )

    def GetItems(self):
        sel = cmds.ls(sl=1)
        if sel:
            for i in sel:
                objType = GH_RefGpuSwitchModules.findObjectType(objectName = i) # "gpuCache", "others"

                if objType == "gpuCache":
                    if i not in self.ReferenceDIC.keys():
                        gpuPath, attrQ = self.getAttrFromSel(i, "gpuCache")
                        self.ReferenceDIC[i] = attrQ
                else:
                    root = GH_RefGpuSwitchModules.findRoot(i)
                    if root not in self.GPUCacheDIC.keys():
                        refPath, refattrQ = self.getAttrFromSel(i, "RootNode")
                        self.GPUCacheDIC[root] = refattrQ
        else:
            print "select object first"

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

        GH_RefGpuSwitchModules.createCache(selList, Stime = int(self.Stime), Etime = int(self.Etime), gpuCacheDir = self.gpuCacheDir, enableSwitch = enableSwitchCondition)

        if enableSwitchCondition:
            GH_RefGpuSwitchModules.switchRefGpu(selList, "reference")

        self.refreshData()

    def DoSwitch(self):
        self.GetItems()
        RefselectedItems = self.GPUCacheDIC.keys()
        GPUselectedItems = self.ReferenceDIC.keys()

        if RefselectedItems:
            GH_RefGpuSwitchModules.switchRefGpu( RefselectedItems, "reference" )

        elif GPUselectedItems:
            GH_RefGpuSwitchModules.switchRefGpu( GPUselectedItems, "gpuCache" )


        self.refreshData()

        cmds.autoKeyframe(state = True)

def Doit():
    global GHRGS
    try:
        GHRGS.close()
    except:
        pass

    GHRGS = GH_RefGpuSwitchUI()

    if sys.platform != "darwin":
        fontPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "OpenSans-Regular.ttf")
        fontId = QFontDatabase.addApplicationFont(fontPath)
        if fontId is not -1:
            family = QFontDatabase.applicationFontFamilies(fontId)
            font = QFont(family[0])
            font.setPointSize(9)
            GHRGS.setFont(font)

    if dockMode:
        cmds.dockControl(label=windowObject, area="right", content = GHRGS, allowedArea=["left", "right"])
    else:
        GHRGS.show()
        GHRGS.resize(200, 100)
