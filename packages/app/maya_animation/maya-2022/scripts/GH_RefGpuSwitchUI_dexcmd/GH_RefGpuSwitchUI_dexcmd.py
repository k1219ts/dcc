__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import os
import sys
import GH_RefGpuSwitchModules_dexcmd as GH_RefGpuSwitchModules
reload(GH_RefGpuSwitchModules)

from PySide2 import QtCore, QtGui, QtWidgets
import dxUI

CURRENTDIR = os.path.dirname(os.path.abspath(__file__))
UIFILE = os.path.join(CURRENTDIR, "GH_RefGpuSwitch.ui")
windowObject = "SwitchReference_v2.0"

#ABSdirName = "OPNpos_0060_ani_v05"

ABSdirName = "_".join( cmds.file(q=1, sn=1, shn=1).split("_")[:4] )

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = GH_RefGpuSwitchUI()
    _win.show()

class GH_RefGpuSwitchUI(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(GH_RefGpuSwitchUI, self).__init__(parent)
        dxUI.setup_ui(UIFILE, self)
        self.setObjectName(windowObject)
        self.connectSlot()

        self.GPUCacheDIC = dict()
        self.ReferenceDIC = dict()

    def connectSlot(self):
        #self.CreateCacheButton.clicked.connect(self.DoCreateCache)
        self.SwitchButton.clicked.connect(self.DoSwitch)

    def refreshData(self):
        self.GPUCacheDIC = dict()
        self.ReferenceDIC = dict()

    def updateData(self):
        self.scenePath = os.path.dirname( cmds.file(q=1, sn=1) )
        self.progressPath = os.sep.join(self.scenePath.split("/")[:-1])
        self.cachePathName = os.path.join(self.progressPath, "data/geoCache/%s" %ABSdirName)

        self.Stime = cmds.playbackOptions(q=1, min=1)
        self.Etime = cmds.playbackOptions(q=1, max=1)

    def getNodeList(self, object = list()):
        self.nodeSwitchDataDIC = dict()

        switchType = "Ref"

        for _obj in object:
            dxNode = _obj

            while (True):
                rootChild = cmds.listRelatives(dxNode, c=1, type="transform")

                if cmds.nodeType(dxNode) == "dxAbcArchive":
                    break;
                elif rootChild:
                    if cmds.nodeType(rootChild[0]) == "dxAbcArchive":
                        dxNode = rootChild
                        break;
                p = cmds.listRelatives(dxNode, ap=1)
                if cmds.nodeType(dxNode) == "dxRig":
                    break;
                elif p:
                    if cmds.nodeType(p[0]) == "dxRig":
                        dxNode = p[0]
                        break;
                elif not p and cmds.nodeType(dxNode) != "dxRig":
                    raise TypeError("please select rig object")

                dxNode = p[0]

            #_objectShape = cmds.listRelatives(_obj, s=1)
            gpuCacheNode = None
            childs = cmds.listRelatives(dxNode, c=1)
            
            for child in childs:
                childType = cmds.nodeType(child)
                childShape = cmds.listRelatives(child, s=1)
                
                if childType and childType == "dxAbcArchive":
                    gpuCacheNode = child
                    print gpuCacheNode + " dxAbcArchive"
                elif childType and childType == "gpuCache":
                    gpuCacheNode = cmds.listRelatives(child, p=1)[0]
                    print gpuCacheNode + " gpuCache_b"
                elif childShape and childShape == "gpuCache":
                    gpuCacheNode = cmds.listRelatives(child, p=1)[0]
                    print gpuCacheNode + " gpuCache_c"

            if gpuCacheNode != None:
            #if _objectShape and cmds.objectType(_objectShape[0]) == "gpuCache":

                # query reference Node Name
                """
                refs = cmds.ls(references=1)

                gpucacheNameSpc = cmds.ls(_obj, showNamespace = 1)[1]

                refNodeDic = dict()

                for referenceNodeName in refs:
                    if (referenceNodeName.find("_UNKNOWN_") == -1) and ( referenceNodeName != 'a_1_elephant4RN'):
                        _nameSpc = cmds.referenceQuery(referenceNodeName, ns=1)[1:]
                        refNodeDic[_nameSpc] = dict()
                        refNodeDic[_nameSpc]["refNodeName"] = referenceNodeName
                        refNodeDic[_nameSpc]["isLoaded"] = cmds.referenceQuery(referenceNodeName, il=1)

                if refNodeDic.has_key(gpucacheNameSpc):
                    self.ReferenceDIC[_obj] = refNodeDic[gpucacheNameSpc]["refNodeName"]
                """
                #refNodeName = cmds.getAttr(_obj + ".dxRigRefPath")
                refNodeName = cmds.getAttr(gpuCacheNode + ".dxRigRefPath")

                print refNodeName

                #self.ReferenceDIC[_obj] = refNodeName
                self.nodeSwitchDataDIC[gpuCacheNode] = refNodeName

                switchType = "gpuCache"
            else:
                abcPath = os.path.join( self.cachePathName, dxNode + ".abc" )
                print abcPath

                if os.path.exists(abcPath):
                    self.nodeSwitchDataDIC[dxNode] = abcPath
                    #self.GPUCacheDIC[dxNode] = abcPath
                else:
                    self.nodeSwitchDataDIC[dxNode] = None
                    #self.GPUCacheDIC[dxNode] = None
                switchType = "Ref"

        return switchType

    def DoSwitch(self):
        self.updateData()
        selNode = cmds.ls(sl=1)
        switchType = self.getNodeList(selNode)
        GH_RefGpuSwitchModules.switchRefGpu(self.nodeSwitchDataDIC, type=switchType)
