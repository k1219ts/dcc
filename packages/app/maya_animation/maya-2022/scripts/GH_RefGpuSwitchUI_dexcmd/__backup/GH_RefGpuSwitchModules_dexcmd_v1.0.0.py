__author__ = 'gyeongheon.jeong'

import re
import maya.cmds as cmds
import dexcmd.aniPub as aniPub
reload(aniPub)

def findRoot(sel = str()):
    rootParent = sel
    while (True):
        p = cmds.listRelatives(rootParent, ap=1)
        if not p:
            break;
        rootParent = p[0]

    return rootParent

def findObjectType( objectName = str() ):
    objectType = "others"
    objShape = cmds.listRelatives(objectName, s=1)

    if not objShape:
        objectType = "Root"
    elif objShape and ( cmds.objectType(objShape[0]) == "gpuCache" ):
        objectType = "gpuCache"

    return objectType

def findRenderMesh(selObj):
    renderMeshList = list()

    if cmds.attributeQuery(selObj + "renderMesh"):
        renderMeshList = cmds.getAttr(selObj + ".renderMesh")

    return renderMeshList

def scriptJobcmd(nodeName, AttrName, _type):
    jobCmd = "import GH_RefGpuSwitchUI_dexcmd.GH_RefGpuSwitchModules_dexcmd as GHRGSmodule\nGHRGSmodule.switchRefGpu(['%s'], '%s')" % (nodeName, _type)
    jobNum = cmds.scriptJob( runOnce=True, attributeChange=['%s.%s' % (nodeName, AttrName), jobCmd] )
    return jobNum

def selectParents(sel_object):
    sel = sel_object
    switchNum = cmds.getAttr("%s.selectparents" % sel)
    if switchNum:
        selectedNum = int( re.search( '\d+', sel).group() )
        for i in range( 1, selectedNum ):
            newSel = sel.replace( str(selectedNum), str(i) )
            if findObjectType(i) == "gpuCache":
                cmds.select(newSel, add=1)
            else:
                break
    else:
        cmds.select(sel, r=False)

def createCache(selList = list(), Stime = int(), Etime = int(), gpuCacheDir = str(), enableSwitch = True):
    """
    import os
    cmd = 'mayapy'
    os.system("gnome-terminal -e '%s'" % cmd)
    """
    panels = cmds.getPanel( type='modelPanel' )

    for Panel in panels:
        cmds.isolateSelect(Panel, state = 1)

    gpuexportClass = aniPub.dxAbcExport( Path = gpuCacheDir, Root = selList, Start = Stime, End = Etime )
    gpuexportClass.doIt()

    for Panel in panels:
        cmds.isolateSelect(Panel,state = 0)
    for selObj in selList:
        if not cmds.attributeQuery("gpuCachePath", node = selObj, ex=True):
            cmds.addAttr(selObj, longName = "gpuCachePath", dataType = "string")

        cmds.setAttr( "%s.gpuCachePath" % selObj, "/".join( [gpuCacheDir, selObj + ".abc"] ), type = "string" )

        if enableSwitch:
            if not cmds.attributeQuery("toGPUcache", node = selObj, ex=True):
                cmds.addAttr(selObj, ln = "toGPUcache", at = "long", min = 0, max = 1, dv = 0)

            cmds.setAttr(selObj + ".toGPUcache", e = True, channelBox = True)

            #scriptJobcmd(selObj, "toGPUcache", "reference")
        #switchRefGpu(selObj)

# switch GPU to Ref

def switchRefGpu(sel = list(), type = str()):

    gpuCacheFileName = list()
    refNodeNameDic = dict()

    if type == "gpuCache":
        for s in sel:
            selShape = cmds.listRelatives( s, s=1 )

            # switch GPU to Ref
            if selShape and ( cmds.objectType(selShape) == "gpuCache" ):
                refNodeNameFromGpuCC = cmds.getAttr(s + ".OriRefPath").split(",")

                gpuNode = refNodeNameFromGpuCC[0]
                refGrpName = refNodeNameFromGpuCC[1]

                Wloc = findRoot(s)

                if Wloc:
                    cmds.delete(Wloc)
                else:
                    cmds.delete(s)

                cmds.file(loadReference = gpuNode)
                cmds.setAttr(refGrpName + ".toGPUcache", 0)
                #scriptJobcmd(refGrpName, "toGPUcache", "reference")
    else:
        # switch Ref to GPU
        for GrpName in sel:
            gpuCacheFileName.append( cmds.getAttr("%s.gpuCachePath" % GrpName) )

            refNodeName = cmds.referenceQuery( GrpName, referenceNode=True )
            cmds.file(unloadReference = refNodeName)
            refNodeNameDic[GrpName] = refNodeName

        abcClass = aniPub.dxAbcImport( Path = gpuCacheFileName, Mode='gpumode', FitTime=False)
        abcClass.doIt()

        gpuShapeList = abcClass.gpuNodes

        for gpuShape in gpuShapeList:
            gpuTransform = cmds.listRelatives(gpuShape, p=1, type = "transform")[0]
            cmds.addAttr(gpuTransform, longName = "OriRefPath", dataType = "string")
            cmds.setAttr(gpuTransform + ".OriRefPath", refNodeNameDic[ gpuTransform ] + "," + gpuTransform, type = "string")
            cmds.addAttr(gpuTransform, ln = "toReference", at = "long", min = 0, max = 1, dv = 0)
            cmds.setAttr(gpuTransform + ".toReference", e = True, channelBox = True)

#            if gpuTransform.find("dajiTail") != -1:
#                cmds.addAttr(gpuTransform, ln = "selectparents", at = "long", min = 0, max = 1, dv = 0)
#                cmds.setAttr(gpuTransform + ".selectparents", e = True, channelBox = True)
#
#                string = """import re\nsel = cmds.ls(sl=1)[0]\nswitchNum = cmds.getAttr("%s.selectparents" % sel)\nif switchNum:\n\tselectedNum = int( re.search( '\d+', sel).group() )\n\tfor i in range( 1, selectedNum ):\n\t\tnewSel = sel.replace( str(selectedNum), str(i) )\n\t\tcmds.select(newSel, add=1)\nelse:\n\tcmds.select(sel, r=False)"""
#
#                cmds.scriptJob( attributeChange=['%s.selectparents' % gpuTransform, string] )
#
#            scriptJobcmd(gpuTransform, "toReference", "gpuCache")

"""
import os, sys
import time
import string
import subprocess
import json
import shutil

import dexcmd.dexCommon as dexCommon
import dexcmd.batchCommon

import maya.mel as mel

class aniPubLocal(aniPub.dxAbcImport):
    def import_gpu( self ):
        gpuNode = aniPub.getGpuNode( self.GpuNodeType )

        for i in self.importMeshFiles:
            basename = os.path.splitext( os.path.basename(i) )[0]
            basename = basename.replace( '_sim', '' )
            basename = basename.replace( '_low', '' )
            if cmds.objExists(basename) and cmds.nodeType(basename)==gpuNode:
                self.gpu = cmds.ls( basename, dag=True, type=gpuNode )[0]
                cmds.setAttr( '%s.cacheFileName' % self.gpuShape, i, type='string' )
                # debug
                mel.eval( 'print "# Result : %s import set file <%s>\\n"' % (gpuNode, i) )
            else:
                self.gpuShape = cmds.createNode( gpuNode, n='%sShape' % basename )
                cmds.setAttr( '%s.visibleInReflections' % self.gpuShape, 1 )
                cmds.setAttr( '%s.visibleInRefractions' % self.gpuShape, 1 )
                if self.DrawingBBox:
                    cmds.setAttr( '%s.overrideEnabled' % self.gpuShape, 1 )
                    cmds.setAttr( '%s.overrideLevelOfDetail' % self.gpuShape, 1 )
                cmds.setAttr( '%s.cacheFileName' % self.gpuShape, i, type='string' )

                self.addGpuRenderAttributes( self.gpuShape )
                # debug
                mel.eval( 'print "# Result : %s import <%s>\\n"' % (gpuNode, i) )

            # fit time range
            if self.FitTime:
            timeRange = eval( 'cmds.%s( "%s", q=True, animTimeRange=True )' % (gpuNode, self.gpuShape) )
            cmds.playbackOptions( minTime=timeRange[0] )
            cmds.playbackOptions( maxTime=timeRange[1] )
            cmds.playbackOptions( animationStartTime=timeRange[0] )
            cmds.playbackOptions( animationEndTime=timeRange[1] )
"""