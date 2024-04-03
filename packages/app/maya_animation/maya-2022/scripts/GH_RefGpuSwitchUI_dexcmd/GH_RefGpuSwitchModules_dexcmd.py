__author__ = 'gyeongheon.jeong'

import re
import maya.cmds as cmds
# import dexcmd.aniPub as aniPub
#import dexcmd.dexAlembic as aniPub

#reload(aniPub)
import cacheCommon


def findRoot(sel=str()):
    rootParent = sel
    while (True):
        p = cmds.listRelatives(rootParent, ap=1)
        if not p:
            break;
        rootParent = p[0]
    return rootParent


def findObjectType(objectName=str()):
    objectType = "others"
    objShape = cmds.listRelatives(objectName, s=1)
    if not objShape:
        objectType = "Root"
    elif objShape and (cmds.objectType(objShape[0]) == "gpuCache"):
        objectType = "gpuCache"
    return objectType


def findRenderMesh(selObj):
    renderMeshList = list()
    if cmds.attributeQuery(selObj + "renderMesh"):
        renderMeshList = cmds.getAttr(selObj + ".renderMesh")
    return renderMeshList


def scriptJobcmd(nodeName, AttrName, _type):
    jobCmd = "import GH_RefGpuSwitchUI_dexcmd.GH_RefGpuSwitchModules_dexcmd as GHRGSmodule\nGHRGSmodule.switchRefGpu(['%s'], '%s')" % (
    nodeName, _type)
    jobNum = cmds.scriptJob(runOnce=True, attributeChange=['%s.%s' % (nodeName, AttrName), jobCmd])
    return jobNum


def selectParents(sel_object):
    sel = sel_object
    AllParents = [sel]

    selectedNum = int(re.search('\d+', sel).group())
    for i in range(1, selectedNum):
        newSel = sel.replace(str(selectedNum), str(i))
        if findObjectType(newSel) == "gpuCache":
            AllParents.append(newSel)
        else:
            break

    return AllParents


# switch GPU to Ref

def switchRefGpu(selDic=dict(), type=str()):
    gpuCacheFileName = list()
    refNodeNameDic = dict()

    if type == "gpuCache":
        for gpuCacheNodeName in selDic.keys():
            gpuCacheArcName = cmds.listRelatives(gpuCacheNodeName, s=1)
            print gpuCacheArcName

            # switch GPU to Ref
            if gpuCacheArcName and (cmds.objectType(gpuCacheArcName) == "gpuCache"):
                WorldCON = findRoot(gpuCacheNodeName)
                if WorldCON:
                    cmds.delete(WorldCON)
                else:
                    cmds.delete(gpuCacheNodeName)
                cmds.file(loadReference=selDic[gpuCacheNodeName])
    else:
        # switch Ref to GPU
        for dxRigNodeName in selDic.keys():
            refNodeName = cmds.referenceQuery(dxRigNodeName, referenceNode=True)
            cmds.file(unloadReference=refNodeName)
            refNodeNameDic[dxRigNodeName] = refNodeName
            if selDic[dxRigNodeName] != None:
                gpuCacheFileName.append(selDic[dxRigNodeName])

        abcClass = aniPub.abcImport(Path=gpuCacheFileName, Mode='gpumode', MeshType='Low', FitTime=False)
        # abcClass = abcImportLocal( Path = gpuCacheFileName, Mode='gpumode', FitTime=False)
        abcClass.doIt()

        # gpuNodeList = abcClass.abcArcNodeList

        for gpuNode in selDic.keys():
            cmds.addAttr(gpuNode, longName="dxRigRefPath", dataType="string")
            cmds.setAttr(gpuNode + ".dxRigRefPath", refNodeNameDic[gpuNode], type="string")
