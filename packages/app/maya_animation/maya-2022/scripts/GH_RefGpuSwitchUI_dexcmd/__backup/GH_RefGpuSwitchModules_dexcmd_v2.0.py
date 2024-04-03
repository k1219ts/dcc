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
    AllParents = [sel]

    selectedNum = int( re.search( '\d+', sel).group() )
    for i in range( 1, selectedNum ):
        newSel = sel.replace( str(selectedNum), str(i) )
        if findObjectType(newSel) == "gpuCache":
            AllParents.append(newSel)
        else:
            break

    return AllParents

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

# switch GPU to Ref

def switchRefGpu(sel, type = str()):

    gpuCacheFileName = list()
    refNodeNameDic = dict()

    if type == "gpuCache":
        for s in sel:
            selShape = cmds.listRelatives( s, s=1 )

            print selShape

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
