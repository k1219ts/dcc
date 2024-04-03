__author__ = 'gyeongheon.jeong'

import re
import maya.cmds as cmds
#import dexcmd.aniPub as aniPub
import dexcmd.dexAlembic as aniPub
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

    gpuexportClass = aniPub.abcExport( FilePath = gpuCacheDir,
                                         Nodes = selList,
                                         Start = Stime, End = Etime, Step = 1, Just = True,
                                         World = True)
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

        #abcClass = aniPub.abcImport( Path = gpuCacheFileName, Mode='gpumode', FitTime=False)
        abcClass = abcImportLocal( Path = gpuCacheFileName, Mode='gpumode', FitTime=False)
        abcClass.doIt()

        gpuNodeList = abcClass.abcArcNodeList

        for gpuNode in gpuNodeList:
            #gpuTransform = cmds.listRelatives(gpuShape, p=1, type = "transform")[0]
            #gpuTransform = findRoot(gpuShape)
            cmds.addAttr(gpuNode, longName = "OriRefPath", dataType = "string")
            cmds.setAttr(gpuNode + ".OriRefPath", refNodeNameDic[ gpuNode ] + "," + gpuNode, type = "string")


import os
import time

# for maya2016
try:
	from alembic.AbcCoreAbstract import *
	from alembic.Abc import *
	from alembic.AbcGeom import *
	from alembic.Util import *
	kWrapExisting = WrapExistingFlag.kWrapExisting
except:
	pass

import maya.mel as mel

import dexcmd.dexCommon as dexCommon

class abcImportLocal(aniPub.abcImport):

    abcArcNodeList = list()

    def doIt( self ):
        startTime = time.time()

        self.getFiles()

        for f in self.m_importFiles:
            baseName = os.path.basename(f)
            splitVer = re.compile( r'_v\d+.abc' ).findall( baseName )
            if splitVer:
                nodeName = baseName.split(splitVer[0])[0]
            else:
                nodeName = baseName.split( '.abc' )[0]

            abcArcNode = cmds.createNode( 'dxAbcArchive', n=nodeName )
            cmds.setAttr( '%s.abcFileName' % abcArcNode, f, type='string' )
            cmds.setAttr( '%s.mode' % abcArcNode, self.m_mode )
            cmds.setAttr( '%s.display' % abcArcNode, self.m_display )
            if self.m_world:
                rfn   = dexCommon.get_reloadFileName( f, 1 )
                wfile = rfn.replace( '.abc', '.world' )
                if os.path.exists( wfile ):
                    cmds.setAttr( '%s.worldFileName' % abcArcNode, wfile, type='string' )

            abcArcClass = dexCommon.dxAbcArchive( abcArcNode )
            if self.m_fitTime:
                abcArcClass.m_fitTime = True
            abcArcClass.doIt()
            self.m_dxAbcNodes.append( abcArcClass.m_curNode )
            self.m_dxArcNodes.append( abcArcClass.m_arcNode )

            if self.m_fitTime:
                self.fitTimeRange( abcArcClass.m_startFrame, abcArcClass.m_endFrame )

            self.abcArcNodeList.append(abcArcNode)

        endTime = time.time()
        mel.eval( 'print "# Result : %.2f sec\\n"' % (endTime-startTime) )
