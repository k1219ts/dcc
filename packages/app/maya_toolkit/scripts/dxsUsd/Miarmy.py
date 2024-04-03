'''
Miarmy cache to USD
'''

import os, sys
import optparse
import math
import string
import glob

import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as OpenMaya

# Miarmy Modules
try:
    import McdGeneral
    import McdSimpleCmd
    import McdPlacementFunctions
    import McdRenderFBXFunctions
    import McdMeshDriveSetup
except:
    pass

from pxr import Usd, Sdf

import dxsMsg
import PathUtils
import dxsMayaUtils
import GeomSkel


def InitializeMiarmy(filename):
    cmds.file(filename, force=True, open=True)
    mel.eval('McdInitMiarmy;')
    McdPlacementFunctions.placementAgent()
    cmds.currentTime(1)

    globalNode = McdGeneral.McdGetMcdGlobalNode()
    enableCache= cmds.getAttr('%s.enableCache' % globalNode)
    if not enableCache:
        dxsMsg.Print('error', 'enableCache is off')
        return False
    return True

def GetCacheInfo():
    globalNode = McdGeneral.McdGetMcdGlobalNode()
    cacheFolder= cmds.getAttr('%s.cacheFolder' % globalNode)
    cacheName  = cmds.getAttr('%s.cacheName' % globalNode)
    frames = list()
    for f in os.listdir(cacheFolder):
        if f.find(cacheName) > -1 and f.find('.mmc') > -1:
            fsplit = f.split('.')
            frames.append(int(fsplit[-2]))
    frames.sort()
    return cacheFolder, cacheName, (frames[0], frames[-1])

def CheckMeshDriveCache():
    globalNode = McdGeneral.McdGetMcdGlobalNode()
    cacheFolder     = cmds.getAttr('%s.cacheFolder' % globalNode)
    meshDriveFolder = cmds.getAttr('%s.outMD2Folder' % globalNode)

    if not meshDriveFolder:
        return False
    if not os.path.exists(meshDriveFolder):
        return False
    if not cacheFolder in meshDriveFolder:
        return False
    return True

def InitMeshDrive():
    globalNode= McdGeneral.McdGetMcdGlobalNode()
    rawPath   = cmds.getAttr('%s.outMD2Folder' % globalNode)
    if not rawPath:
        return
    if not os.path.exists(rawPath):
        return

    McdPlacementFunctions.dePlacementAgent()

    #--------------------------------------------------------------------------- McdMeshDriveSetup.MDDuplicate Modified
    md3f = McdGeneral.envPath2AbsPath(rawPath)
    md3n = cmds.getAttr(globalNode + ".outMD2Name")

    dupGeomGrp = cmds.ls("MDGGrp_*", l = True)
    McdMeshDriveSetup.McdMeshDrive2Clear()

    renameCommand = '\
        proc prefixNode(string $prefix, string $node){\
            string $isType[]	= `ls -type transform $node`;\
            if (size($isType) > 0 ) {\
                string $nodeName = `substitute ".*|" $node ""`;\
                string $newName = `rename $node ( $prefix + $nodeName )`;\
            }\
        }\
        string $prefix = "MDG_";\
        string $currentNodes[] = eval("listRelatives -pa -ad `ls -sl -l`");\
        if ( size( $currentNodes ) > 0 ) {\
            for( $i=0; $i < size( $currentNodes ); $i++ ) {\
                prefixNode( $prefix, $currentNodes[$i] );\
            }\
        }\
        $currentNodes = `ls -sl -l`;\
        if ( size( $currentNodes ) > 0 ) {\
            for( $i=0; $i < size( $currentNodes ); $i++ ) {\
                prefixNode( $prefix, $currentNodes[$i] );\
            }\
        }\
    '
    renamePrefixCommandPre = '\
        proc prefixNode(string $prefix, string $node){\
            string $isType[]	= `ls -type transform $node`;\
            if (size($isType) > 0 ) {\
                string $nodeName = `substitute ".*|" $node ""`;\
                string $newName = `rename $node ( $prefix + $nodeName )`;\
            }\
        }\
        string $prefix = "'

    renamePrefixCommandPost = '";\
        string $currentNodes[] = eval("listRelatives -pa -ad `ls -sl -l`");\
        if ( size( $currentNodes ) > 0 ) {\
            for( $i=0; $i < size( $currentNodes ); $i++ ) {\
                prefixNode( $prefix, $currentNodes[$i] );\
            }\
        }\
        $currentNodes = `ls -sl -l`;\
        if ( size( $currentNodes ) > 0 ) {\
            for( $i=0; $i < size( $currentNodes ); $i++ ) {\
                prefixNode( $prefix, $currentNodes[$i] );\
            }\
        }\
    '

    # duplicate the geometry_<agt>
    allAgtGrpNode = cmds.ls(type = "McdAgentGroup")
    if allAgtGrpNode == [] or allAgtGrpNode == None:
        return

    mainDupGeomGrp = cmds.ls("MDG_MDG_Geometry_*", l = True)
    if mainDupGeomGrp != [] and mainDupGeomGrp != None:
        for i in range(len(mainDupGeomGrp)):
            try:
                cmds.delete(mainDupGeomGrp[i])
            except:
                pass

    oddRecordList = [[],[]]

    for i in range(len(allAgtGrpNode)):
        allAgtGrpChd = cmds.listRelatives(allAgtGrpNode[i], c = True)
        if allAgtGrpChd == [] or allAgtGrpChd == None:
            continue;
        for j in range(len(allAgtGrpChd)):
            try:
                if allAgtGrpChd[j].find("Geometry_") == 0:
                    dupNode = cmds.ls("MDG_MDG_" + allAgtGrpChd[j])
                    if dupNode != [] and dupNode != None:
                        continue;

                    if McdGeneral.McdCheckSubNodesNaming(allAgtGrpChd[j]) != 0:
                        if McdGeneral.McdCheckSubNodesNaming(allAgtGrpChd[j]) == 2:
                            continue
                        else:
                            return

                    dupNode = cmds.duplicate(allAgtGrpChd[j], name = "MDG_" + allAgtGrpChd[j])
                    McdMeshDriveSetup.McdCheckShapeNodeInHi(dupNode[0])
                    cmds.hide(dupNode[0])
                    cmds.parent(dupNode[0], w = True)

                    oddRecord = McdGeneral.McdCheckAndFixName(allAgtGrpChd[j], dupNode)
                    oddRecordList[0].extend(oddRecord[0])
                    oddRecordList[1].extend(oddRecord[1])

                    cmds.select(clear = True)
                    cmds.select("MDG_" + allAgtGrpChd[j])
                    mel.eval(renameCommand)

                    # re-check the hierarchy:
                    allDupNodes = cmds.listRelatives(ad = True, path = True)
                    for k in range(len(allDupNodes)):
                        if cmds.getAttr(allDupNodes[k] + ".intermediateObject") == 0:
                            if allDupNodes[k].find("|") >= 0:
                                realName = allDupNodes[k].split("|")[-1]
                                if realName.find("MDG_") != 0:
                                    cmds.rename(allDupNodes[k], "MDG_" + realName)

                    # clear the geo history and useless shapes!
                    cmds.select("MDG_MDG_" + allAgtGrpChd[j], hi = True)
                    allSelObj = cmds.ls(sl = True, l = True)
                    if allSelObj == [] or allSelObj == None:
                        continue
                    for k in range(len(allSelObj)):
                        # delete history:
                        cmds.delete(allSelObj[k], ch = True)

                        # delete extra shapes:
                        if cmds.nodeType(allSelObj[k]) == "transform":
                            allGeoShapes = cmds.listRelatives(allSelObj[k], c = True, path = True)
                            if allGeoShapes != [] and allGeoShapes != None:
                                if cmds.nodeType(allGeoShapes[0]) == "mesh":
                                    for l in range(len(allGeoShapes)):
                                        if cmds.getAttr(allGeoShapes[l] + ".intermediateObject") == 1:
                                            try:
                                                cmds.delete(allGeoShapes[l])
                                            except:
                                                pass
                                        else:
                                            # set to bounding box mode:
                                            try:
                                                cmds.setAttr(allGeoShapes[l] + ".overrideEnabled", 1)
                                                cmds.setAttr(allGeoShapes[l] + ".overrideLevelOfDetail", 1)
                                            except:
                                                try:
                                                    cmds.setAttr(allGeoShapes[l] + ".overrideLevelOfDetail", 1)
                                                except:
                                                    pass

                elif allAgtGrpChd[j].find(":Geometry_") > 0:
                    nmSpStr = allAgtGrpChd[j].split(":")[0]
                    nameStr = allAgtGrpChd[j].split(":")[1]

                    endingName = allAgtGrpChd[j].split(":")[-1]

                    dupNode = cmds.ls("MDG_" + nmSpStr + "_MDG_" + endingName)
                    if dupNode != [] and dupNode != None:
                        continue;

                    if McdGeneral.McdCheckSubNodesNaming(allAgtGrpChd[j]) != 0:
                        if McdGeneral.McdCheckSubNodesNaming(allAgtGrpChd[j]) == 2:
                            continue
                        else:
                            return

                    dupNode = cmds.duplicate(allAgtGrpChd[j], name = "MDG_" + nameStr)

                    McdMeshDriveSetup.McdCheckShapeNodeInHi(dupNode[0])

                    cmds.hide(dupNode[0])
                    cmds.parent(dupNode[0], w = True)

                    renameCmd2 = renamePrefixCommandPre + "MDG_" + nmSpStr + "_" + renamePrefixCommandPost

                    cmds.select(clear = True)
                    cmds.select("MDG_" + nameStr)
                    mel.eval(renameCmd2)

                    for x in range(len(dupNode)):
                        dupNode[x] = "MDG_" + nmSpStr + "_" + dupNode[x]

                    oddRecord = McdGeneral.McdCheckAndFixName(allAgtGrpChd[j], dupNode, "MDG_" + nmSpStr + "_")
                    oddRecordList[0].extend(oddRecord[0])
                    oddRecordList[1].extend(oddRecord[1])

                    # re-check the hierarchy:
                    allDupNodes = cmds.listRelatives(ad = True, path = True)
                    for k in range(len(allDupNodes)):
                        if cmds.getAttr(allDupNodes[k] + ".intermediateObject") == 0:
                            if allDupNodes[k].find("|") >= 0:
                                realName = allDupNodes[k].split("|")[-1]
                                if realName.find("MDG_") != 0:
                                    cmds.rename(allDupNodes[k], "MDG_" + nmSpStr + realName)

                    # clear the geo history and useless shapes!
                    cmds.select("MDG_" + nmSpStr + "_MDG_" + nameStr, hi = True)
                    allSelObj = cmds.ls(sl = True, l = True)
                    if allSelObj == [] or allSelObj == None:
                        continue
                    for k in range(len(allSelObj)):
                        # delete history:
                        cmds.delete(allSelObj[k], ch = True)

                        # delete extra shapes:
                        if cmds.nodeType(allSelObj[k]) == "transform":
                            allGeoShapes = cmds.listRelatives(allSelObj[k], c = True, path = True)
                            if allGeoShapes != [] and allGeoShapes != None:
                                if cmds.nodeType(allGeoShapes[0]) == "mesh":
                                    for l in range(len(allGeoShapes)):
                                        if cmds.getAttr(allGeoShapes[l] + ".intermediateObject") == 1:
                                            try:
                                                cmds.delete(allGeoShapes[l])
                                            except:
                                                pass
                                        else:
                                            # set to bounding box mode:
                                            try:
                                                cmds.setAttr(allGeoShapes[l] + ".overrideEnabled", 1)
                                                cmds.setAttr(allGeoShapes[l] + ".overrideLevelOfDetail", 1)
                                            except:
                                                try:
                                                    cmds.setAttr(allGeoShapes[l] + ".overrideLevelOfDetail", 1)
                                                except:
                                                    pass

            except:
                return

    # get shapes to be duplicate:
    meshListRaw = []
    meshListRaw = mel.eval("McdGetRenderGeoCmd -rec 3;") # get and storing

    if meshListRaw == []:
        return
    # parse string list:
    agentNameList = []
    meshList = []

    isGetName = False # flag
    meshListUnit = [] # flag
    counter = 0;
    for i in range(len(meshListRaw)):
        if not isGetName:
            agentNameList.append("MDGGrp_" + str(counter))
            counter +=1
            isGetName = True
        if meshListRaw[i] != "#":

            if meshListRaw[i].find(":") <= 0:
                if meshListRaw[i].find('|') > 0:
                    meshListUnit.append("MDG_MDG_" + meshListRaw[i].replace("|", "|MDG_"))
                else:
                    meshListUnit.append("MDG_" + meshListRaw[i])
            else:

                if meshListRaw[i].find('|') > 0:
                    nmsp = meshListRaw[i].split(":")[0]
                    newName = meshListRaw[i].replace(nmsp + ":", "")

                    firstName = newName.split("|")[0]
                    firstNameNew = "MDG_" + nmsp + "_MDG_" + firstName

                    newNameSegs = newName.split("|")
                    newNameNew = ""
                    for j in range(len(newNameSegs)):
                        if j == 0:
                            newNameNew += firstNameNew
                        else:
                            newNameNew += "|MDG_" + nmsp + "_" + newNameSegs[j]

                    meshListUnit.append(newNameNew)
                else:
                    newName = meshListRaw[i].split(":")[1]
                    nameSpc = meshListRaw[i].split(":")[0]
                    meshListUnit.append("MDG_" + nameSpc + "_" + newName)

        else:
            isGetName = False
            meshList.append(meshListUnit)
            meshListUnit = []

    # duplicate and renaming:
    needHideSomeAgents = False
    if (cmds.getAttr(globalNode + ".hideList[0]") == 1):
        needHideSomeAgents = True

    dupInfo_Org = []
    dupInfo_Dup = []
    dupInfo_Aid = []

    for i in range(len(agentNameList)):
        thisAgentDupInfo = []
        stri = str(i)

        thisNeedHide = False
        if needHideSomeAgents:
            if (cmds.getAttr(globalNode + ".hideList[" + str(i+1) + "]") == 1):
                thisNeedHide = True

        mel.eval("flushUndo;")
        cmds.group(n = agentNameList[i], em = True)
        cmds.addAttr(agentNameList[i], ln = "agentId", at = "long", dv = i)
        if thisNeedHide:
            cmds.hide(agentNameList[i])

        if oddRecordList != [[],[]]:
            for j in range(len(meshList[i])):
                try:
                    idx = oddRecordList[0].index(meshList[i][j])
                    meshList[i][j] = oddRecordList[1][idx]
                except:
                    pass

        try:
            dupListTemp = cmds.duplicate(meshList[i], rr = True)
        except:
            return

        # add new feature, rename to better name:
        dupList = []
        dupList = dupListTemp

        for j in range(len(dupList)):
            try:
                currentShape = cmds.listRelatives(dupList[j], c = True, path = True)[0]
                if currentShape.find("|") >= 0:
                    realName = currentShape.split("|")[-1]
                    cmds.rename(currentShape, realName + "agent" + str(i))
                    cmds.parent(realName + "agent" + str(i), agentNameList[i], r = True, s = True)
                    thisAgentDupInfo.append(realName + "agent" + str(i))
                else:
                    cmds.parent(currentShape, agentNameList[i], r = True, s = True)
                    thisAgentDupInfo.append(currentShape)

                cmds.delete(dupList[j])
            except:
                return

        for j in range(len(meshList[i])):
            if meshList[i][j] not in dupInfo_Org:
                dupInfo_Org.append(meshList[i][j])
                dupInfo_Dup.append([])
                dupInfo_Aid.append([])

            meshIndex = dupInfo_Org.index(meshList[i][j])
            dupInfo_Dup[meshIndex].append(thisAgentDupInfo[j])
            dupInfo_Aid[meshIndex].append(i)

    dupInfo = []
    dupInfo.append(dupInfo_Org)
    dupInfo.append(dupInfo_Dup)
    dupInfo.append(dupInfo_Aid)

    # auto parenting:
    masterNode = cmds.ls("MDGGRPMASTER")
    if McdGeneral.McdIsBlank(masterNode):
        cmds.createNode("transform", n = "MDGGRPMASTER")
    allDupNodes = cmds.ls("MDGGrp_*")
    cmds.parent(allDupNodes, "MDGGRPMASTER")

    mel.eval("flushUndo;")
    #--------------------------------------------------------------------------- McdMeshDriveSetup.MDDuplicate Modified

    McdMeshDriveSetup.McdCreateMeshDriveIMNode(1)
    return True



class SceneParse:
    def __init__(self, outDir):
        self.outShowDir, self.outShowName = PathUtils.GetRootPath(outDir)

        # Member Variables
        self.m_agentData = dict()
        self.m_simData   = dict()
        self.m_error = list()

        self.doIt()

    def doIt(self):
        # Agent data
        self.getAgentData()
        if self.m_error:
            return
        # Randomize data
        randomizeData = GeomSkel.GetRandomizeData(self.m_agentData['assetList'], self.m_agentData['assetFiles'])
        self.m_agentData['randomizeData'] = randomizeData
        # Agent Hide List
        self.getHideList()
        # Simulation data
        self.getSimulationData()

        # Debug message
        dxsMsg.Print('info', '[ Agent AssetFile ]')
        for f in self.m_agentData['assetFiles']:
            dxsMsg.Print('info', f)


    def getAgentData(self):
        allAgents = cmds.ls(type='McdAgent')

        agentAssetList = list() # ['OriginalAgent_$AGTYPE', ...]
        agentAssetFiles= list()
        agentAssetJointsList = list()
        agentAssetJointsOrientList = list()

        allAgentGroups = cmds.listRelatives('Miarmy_Contents', c=True, type='McdAgentGroup')
        for ag in allAgentGroups:
            allChildren = cmds.listRelatives(ag, c=True, p=False, path=True)
            for c in allChildren:
                if c.find('OriginalAgent_') > -1:
                    agentAssetList.append(c)
                    agentAssetFiles.append(self.getAgentAssetFile(c))

                    jt = GeomSkel.GetJoints(c)
                    agentAssetJointsList.append(jt.allJointsPath)
                    agentAssetJointsOrientList.append(jt.getOrientList())

        self.m_agentData = {
            'allAgents': allAgents,
            'assetList': agentAssetList,
            'assetFiles': agentAssetFiles,
            'jointsList': agentAssetJointsList,
            'jointsOrientList': agentAssetJointsOrientList
        }


    def getHideList(self):
        globalNode = McdGeneral.McdGetMcdGlobalNode()
        if cmds.getAttr('%s.hideList[0]' % globalNode) == 1:
            hideList = dict()

            allPlaceNodes = cmds.ls(type='McdPlace')
            allPlacePlaceID = list()
            allPlaceNBAgents= list()
            for i in range(len(allPlaceNodes)):
                allPlacePlaceID.append(cmds.getAttr(allPlaceNodes[i] + '.plid'))
                allPlaceNBAgents.append(cmds.getAttr(allPlaceNodes[i] + '.numOfAgent'))

            for agent in self.m_agentData['allAgents']:
                realAgentID = McdSimpleCmd.McdGetRealAgentIDInScene1(agent, allPlacePlaceID, allPlaceNBAgents)
                ishide = cmds.getAttr('%s.hideList[%d]' % (globalNode, realAgentID))
                hideList[agent] = int(ishide)

            self.m_agentData['hideList'] = hideList


    def getAgentAssetFile(self, node):  # node : OriginalAgent Node
        if not cmds.referenceQuery(node, isNodeReferenced=True):
            assert False, '[only support referenced node] -> %s' % node

        agentType = node.split('|')[-1].replace('OriginalAgent_', '')
        splitStr  = agentType.split('_')
        assetName = splitStr[0]
        if len(splitStr) > 1:
            assetName = string.join(splitStr[:-1], '_')

        dirRule = '{SHOW}' + '/asset/{ASSET}/agent/{AGTYPE}/{AGTYPE}.usd'.format(ASSET=assetName, AGTYPE=agentType)

        usdfile = dirRule.format(SHOW=self.outShowDir)
        if os.path.exists(usdfile): # 1. find by outdir
            return usdfile
        else:                       # 2. find by referenced OA file
            filename = cmds.referenceQuery(node, filename=True, withoutCopyNumber=True)
            showDir, showName = PathUtils.GetRootPath(filename)
            usdfile = dirRule.format(SHOW=showDir)
            if os.path.exists(usdfile):
                return usdfile
            else:
                if cmds.about(batch=True):
                    self.m_error.append(usdfile)
                else:
                    assert False, '[not found AgentAssetFile] -> %s' % usdfile


    def getSimulationData(self):
        self.m_simData = {
            'assetData': dict(),    # {'tid': [shapename, ...], }
            'indexMap': dict()      # {'index': {'tid': xx, 'aid': xx}, ..}
        }
        # cloth
        mcdClothes = cmds.ls(type='McdCloth')
        if mcdClothes:
            self.getSimulationCloth(mcdClothes)

        simTypeIDs = self.m_simData['assetData']
        if not simTypeIDs:
            return

        allAgents  = list()
        for n in self.m_agentData['allAgents']:
            if cmds.getAttr('%s.visibility' % n):
                allAgents.append(n)
        for i in range(len(allAgents)):
            tid = cmds.getAttr('%s.tempTypeId' % allAgents[i])
            aid = cmds.getAttr('%s.agentId' % allAgents[i])
            if tid in simTypeIDs:
                self.m_simData['indexMap'][i] = {'tid': tid, 'aid': aid}

    def getSimulationCloth(self, mcdClothes):
        for n in mcdClothes:
            meshes = cmds.listConnections('%s.cloth' % n, s=True, d=False)
            if meshes:
                fullPath  = cmds.ls(meshes[0], l=True)[0]
                agentGroup= fullPath.split('|')[2]
                originalAgent = agentGroup.replace('Agent_', 'OriginalAgent_')
                tid = self.m_agentData['assetList'].index(originalAgent)
                if self.m_simData['assetData'].has_key(tid):
                    self.m_simData['assetData'][tid] += meshes
                else:
                    self.m_simData['assetData'][tid] = meshes




#-------------------------------------------------------------------------------
#
#   MeshDrive
#
#-------------------------------------------------------------------------------
def MiarmyMeshDriveExport():
    '''
    MeshDrive cache export
    '''
    # import McdGeneral
    globalNode = McdGeneral.McdGetMcdGlobalNode()

    cacheFolder, cacheName, cachefr = GetCacheInfo()

    meshDriveFolder = cmds.getAttr('%s.outMD2Folder' % globalNode)
    meshDriveName   = cmds.getAttr('%s.outMD2Name' % globalNode)
    if (not meshDriveFolder) or (not cacheFolder in meshDriveFolder):
        meshDriveFolder = cacheFolder + '_meshDrive'
        meshDriveName   = cacheName
        if not os.path.exists(meshDriveFolder):
            os.makedirs(meshDriveFolder)
        cmds.setAttr('%s.outMD2Folder' % globalNode, meshDriveFolder, type='string')
        cmds.setAttr('%s.outMD2Name' % globalNode, meshDriveName, type='string')

    cmds.file(save=True)

    startFile = os.path.join(meshDriveFolder, '%s.%d.mbc' % (meshDriveName, cachefr[0]))
    endFile   = os.path.join(meshDriveFolder, '%s.%d.mbc' % (meshDriveName, cachefr[1]))
    if not os.path.exists(startFile) or not os.path.exists(endFile):
        mel.eval('McdMakeJointCacheCmd -actionMode 0;')

        brainNode = mel.eval('McdSimpleCommand -execute 3;')
        solverFrame = cmds.getAttr('%s.startTime' % brainNode)
        solverFrame-= 1
        if solverFrame > cachefr[0]:
            solverFrame = cachefr[0]

        cmds.currentTime(solverFrame - 1)
        cmds.currentTime(solverFrame)

        while (solverFrame <= cachefr[1]):
            if solverFrame >= cachefr[0]:
                cmds.currentTime(solverFrame)
                mel.eval('McdMakeJointCacheCmd -actionMode 1;')
            solverFrame += 1


def MiarmyBatchMeshDriveExport(srcFile):
    '''
    MeshDrive mesh export
    '''
    if not InitializeMiarmy(srcFile):
        return
    ms = SceneParse()
    if ms.m_simData and ms.m_simData['indexMap']:
        MiarmyMeshDriveExport()
