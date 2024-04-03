## For non_stop publish

import maya.cmds as cmds
import maya.mel as mel
from McdGeneral import *
from McdSimpleCmd import *
from McdRenderFBXFunctions import *
import os

if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
    try:
        import mentalray.renderProxyUtils
    except:
        pass
else:
    try:
        import maya.app.mentalray.renderProxyUtils
    except:
        pass

def McdCheckShapeNodeInHi(rootNode):
    cmds.select(rootNode, hi=True)

    allSelObj = cmds.ls(sl=True)

    allTransformNode = []
    # delete extra shapes:
    for i in range(len(allSelObj)):
        if cmds.nodeType(allSelObj[i]) == "transform":
            allTransformNode.append(allSelObj[i])

    for i in range(len(allTransformNode)):
        allGeoShapes = cmds.listRelatives(allTransformNode[i], c=True, path=True)
        if allGeoShapes != [] and allGeoShapes != None:
            if cmds.nodeType(allGeoShapes[0]) == "mesh":
                for l in range(len(allGeoShapes)):
                    if cmds.getAttr(allGeoShapes[l] + ".intermediateObject") == 0:
                        # set to bounding box mode:
                        shapeName = allGeoShapes[l].split("|")[-1]
                        transName = allTransformNode[i].split("|")[-1]
                        if shapeName != transName + "Shape":
                            try:
                                cmds.rename(shapeName, transName + "Shape")
                                break;
                            except:
                                pass

def MDDuplicate(showMsg=True, keepStructure=False):
    try:
        cmds.evaluationManager(mode="off")
    except:
        pass
    checkSkinForAll()
    allAgents = cmds.ls(type="McdAgent")
    if not McdIsBlank(allAgents):
        cmds.confirmDialog(t="De-Place Needed", m="Please de-place agent. Miarmy > Placement > De-Place")
        return
    globalNode = McdListMcdGlobal()
    rawPath = cmds.getAttr(globalNode + ".outMD2Folder")
    md3f = envPath2AbsPath(rawPath)
    md3n = cmds.getAttr(globalNode + ".outMD2Name")
    if MIsBlank(md3f):
        cmds.confirmDialog(t="Cache Error", m="Please make Mesh Drive 3 cache firstly.")
        return
    if MIsBlank(md3n):
        cmds.confirmDialog(t="Cache Error", m="Please make Mesh Drive 3 cache firstly.")
        return
    McdMeshDrive2Clear()
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
    allAgtGrpNode = cmds.ls(type="McdAgentGroup")
    if allAgtGrpNode == [] or allAgtGrpNode == None:
        return
    refMode = False
    for i in range(len(allAgtGrpNode)):
        if i == 0:
            if allAgtGrpNode[i].find(":") > 0:
                refMode = True

        if refMode:
            if allAgtGrpNode[i].find(":") <= 0:
                cmds.confirmDialog(t="Naming Error",
                                   m="Hybrid naming with real name and namespace mode.\nAuto selected, check outliner.")
                cmds.select(allAgtGrpNode[i])
                return 1
        else:
            if allAgtGrpNode[i].find(":") > 0:
                cmds.confirmDialog(t="Naming Error",
                                   m="Hybrid naming with real name and namespace mode.\nAuto selected, check outliner.")
                cmds.select(allAgtGrpNode[i])
                return 1

    mainDupGeomGrp = cmds.ls("MDG_MDG_Geometry_*", l=True)
    if mainDupGeomGrp != [] and mainDupGeomGrp != None:
        for i in range(len(mainDupGeomGrp)):
            try:
                cmds.delete(mainDupGeomGrp[i])
            except:
                pass

    oddRecordList = [[], []]

    for i in range(len(allAgtGrpNode)):
        allAgtGrpChd = cmds.listRelatives(allAgtGrpNode[i], c=True)
        if allAgtGrpChd == [] or allAgtGrpChd == None:
            continue;
        for j in range(len(allAgtGrpChd)):
            try:
                if allAgtGrpChd[j].find("Geometry_") == 0:
                    ##########################
                    dupNode = cmds.ls("MDG_MDG_" + allAgtGrpChd[j])
                    if dupNode != [] and dupNode != None:
                        continue;

                    if McdCheckSubNodesNaming(allAgtGrpChd[j]) != 0:
                        return

                    dupNode = cmds.duplicate(allAgtGrpChd[j], name="MDG_" + allAgtGrpChd[j])
                    McdCheckShapeNodeInHi(dupNode[0])
                    cmds.hide(dupNode[0])
                    cmds.parent(dupNode[0], w=True)

                    oddRecord = McdCheckAndFixName(allAgtGrpChd[j], dupNode)
                    oddRecordList[0].extend(oddRecord[0])
                    oddRecordList[1].extend(oddRecord[1])

                    cmds.select(clear=True)
                    cmds.select("MDG_" + allAgtGrpChd[j])
                    mel.eval(renameCommand)

                    # re-check the hierarchy:
                    allDupNodes = cmds.listRelatives(ad=True, path=True)
                    for k in range(len(allDupNodes)):
                        if cmds.getAttr(allDupNodes[k] + ".intermediateObject") == 0:
                            if allDupNodes[k].find("|") >= 0:
                                realName = allDupNodes[k].split("|")[-1]
                                if realName.find("MDG_") != 0:
                                    cmds.rename(allDupNodes[k], "MDG_" + realName)

                    # clear the geo history and useless shapes!
                    cmds.select("MDG_MDG_" + allAgtGrpChd[j], hi=True)
                    allSelObj = cmds.ls(sl=True, l=True)
                    if allSelObj == [] or allSelObj == None:
                        continue
                    for k in range(len(allSelObj)):
                        # delete history:
                        cmds.delete(allSelObj[k], ch=True)

                        # delete extra shapes:
                        if cmds.nodeType(allSelObj[k]) == "transform":
                            allGeoShapes = cmds.listRelatives(allSelObj[k], c=True, path=True)
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
                    ##########################
                    nmSpStr = allAgtGrpChd[j].split(":")[0]
                    nameStr = allAgtGrpChd[j].split(":")[1]

                    endingName = allAgtGrpChd[j].split(":")[-1]

                    dupNode = cmds.ls("MDG_" + nmSpStr + "_MDG_" + endingName)
                    if dupNode != [] and dupNode != None:
                        continue;

                    if McdCheckSubNodesNaming(allAgtGrpChd[j]) != 0:
                        return

                    dupNode = cmds.duplicate(allAgtGrpChd[j], name="MDG_" + nameStr)

                    if not keepStructure:
                        McdCheckShapeNodeInHi(dupNode[0])

                    cmds.hide(dupNode[0])
                    cmds.parent(dupNode[0], w=True)

                    renameCmd2 = renamePrefixCommandPre + "MDG_" + nmSpStr + "_" + renamePrefixCommandPost

                    cmds.select(clear=True)
                    cmds.select("MDG_" + nameStr)
                    mel.eval(renameCmd2)

                    for x in range(len(dupNode)):
                        dupNode[x] = "MDG_" + nmSpStr + "_" + dupNode[x]

                    oddRecord = McdCheckAndFixName(allAgtGrpChd[j], dupNode, "MDG_" + nmSpStr + "_")
                    oddRecordList[0].extend(oddRecord[0])
                    oddRecordList[1].extend(oddRecord[1])

                    # re-check the hierarchy:
                    allDupNodes = cmds.listRelatives(ad=True, path=True)
                    for k in range(len(allDupNodes)):
                        if cmds.getAttr(allDupNodes[k] + ".intermediateObject") == 0:
                            if allDupNodes[k].find("|") >= 0:
                                realName = allDupNodes[k].split("|")[-1]
                                if realName.find("MDG_") != 0:
                                    cmds.rename(allDupNodes[k], "MDG_" + nmSpStr + realName)

                    # clear the geo history and useless shapes!
                    cmds.select("MDG_" + nmSpStr + "_MDG_" + nameStr, hi=True)
                    allSelObj = cmds.ls(sl=True, l=True)
                    if allSelObj == [] or allSelObj == None:
                        continue
                    for k in range(len(allSelObj)):
                        # delete history:
                        cmds.delete(allSelObj[k], ch=True)

                        # delete extra shapes:
                        if cmds.nodeType(allSelObj[k]) == "transform":
                            allGeoShapes = cmds.listRelatives(allSelObj[k], c=True, path=True)
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
                if showMsg:
                    cmds.confirmDialog(t="Naming Error", m="Please check the naming conventions")
                return

    # get shapes to be duplicate:
    meshListRaw = []
    meshListRaw = mel.eval("McdGetRenderGeoCmd -rec 3;")  # get and storing

    if meshListRaw == []:
        return
    # parse string list:
    agentNameList = []
    meshList = []

    isGetName = False  # flag
    meshListUnit = []  # flag
    counter = 0;
    for i in range(len(meshListRaw)):
        if not isGetName:
            agentNameList.append("MDGGrp_" + str(counter))
            counter += 1
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
    amount = 0
    counter = 0
    totalCount = len(agentNameList)
    if showMsg:
        cmds.progressWindow(title="Duplicating:", progress=0, \
                            min=0, max=100, \
                            status='Copying', isInterruptable=True)

    nbAgents = len(agentNameList)
    tenPercent = nbAgents / 10
    if tenPercent == 0:
        tenPercent = 1
    progress = 0

    needHideSomeAgents = False
    if (cmds.getAttr(globalNode + ".hideList[0]") == 1):
        needHideSomeAgents = True

    for i in range(len(agentNameList)):

        stri = str(i)

        thisNeedHide = False
        if needHideSomeAgents:
            if (cmds.getAttr(globalNode + ".hideList[" + str(i + 1) + "]") == 1):
                thisNeedHide = True

        if i % tenPercent == 0:
            print "Duplicating: " + str(progress) + " %";
            progress += 10;

        mel.eval("flushUndo;")
        cmds.group(n=agentNameList[i], em=True)
        cmds.addAttr(agentNameList[i], ln="agentId", at="long", dv=i)
        if thisNeedHide:
            cmds.hide(agentNameList[i])

        if oddRecordList != [[], []]:
            for j in range(len(meshList[i])):
                try:
                    idx = oddRecordList[0].index(meshList[i][j])
                    meshList[i][j] = oddRecordList[1][idx]
                except:
                    pass

        try:
            dupListTemp = cmds.duplicate(meshList[i], rr=True)
        except:
            cmds.progressWindow(endProgress=1)

        # add new feature, rename to better name:
        dupList = []
        if keepStructure:
            for j in range(len(dupListTemp)):
                currentShape = cmds.listRelatives(dupListTemp[j], c=True, path=True)[0]

                # after rename:
                newShapeName = meshList[i][j] + "_" + stri
                newShapeNameConfirm = cmds.rename(currentShape, newShapeName)
                currentTrans = cmds.listRelatives(newShapeNameConfirm, p=True, c=False, path=True)[0]
                # print currentTrans
                dupList.append(currentTrans)
        else:
            dupList = dupListTemp

        for j in range(len(dupList)):
            try:
                currentShape = cmds.listRelatives(dupList[j], c=True, path=True)[0]
                if currentShape.find("|") >= 0:
                    realName = currentShape.split("|")[-1]
                    cmds.rename(currentShape, realName + "agent" + str(i))
                    cmds.parent(realName + "agent" + str(i), agentNameList[i], r=True, s=True)
                else:
                    cmds.parent(currentShape, agentNameList[i], r=True, s=True)

                cmds.delete(dupList[j])
            except:
                if showMsg:
                    cmds.confirmDialog(t="Error", m="Please check the geometry naming error")
                else:
                    print "Please check the geometry error"

        ## progress operation: ////////////////////////////////////////////////
        if cmds.progressWindow(query=True, isCancelled=True):
            break
        counter += 1
        amount = float(counter) / float(totalCount) * 100.0
        if showMsg:
            cmds.progressWindow(edit=True, progress=amount)

    if showMsg:
        cmds.progressWindow(endProgress=1)
    McdRandomizeTexturesDuplicate()

    # auto parenting:
    masterNode = cmds.ls("MDGGRPMASTER")
    if McdIsBlank(masterNode):
        cmds.createNode("transform", n="MDGGRPMASTER")
    allDupNodes = cmds.ls("MDGGrp_*")
    cmds.parent(allDupNodes, "MDGGRPMASTER")

    McdDuplicateShaderForAutoTexGeo()

    mel.eval("flushUndo;")


def McdMeshDrive2Clear():
    turn_On_1_Off_0_ClothSkin(0)

    allMDGGrp = cmds.ls("MDGGrp_*")
    if not McdIsBlank(allMDGGrp):
        for i in range(len(allMDGGrp)):
            try:
                cmds.delete(allMDGGrp[i])
            except:
                pass

    try:
        cmds.delete("MDGGRPMASTER")
    except:
        pass

    allMDGGrp = cmds.ls("MDG_*")
    if not McdIsBlank(allMDGGrp):
        for i in range(len(allMDGGrp)):
            try:
                cmds.delete(allMDGGrp[i])
            except:
                pass

    dupRoot = cmds.ls("McdRoot_*")
    if not MIsBlank(dupRoot):
        cmds.delete(dupRoot)

    dupAgentGeo = cmds.ls("McdAgentGeometry_*")
    if not MIsBlank(dupAgentGeo):
        cmds.delete(dupAgentGeo)

    allMDNodes = cmds.ls(type="McdMeshDrive")
    if not McdIsBlank(allMDNodes):
        for i in range(len(allMDNodes)):
            try:
                cmds.delete(allMDNodes[i])
            except:
                pass

    allMDNodes = cmds.ls(type="McdMeshDriveIM")
    if not McdIsBlank(allMDNodes):
        for i in range(len(allMDNodes)):
            try:
                cmds.delete(allMDNodes[i])
            except:
                pass

    # agent return:
    try:
        cmd = "McdAgentMatchCmd -mm 0;"
        mel.eval(cmd)
    except:
        pass

    McdClearUselessShader()

    try:
        cmds.showHidden("Miarmy_Contents")
    except:
        pass


def McdSetupBatchRender():
    stat = cmds.confirmDialog(t="Randomize Shader", m="Do you want to setup mesh drive for batch render:" +
                                                      "\n* clear duplicated mesh", \
                              b=["Clear and Setup", "Just Setup MEL", "Cancel"])
    if stat == "Clear and Setup":
        McdMeshDrive2Clear()

    if stat == "Just Setup MEL" or stat == "Clear and Setup":
        cmds.setAttr("defaultRenderGlobals.preMel", "McdBatchMeshDrive2", type="string")
        cmds.setAttr("defaultRenderGlobals.preRenderMel", "McdBatchMeshDrive2Frame", type="string")


def turn_On_1_Off_0_ClothSkin(on_1_off_0):
    allClothNode = cmds.ls(type="McdCloth")

    if MIsBlank(allClothNode):
        return

    for i in range(len(allClothNode)):
        allHis = cmds.listHistory(allClothNode[i])
        if MIsBlank(allHis):
            continue
        for j in range(len(allHis)):
            if cmds.nodeType(allHis[j]) == "skinCluster":
                if on_1_off_0 == 1:
                    cmds.setAttr(allHis[j] + ".envelope", 0)
                else:
                    cmds.setAttr(allHis[j] + ".envelope", 1)


def McdCreateMeshDriveNode(mode, isShowProgress=0):
    turn_On_1_Off_0_ClothSkin(mode)

    if mode == 1:
        # create node
        allMDNodes = cmds.ls(type="McdMeshDrive")
        if not McdIsBlank(allMDNodes):
            return
        mel.eval("McdSimpleCommand -exe 25;")

        if isShowProgress == 1:
            allMDNodes = cmds.ls(type="McdMeshDrive")
            try:
                for i in range(len(allMDNodes)):
                    cmds.setAttr(allMDNodes[i] + ".showProgress", 1)
            except:
                pass
    else:
        # clear
        allMDNodes = cmds.ls(type="McdMeshDrive")
        if McdIsBlank(allMDNodes):
            return

        for i in range(len(allMDNodes)):
            try:
                cmds.delete(allMDNodes[i])
            except:
                print "Some nodes McdMeshDrive type cannot be deleted!"

    try:
        cmds.hide("Miarmy_Contents")
    except:
        pass


def McdCreateMeshDriveIMNode(mode, isShowProgress=0):
    globalNode = McdGetMcdGlobalNode()
    notShowProgress = cmds.getAttr(globalNode + ".boolMaster[21]")

    turn_On_1_Off_0_ClothSkin(mode)

    if mode == 1:

        allMDGrps = cmds.ls("MDGGrp_*")
        if MIsBlank(allMDGrps):
            cmds.confirmDialog(t="Abort", m="Need duplicate mesh firstly.")
            return

        # if exist, turn off disable
        # if not, create and link
        allMDNodes = cmds.ls(type="McdMeshDriveIM")
        if not McdIsBlank(allMDNodes):
            cmds.setAttr(allMDNodes[0] + ".disable", 0)
            return
        md3Node = cmds.createNode("McdMeshDriveIM")

        if isShowProgress == 1 and notShowProgress == 0:
            allMDNodes = cmds.ls(type="McdMeshDriveIM")
            try:
                for i in range(len(allMDNodes)):
                    cmds.setAttr(allMDNodes[i] + ".showProgress", 1)
            except:
                pass

        # ---------------------- connect node! --------------------#
        # 1. connect time:
        timeNode = cmds.ls(type="time")[0]
        cmds.connectAttr(timeNode + ".outTime", md3Node + ".timeValue")
        # 2. connect mesh:
        geoCount = 0
        for i in range(len(allMDGrps)):
            allChildren = cmds.listRelatives(allMDGrps[i], c=True, p=False)
            if not MIsBlank(allChildren):
                for j in range(len(allChildren)):
                    cmds.connectAttr(md3Node + ".outputMeshes[" + str(geoCount) + "]", allChildren[j] + ".inMesh")
                    geoCount += 1

    else:
        # just turn on disable
        allMDNodes = cmds.ls(type="McdMeshDriveIM")
        if McdIsBlank(allMDNodes):
            return
        cmds.setAttr(allMDNodes[0] + ".disable", 1)

    try:
        cmds.hide("Miarmy_Contents")
    except:
        pass


def McdExportMD2Cache():
    McdLicenseCheck()
    performMD2CachePreCheck()

    # export agent type list:
    mel.eval("McdMakeJointCacheCmd -actionMode 0;")

    # batch make cache: --------------------------------------------------------
    startFrame = cmds.playbackOptions(q=True, min=True)
    endFrame = cmds.playbackOptions(q=True, max=True)
    brainNode = mel.eval("McdSimpleCommand -execute 3")
    solverFrame = cmds.getAttr(brainNode + ".startTime")
    solverFrame -= 1
    if solverFrame > startFrame:
        solverFrame = startFrame

    cmds.currentTime(solverFrame - 1)
    cmds.currentTime(solverFrame)

    amount = 0
    counter = 0
    totalCount = endFrame - startFrame
    cmds.progressWindow(title="Caching:", progress=0, \
                        min=0, max=100, \
                        status="caching", isInterruptable=True)

    # from solverFrame to endFrame:
    while (solverFrame <= endFrame):

        if solverFrame >= startFrame:
            counter += 1
            cmds.currentTime(solverFrame)
            # deal with batch cache
            mel.eval("McdMakeJointCacheCmd -actionMode 1;")

        solverFrame += 1

        ## progress operation: ////////////////////////////////////////////////
        if cmds.progressWindow(query=True, isCancelled=True):
            break

        amount = float(counter) / float(totalCount) * 100.0
        cmds.progressWindow(edit=True, progress=amount)

    cmds.progressWindow(endProgress=1)


def McdExportMD2CacheNoPDW():
    McdLicenseCheck()
    performMD2CachePreCheck()

    # export agent type list:
    mel.eval("McdMakeJointCacheCmd -actionMode 0;")

    # batch make cache: --------------------------------------------------------
    startFrame = cmds.playbackOptions(q=True, min=True)
    endFrame = cmds.playbackOptions(q=True, max=True)

    brainNode = mel.eval("McdSimpleCommand -execute 3")
    solverFrame = cmds.getAttr(brainNode + ".startTime")
    solverFrame -= 1
    if solverFrame > startFrame:
        solverFrame = startFrame

    cmds.currentTime(solverFrame - 1)
    cmds.currentTime(solverFrame)

    amount = 0
    counter = 0
    totalCount = endFrame - startFrame

    # from solverFrame to endFrame:
    while (solverFrame <= endFrame):

        if solverFrame >= startFrame:
            counter += 1
            cmds.currentTime(solverFrame)
            # deal with batch cache
            mel.eval("McdMakeJointCacheCmd -actionMode 1;")

        solverFrame += 1


def performMD2CachePreCheck():
    # agent exist
    allAgents = cmds.ls(type="McdAgent")
    if allAgents == [] or allAgents == None:
        raise Exception("There is no agent in scene for making cache.")

    # path writable:
    globalNode = mel.eval("McdSimpleCommand -execute 2")
    rawPath = cmds.getAttr(globalNode + ".outMD2Folder")

    cacheFolder = envPath2AbsPath(rawPath)

    if cacheFolder == "" or cacheFolder == None:
        raise Exception("The output folder is not exist.")

    if not os.access(cacheFolder, os.W_OK):
        raise Exception("The output folder is not exist.")

def MDExpandMRProxy(isAllScene):
    if isAllScene == 0:
        # single frame

        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputMIFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputMIName")

        isGZip = cmds.getAttr(globalNode + ".outputMIGzip")
        isBlockMat = cmds.getAttr(globalNode + ".blockMRMat")
        isBothMat = cmds.getAttr(globalNode + ".bothMRMat")
        isRelTexPath = cmds.getAttr(globalNode + ".mrAttrList[0]")
        xpScheme = "3313333333"
        if isRelTexPath:
            xpScheme = "3323333333"

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        # create extra path
        try:
            os.makedirs(outputPath + "/" + outputName)
        except:
            pass

        if not os.access(outputPath + "/" + outputName, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if isBothMat == 1:
            # execute
            try:
                cmds.select("MDGGRPMASTER")
            except:
                cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                return
            exepandDisplaySelection()

            # 11111111111111111111111111
            if isGZip:
                melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 -fem -fma \
                        -fis  -fcd  -pcm  -as  -asn "' + outputName + '_1' + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '_1' + '.mi' + '"'
            else:
                melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -fem -fma \
                        -fis  -fcd  -pcm  -as  -asn "' + outputName + '_1' + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '_1' + '.mi' + '"'
            mel.eval(melCmd)

            # execute
            try:
                cmds.select("MDGGRPMASTER")
            except:
                cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                return
            exepandDisplaySelection()

            # 22222222222222222222222222
            if isGZip:
                melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 \
                        -fis  -fcd  -pcm  -as  -asn "' + outputName + '_2' + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '_2' + '.mi' + '"'
            else:
                melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe \
                        -fis  -fcd  -pcm  -as  -asn "' + outputName + '_2' + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '_2' + '.mi' + '"'
            mel.eval(melCmd)

            # create box and naming it to McdMRRenderDummy
            dummyCubeTemp = cmds.ls("McdMRRenderDummy")
            dummyCube = ""
            if dummyCubeTemp != [] and dummyCubeTemp != None:
                dummyCube = dummyCubeTemp[0]
            if dummyCube == "":
                cmds.polyCube(n="McdMRRenderDummy", sx=1, sy=1)

            dummyCubeTemp = cmds.ls("McdMRRenderDummy_")
            dummyCube = ""
            if dummyCubeTemp != [] and dummyCubeTemp != None:
                dummyCube = dummyCubeTemp[0]
            if dummyCube == "":
                cmds.polyCube(n="McdMRRenderDummy_", sx=1, sy=1)

            # pading:
            currentFrame = cmds.currentTime(q=True)
            frameNumber = str(int(currentFrame))
            while (len(frameNumber) < 4):
                frameNumber = "0" + frameNumber

            # link this to MR contents
            if isGZip:
                cmds.setAttr("McdMRRenderDummy.miProxyFile", outputPath + "/" + outputName + "_1" + ".mi.gz",
                             type="string")
            else:
                cmds.setAttr("McdMRRenderDummy.miProxyFile", outputPath + "/" + outputName + "_1" + ".mi",
                             type="string")

            if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummyShape")
            else:
                maya.app.mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummyShape")

            if isGZip:
                cmds.setAttr("McdMRRenderDummy_.miProxyFile", outputPath + "/" + outputName + "_2" + ".mi.gz",
                             type="string")
            else:
                cmds.setAttr("McdMRRenderDummy_.miProxyFile", outputPath + "/" + outputName + "_2" + ".mi",
                             type="string")

            if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummy_Shape")
            else:
                maya.app.mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummy_Shape")

        else:
            # execute
            try:
                cmds.select("MDGGRPMASTER")
            except:
                cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                return
            exepandDisplaySelection()

            if isGZip:
                if isBlockMat:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '.mi' + '"'
                else:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 -fem -fma \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '.mi' + '"'
            else:
                if isBlockMat:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '.mi' + '"'
                else:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -fem -fma \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + outputName + '.mi' + '"'

            mel.eval(melCmd)

            # create box and naming it to McdMRRenderDummy
            dummyCubeTemp = cmds.ls("McdMRRenderDummy")
            dummyCube = ""
            if dummyCubeTemp != [] and dummyCubeTemp != None:
                dummyCube = dummyCubeTemp[0]
            if dummyCube == "":
                cmds.polyCube(n="McdMRRenderDummy", sx=1, sy=1)

            # pading:
            currentFrame = cmds.currentTime(q=True)
            frameNumber = str(int(currentFrame))
            while (len(frameNumber) < 4):
                frameNumber = "0" + frameNumber

            # link this to MR contents
            if isGZip:
                cmds.setAttr("McdMRRenderDummy.miProxyFile", outputPath + "/" + outputName + ".mi.gz", type="string")
            else:
                cmds.setAttr("McdMRRenderDummy.miProxyFile", outputPath + "/" + outputName + ".mi", type="string")

            if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummyShape")
            else:
                maya.app.mentalray.renderProxyUtils.resizeToBoundingBox("McdMRRenderDummyShape")


    elif isAllScene == 1:
        # all frames


        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputMIFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputMIName")

        isGZip = cmds.getAttr(globalNode + ".outputMIGzip")
        isBlockMat = cmds.getAttr(globalNode + ".blockMRMat")
        isBothMat = cmds.getAttr(globalNode + ".bothMRMat")
        isRelTexPath = cmds.getAttr(globalNode + ".mrAttrList[0]")
        xpScheme = "3313333333"
        if isRelTexPath:
            xpScheme = "3323333333"

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        # create extra path
        try:
            os.makedirs(outputPath + "/" + outputName)
        except:
            pass

        if not os.access(outputPath + "/" + outputName, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        minFrame = cmds.playbackOptions(q=True, min=True)
        maxFrame = cmds.playbackOptions(q=True, max=True)
        nbFrame = int(maxFrame - minFrame + 1)

        for i in range(nbFrame):

            frameNumberNum = int(minFrame + i)
            cmds.currentTime(frameNumberNum)

            stri = str(frameNumberNum)

            while len(stri) < 4:
                stri = "0" + stri

            if isBothMat == 1:
                # execute 1111111111111111111111111111
                try:
                    cmds.select("MDGGRPMASTER")
                except:
                    cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                    return
                exepandDisplaySelection()

                if isGZip:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 -fem -fma \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '_1.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                             outputName + '_1.' + stri + '.mi' + '"'
                else:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -fem -fma \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '_1.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                             outputName + '_1.' + stri + '.mi' + '"'
                mel.eval(melCmd)

                # execute 22222222222222222222222222222
                try:
                    cmds.select("MDGGRPMASTER")
                except:
                    cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                    return
                exepandDisplaySelection()

                if isGZip:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '_2.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                             outputName + '_2.' + stri + '.mi' + '"'
                else:
                    melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe \
                            -fis  -fcd  -pcm  -as  -asn "' + outputName + '_2.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                             outputName + '_2.' + stri + '.mi' + '"'
                mel.eval(melCmd)

                # create box and naming it to McdMRRenderDummy
                dummyCubeTemp = cmds.ls("McdMRRenderDummy" + stri)
                dummyCube = ""
                if dummyCubeTemp != [] and dummyCubeTemp != None:
                    dummyCube = dummyCubeTemp[0]
                if dummyCube == "":
                    cmds.polyCube(n="McdMRRenderDummy" + stri, sx=1, sy=1)

                # pading:
                currentFrame = cmds.currentTime(q=True)
                frameNumber = str(int(currentFrame))
                while (len(frameNumber) < 4):
                    frameNumber = "0" + frameNumber

                # link this to MR contents
                if isGZip:
                    cmds.setAttr("McdMRRenderDummy" + stri + ".miProxyFile",
                                 outputPath + "/" + outputName + "_1." + stri + ".mi.gz", type="string")
                else:
                    cmds.setAttr("McdMRRenderDummy" + stri + ".miProxyFile",
                                 outputPath + "/" + outputName + "_1." + stri + ".mi", type="string")

                cmds.setKeyframe("McdMRRenderDummy" + stri, v=0, at='v', t=frameNumberNum - 1)
                cmds.setKeyframe("McdMRRenderDummy" + stri, v=1, at='v', t=frameNumberNum)
                cmds.setKeyframe("McdMRRenderDummy" + stri, v=0, at='v', t=frameNumberNum + 1)

                childNode = cmds.listRelatives("McdMRRenderDummy" + stri, c=True, p=False)[0]

                if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                    mentalray.renderProxyUtils.resizeToBoundingBox(childNode)
                else:
                    maya.app.mentalray.renderProxyUtils.resizeToBoundingBox(childNode)

                # 2222222222222222222222222222222222222222222222222222222222222222222222222222222
                # create box and naming it to McdMRRenderDummy_
                dummyCubeTemp = cmds.ls("McdMRRenderDummy" + stri + "_")
                dummyCube = ""
                if dummyCubeTemp != [] and dummyCubeTemp != None:
                    dummyCube = dummyCubeTemp[0]
                if dummyCube == "":
                    cmds.polyCube(n="McdMRRenderDummy" + stri + "_", sx=1, sy=1)

                # pading:
                currentFrame = cmds.currentTime(q=True)
                frameNumber = str(int(currentFrame))
                while (len(frameNumber) < 4):
                    frameNumber = "0" + frameNumber

                # link this to MR contents
                if isGZip:
                    cmds.setAttr("McdMRRenderDummy" + stri + "_" + ".miProxyFile",
                                 outputPath + "/" + outputName + "_2." + stri + ".mi.gz", type="string")
                else:
                    cmds.setAttr("McdMRRenderDummy" + stri + "_" + ".miProxyFile",
                                 outputPath + "/" + outputName + "_2." + stri + ".mi", type="string")

                cmds.setKeyframe("McdMRRenderDummy" + stri + "_", v=0, at='v', t=frameNumberNum - 1)
                cmds.setKeyframe("McdMRRenderDummy" + stri + "_", v=1, at='v', t=frameNumberNum)
                cmds.setKeyframe("McdMRRenderDummy" + stri + "_", v=0, at='v', t=frameNumberNum + 1)

                childNode = cmds.listRelatives("McdMRRenderDummy" + stri + "_", c=True, p=False)[0]

                if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                    mentalray.renderProxyUtils.resizeToBoundingBox(childNode)
                else:
                    maya.app.mentalray.renderProxyUtils.resizeToBoundingBox(childNode)

            else:

                # execute
                try:
                    cmds.select("MDGGRPMASTER")
                except:
                    cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                    return
                exepandDisplaySelection()

                if isGZip:
                    if isBlockMat:
                        melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 \
                                -fis  -fcd  -pcm  -as  -asn "' + outputName + '.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                                 outputName + '.' + stri + '.mi' + '"'
                    else:
                        melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -compression 1 -fem -fma \
                                -fis  -fcd  -pcm  -as  -asn "' + outputName + '.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                                 outputName + '.' + stri + '.mi' + '"'
                else:
                    if isBlockMat:
                        melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe \
                                -fis  -fcd  -pcm  -as  -asn "' + outputName + '.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                                 outputName + '.' + stri + '.mi' + '"'
                    else:
                        melCmd = 'Mayatomr -mi -exportFilter 721600 -active -binary -fe -fem -fma \
                                -fis  -fcd  -pcm  -as  -asn "' + outputName + '.' + stri + '" -xp "' + xpScheme + '" -file "' + outputPath + '/' + \
                                 outputName + '.' + stri + '.mi' + '"'

                mel.eval(melCmd)

                # create box and naming it to McdMRRenderDummy
                dummyCubeTemp = cmds.ls("McdMRRenderDummy" + stri)
                dummyCube = ""
                if dummyCubeTemp != [] and dummyCubeTemp != None:
                    dummyCube = dummyCubeTemp[0]
                if dummyCube == "":
                    cmds.polyCube(n="McdMRRenderDummy" + stri, sx=1, sy=1)

                # pading:
                currentFrame = cmds.currentTime(q=True)
                frameNumber = str(int(currentFrame))
                while (len(frameNumber) < 4):
                    frameNumber = "0" + frameNumber

                # link this to MR contents
                if isGZip:
                    cmds.setAttr("McdMRRenderDummy" + stri + ".miProxyFile",
                                 outputPath + "/" + outputName + "." + stri + ".mi.gz", type="string")
                else:
                    cmds.setAttr("McdMRRenderDummy" + stri + ".miProxyFile",
                                 outputPath + "/" + outputName + "." + stri + ".mi", type="string")

                cmds.setKeyframe("McdMRRenderDummy" + stri, v=0, at='v', t=frameNumberNum - 1)
                cmds.setKeyframe("McdMRRenderDummy" + stri, v=1, at='v', t=frameNumberNum)
                cmds.setKeyframe("McdMRRenderDummy" + stri, v=0, at='v', t=frameNumberNum + 1)

                childNode = cmds.listRelatives("McdMRRenderDummy" + stri, c=True, p=False)[0]

                if int(mel.eval("getApplicationVersionAsFloat")) >= 2013:
                    mentalray.renderProxyUtils.resizeToBoundingBox(childNode)
                else:
                    maya.app.mentalray.renderProxyUtils.resizeToBoundingBox(childNode)


def MDExpandMRRScene(isAllScene):
    if isAllScene == 0:
        # single frame

        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputMIFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputMIName")

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        # create extra path
        try:
            os.makedirs(outputPath + "/" + outputName)
        except:
            pass

        if not os.access(outputPath + "/" + outputName, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        melCmd = 'mentalrayBatchExportProcedure( "' + outputPath + '/' + outputName + '.mi' + '"," -binary -pcm  -pud  -xp \\\"3313323333\\\"");'
        print melCmd

        mel.eval(melCmd)



    elif isAllScene == 1:
        # all frames


        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputMIFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputMIName")

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        # create extra path
        try:
            os.makedirs(outputPath + "/" + outputName)
        except:
            pass

        if not os.access(outputPath + "/" + outputName, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write mi file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        minFrame = cmds.playbackOptions(q=True, min=True)
        maxFrame = cmds.playbackOptions(q=True, max=True)
        nbFrame = int(maxFrame - minFrame + 1)

        for i in range(nbFrame):

            frameNumberNum = int(minFrame + i)
            cmds.currentTime(frameNumberNum)

            stri = str(frameNumberNum)

            while len(stri) < 4:
                stri = "0" + stri

            # execute
            melCmd = 'mentalrayBatchExportProcedure( "' + outputPath + '/' + outputName + '.' + stri + '.mi' + '"," -binary -pcm  -pud  -xp \\\"3313323333\\\"");'
            print melCmd

            mel.eval(melCmd)


def exepandDisplaySelection():
    selObj = cmds.ls(sl=True)[0]
    allChildren = getAllChildren(selObj)
    shouldSel = []
    for i in range(len(allChildren)):
        if cmds.getAttr(allChildren[i] + ".v") == 1:
            shouldSel.append(allChildren[i])

    cmds.select(shouldSel)


################################################################################
def MDExpandVRProxy(isAllScene):
    if isAllScene == 0:
        # single frame

        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputVRFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputVRName")

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write vrmesh file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write vrmesh file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

            # execute
        try:
            cmds.select("MDGGRPMASTER")
        except:
            cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
            return

        melCmd = 'vrayCreateProxy -exportType 1 -previewFaces 0 -dir "' + outputPath + '" -fname "' + outputName + '.vrmesh" -overwrite'
        mel.eval(melCmd)

        dummyCubeTemp = cmds.ls("McdVRRenderDummy")
        if dummyCubeTemp != [] and dummyCubeTemp != None:
            cmds.delete(dummyCubeTemp[0])

        melCmd = 'vrayCreateProxy -node "McdVRRenderDummy" -dir "' + outputPath + "/" + outputName + '.vrmesh" -existing -createProxyNode'
        mel.eval(melCmd)

        vrShape = cmds.listRelatives("McdVRRenderDummy", p=False, c=True)[0]
        allConns = cmds.listConnections(vrShape, s=True, d=False)
        for j in range(len(allConns)):
            if cmds.nodeType(allConns[j]) == "VRayMesh":
                cmds.setAttr(allConns[j] + ".showBBoxOnly", 1)


    elif isAllScene == 1:
        # all frames

        try:
            globalNode = mel.eval("McdSimpleCommand -exe 2;")
            if globalNode == "_NULL_":
                raise
        except:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy Global Node, please create it in Miarmy > Miarmy Ready")
            return;

        # read export path name from MGlobal
        outputPath = cmds.getAttr(globalNode + ".outputVRFolder")
        # read export file name from MGlobal
        outputName = cmds.getAttr(globalNode + ".outputVRName")

        # check availablity
        if outputPath == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot write vrmesh file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        if not os.access(outputPath, os.W_OK):
            cmds.confirmDialog(t="Error",
                               m="Cannot write vrmesh file to disk, specify right path in: \nMiarmy > Render Global > Other Renders Tab")
            return;

        minFrame = cmds.playbackOptions(q=True, min=True)
        maxFrame = cmds.playbackOptions(q=True, max=True)
        nbFrame = int(maxFrame - minFrame + 1)

        for i in range(nbFrame):

            frameNumberNum = int(minFrame + i)
            cmds.currentTime(frameNumberNum)
            stri = str(frameNumberNum)

            while len(stri) < 4:
                stri = "0" + stri

            # execute
            try:
                cmds.select("MDGGRPMASTER")
            except:
                cmds.confirmDialog(t="Error", m="Please duplicate mesh and try again.")
                return

            melCmd = 'vrayCreateProxy -exportType 1 -previewFaces 0 -dir "' + outputPath + '" -fname "' + outputName + '.' + stri + '.vrmesh" -overwrite'
            mel.eval(melCmd)

            dummyCubeTemp = cmds.ls("McdVRRenderDummy" + stri)
            if dummyCubeTemp != [] and dummyCubeTemp != None:
                cmds.delete(dummyCubeTemp[0])

            melCmd = 'vrayCreateProxy -node "McdVRRenderDummy' + stri + '" -dir "' + outputPath + "/" + outputName + '.' \
                     + stri + '.vrmesh" -existing -createProxyNode'
            mel.eval(melCmd)

            cmds.setKeyframe("McdVRRenderDummy" + stri, v=0, at='v', t=frameNumberNum - 1)
            cmds.setKeyframe("McdVRRenderDummy" + stri, v=1, at='v', t=frameNumberNum)
            cmds.setKeyframe("McdVRRenderDummy" + stri, v=0, at='v', t=frameNumberNum + 1)

            vrShape = cmds.listRelatives("McdVRRenderDummy" + stri, p=False, c=True)[0]
            allConns = cmds.listConnections(vrShape, s=True, d=False)
            for j in range(len(allConns)):
                if cmds.nodeType(allConns[j]) == "VRayMesh":
                    cmds.setAttr(allConns[j] + ".showBBoxOnly", 1)


def displayGeoMD():
    allSelObjs = cmds.ls(sl=True)
    if MIsBlank(allSelObjs):
        cmds.confirmDialog(t="Abort", m="Please select some duplicated agents. (MDGGrp_*)")
        return

    allMDObj = []
    for i in range(len(allSelObjs)):
        if allSelObjs[i].find("MDGGrp_") >= 0:
            allMDObj.append(allSelObjs[i])

    if MIsBlank(allMDObj):
        cmds.confirmDialog(t="Abort", m="Please select some duplicated agents (MDGGrp_*)")
        return

    for i in range(len(allMDObj)):
        allMesh = cmds.listRelatives(allMDObj[i], c=True, p=False)
        if MIsBlank(allMesh):
            continue

        for j in range(len(allMesh)):
            try:
                cmds.setAttr(allMesh[j] + ".overrideEnabled", 0)
            except:
                pass


def hideGeoMD():
    allSelObjs = cmds.ls("MDGGrp_*")
    if MIsBlank(allSelObjs):
        return

    for i in range(len(allSelObjs)):
        allMesh = cmds.listRelatives(allSelObjs[i], c=True, p=False)
        if MIsBlank(allMesh):
            continue

        for j in range(len(allMesh)):
            try:
                cmds.setAttr(allMesh[j] + ".overrideEnabled", 1)
            except:
                pass


def McdAddAgentSeedToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.addAttr(childNode, ln="agSeed", at="long")
        except:
            try:
                cmds.setAttr(childNode + ".agSeed", 1)
            except:
                pass


def McdDelAgentSeedToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.deleteAttr(n=childNode, at="agSeed")
        except:
            try:
                cmds.setAttr(childNode + ".agSeed", -1)
            except:
                pass


def McdAddReadTCToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.addAttr(childNode, ln="readTC", at="bool")
            cmds.setAttr(childNode + ".readTC", 1)
            cmds.setAttr(childNode + ".readTC", k=True)
        except:
            try:
                cmds.setAttr(childNode + ".readTC", 1)
            except:
                pass


def McdDelReadTCToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.setAttr(childNode + ".readTC", 0)
        except:
            pass


def McdAddReadAIDToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.addAttr(childNode, ln="readPlaid", at="bool")
            cmds.setAttr(childNode + ".readPlaid", 1)
            cmds.setAttr(childNode + ".readPlaid", k=True)
        except:
            try:
                cmds.setAttr(childNode + ".readPlaid", 1)
            except:
                pass


def McdDelReadAIDToSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.setAttr(childNode + ".readPlaid", 0)
        except:
            pass


def McdAddControlTexSelAttrNameSelected():
    selObj = cmds.ls(sl=True)
    if MIsBlank(selObj):
        return

    attrName = ""
    stat = cmds.promptDialog(t="Attr Name", m="Please specify the attribute name which control the texture selection", \
                             button=["OK", "Cancel"], \
                             defaultButton="OK", cancelButton="Cancel", dismissString="Cancel")
    if stat == "OK":
        attrName = cmds.promptDialog(query=True, text=True)
    if stat == "Cancel" or attrName == "":
        return;

    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c=True, p=False)[0]
            cmds.addAttr(childNode, ln=attrName, at="bool")
            cmds.setAttr(childNode + "." + attrName, 1)
            cmds.setAttr(childNode + "." + attrName, k=True)
        except:
            try:
                cmds.setAttr(childNode + "." + attrName, 1)
            except:
                pass


def McdClearRearrangedMeshes():
    dupRoot = cmds.ls("McdRoot_*")
    if not MIsBlank(dupRoot):
        cmds.delete(dupRoot)


def McdRearrangeDuplicatedMeshes():
    # list all Geometry_XXX
    # list all type ids

    dupAgent0 = cmds.ls("MDGGrp_0")
    if MIsBlank(dupAgent0):
        cmds.confirmDialog(t="Error", m="Need to duplicat mesh firstly.")
        return

    dupRoot = cmds.ls("McdRoot_*")
    if not MIsBlank(dupRoot):
        cmds.confirmDialog(t="Error",
                           m="Already re-arranged. Please clean and duplicate mesh, and then re-arrange again.")
        return

    allGeoGrp = []
    allTypeId = []
    allNamespace = []

    melcmdgetat = "McdSimpleCommand -exe 19"
    allAgtGrp = mel.eval(melcmdgetat)
    for i in range(len(allAgtGrp)):
        allChildren = cmds.listRelatives(allAgtGrp[i], p=False, c=True)
        if not MIsBlank(allChildren):
            for j in range(len(allChildren)):
                if allChildren[j].find("OriginalMesh_") >= 0:

                    allGeoGrp.append(allChildren[j])
                    allTypeId.append(cmds.getAttr(allAgtGrp[i] + ".tempTypeId"))
                    if allAgtGrp[i].find(":") > 0:
                        allNamespace.append(allAgtGrp[i].split(":")[0])
                    else:
                        allNamespace.append(".")

                    break
                else:
                    if allChildren[j].find("Geometry_") >= 0:
                        allGeoGrp.append(allChildren[j])
                        allTypeId.append(cmds.getAttr(allAgtGrp[i] + ".tempTypeId"))
                        if allAgtGrp[i].find(":") > 0:
                            allNamespace.append(allAgtGrp[i].split(":")[0])
                        else:
                            allNamespace.append(".")

                        break

    if allGeoGrp == []:
        return

    allAgentTypeNames = []
    allTypeIdIndices = []
    for i in range(len(allAgtGrp)):
        agentTypeName = allAgtGrp[i].split("gent_")[1]
        allAgentTypeNames.append(agentTypeName)

        tid = cmds.getAttr(allAgtGrp[i] + ".tid")

        allTypeIdIndices.append(tid)

    # print allAgtGrp
    # print allNamespace
    # print allTypeIdIndices;



    # build root of each type with planes
    for i in range(len(allGeoGrp)):
        # duplicate all OriginalMesh_XXX
        dupNode = cmds.duplicate(allGeoGrp[i])

        rootTemplate = "McdRoot_" + str(allTypeId[i])
        cmds.rename(dupNode[0], rootTemplate)
        # parent plane into it
        cmds.parent(rootTemplate, w=True)

        allSubNodes = cmds.listRelatives(rootTemplate, p=False, c=True, ad=True, fullPath=True)
        for j in range(len(allSubNodes)):
            if cmds.getAttr(allSubNodes[j] + ".intermediateObject") == 1:
                cmds.delete(allSubNodes[j])
                continue

            if cmds.nodeType(allSubNodes[j]) == "mesh":
                shapeNode = allSubNodes[j]
                shapeTransNode = cmds.listRelatives(allSubNodes[j], p=True, c=False, path=True)[0]
                planeNodes = cmds.polyPlane(sx=1, sy=1)
                planeTrans = planeNodes[0]
                planeShape = cmds.listRelatives(planeTrans, p=False, c=True, path=True)[0]

                cmds.parent(planeShape, shapeTransNode, r=True, s=True)
                cmds.delete(planeTrans)
                cmds.delete(shapeNode)

                shapeNodeShortName = shapeNode.split("|")[-1]
                cmds.rename(planeShape, shapeNodeShortName)

    melCmd = "McdMakeJointCacheCmd -am 2;"
    typeList = mel.eval(melCmd)

    # print typeList

    for i in range(len(typeList)):
        agentTypeName = allAgentTypeNames[typeList[i]]
        agentGeoRoot = "McdRoot_" + str(typeList[i])
        namespace = allNamespace[typeList[i]]
        dupNode = cmds.duplicate(agentGeoRoot)

        stri = str(i)
        agentMDRoot = "McdAgentGeometry_" + stri + "_" + str(typeList[i])
        cmds.rename(dupNode[0], agentMDRoot)

        agentDupGrp = "MDGGrp_" + stri
        allDupGeo = cmds.listRelatives(agentDupGrp, c=True, p=False)
        allDupGeoShortName = []

        for j in range(len(allDupGeo)):
            a = allDupGeo[j]
            if namespace == ".":
                b = a.split("MDG_")
            else:
                b = a.split("MDG_" + namespace + "_")

            # print namespace
            # print agentTypeName
            if b[1].find("_" + agentTypeName + "Geo") >= 0:
                c = b[1].split("_" + agentTypeName + "Geo")
                geoName = c[0] + "Shape"
                allDupGeoShortName.append(geoName)
            else:
                lastUD = b[1].rfind("_")
                geoName = b[1][0:lastUD]
                allDupGeoShortName.append(geoName)

        allChildren = cmds.listRelatives(agentMDRoot, c=True, p=False, ad=True)
        allChildren2 = cmds.listRelatives(agentMDRoot, c=True, p=False, ad=True, fullPath=True)
        for j in range(len(allChildren2)):
            if cmds.nodeType(allChildren2[j]) == "mesh":
                try:
                    idx = allDupGeoShortName.index(allChildren[j])
                except:
                    idx = -1
                if idx != -1:
                    parentNode = cmds.listRelatives(allChildren2[j], c=False, p=True, path=True)[0]

                    cmds.parent(allDupGeo[idx], parentNode, r=True, s=True)
                    randomizeAttrVal(i, j, allDupGeo[idx], parentNode)

                    cmds.delete(allChildren2[j])
                    cmds.rename(allDupGeo[idx], allChildren[j])


def McdVisControlBasedOnActionCache():
    # read vis file
    globalNode = McdGetMcdGlobalNode()

    rawPath = cmds.getAttr(globalNode + ".cacheFolder")
    cacheFolder = envPath2AbsPath(rawPath)

    cacheName = cmds.getAttr(globalNode + ".cacheName")

    currentFrameRead = cmds.currentTime(q=True)
    frameStr = str(int(currentFrameRead))

    cachePath = cacheFolder + "/" + cacheName + "." + frameStr + ".mac"

    if not os.access(cachePath, os.R_OK):
        return;

    f = open(cachePath)
    contents = f.read()
    f.close()

    if contents.find("\r\r\n"):
        contentsList = contents.split("\r\r\n")
    else:
        contentsList = contents.split("\r\n")

    # find all vis rules:
    allHideNodes = cmds.ls(type="McdGeoVisInfo")
    if MIsBlank(allHideNodes):
        return

    geoNameList = []
    actionNameList = []
    startFrameList = []
    endFrameList = []
    for i in range(len(allHideNodes)):
        geoName = cmds.getAttr(allHideNodes[i] + ".geometryName")
        geoNameList.append(geoName)

        actionName = cmds.getAttr(allHideNodes[i] + ".actionName")
        actionNameList.append(actionName)

        startFrame = cmds.getAttr(allHideNodes[i] + ".startFrame")
        startFrameList.append(startFrame)

        endFrame = cmds.getAttr(allHideNodes[i] + ".endFrame")
        endFrameList.append(endFrame)

    # print geoNameList
    # print actionNameList
    # print startFrameList
    # print endFrameList


    # print "-------------------------------------------------------------------"
    agentCounter = 0
    for i in range(len(contentsList)):
        if len(contentsList[i]) > 6:
            agentName = "MDGGrp_" + str(agentCounter)
            agentCounter += 1
            agentNodes = cmds.ls(agentName)
            if MIsBlank(agentNodes):
                continue
            if len(agentNodes) != 1:
                continue

            actionInfo = contentsList[i].split(" ")
            if len(actionInfo) >= 3:
                actionNameRaw = actionInfo[0].split('|')[-1]
                actionNameRaw2 = actionNameRaw.split(':')[-1]

                action = actionNameRaw2.split("_action_")[0]
                frame = int(float(actionInfo[1]))
            else:
                continue

            agentNode = agentNodes[0]

            # print "    >>>>>>>>>>>   " + agentNode

            allChild = cmds.listRelatives(agentNode, p=False, c=True)

            for j in range(len(allChild)):
                shapeName = allChild[j].split("MDG_")[-1]
                foundThisNode = False
                for k in range(len(geoNameList)):
                    if shapeName.find(geoNameList[k]) >= 0:
                        # find a shape:
                        # print "xxxx"
                        # print actionNameList[k]
                        # print action
                        # print frame
                        # print startFrameList[k]
                        # print endFrameList[k]
                        if (actionNameList[k] == action) and (frame >= startFrameList[k]) and (
                            frame <= endFrameList[k]):
                            foundThisNode = True
                            break
                if foundThisNode:
                    cmds.setAttr(allChild[j] + ".v", 0)
                else:
                    cmds.setAttr(allChild[j] + ".v", 1)


def putAllShapeOutToGroup():
    # list all Geometry_XXX
    # list all type ids

    dupAgent0 = cmds.ls("MDGGrp_0")
    if MIsBlank(dupAgent0):
        cmds.confirmDialog(t="Error", m="Need to duplicat mesh firstly.")
        return

    allDupAgents = cmds.ls("MDGGrp_*")

    for i in range(len(allDupAgents)):
        stri1 = str(i + 1)
        newParent = "MDGAgent_" + stri1
        cmds.group(n=newParent, em=True)
        cmds.parent(newParent, "MDGGRPMASTER")

        allShapes = cmds.listRelatives(allDupAgents[i], c=True, p=False)

        for j in range(len(allShapes)):
            print allShapes[j]

            newName = allShapes[j].replace("MDG_", "MDAG_")
            cmds.group(n=newName, em=True)
            cmds.parent(newName, newParent)

            cmds.parent(allShapes[j], newName, r=True, s=True)

        cmds.delete(allDupAgents[i])


def randomizeAttrVal(ii, jj, childNode, parentNode):
    allAttrs = cmds.listAttr(childNode, ud=True)

    if allAttrs == None or allAttrs == []:
        return

    for i in range(len(allAttrs)):
        if allAttrs[i].find("MID_") == 0:
            val = cmds.getAttr(childNode + "." + allAttrs[i])

            min = 0
            max = int(val)

            seed = int((float(ii) + 0.889) * 1.71512 + (float(jj) + 0.789) * 1.77155)

            randNum = int(McdSolveASeedMinMax(seed, 11.457, min, max))

            try:
                cmds.addAttr(parentNode, ln=allAttrs[i], at="long")
            except:
                pass

            cmds.setAttr(parentNode + "." + allAttrs[i], randNum)
            cmds.setAttr(childNode + "." + allAttrs[i], randNum)