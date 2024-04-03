#!/usr/bin/env python

import maya.cmds as cmds
import maya.mel as mel
import McdOriginalAgentFunctions
from McdMakeAgentCache import *
from McdGeneral import *
from McdRenderFBXFunctions import *
from McdSimpleCmd import *
import time

def McdCheckAgentGroupNode():
    allAGs = cmds.ls(type = "McdAgentGroup")
    
    if MIsBlank(allAGs):
        raise Exception('There is no Agent Group Found')
        
    for i in range(len(allAGs)):
        allParents = cmds.listRelatives(allAGs[i], c = False, p = True)
        if allParents == [] or allParents == None:
            raise Exception('Agent Group\'s Parent should be Miarmy_Contents.')
            
        if allParents[0] != "Miarmy_Contents":
            parentOfParent = cmds.listRelatives(allParents[0], c = False, p = True)
            if parentOfParent == [] or parentOfParent == None:
                raise Exception('Agent Group\'s Parent should be Miarmy_Contents.')
                
            if parentOfParent[0] != "Miarmy_Contents":
                raise Exception('Agent Group\'s Parent should be Miarmy_Contents.')
        
    # copy old version setup if existed:
    try:
        McdCopyOldSetupIfExist(allAGs)
    except:
        pass
    
    
    # adding extra attr for upgrading to latest version if not have:
    allAGs = cmds.ls("*_dummyShape_*")
    
    isUp2Date = False
    for i in range(len(allAGs)):
        try:
            a = cmds.getAttr(allAGs[i] + ".endBoneDraw")
            isUp2Date = True
            break
        except:
            pass
        
    
    if not isUp2Date:
        McdOriginalAgentFunctions.McdUpgradeOAgent()
    
        
def McdCopyOldSetupIfExist(allAGs):
    
    for i in range(len(allAGs)):
        oldSphereRange = cmds.getAttr(allAGs[i] + ".soundRange")
        newSphereRange = cmds.getAttr(allAGs[i] + ".sphereRange")
        
        if oldSphereRange > 0.1 and newSphereRange < 0.1:
            print "copy old sphere range: " + str(oldSphereRange)
            cmds.setAttr(allAGs[i] + ".sphereRange", oldSphereRange)
        
        oldSphereColor = cmds.getAttr(allAGs[i] + ".soundFreq")
        newSphereColor = cmds.getAttr(allAGs[i] + ".sphereColor")
        if oldSphereColor != 1 and newSphereColor == 1:
            print "copy old sphere color: " + str(oldSphereColor)
            cmds.setAttr(allAGs[i] + ".sphereColor", oldSphereColor)
        elif newSphereColor == 0:
            print "copy old sphere color: " + str(oldSphereColor)
            cmds.setAttr(allAGs[i] + ".sphereColor", oldSphereColor)
    
        
def McdCheckCompoundRBDFlags():
    # for all joints
    allJnts = cmds.ls(type = "joint")
    
    # find isCompound 1, feel collide 1,
    # if found, find until the is NOT compund 
    for i in range(len(allJnts)):
        try:
            isCom = cmds.getAttr(allJnts[i] + ".compoundRBD")
            isfC = cmds.getAttr(allJnts[i] + ".collideFeel")
            
            if isCom and isfC:
                compoundRoot = findCompoundRoot(allJnts[i])
                if compoundRoot != None:
                    allChildren = cmds.listRelatives(compoundRoot, c = True, p = False)
                    if not MIsBlank(allChildren):
                        for j in range(len(allChildren)):
                            try:
                                cmds.setAttr(allChildren[j] + ".collideFeel", True)
                            except:
                                pass
            
        except:
            continue
            
    # find isCompound 1, feel collide 1,
    # if found, find until the is NOT compund 
    for i in range(len(allJnts)):
        try:
            isCom = cmds.getAttr(allJnts[i] + ".compoundRBD")
            isfC = cmds.getAttr(allJnts[i] + ".collideCloth")
            
            if isCom and isfC:
                compoundRoot = findCompoundRoot(allJnts[i])
                if compoundRoot != None:
                    allChildren = cmds.listRelatives(compoundRoot, c = True, p = False)
                    if not MIsBlank(allChildren):
                        for j in range(len(allChildren)):
                            try:
                                cmds.setAttr(allChildren[j] + ".collideCloth", True)
                            except:
                                pass
            
        except:
            continue
        
        
    
def findCompoundRoot(inJnt):
    
    inputNode = inJnt;
    while True:
        allParents = cmds.listRelatives(inputNode, c = False, p = True, path = True)
        if allParents != None:
            inputNode = allParents[0]
        else:
            break
        try:
            isCom = cmds.getAttr(inputNode + ".compoundRBD")
            if not isCom:
                return inputNode
        except:
            return None
        


def McdCreatePlacementNode():
    
    miarmyMain = cmds.ls("Miarmy_Contents")
    if miarmyMain == [] or miarmyMain == None:
        cmds.confirmDialog(t = "Error", m = 'Cannot find "Miarmy_Contents" Group.')
        raise Exception('Cannot find "Miarmy_Contents" Group.')
    placementGrp = cmds.ls("Placement_Set")
    if placementGrp == [] or placementGrp == None:
        #create one and parent
        cmds.group(n = "Placement_Set", em = True)
        cmds.parent("Placement_Set", "Miarmy_Contents")
    else:
        #try to parent
        try:
            placeParent = cmds.listRelatives("Placement_Set", c = False, p = True)[0]
            if placeParent != "Miarmy_Contents":
                cmds.parent("Placement_Set", "Miarmy_Contents")
        except:
            pass
        
        
    #create node and place it to group
    nbPlace = str(McdGetNumOfThisType("McdPlace"))
    newNodeTrans = cmds.createNode("transform", n = "McdPlace" + nbPlace)
    newNode = cmds.createNode("McdPlace", n = "McdPlace" + nbPlace + "Shape", p = newNodeTrans)
    
    tranNode = cmds.listRelatives(newNode, parent = True, c = False, path = True)[0]
    cmds.setAttr(tranNode + ".placeType", 4)
    
    cmds.connectAttr(tranNode + ".tx", newNode + ".localPositionX")
    cmds.connectAttr(tranNode + ".ty", newNode + ".localPositionY")
    cmds.connectAttr(tranNode + ".tz", newNode + ".localPositionZ")
    
    cmds.setAttr(tranNode + ".rx", lock = True, k = False)
    cmds.setAttr(tranNode + ".ry", lock = True, k = False)
    cmds.setAttr(tranNode + ".rz", lock = True, k = False)
    
    cmds.setAttr(tranNode + ".sx", lock = True, k = False)
    cmds.setAttr(tranNode + ".sy", lock = True, k = False)
    cmds.setAttr(tranNode + ".sz", lock = True, k = False)
    
    cmds.setAttr(newNode + ".localPositionX", cb = False)
    cmds.setAttr(newNode + ".localPositionY", cb = False)
    cmds.setAttr(newNode + ".localPositionZ", cb = False)
    
    cmds.setAttr(newNode + ".localScaleX", cb = False)
    cmds.setAttr(newNode + ".localScaleY", cb = False)
    cmds.setAttr(newNode + ".localScaleZ", cb = False)
    
    cmds.setAttr(newNode + ".proportion[0]", 0)
    cmds.setAttr(newNode + ".proportion[1]", 0)
    cmds.setAttr(newNode + ".proportion[2]", 1)
    cmds.setAttr(newNode + ".proportion[3]", 0)
    cmds.setAttr(newNode + ".proportion[4]", 0)
    cmds.setAttr(newNode + ".proportion[5]", 0)
    
    cmds.parent(tranNode, "Placement_Set")

def unhideTargetPlaceNode():
    allPlaceNodes = cmds.ls(type = "McdPlace")
    if not MIsBlank(allPlaceNodes):
        for i in range(len(allPlaceNodes)):
            try:
                if cmds.getAttr(allPlaceNodes[i] + ".asTarget") == 1:
                    transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
                    cmds.showHidden(transform)
            except:
                pass
            


def placementAgent():
    
    #newEngine = cmds.optionVar( q='NewMiarmyFlag' )
    #if newEngine < 35:
    #    # prompt:
    #    stat = cmds.confirmDialog(t = "New Miarmy Note", m = "Dear Miarmy User, \nWe detected you're using new Miarmy 3.6 or higher." +  \
    #                                                        "\n\nSince 3.6, we added a new \"Human Language\" logic engine into our system" + \
    #                                                        "\nAnd in future, we're going to use and document new \"Language Mode\" Engine" + \
    #                                                        "\n\nPlease read more about this if you don't know the details", \
    #                              
    #                                        b = ["I know, do not mention again", "Read the Detail", "Cancel"])
    #    
    #    # store new flag
    #    if stat == "I know, do not mention again":
    #        cmds.optionVar( iv=('NewMiarmyFlag', 36) )
    #        cmds.confirmDialog(t = "About", m = 'If you want to know later, you can also get help in Miarmy Global')
    #        cmds.launch(web="https://basefount.atlassian.net/wiki/display/MDE/Miarmy+3.6+Engine+Upgrade")
    #    elif stat == "Read the Detail":
    #        cmds.launch(web="https://basefount.atlassian.net/wiki/display/MDE/Miarmy+3.6+Engine+Upgrade")
    #    else:
    #        pass
    allGlobalNodes = cmds.ls(type = "McdGlobal")
    if len(allGlobalNodes) > 1:
        for i in range(len(allGlobalNodes)):
            if i == 0:
                continue
            try:
                cmds.delete(allGlobalNodes[i])
                cmds.confirmDialog(t = "Note", m = 'Detected trash McdGlobal node, already remove it for you.')
            except:
                pass
            
    allGlobalNodes = cmds.ls(type = "McdBrain")
    if len(allGlobalNodes) > 1:
        for i in range(len(allGlobalNodes)):
            if i == 0:
                continue
            try:
                cmds.delete(allGlobalNodes[i])
                cmds.confirmDialog(t = "Note", m = 'Detected trash McdBrain node, already remove it for you.')
            except:
                pass
    
    try:
        allPS = cmds.ls("*Perception_Set*")
        if not MIsBlank(allPS):
            for i in range(len(allPS)):
                cmds.setAttr(allPS[i] + ".t", 0, 0, 0)
                cmds.setAttr(allPS[i] + ".r", 0, 0, 0)
                cmds.setAttr(allPS[i] + ".s", 1, 1, 1)
                cmds.setAttr(allPS[i] + ".t", lock = True)
                cmds.setAttr(allPS[i] + ".r", lock = True)
                cmds.setAttr(allPS[i] + ".s", lock = True)
    except:
        pass
    try:
        allPS = cmds.ls("*Placement_Set*")
        if not MIsBlank(allPS):
            for i in range(len(allPS)):
                cmds.setAttr(allPS[i] + ".t", 0, 0, 0)
                cmds.setAttr(allPS[i] + ".r", 0, 0, 0)
                cmds.setAttr(allPS[i] + ".s", 1, 1, 1)
                cmds.setAttr(allPS[i] + ".t", lock = True)
                cmds.setAttr(allPS[i] + ".r", lock = True)
                cmds.setAttr(allPS[i] + ".s", lock = True)
    except:
        pass
    
    
    
    globalNode = McdGetMcdGlobalNode()
    if not cmds.getAttr(globalNode + ".enableCache"):
        if checkPlaceFromSelection():
            
            stat = cmds.confirmDialog(t = "Question", m = 'Detected one Place Node selected. Do you want to place from selction?', \
                                                        b = ["Place Selection", "Place all", "Cancel"])
            if stat == "Place Selection":
                placementAgentFromSelect()
                return
            elif stat == "Cancel":
                return
    
    McdCheckAgentGroupNode()
    McdCheckCompoundRBDFlags()
    McdDeleteExtraConnectionsASNotActive()
    
    # set time:
    globalNode = McdGetMcdGlobalNode()
    startTime = cmds.playbackOptions(q = True, min = True)
    allBrainNodes = cmds.ls(type = "McdBrain")
    if allBrainNodes != [] and allBrainNodes != None:
        startTimeInNode = cmds.getAttr(allBrainNodes[0] + ".startTime")
        if startTime > startTimeInNode + 0.1:
            stat = cmds.confirmDialog(t = "Question", m = 'Detected StartTime in Brain Node is smaller than value in timeslider, fix it??', \
                                                        b = ["Yes", "Cancel"])
            if stat == "Yes":
                cmds.setAttr(allBrainNodes[0] + ".startTime", startTime)
    
    # check whether inverse placement and normal place node both exited
    # stat = "none"
    # placed = False
    # if checkInverseAndNormalExist() == True:
    #     stat = cmds.confirmDialog(t = "Question", m = 'Detected both "normal place node" and "place node generated from inverse place" existed\n' + \
    #                                                 'Please choose the the way you want to place: ', \
    #                                                 b = ["Place Both", "Place Only from Inverse Placement", "Cancel"])
    #     
    #     if stat == "Place Only from Inverse Placement":
    #         cmd = "McdPlacementCmd -am 0 -ign 1;"
    #         placed = mel.eval(cmd)
    #         McdAfterPlaceFunction()
    #     elif stat == "Place Both":
    #         cmd = "McdPlacementCmd -am 0 -ign 0;"
    #         placed = mel.eval(cmd)
    #         McdAfterPlaceFunction()
    # else:
    #     cmd = "McdPlacementCmd -am 0 -ign 0;"
    #     placed = mel.eval(cmd)
    #     McdAfterPlaceFunction()
    
    cmd = "McdPlacementCmd -am 0 -ign 0;"
    placed = mel.eval(cmd)
    McdAfterPlaceFunction()
        
    if placed:
        isHide = False
        
        hideAttr = cmds.getAttr(globalNode + ".funcParamList[0]")
        if hideAttr == 1:
            isHide = True
        elif hideAttr == 2:
            isHide = False
        else:
            stat = cmds.confirmDialog(t = "Question", m = 'Are you willing to hide all Original Agent and Place Node?', \
                                                    b = ["Hide", "Hide and not mention again", "Show it and not mention again", "No"])
            if stat == "Hide":
                isHide = True
            elif stat == "Hide and not mention again":
                isHide = True
                cmds.setAttr(globalNode + ".funcParamList[0]", 1)
            elif stat == "Show it and not mention again":
                cmds.setAttr(globalNode + ".funcParamList[0]", 2)
        
        if isHide == True:
            allAgentGrp = cmds.ls(type = "McdAgentGroup")
            try:
                cmds.hide(allAgentGrp)
            except:
                pass
            
            allPlaceNodes = cmds.ls(type = "McdPlace")
            try:
                for i in range(len(allPlaceNodes)):
                    transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
                    cmds.hide(transform)
            except:
                pass
    
    # check "PhysX" plugin:
    if cmds.pluginInfo( "physx.mll", query = True, loaded = True ):
        cmds.confirmDialog(t = "Warning", m = "Load both Miarmy and Maya Physx together may cause unstable.\nFor disable this warning, please:\n" + \
                         "* Unload physx.mll plugin\n* Or modify the McdPlacementFunctions.py in you Miarmy installation place.")

    unhideTargetPlaceNode()

def simplePlacementAgent():
    
    globalNode = McdGetMcdGlobalNode()
    
    McdCheckAgentGroupNode()
    McdCheckCompoundRBDFlags()
    McdDeleteExtraConnectionsASNotActive()
    
    # set time:
    globalNode = McdGetMcdGlobalNode()
    startTime = cmds.playbackOptions(q = True, min = True)
    allBrainNodes = cmds.ls(type = "McdBrain")
    if allBrainNodes != [] and allBrainNodes != None:
        startTimeInNode = cmds.getAttr(allBrainNodes[0] + ".startTime")
        if startTime > startTimeInNode + 0.1:
            stat = cmds.confirmDialog(t = "Question", m = 'Detected StartTime in Brain Node is smaller than value in timeslider, fix it??', \
                                                        b = ["Yes", "Cancel"])
            if stat == "Yes":
                cmds.setAttr(allBrainNodes[0] + ".startTime", startTime)
    
    # check whether inverse placement and normal place node both exited
    stat = "none"
    placed = False
    if checkInverseAndNormalExist() == True:
        stat = cmds.confirmDialog(t = "Question", m = 'Detected both "normal place node" and "place node generated from inverse place" existed\n' + \
                                                    'Please choose the the way you want to place: ', \
                                                    b = ["Place Both", "Place Only from Inverse Placement", "Cancel"])
        
        if stat == "Place Only from Inverse Placement":
            cmd = "McdPlacementCmd -am 5 -ign 1;"
            placed = mel.eval(cmd)
            
        elif stat == "Place Both":
            cmd = "McdPlacementCmd -am 5 -ign 0;"
            placed = mel.eval(cmd)

    else:
        cmd = "McdPlacementCmd -am 5 -ign 0;"
        placed = mel.eval(cmd)

        
    if placed:
        isHide = False
        
        hideAttr = cmds.getAttr(globalNode + ".funcParamList[0]")
        if hideAttr == 1:
            isHide = True
        elif hideAttr == 2:
            isHide = False
        else:
            stat = cmds.confirmDialog(t = "Question", m = 'Are you willing to hide all Original Agent and Place Node?', \
                                                    b = ["Hide", "Hide and not mention again", "Show it and not mention again", "No"])
            if stat == "Hide":
                isHide = True
            elif stat == "Hide and not mention again":
                isHide = True
                cmds.setAttr(globalNode + ".funcParamList[0]", 1)
            elif stat == "Show it and not mention again":
                cmds.setAttr(globalNode + ".funcParamList[0]", 2)
        
        if isHide == True:
            allAgentGrp = cmds.ls(type = "McdAgentGroup")
            try:
                cmds.hide(allAgentGrp)
            except:
                pass
            
            allPlaceNodes = cmds.ls(type = "McdPlace")
            try:
                for i in range(len(allPlaceNodes)):
                    transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
                    cmds.hide(transform)
            except:
                pass
    
    # check "PhysX" plugin:
    if cmds.pluginInfo( "physx.mll", query = True, loaded = True ):
        cmds.confirmDialog(t = "Warning", m = "Load both Miarmy and Maya Physx together may cause unstable.\nFor disable this warning, please:\n" + \
                         "* Unload physx.mll plugin\n* Or modify the McdPlacementFunctions.py in you Miarmy installation place.")

    unhideTargetPlaceNode()

def placementAgentRange():
    
    McdCheckAgentGroupNode()
    McdCheckCompoundRBDFlags()
    McdDeleteExtraConnectionsASNotActive()
    
    # set time:
    globalNode = McdGetMcdGlobalNode()
    startTime = cmds.playbackOptions(q = True, min = True)
    allBrainNodes = cmds.ls(type = "McdBrain")
    if allBrainNodes != [] and allBrainNodes != None:
        startTimeInNode = cmds.getAttr(allBrainNodes[0] + ".startTime")
        if startTime > startTimeInNode + 0.1:
            stat = cmds.confirmDialog(t = "Question", m = 'Detected StartTime in Brain Node is smaller than value in timeslider, fix it??', \
                                                        b = ["Yes", "Cancel"])
            if stat == "Yes":
                cmds.setAttr(allBrainNodes[0] + ".startTime", startTime)
    
    # check whether inverse placement and normal place node both exited
    stat = "none"
    placed = False
    if checkInverseAndNormalExist() == True:
        stat = cmds.confirmDialog(t = "Question", m = 'Detected both "normal place node" and "place node generated from inverse place" existed\n' + \
                                                    'Please choose the the way you want to place: ', \
                                                    b = ["Place Both", "Place Only from Inverse Placement", "Cancel"])
        
        if stat == "Place Only from Inverse Placement":
            cmd = "McdPlacementCmd -am 4 -ign 1;"
            placed = mel.eval(cmd)
            McdAfterPlaceFunction()
        elif stat == "Place Both":
            cmd = "McdPlacementCmd -am 4 -ign 0;"
            placed = mel.eval(cmd)
            McdAfterPlaceFunction()
    else:
        cmd = "McdPlacementCmd -am 4 -ign 0;"
        placed = mel.eval(cmd)
        McdAfterPlaceFunction()
        
    if placed:
        isHide = False
        
        hideAttr = cmds.getAttr(globalNode + ".funcParamList[0]")
        if hideAttr == 1:
            isHide = True
        elif hideAttr == 2:
            isHide = False
        else:
            stat = cmds.confirmDialog(t = "Question", m = 'Are you willing to hide all Original Agent and Place Node?', \
                                                    b = ["Hide", "Hide and not mention again", "Show it and not mention again", "No"])
            if stat == "Hide":
                isHide = True
            elif stat == "Hide and not mention again":
                isHide = True
                cmds.setAttr(globalNode + ".funcParamList[0]", 1)
            elif stat == "Show it and not mention again":
                cmds.setAttr(globalNode + ".funcParamList[0]", 2)
        
        if isHide == True:
            allAgentGrp = cmds.ls(type = "McdAgentGroup")
            try:
                cmds.hide(allAgentGrp)
            except:
                pass
            
            allPlaceNodes = cmds.ls(type = "McdPlace")
            try:
                for i in range(len(allPlaceNodes)):
                    transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
                    cmds.hide(transform)
            except:
                pass
    
    # check "PhysX" plugin:
    if cmds.pluginInfo( "physx.mll", query = True, loaded = True ):
        cmds.confirmDialog(t = "Warning", m = "Load both Miarmy and Maya Physx together may cause unstable.\nFor disable this warning, please:\n" + \
                         "* Unload physx.mll plugin\n* Or modify the McdPlacementFunctions.py in you Miarmy installation place.")

    unhideTargetPlaceNode()

def checkPlaceFromSelection():
    
    allPlace = cmds.ls(type = "McdPlace")
    if not MIsBlank(allPlace):
        if len(allPlace) == 1:
            return False
    
    allSels = cmds.ls(sl = True, type = "transform")
    if not MIsBlank(allSels):
        allChild = cmds.listRelatives(allSels[0], c = True, path = True)
        if not MIsBlank(allChild):
            if cmds.nodeType(allChild[0]) == "McdPlace":
                return True
                    
    return False

def placementAgentFromSelect():
    
    McdCheckAgentGroupNode()
    McdCheckCompoundRBDFlags()
    McdDeleteExtraConnectionsASNotActive()
    
    # set time:
    globalNode = McdGetMcdGlobalNode()
    startTime = cmds.playbackOptions(q = True, min = True)
    allBrainNodes = cmds.ls(type = "McdBrain")
    if allBrainNodes != [] and allBrainNodes != None:
        startTimeInNode = cmds.getAttr(allBrainNodes[0] + ".startTime")
        if startTime > startTimeInNode + 0.1:
            stat = cmds.confirmDialog(t = "Question", m = 'Detected StartTime in Brain Node is smaller than value in timeslider, fix it??', \
                                                        b = ["Yes", "Cancel"])
            if stat == "Yes":
                cmds.setAttr(allBrainNodes[0] + ".startTime", startTime)
    
    # check whether inverse placement and normal place node both exited
    stat = "none"
    placed = False

    cmd = "McdPlacementCmd -am 2 -ign 0;"
    placed = mel.eval(cmd)
    McdAfterPlaceFunction()
        
    if placed:
        isHide = False
        
        hideAttr = cmds.getAttr(globalNode + ".funcParamList[0]")
        if hideAttr == 1:
            isHide = True
        elif hideAttr == 2:
            isHide = False
        else:
            stat = cmds.confirmDialog(t = "Question", m = 'Are you willing to hide all Original Agent and Place Node?', \
                                                    b = ["Hide", "Hide and not mention again", "Show it and not mention again", "No"])
            if stat == "Hide":
                isHide = True
            elif stat == "Hide and not mention again":
                isHide = True
                cmds.setAttr(globalNode + ".funcParamList[0]", 1)
            elif stat == "Show it and not mention again":
                cmds.setAttr(globalNode + ".funcParamList[0]", 2)
        
        if isHide == True:
            allAgentGrp = cmds.ls(type = "McdAgentGroup")
            try:
                cmds.hide(allAgentGrp)
            except:
                pass
            
            allPlaceNodes = cmds.ls(type = "McdPlace")
            try:
                for i in range(len(allPlaceNodes)):
                    transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
                    cmds.hide(transform)
            except:
                pass
    
    # check "PhysX" plugin:
    if cmds.pluginInfo( "physx.mll", query = True, loaded = True ):
        cmds.confirmDialog(t = "Warning", m = "Load both Miarmy and Maya Physx together may cause unstable.\nFor disable this warning, please:\n" + \
                         "* Unload physx.mll plugin\n* Or modify the McdPlacementFunctions.py in you Miarmy installation place.")
        
    unhideTargetPlaceNode()
    
def dePlacementAgent():
    
    # #################################
    # delete agents
    # unhide
    # agent return
    # delete all geo without geo cache
    #     # check pre 10 objects
    # clear all MDGGrp_*
    
    McdMarkClothMeshSkinClusterOnAndOff(1)
    
    allMDNodes = cmds.ls(type = "McdMeshDrive")
    if not McdIsBlank(allMDNodes):
        for i in range(len(allMDNodes)):
            try:
                cmds.delete(allMDNodes[i])
            except:
                pass
        
    allAgentShapes = cmds.ls(type = "McdAgent")
    if allAgentShapes != [] and allAgentShapes != None:
        
        # progress window
        counter = 0
        totalCount = len(allAgentShapes)
        cmds.progressWindow( title = "De-place Agents...", progress = 0, min = 0, max = totalCount, status = 'Complete', isInterruptable = False )
        
        cmds.select(clear = True)
        for i in range(len(allAgentShapes)):
            agentParent = cmds.listRelatives(allAgentShapes[i], p = True)[0]
            try:
                cmds.delete(agentParent)
            except:
                pass
            cmds.progressWindow( edit = True, progress = i)
        
        cmds.progressWindow(endProgress=1)
    
    cmds.flushUndo() # clear dump
    

    allAgentDShapes = cmds.ls(type = "McdAgentDummy")
    if not MIsBlank(allAgentDShapes):
        for i in range(len(allAgentDShapes)):
            agentParent = cmds.listRelatives(allAgentDShapes[i], p = True)[0]
            try:
                cmds.delete(agentParent)
            except:
                pass

    
    # showHidden
    allAgentGrp = cmds.ls(type = "McdAgentGroup")
    try:
        cmds.showHidden(allAgentGrp)
    except:
        pass
    
    allPlaceNodes = cmds.ls(type = "McdPlace")
    try:
        for i in range(len(allPlaceNodes)):
            transform = cmds.listRelatives(allPlaceNodes[i], p = True, c = False)[0]
            cmds.showHidden(transform)
    except:
        pass
    
    # agent return:
    try:
        cmd = "McdAgentMatchCmd -mm 0;"
        mel.eval(cmd)
    except:
        pass
    
    allMDMesh = cmds.ls("MDG_*", type = "mesh")
    if allMDMesh == [] or allMDMesh == None:
        cmds.flushUndo() # clear memory
        return
    
    counter = 10
    cached = False
    for i in range(len(allMDMesh)):
        allHis = cmds.listConnections(allMDMesh, s = True, d = False)
        try:
            if allHis != None and allHis != []:
                for j in range(len(allHis)):
                    if cmds.nodeType(allHis[j]) == "historySwitch":
                        cached = True
        except:
            pass
        
        if counter < 0:
            break
        counter -= 1
        
    if not cached:
        stat = cmds.confirmDialog(t = "Question", m = "Do you want to delete duplicated meshes?", b = ["Yes", "No"])
        if stat == "No":
            cmds.flushUndo() # clear memory
            return
    
        allGrps = cmds.ls("MDGGrp_*", l = True)
        if allGrps != [] and allGrps != None:
            for i in range(len(allGrps)):
                try:
                    cmds.delete(allGrps[i])
                except:
                    pass

        allGrps = cmds.ls("MDG_*", l = True)
        if allGrps != [] and allGrps != None:
            for i in range(len(allGrps)):
                try:
                    cmds.delete(allGrps[i])
                except:
                    pass
        
        stat = cmds.confirmDialog(t = "Question", m = "Do you want to delete useless shader?", b = ["Yes", "No"])
        if stat == "Yes":
            McdClearUselessShader()
        
        
    cmds.flushUndo() # clear dump

def McdAttachTerrain():
    selObj = cmds.ls(sl = True)
    if selObj == [] or selObj == None:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
        raise Exception('First select placement node, then terrain geometry.')
    if len(selObj) < 2:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
        raise Exception('First select placement node, then terrain geometry.')
    
    placeNode = ""
    placeTransform = ""
    terrainNode = ""
    terrainTransform = ""
    if cmds.nodeType(selObj[0]) == "McdPlace":
        placeNode = selObj[0]
        placeTransform = cmds.listRelatives(placeNode, c = False, p = True)[0]
    else:
        allChild = cmds.listRelatives(selObj[0]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
            raise Exception('First select placement node, then terrain geometry.')
        if cmds.nodeType(allChild[0]) == "McdPlace":
            placeNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
            raise Exception('First select placement node, then terrain geometry.')
        placeTransform = selObj[0]
            
    if cmds.nodeType(selObj[1]) == "mesh":
        terrainNode = selObj[1]
        terrainTransform = cmds.listRelatives(terrainNode, c = False, p = True)[0]
    else:
        allChild = cmds.listRelatives(selObj[1]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
            raise Exception('First select placement node, then terrain geometry.')
        if cmds.nodeType(allChild[0]) == "mesh":
            terrainNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then terrain geometry, and try again.')
            raise Exception('First select placement node, then terrain geometry.')
        terrainTransform = selObj[1]
        
    # test terrain world transform:
    nillMat = True;
    trnWMat = cmds.getAttr(terrainTransform + ".worldMatrix")
    for i in range(16):
        if i == 0 or i == 5 or i == 10 or i == 15:
            if not isFloatEqual(trnWMat[i], 1.0):
                nillMat = False
        else:
            if not isFloatEqual(trnWMat[i], 0.0):
                nillMat = False
             
    stat = ""   
    if not nillMat:
        # check already parented.
        allParents = cmds.listRelatives(placeTransform, c = False, p = True)
        if allParents == [] or allParents == None:
            allParents = []
        if terrainTransform not in allParents:
            stat = cmds.confirmDialog(t = "Warning", m = 'System detect your terrain mesh with translate, rotate, or scale. ' + \
                                                        'Attach this terrain may cause "shift" problem. \nTo solve this, you can parent ' + \
                                                        'your place node to the terrain. Are you willing to parent it?', \
                                                        b = ["Yes", "Help", "Cancel"])
            if stat == "Yes":
                try:
                    cmds.parent(placeTransform, terrainTransform)
                except:
                    pass
                try:
                    cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inTerrain", f = True)
                except:
                    pass
            
            elif stat == "Help":
                cmds.launch(web="https://basefount.atlassian.net/wiki/pages/viewpage.action?pageId=524460")
                return
            else:
                return
            
        else:
            try:
                # connect directly:
                cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inTerrain", f = True)
            except:
                pass

        clearTransform(placeTransform)
    else:
        try:
            # connect directly:
            cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inTerrain", f = True)
        except:
            pass
          
          
def McdDetachTerrain():
    selObj = getSelection("McdPlace")
    allConns = cmds.listConnections(selObj, s = True, d = False, p = True, c = True)
    
    if MIsBlank(allConns):
        return
    
    for i in range(len(allConns) / 2):
        if (allConns[i*2].find(".inTerrain") > 0):
            try:
                cmds.disconnectAttr(allConns[i*2+1], allConns[i*2])
            except:
                cmds.confirmDialog(t = "Error", m = 'Cannot break connections')
        
def McdAttachRangeMesh():
    selObj = cmds.ls(sl = True)
    if selObj == [] or selObj == None:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
        return
    if len(selObj) < 2:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
        return
    
    placeNode = ""
    placeTransform = ""
    terrainNode = ""
    terrainTransform = ""
    if cmds.nodeType(selObj[0]) == "McdPlace":
        placeNode = selObj[0]
        placeTransform = cmds.listRelatives(placeNode, c = False, p = True)[0]
    else:
        allChild = cmds.listRelatives(selObj[0]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "McdPlace":
            placeNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
            return
        placeTransform = selObj[0]
            
    if cmds.nodeType(selObj[1]) == "mesh":
        terrainNode = selObj[1]
        terrainTransform = cmds.listRelatives(terrainNode, c = False, p = True)[0]
    else:
        allChild = cmds.listRelatives(selObj[1]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "mesh":
            terrainNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then polygon mesh, and try again.')
            return
        terrainTransform = selObj[1]
        
    # test terrain world transform:
    nillMat = True;
    trnWMat = cmds.getAttr(terrainTransform + ".worldMatrix")
    for i in range(16):
        if i == 0 or i == 5 or i == 10 or i == 15:
            if not isFloatEqual(trnWMat[i], 1.0):
                nillMat = False
        else:
            if not isFloatEqual(trnWMat[i], 0.0):
                nillMat = False
             
    stat = ""   
    if not nillMat:
        # check already parented.
        allParents = cmds.listRelatives(placeTransform, c = False, p = True)
        if allParents == [] or allParents == None:
            allParents = []
        if terrainTransform not in allParents:
            stat = cmds.confirmDialog(t = "Warning", m = 'System detect your range mesh with translate, rotate, or scale. ' + \
                                                        'Attach this terrain may cause "shift" problem. \nTo solve this, you can parent ' + \
                                                        'your place node to the terrain. Are you willing to parent it?', \
                                                        b = ["Yes", "Help", "Cancel"])
            if stat == "Yes":
                try:
                    cmds.parent(placeTransform, terrainTransform)
                except:
                    pass
                try:
                    cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inPolygon", f = True)
                except:
                    pass
        else:
            try:
                # connect directly:
                cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inPolygon", f = True)
            except:
                pass

        clearTransform(placeTransform)
    else:
        try:
            # connect directly:
            cmds.connectAttr(terrainNode + ".outMesh", placeNode + ".inPolygon", f = True)
        except:
            pass
          
          
def McdDetachRangeMesh():
    selObj = getSelection("McdPlace")

    allConns = cmds.listConnections(selObj, s = True, d = False, p = True, c = True)
    
    if MIsBlank(allConns):
        return
    
    for i in range(len(allConns) / 2):
        if (allConns[i*2].find(".inPolygon") > 0):

            try:
                cmds.disconnectAttr(allConns[i*2+1], allConns[i*2])
            except:
                cmds.confirmDialog(t = "Error", m = 'Cannot break connections')

        
def McdAttachCurve():
    selObj = cmds.ls(sl = True)
    if selObj == [] or selObj == None:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
        return
    if len(selObj) < 2:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
        return
        
    placeNode = ""
    curveNode = ""
    if cmds.nodeType(selObj[0]) == "McdPlace":
        placeNode = selObj[0]
    else:
        allChild = cmds.listRelatives(selObj[0]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "McdPlace":
            placeNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
            
    if cmds.nodeType(selObj[1]) == "nurbsCurve":
        curveNode = selObj[1]
    else:
        allChild = cmds.listRelatives(selObj[1]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "nurbsCurve":
            curveNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
            
    try:
        cmds.connectAttr(curveNode + ".worldSpace", placeNode + ".inCurve", f = True)
    except:
        pass
            
            

def McdAttachAimCurve():
    selObj = cmds.ls(sl = True)
    if selObj == [] or selObj == None:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
        return
    if len(selObj) < 2:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
        return
        
    placeNode = ""
    curveNode = ""
    if cmds.nodeType(selObj[0]) == "McdPlace":
        placeNode = selObj[0]
    else:
        allChild = cmds.listRelatives(selObj[0]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "McdPlace":
            placeNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
            
    if cmds.nodeType(selObj[1]) == "nurbsCurve":
        curveNode = selObj[1]
    else:
        allChild = cmds.listRelatives(selObj[1]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "nurbsCurve":
            curveNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then nurbs curve, and try again.')
            return
            
    try:
        cmds.connectAttr(curveNode + ".worldSpace", placeNode + ".aimCurve", f = True)
    except:
        pass
    
    #drawoverride for control the links
    cmds.setAttr(curveNode + ".overrideEnabled", 1)
    
    cmds.setAttr(placeNode + ".aimType", 1)

    cmds.confirmDialog(t = "Known Issue", m = "You'd better make your agents number in this place node more than 1")
    

            
def McdDetachCurve():
    selObj = getSelection("McdPlace")

    allConns = cmds.listConnections(selObj, s = True, d = False, p = True, c = True)
    
    if MIsBlank(allConns):
        return
    
    for i in range(len(allConns) / 2):
        if (allConns[i*2].find(".inCurve") > 0):

            try:
                cmds.disconnectAttr(allConns[i*2+1], allConns[i*2])
            except:
                cmds.confirmDialog(t = "Error", m = 'Cannot break connections')
            
def McdDetachAimCurve():
    selObj = getSelection("McdPlace")

    allConns = cmds.listConnections(selObj, s = True, d = False, p = True, c = True)
    
    if MIsBlank(allConns):
        return
    
    for i in range(len(allConns) / 2):
        if (allConns[i*2].find(".aimCurve") > 0):

            try:
                cmds.disconnectAttr(allConns[i*2+1], allConns[i*2])
            except:
                cmds.confirmDialog(t = "Error", m = 'Cannot break connections')
            
def McdAttachParticle():
    selObj = cmds.ls(sl = True)
    if selObj == [] or selObj == None:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
        return
    if len(selObj) < 2:
        cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
        return
        
    placeNode = ""
    curveNode = ""
    if cmds.nodeType(selObj[0]) == "McdPlace":
        placeNode = selObj[0]
    else:
        allChild = cmds.listRelatives(selObj[0]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "McdPlace":
            placeNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
            return
            
    if cmds.nodeType(selObj[1]) == "particle" or cmds.nodeType(selObj[1]) == "nParticle":
        curveNode = selObj[1]
    else:
        allChild = cmds.listRelatives(selObj[1]);
        if allChild == [] or allChild == None:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
            return
        if cmds.nodeType(allChild[0]) == "particle" or cmds.nodeType(allChild[0]) == "nParticle":
            curveNode = allChild[0]
        else:
            cmds.confirmDialog(t = "Error", m = 'First select placement node, then particle node, and try again.')
            return
            
    try:
        cmds.connectAttr(curveNode + ".count", placeNode + ".localScaleX", f = True)
    except:
        pass
            
def McdDetachParticle():
    selObj = getSelection("McdPlace")

    allConns = cmds.listConnections(selObj, s = True, d = False, p = True, c = True)
    
    if MIsBlank(allConns):
        return
    
    for i in range(len(allConns) / 2):
        if (allConns[i*2].find(".localScaleX") > 0):
            nodeName = allConns[i*2+1].split(".")[0]
            if cmds.nodeType(nodeName) == "particle" or cmds.nodeType(nodeName) == "nParticle":
                try:
                    cmds.disconnectAttr(allConns[i*2+1], allConns[i*2])
                except:
                    cmds.confirmDialog(t = "Error", m = 'Cannot break connections')
            
def inversePlacementAgent():
    cmd = "McdInvPlacementCmd;"
    placeNode = mel.eval(cmd)
    
    
def duplicatePlacement():
    allGlobal = cmds.ls(type = "McdGlobal")
    
    if MIsBlank(allGlobal):
        cmds.confirmDialog(t = "Abort", m = 'No found McdGlobal node, please click Miarmy Ready.')
        return
    
    selObj = getSelection("McdPlace")
    
    # type filter
    ptype = cmds.getAttr(selObj + ".placeType")
    if ptype == 1 or ptype == 2 or ptype == 3:
        cmds.confirmDialog(t = "Abort", m = 'Cannot duplicate circle, curve, and polygon type.')
        return
    
    # McdGlobal enable
    for i in range(len(allGlobal)):
        cmds.setAttr(allGlobal[i] + ".boolMaster[4]", 1)
    
    # duplicate selObj
    dupNode = cmds.duplicate(selObj, rr = True)[0]
    
    # connect attr:
    cmds.connectAttr(dupNode + ".translateX", dupNode + ".localPositionX")
    cmds.connectAttr(dupNode + ".translateY", dupNode + ".localPositionY")
    cmds.connectAttr(dupNode + ".translateZ", dupNode + ".localPositionZ")
    
    # turn off all caches.
    cmds.setAttr(dupNode + ".enableCache", 0)
    
    
    # get terrain:
    allConns = cmds.listConnections(selObj, p = True)
    if not MIsBlank(allConns):
        for i in range(len(allConns)):
            if allConns[i].find(".outMesh") > 0:
                cmds.connectAttr(allConns[i],  dupNode + ".inTerrain")
    
    
    # McdGlobal disable
    for i in range(len(allGlobal)):
        cmds.setAttr(allGlobal[i] + ".boolMaster[4]", 0)
    


def checkInverseAndNormalExist():
    allPlaceNode = cmds.ls(type = "McdPlace")
    if allPlaceNode == [] or allPlaceNode == None:
        return False
    if len(allPlaceNode) < 2:
        return False
    
    gotNormalPlace = False
    gotInversePlace = False
    
    for i in range(len(allPlaceNode)):
        placeType = cmds.getAttr(allPlaceNode[i] + ".placeType")
        if placeType != 5:
            # we got normal node
            gotNormalPlace = True
        else:
            # we got custom place node:
            parent0 = cmds.getAttr(allPlaceNode[i] + ".parentSet[0]")
            if parent0 != "" and parent0 != None:
                # we got inverse place node:
                gotInversePlace = True
            else:
                gotNormalPlace = True
    
    if gotNormalPlace and gotInversePlace:
        return True
    
    return False

def McdFromAgentsGetPlace(allSelAgents):
    
    allPlaceNodes = cmds.ls(type = "McdPlace")
    allPlacePlid = []
    for i in range(len(allPlaceNodes)):
        plid = cmds.getAttr(allPlaceNodes[i] + ".plid")
        allPlacePlid.append(plid)
        
    result = []
    for i in range(len(allSelAgents)):
        plid = cmds.getAttr(allSelAgents[i] + ".plid")
        plaid = cmds.getAttr(allSelAgents[i] + ".plaid")
        for j in range(len(allPlacePlid)):
            if plid == allPlacePlid[j]:
                result.append(allPlaceNodes[j])
                result.append(plaid)
        
    return result

def McdReverseOldNew(globalNode):
    
    globalNode = McdGetMcdGlobalNode()
    rawPath = cmds.getAttr(globalNode + ".cacheFolder")
    cacheFolder = envPath2AbsPath(rawPath)
    if cacheFolder == "" or cacheFolder == None:
        return False
         
    if not os.access(cacheFolder, os.W_OK):
        return False
     
    if not os.access(cacheFolder, os.R_OK):
        try:
            os.mkdir(cacheFolder)
        except:
            return False
         
    cacheName = cmds.getAttr(globalNode + ".cacheName")
    cachenameNew = cacheName + "_new"
     
    cacheFileOld = cacheFolder + "/" +  cacheName
    cacheFileNew = cacheFolder + "/" + cachenameNew
     
    startFrame = int(cmds.playbackOptions(q =True, min = True) + .1)
    endFrame = int(cmds.playbackOptions(q =True, max = True) + .1)
    
    sumFrame = endFrame - startFrame + 1
    
    hisStr = str(int(time.time()))
    
    for i in range(sumFrame):
        frameNumber = str(int(startFrame + i))
        cacheFileOldName = cacheFileOld + "." + frameNumber + ".mmc"
        cacheFileOldName_= cacheFileOld + "." + hisStr + "." + frameNumber + ".mmc"
        
        os.rename(cacheFileOldName, cacheFileOldName_)
        
        cacheFileNewName = cacheFileNew + "." + frameNumber + ".mmc"
        cacheFileNewName_= cacheFileOldName
        
        os.rename(cacheFileNewName, cacheFileNewName_)
        
        # print ""
        # print cacheFileOldName
        # print cacheFileOldName_
        # print "->"
        # print cacheFileNewName
        # print cacheFileNewName_
    
    
    # e:/abc/testcache.20.mmc
    # e:/abc/testcache_new.20.mmc
    return True
    
def McdMarkAgentOut():
    selAgents = cmds.ls(sl = True)
    if McdIsBlank(selAgents):
        cmds.confirmDialog(t = "Error", m = "Please firstly select some agents node, and try again.")
        return
    
    haveError = False
    agentShapes = []
    for i in range(len(selAgents)):
        shapeNode = cmds.listRelatives(selAgents[i], c = True, p = False)
        if McdIsBlank(shapeNode):
            haveError = True
            break
        if cmds.nodeType(shapeNode[0]) != "McdAgent":
            haveError = True
            break
        else:
            agentShapes.append(shapeNode[0])
            
    if haveError:
        cmds.confirmDialog(t = "Error", m = "Some of your selected, are not agent transform node")
        return
    
    agent_place_Info = McdFromAgentsGetPlace(agentShapes)
    
    allPlaceNodes = cmds.ls(type = "McdPlace")
    allPlacePlid = []
    for i in range(len(allPlaceNodes)):
        plid = cmds.getAttr(allPlaceNodes[i] + ".plid")
        allPlacePlid.append(plid)
    
    
    # mark out
    nbMarkOut = len(agent_place_Info) / 2
    for i in range(nbMarkOut):
        cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
    
    # if have cache enable, prompt , and redo the cache
    globalNode = McdGetMcdGlobalNode()
    enableCache = cmds.getAttr(globalNode + ".enableCache")
    
    if enableCache == 1:
        # hide agent and redo cache
        stat1 = "Rebuild"
        stat = cmds.confirmDialog(t = "Warning!!", m = "We detected you're using agent cache, rebuild it now??", b = ["Yes", "No", "Cancel Operation"])
        if stat == "Cancel Operation":
            return;
        if stat == "No":
            stat1 = cmds.confirmDialog(t = "Warning!! Warning!!", m = "If you not rebuild your agent cache, your old cache will useless, because agent number changes", b = ["Rebuild", "No"])
        
        if stat1 == "Rebuild":
            nbMarkOut = len(agent_place_Info) / 2
            for i in range(nbMarkOut):
                cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
                
            cmds.hide() # hide all selected agents:
            cmds.setAttr(globalNode + ".funcParamList[1]", 1)
            if mel.eval("McdMakeCacheCmd;") == True:
                # re-set the caceh folder;
                if not McdReverseOldNew(globalNode):
                    cacheName = cmds.getAttr(globalNode + ".cacheName")
                    cachenameNew = cacheName + "_new"
                    cmds.setAttr(globalNode + ".cacheName", cachenameNew, type = "string")
            
            # deplace
            cmds.confirmDialog(t = "Warning", m = "Please save your scene, \notherwise agent place skip flags will discard and agent cache will useless.")
            dePlacementAgent()
                
        
        
    else:
        nbMarkOut = len(agent_place_Info) / 2
        for i in range(nbMarkOut):
            cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
        cmds.confirmDialog(t = "Warning", m = "Please save your scene, and re-place agents")
        dePlacementAgent()
    
    
def McdMarkAgentOutLayer():
    selAgents = cmds.ls(sl = True)
    if McdIsBlank(selAgents):
        cmds.confirmDialog(t = "Error", m = "Please firstly select some agents node, and try again.")
        return
    
    haveError = False
    agentShapes = []
    for i in range(len(selAgents)):
        shapeNode = cmds.listRelatives(selAgents[i], c = True, p = False)
        if McdIsBlank(shapeNode):
            haveError = True
            break
        if cmds.nodeType(shapeNode[0]) != "McdAgent":
            haveError = True
            break
        else:
            agentShapes.append(shapeNode[0])
            
    if haveError:
        cmds.confirmDialog(t = "Error", m = "Some of your selected, are not agent transform node")
        return
    
    agent_place_Info = McdFromAgentsGetPlace(agentShapes)
    
    allPlaceNodes = cmds.ls(type = "McdPlace")
    allPlacePlid = []
    for i in range(len(allPlaceNodes)):
        plid = cmds.getAttr(allPlaceNodes[i] + ".plid")
        allPlacePlid.append(plid)
    
    
    # mark out
    nbMarkOut = len(agent_place_Info) / 2
    for i in range(nbMarkOut):
        cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
    
    # if have cache enable, prompt , and redo the cache
    globalNode = McdGetMcdGlobalNode()
    enableCache = cmds.getAttr(globalNode + ".enableCache")
    
    if enableCache == 1:
        # hide agent and redo cache
        stat1 = "Rebuild"
        stat = cmds.confirmDialog(t = "Warning!!", m = "We detected you're using agent cache, rebuild it now??", b = ["Yes", "No", "Cancel Operation"])
        if stat == "Cancel Operation":
            return;
        if stat == "No":
            stat1 = cmds.confirmDialog(t = "Warning!! Warning!!", m = "If you not rebuild your agent cache, your old cache will useless, because agent number changes", b = ["Rebuild", "No"])
        
        if stat1 == "Rebuild":
            nbMarkOut = len(agent_place_Info) / 2
            for i in range(nbMarkOut):
                cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
                
            cmds.hide() # hide all selected agents:
            cmds.setAttr(globalNode + ".funcParamList[1]", 1)
            if mel.eval("McdMakeCacheCmd;") == True:
                # re-set the caceh folder;
                cacheName = cmds.getAttr(globalNode + ".cacheName")
                cachenameNew = cacheName + "_new"
                cmds.setAttr(globalNode + ".cacheName", cachenameNew, type = "string")
            
            # deplace
            cmds.confirmDialog(t = "Warning", m = "Please save your scene, \notherwise agent place skip flags will discard and agent cache will useless.")
            dePlacementAgent()
                
        
        
    else:
        nbMarkOut = len(agent_place_Info) / 2
        for i in range(nbMarkOut):
            cmds.setAttr(agent_place_Info[i*2] + ".epSkip[" + str(agent_place_Info[i*2+1]) + "]", 1)
        cmds.confirmDialog(t = "Warning", m = "Please save your scene, and re-place agents")
        dePlacementAgent()

    
def McdUnmarkAgentOut():
    placeNode = getSelection('McdPlace')
    
    globalNode = McdGetMcdGlobalNode()
    if cmds.getAttr(globalNode + ".enableCache") == 1:
        op = cmds.confirmDialog(t = "Warning", m = "We detected you enabled the Agent Cache\nThis process will make agent cache useless\nDo you want to proceed.",\
                                b = ["OK", "Cancel"])
        
        if op == "OK":
            cmds.setAttr(globalNode + ".enableCache", 0)
        else:
            return;
            
            
    nbAgt = cmds.getAttr(placeNode + ".numOfAgent")
    for i in range(nbAgt):
        cmds.setAttr(placeNode + ".epSkip[" + str(i) + "]", 0)

    
def McdCreatePlacementNodeMesh():
    
    # test mesh selected?
    # read info from mesh
    
    meshNode = getSelection("mesh")
    allPointsData = mel.eval("McdSimpleCommand -exe 32");
    if MIsBlank(allPointsData):
        cmds.confirmDialog(t = "Error", m = "Mesh data error");
        return;
    
    McdCreatePlacementNode()
    
    selObj = getSelection()
    
    # set to posLock Mode
    nbPoints = len(allPointsData) / 3
    
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", nbPoints)
    
    for i in range(nbPoints):
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", allPointsData[i * 3 + 1])
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
        
    # important!! connect message!
    allConns = cmds.listConnections(meshNode, s = False, d = True, shapes = True, type = "McdFormation")
    if MIsBlank(allConns):
        cmds.confirmDialog(t = "Warning!", m = "cannot detect formation node, may cause formation not useable")
        return
    formationNode = allConns[0]
    placeNode = selObj
    try:
        cmds.connectAttr(formationNode + ".outEntity", placeNode + ".formationEntity", force = True)
    except:
        cmds.confirmDialog(t = "Warning!", m = "Mesh entity connection failure, may cause formation not useable")
        
    
def McdCreatePlacementNodeLattice():
    # test lattice selected?
    # read info from mesh
    
    latticeNode = getSelection("lattice")
    allPointsData = mel.eval("McdSimpleCommand -exe 33");
    if MIsBlank(allPointsData):
        cmds.confirmDialog(t = "Error", m = "lattice data error");
        return;
    
    McdCreatePlacementNode()
    
    selObj = getSelection()
    
    # set to posLock Mode
    nbPoints = len(allPointsData) / 3
    
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", nbPoints)
    
    for i in range(nbPoints):
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", allPointsData[i * 3 + 1])
        cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
        
    # important!! connect message!
    allConns = cmds.listConnections(latticeNode, s = False, d = True, shapes = True, type = "McdFormation")
    if MIsBlank(allConns):
        cmds.confirmDialog(t = "Warning!", m = "cannot detect formation node, may cause formation not useable")
        return
    formationNode = allConns[0]
    placeNode = selObj
        
    try:
        cmds.connectAttr(formationNode + ".outEntity", placeNode + ".formationEntity", force = True)
    except:
        cmds.confirmDialog(t = "Warning!", m = "Lattice entity connection failure, may cause formation not useable")
    
    
    
def McdCreatePlacementNodeParticle(is3D):
    # test lattice selected?
    # read info from mesh
    
    ptNode = getSelection("particle")
    allPointsData = []
    ptCount = cmds.getAttr(ptNode + ".count")
    
    if ptCount == 0:
        cmds.confirmDialog(t = "Abort", m = "No particle instance in particle shape.");
        return; 
    
    for i in range(ptCount):
        stri = str(i)
        currentPos = cmds.xform(ptNode + '.pt[' + stri + ']', q = True, t = True)
        for j in range(3):
            allPointsData.append(currentPos[j])

    McdCreatePlacementNode()
    
    selObj = getSelection()
    
    # set to posLock Mode
    nbPoints = len(allPointsData) / 3
    
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", nbPoints)
    
    if is3D == 0:
        for i in range(nbPoints):
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", 0.0)
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
    else:
        for i in range(nbPoints):
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", allPointsData[i * 3 + 1])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
    
    
    stat = cmds.confirmDialog(t = "Particle Following", m = "Do you want to link particle system to place node??\n" + \
                     "This can enable particle following feature.\n\n\n" + \
                     "If you dont want to use it please do not link for saving memory.", b = ["Link", "Do not Link"]);
    
    if stat == "Link":
        placeNode = cmds.listRelatives(selObj, c = True, p = False)[0]
        cmds.connectAttr(ptNode + ".count", placeNode + ".localScaleX")
    
    
    
    
def McdCreatePlacementNodeNParticle(is3D):
    # test lattice selected?
    # read info from mesh
    
    ptNode = getSelection("nParticle")
    allPointsData = []
    ptCount = cmds.getAttr(ptNode + ".count")
    
    if ptCount == 0:
        cmds.confirmDialog(t = "Abort", m = "No nParticle instance in particle shape.");
        return; 
    
    for i in range(ptCount):
        stri = str(i)
        currentPos = cmds.xform(ptNode + '.pt[' + stri + ']', q = True, t = True)
        for j in range(3):
            allPointsData.append(currentPos[j])

    McdCreatePlacementNode()
    
    selObj = getSelection()
    
    # set to posLock Mode
    nbPoints = len(allPointsData) / 3
    
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", nbPoints)
    
    if is3D == 0:
        for i in range(nbPoints):
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", 0.0)
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
    else:
        for i in range(nbPoints):
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[1]", allPointsData[i * 3])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[2]", allPointsData[i * 3 + 1])
            cmds.setAttr(selObj + ".placement[" + str(i) + "].agentPlace[3]", allPointsData[i * 3 + 2])
    
    stat = cmds.confirmDialog(t = "Particle Following", m = "Do you want to link particle system to place node??\n" + \
                     "This can enable particle following feature.\n\n\n" + \
                     "If you dont want to use it please do not link for saving memory.", b = ["Link", "Do not Link"]);
    
    if stat == "Link":
        placeNode = cmds.listRelatives(selObj, c = True, p = False)[0]
        cmds.connectAttr(ptNode + ".count", placeNode + ".localScaleX")
        
        
def McdCreatePlacementNodeNParticleM(is3D):
    # test lattice selected?
    # read info from mesh    

    allLoc=cmds.ls(sl=1)
    sumCount = 0
    for i in range(len(allLoc)):
        shapeNode = cmds.listRelatives(allLoc[i], p = False, c = True)[0]
        nbParticle = cmds.getAttr( shapeNode + '.count' )
        sumCount += nbParticle
        
          
    McdCreatePlacementNode()
    selObj = getSelection()
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", sumCount)
 
        
    pCount = 0
    for i in range(len(allLoc)):
        shapeNode = cmds.listRelatives(allLoc[i], p = False, c = True)[0]
        nbParticle = cmds.getAttr( shapeNode + '.count' )
        for j in range(nbParticle):
            strj = str(j)
            strp = str(pCount)
            tdata=cmds.xform(shapeNode + '.pt[' + strj + ']', q = True, t = True, ws = True)

            cmds.setAttr(selObj+'.placement['+strp+'].agentPlace[1]',tdata[0])
            cmds.setAttr(selObj+'.placement['+strp+'].agentPlace[2]',tdata[1])
            cmds.setAttr(selObj+'.placement['+strp+'].agentPlace[3]',tdata[2])
            pCount += 1

def McdCreatePlaceNodeSelectTransform():
 
    AllLoc=cmds.ls(sl=1)
    
    McdCreatePlacementNode()
    selObj = getSelection()
    cmds.setAttr(selObj + ".placeType", 5)
    cmds.setAttr(selObj + ".numOfAgent", len(AllLoc))
    
    for i in range(0,len(AllLoc)):
        tdata=cmds.xform(AllLoc[i],q=1,ws=1,piv=1)
        cmds.setAttr(placeName+'.placement['+str(i)+'].agentPlace[1]',tdata[0])
        cmds.setAttr(placeName+'.placement['+str(i)+'].agentPlace[2]',tdata[1])
        cmds.setAttr(placeName+'.placement['+str(i)+'].agentPlace[3]',tdata[2])
        cmds.setAttr(placeName+'.placement['+str(i)+'].agentPlace[5]',cmds.getAttr(AllLoc[i]+'.rotateY'))


    
def McdUpdateTexPlacement(mode):
    melCmd = "McdSimpleCommand -exe 43;"
    if mode == 3:
        melCmd = "McdSimpleCommand -exe 44"
        
    mel.eval(melCmd)
    
    
def McdAddDeleteZoneToSelect():
    selObj = cmds.ls(sl = True)
    if MIsBlank(selObj):
        return
    
    for i in range(len(selObj)):
        try:
            childNode = cmds.listRelatives(selObj[i], c = True, p = False)[0]
            cmds.addAttr(childNode, ln = "deleteInside", at = "bool", dv = True)
            cmds.setAttr(childNode + ".deleteInside", k = False, cb = True)
        except:
            try:
                cmds.setAttr(childNode + ".deleteInside", True)
            except:
                pass
            
    
    
def storeAndSetupUVLock():
    melCmd = "McdSimpleCommand -exe 20;"
    mel.eval(melCmd)
    
    selObj = cmds.ls(sl = True)
    
    # setup to mesh mode
    cmds.setAttr(selObj[0] + ".placeType", 2)
    
    placeNode = cmds.listRelatives(selObj[0], c = True, p = False)[0]
    
    # if curve connected, detach it!
    allConns = cmds.listConnections(placeNode, s = True, d = False, p = True)
    
    if not MIsBlank(allConns):
        for i in range(len(allConns)):
            connNode = allConns[i].split('.')[0]
            if cmds.nodeType(connNode) == "nurbsCurve":
                cmds.disconnectAttr(allConns[i], placeNode + ".inCurve")
    




def clearUVLockPlace():
    selObj = getSelection("McdPlace")
    
    ulist0 = cmds.setAttr(selObj + ".uList[0]", 0.0)
    ulist2 = cmds.setAttr(selObj + ".uList[2]", 0.0)
    ulist4 = cmds.setAttr(selObj + ".uList[4]", 0.0)
    
    cmds.setAttr(selObj + ".placeType", 4)



def McdDeleteExtraConnectionsASNotActive():
        
    # check code 17, if 1 do it:
    globalNode = McdGetMcdGlobalNode();
    if not cmds.getAttr(globalNode + ".boolMaster[17]"):
        return
    
    print "delete extra as..."
    
    allAgentGrp = cmds.ls(type = "McdAgentGroup")
    if MIsBlank(allAgentGrp):
        return
    
    for i in range(len(allAgentGrp)):
        allStates = []
        allTreeNodes = cmds.listRelatives(allAgentGrp[i], ad = True, path = True, c = True, p = False)
        if MIsBlank(allTreeNodes):
            continue
        for j in range(len(allTreeNodes)):
            if cmds.nodeType(allTreeNodes[j]) == "McdState":
                allStates.append(allTreeNodes[j])
        
        if len(allStates) > 0:
            allStates1 = allStates
            for j in range(len(allStates)):
                for k in range(len(allStates1)):
                    allTransAct = []
                    if allStates[j] == allStates1[k]:
                        continue
                    allS01Out = cmds.listConnections(allStates[j], d = True, s = False)
                    allS02In = cmds.listConnections(allStates1[k], d = False, s = True)
                    if MIsBlank(allS01Out):
                        continue
                    if MIsBlank(allS02In):
                        continue
                    
                    for m in range(len(allS01Out)):
                        if MIndexOf(allS01Out[m], allS02In) > -1:
                            allTransAct.append(allS01Out[m])
                            
                    if not MIsBlank(allTransAct):
                        active = False
                        for m in range(len(allTransAct)):
                            try:
                                if cmds.getAttr(allTransAct[m] + ".active") == 1:
                                    active = True
                                    break
                            except:
                                raise Exception("Version not correct, actSh no act attr.")
                        
                        if not active:
                            print "\nREMOVE Action Shells:"
                            print allTransAct
                            print ""
                            try:
                                cmds.delete(allTransAct)
                            except:
                                pass


















    
    
    
    
    
    
    

