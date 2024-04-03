import maya.cmds as cmds
import maya.mel as mel
from McdGeneral import *
from McdSimpleCmd import *

def McdLoadActions(actionName):
    dir = mel.eval("getenv MCD_ACTION_PATH;")
    if dir == "" or dir == None:
        dir = ""
    
    fileName = actionName

    if fileName == None or fileName == "" or fileName == []:
        return

    activeAgentName = McdGetActiveAgentName()
    if activeAgentName == "":
        cmds.confirmDialog(t = "Abort", m = "Please active agent type in Agent Manager.")
        return
        
    decRoot = cmds.ls("Action_" + activeAgentName)
    if MIsBlank(decRoot):
        stat = cmds.confirmDialog(t = "Warning!", m = "Cannot find the node root node:\n\n Action_" + activeAgentName, b = ["Continue", "Cancel"])
        if stat == "Cancel":
            return

    cmds.file( fileName, i = True, type = "mayaAscii", ra = True, rpr = "MCDIMPORTACTION", lrd = "all")
    
    allImpObjs = cmds.ls("MCDIMPORTACTION*", type = "McdAction")
    if MIsBlank(allImpObjs):
        cmds.confirmDialog(t = "Abort", m = "Nothing imported.")
        return 
    
    exeObjs = []
    for i in range(len(allImpObjs)):
        allParents = cmds.listRelatives(allImpObjs[i], c = 0, p = 1)
        if MIsBlank(allParents):
            exeObjs.append(allImpObjs[i])
            
    cmds.select(exeObjs)
    # renaming selected:
    allRenameObj = cmds.ls(sl = True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        nakedName = allRenameObj[i].split("|")[-1]
        newName = nakedName + "_action_" + activeAgentName
        cmds.rename(allRenameObj[i], newName)
    
    # renaming children:    
    allRenameObj = cmds.ls(sl = True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        allCN = cmds.listRelatives(allRenameObj[i], ad = True, c = True, p = False, f = True)
        
        if not MIsBlank(allCN):
            for j in range(len(allCN)):
                nakedName = allCN[j].split("|")[-1]
                newName = nakedName + "_action_" + activeAgentName
                cmds.rename(allCN[j], newName)
    
    if not MIsBlank(decRoot):
        allRenameObj = cmds.ls(sl = True)
        cmds.parent(allRenameObj, decRoot[0])
    
    # ------------------------------------------
    try:
        # delete the suffix:
        allRenameObj = cmds.ls(sl = True)
        if MIsBlank(allRenameObj):
            return
        for i in range(len(allRenameObj)):
            nakedName = allRenameObj[i].split("|")[-1]
            newName = nakedName.split("MCDIMPORTACTION_")[-1]
            cmds.rename(allRenameObj[i], newName)
        
        # renaming children:    
        allRenameObj = cmds.ls(sl = True)
        if MIsBlank(allRenameObj):
            return
        for i in range(len(allRenameObj)):
            allCN = cmds.listRelatives(allRenameObj[i], ad = True, c = True, p = False, f = True)
            
            if not MIsBlank(allCN):
                for j in range(len(allCN)):
                    nakedName = allCN[j].split("|")[-1]
                    newName = nakedName.split("MCDIMPORTACTION_")[-1]
                    cmds.rename(allCN[j], newName)
    except:
        cmds.confirmDialog(t = "Abort", m = "Cannot rename automatically, please rename manually.")

def McdSaveAction(pathD):
    dir = mel.eval("getenv MCD_ACTION_PATH;")
    if dir == "" or dir == None:
        dir = ""
    
    selObjs = cmds.ls(sl = True)
    if MIsBlank(selObjs):
        cmds.confirmDialog(t = "Abort", m = "Please select one or more action nodes")
        return
    
    fileName = pathD
    
    if fileName == None or fileName == "" or fileName == []:
        return
    
    exeObjs = []
    for i in range(len(selObjs)):
        if cmds.nodeType(selObjs[i]) == "McdAction":
            exeObjs.append(selObjs[i])
            
    dupObjs = cmds.duplicate(exeObjs, rr = True)
    if MIsBlank(dupObjs):
        cmds.confirmDialog(t = "Abort", m = "Cannot duplicate and save out.")
        return
    
    cmds.parent(w = True)
    
    # renaming selected:
    allRenameObj = cmds.ls(sl = True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        newName1 = allRenameObj[i].split('|')[-1]
        if newName1.find("_action_") > 0:
            newName = newName1.split("_action_")[0]
            cmds.rename(allRenameObj[i], newName)
    
    # renaming children:    
    allRenameObj = cmds.ls(sl = True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        allCN = cmds.listRelatives(allRenameObj[i], ad = True, c = True, p = False, f = True)
        
        if not MIsBlank(allCN):
            for j in range(len(allCN)):
                newName1 = allCN[j].split('|')[-1]
                if newName1.find("_action_") > 0:
                    newName = newName1.split("_action_")[0]
                    cmds.rename(allCN[j], newName)
        
    cmds.file(fileName, exportSelected = True, type = "mayaAscii")
    
    cmds.delete()


def McdSelectMcdGlobal():
    McdClearUselessNodes()

    cmds.playbackOptions(view="all")  # update all, feedback usage
    cmds.playbackOptions(playbackSpeed=0)  # play every frame
    cmds.playbackOptions(by=1.0)  # by 1.0 for each update
    cmds.playbackOptions(maxPlaybackSpeed=1)  # max speed, 24fps

    # create McdBrain and McdBrainPost
    allSolveNode = cmds.ls(type="McdBrain")
    if allSolveNode == [] or allSolveNode == None:
        cmds.createNode("McdBrain")

    allSolveNode = cmds.ls(type="McdBrainPost")
    if allSolveNode == [] or allSolveNode == None:
        cmds.createNode("McdBrainPost")

    # create McdGlobal and contents
    allMcdGlobal = cmds.ls(type="McdGlobal");
    allContentsNodes = cmds.ls("Miarmy_Contents")
    if allContentsNodes == [] or allContentsNodes == None:
        if allMcdGlobal != [] and allMcdGlobal != None:
            for i in range(len(allMcdGlobal)):
                try:
                    cmds.delete(allMcdGlobal[i])
                except:
                    pass

    allMcdGlobal = cmds.ls(type="McdGlobal");
    if allMcdGlobal == [] or allMcdGlobal == None:
        McdCreateMcdGlobal()
        allMcdGlobal = cmds.ls(type="McdGlobal");
        cmds.select(allMcdGlobal[0]);
    else:
        mel.eval("McdSimpleCommand -exe 1;")
