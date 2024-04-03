import maya.cmds as cmds
import maya.mel as mel
from McdGeneral import *

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
