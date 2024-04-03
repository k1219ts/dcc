import maya.cmds as cmds
import maya.mel as mel
from McdGeneral import *

def McdCreateActionCmd(showAn):                                         # From McdActionFunctions
    # parse origianl agent group
    # and the root bone:(select the root bone)

    allMcdGlobal = cmds.ls(type="McdGlobal");
    if allMcdGlobal == [] or allMcdGlobal == None:
        cmds.confirmDialog(t="Error", m="No found Miarmy Global, please create it in \"Miarmy > Miarmy Global\"")
        raise Exception("No found Miarmy Global, please create it in \"Miarmy > Miarmy Global\"")

    globalNode = allMcdGlobal[0]
    activeAgentName = cmds.getAttr(globalNode + ".activeAgentName")
    isValid = CheckStringIsValid(activeAgentName)
    if isValid == True:
        miarmyGrp = cmds.ls("Miarmy_Contents")
        if miarmyGrp == [] or miarmyGrp == None:
            cmds.confirmDialog(t="Error", m="Cannot find Miarmy_Contents group.")
            raise Exception("Cannot find Miarmy_Contents group.")

        agentGrp = cmds.ls("Agent_" + activeAgentName)
        if agentGrp == [] or agentGrp == None:
            cmds.confirmDialog(t="Error", m="Cannot find Agent group, cannot create action node for this active agent.")
            raise Exception("Cannot find Agent group, cannot create action node for this active agent.")

        setupGrp = cmds.ls("Setup_" + activeAgentName)
        if setupGrp == [] or setupGrp == None:
            cmds.confirmDialog(t="Error",
                               m="Cannot find Agent setup rig group, cannot create action node for this active agent.")
            raise Exception("Cannot find Agent setup rig group, cannot create action node for this active agent.")

        actGrp = cmds.ls("Action_" + activeAgentName)
        if actGrp == [] or actGrp == None:
            cmds.group(n="Action_" + activeAgentName, em=True)
            try:
                cmds.parent("Action_" + activeAgentName, "Agent_" + activeAgentName)
            except:
                cmds.confirmDialog(t="Error",
                                   m="May be you have naming problem, check it firstly in \"Miarmy > Miarmy Contents Check.\"")
                raise Exception(
                    "May be you have naming problem, check it firstly in \"Miarmy > Miarmy Contents Check.\"")

        McdCheckRepeatNameTreeAct(setupGrp, "joint")

        cmds.select("Setup_" + activeAgentName)
        # set name
        startTime = cmds.playbackOptions(q=True, min=True)
        endTime = cmds.playbackOptions(q=True, max=True)

        option = cmds.promptDialog(t="Information", m="Agent Name: " + activeAgentName + "\n" + \
                                                      "Start Frame: " + str(startTime) + "\n" + \
                                                      "End Frame: " + str(endTime) + "\n" + \
                                                      "Please specify an action name:", \
                                   button=["Proceed", "Cancel"], \
                                   defaultButton="Proceed", cancelButton="Cancel", \
                                   dismissString="Cancel", tx=str(showAn))
        if option == "Proceed":
            newAction = cmds.promptDialog(query=True, text=True)
            isVaild = CheckStringIsValid(newAction)
            if isVaild == True:
                newActionNodeName = newAction + "_action_" + activeAgentName
                actGrp = cmds.ls(newActionNodeName)
                if actGrp == [] or actGrp == None:
                    newNodeName = mel.eval("McdCreateActionCmd;")
                    try:
                        cmds.select(newNodeName)
                        cmds.parent(newNodeName, "Action_" + activeAgentName)
                    except:
                        cmds.confirmDialog(t="Error",
                                           m="May be you have naming problem, check it firstly in \"Miarmy > Miarmy Contents Check.\"")
                        raise Exception(
                            "May be you have naming problem, check it firstly in \"Miarmy > Miarmy Contents Check.\"")
                    try:
                        selObj = cmds.ls(sl=True)[0]
                        cmds.rename(selObj, newActionNodeName)
                    except:
                        cmds.confirmDialog(t="Warning",
                                           m="Naming node error, please rename it manually: <actionName>_action_<agentName>")

                    actionNode = cmds.ls(sl=True)[0]
                    stepIntoActionSetupWizard(actionNode)

                else:
                    cmds.confirmDialog(t="Abort", m="Action name exist. System select it(them) automatically.")
                    cmds.select(actGrp)
            else:
                cmds.confirmDialog(t="Abort", m="The new action name: \"" + newAction + "\" you specified is invalid.")

            try:
                nodeName = newActionNodeName
                lenth = cmds.getAttr(nodeName + ".length")
                setValue = int(0.80 * float(lenth))
                cmds.setAttr(nodeName + ".entryMax", setValue)
            except:
                pass


def McdCheckRepeatNameTreeAct(rootNode, tNode):                         # From McdActionFunctions
    allChildren = cmds.listRelatives(rootNode, ad=True, pa=True)
    nameList = []
    for i in range(len(allChildren)):
        if cmds.nodeType(allChildren[i]) == tNode:
            if allChildren[i].find("|") >= 0:
                nameList.append(allChildren[i])

    if len(nameList) > 0:
        print ""
        print "****** Repeat Name ******"
        for i in range(len(nameList)):
            print nameList[i]

        print ""
        print "*************************"
        print ""

        option = cmds.confirmDialog(t="Error",
                                    m="Repeat name detected in Rig\nplease check detail in feedback of Window > General Editors > Script Editor", \
                                    b=["Create Anyway", "Cancel"])
        if option == "Cancel":
            raise Exception("Repeat name detected in Rig.")


def stepIntoActionSetupWizard(actionNode):                              # From McdActionFunctions
    stat = cmds.confirmDialog(t="Action Setup Wizard", m="Do you want to setup this action now?", \
                              b=["Setup", "Later"])
    if stat == "Later":
        return

    stat = cmds.confirmDialog(t="Agent Speed Type:", m="Which type of agent speed you will apply?", \
                              b=["Still", "Move Forward(Z+)", "Move Forward", "Move Up", "Turning", "Cancel"])
    cmds.setAttr(actionNode + ".txState", 0)
    cmds.setAttr(actionNode + ".tyState", 0)
    cmds.setAttr(actionNode + ".tzState", 0)
    cmds.setAttr(actionNode + ".rxState", 0)
    cmds.setAttr(actionNode + ".ryState", 0)
    cmds.setAttr(actionNode + ".rzState", 0)

    if stat == "Move Forward(Z+)":
        cmds.setAttr(actionNode + ".tzState", 1)  # tz
    elif stat == "Move Forward":
        cmds.setAttr(actionNode + ".txState", 1)  # tx
        cmds.setAttr(actionNode + ".tzState", 1)  # tz
    elif stat == "Move Up":
        cmds.setAttr(actionNode + ".tyState", 1)  # ty
        cmds.setAttr(actionNode + ".tzState", 1)  # tz
    elif stat == "Turning":
        cmds.setAttr(actionNode + ".txState", 1)  # tx
        cmds.setAttr(actionNode + ".tzState", 1)  # tz
        cmds.setAttr(actionNode + ".ryState", 1)  # ry
    elif stat == "Ramp":
        cmds.setAttr(actionNode + ".txState", 1)  # tx
        cmds.setAttr(actionNode + ".tyState", 1)  # tx
        cmds.setAttr(actionNode + ".tzState", 1)  # tz
        cmds.setAttr(actionNode + ".rxState", 1)  # ry

    elif stat == "Cancel":
        # free all:
        cmds.setAttr(actionNode + ".txState", 1)
        cmds.setAttr(actionNode + ".tyState", 1)
        cmds.setAttr(actionNode + ".tzState", 1)
        cmds.setAttr(actionNode + ".rxState", 1)
        cmds.setAttr(actionNode + ".ryState", 1)
        cmds.setAttr(actionNode + ".rzState", 1)
        return

    cycFlag = True
    stat = cmds.confirmDialog(t="Action Playback Type:", m="Which type of playback you will apply?", \
                              b=["Cycle Action", "Transition Action", "Cancel"])

    if stat == "Transition Action":
        cmds.setAttr(actionNode + ".isCycle", 0)
        cycFlag = False
    elif stat == "Cancel":
        return

    if cycFlag:
        stat = cmds.promptDialog(t="Self Cycle Range",
                                 m="Please specify percent of self cycle range:\nValid Number: 1-30 integer", \
                                 button=["Confirm", "Use Default (10%)", "Cancel"], \
                                 defaultButton="Confirm", cancelButton="Cancel", \
                                 dismissString="Cancel")

        if stat == "Confirm":
            cycRange = cmds.promptDialog(query=True, text=True)
            if cycRange.isdigit():
                if int(cycRange) <= 30:
                    cmds.setAttr(actionNode + ".cycleFilter", float(cycRange) / 100.0)
                else:
                    cmds.setAttr(actionNode + ".cycleFilter", .3)
            else:
                cmds.confirmDialog(t="Incorrect Input",
                                   m="Your input is invalid, use default, modify it in Action Editor if you want.")

        elif stat == "Use Default (10%)":
            cmds.setAttr(actionNode + ".cycleFilter", .1)

        elif stat == "Cancel":
            return

    stat = cmds.promptDialog(t="Transition In Range",
                             m="Please specify before XX percent, can transit in:\nValid Number: 1-100 integer", \
                             button=["Confirm", "Use Default (10%)", "Cancel"], \
                             defaultButton="Confirm", cancelButton="Cancel", \
                             dismissString="Cancel")
    if stat == "Confirm":
        transIn = cmds.promptDialog(query=True, text=True)
        if transIn.isdigit():
            if int(transIn) <= 100:
                cmds.setAttr(actionNode + ".transIn", float(transIn) / 100.0)
            else:
                cmds.setAttr(actionNode + ".transIn", 1)
        else:
            cmds.confirmDialog(t="Incorrect Input",
                               m="Your input is invalid, use default, modify it in Action Editor if you want.")
    elif stat == "Cancel":
        return

    stat = cmds.promptDialog(t="Transition Out Range",
                             m="Please specify after XX percent, can transit out:\nValid Number: 1-100 integer", \
                             button=["Confirm", "Use Default (80%)", "Cancel"], \
                             defaultButton="Confirm", cancelButton="Cancel", \
                             dismissString="Cancel")
    if stat == "Confirm":
        transOut = cmds.promptDialog(query=True, text=True)
        if transOut.isdigit():
            transIn = cmds.getAttr(actionNode + ".transIn")
            transIn *= 100.0

            if float(transOut) <= 100.0:
                if float(transOut) < transIn:
                    cmds.confirmDialog(t="Range Invalid",
                                       m="Your input is smaller than transition in, we use transition in")
                    cmds.setAttr(actionNode + ".transOut", transIn / 100.0)
                else:
                    cmds.setAttr(actionNode + ".transOut", float(transOut) / 100.0)
            else:
                cmds.setAttr(actionNode + ".transOut", 1)
        else:
            cmds.confirmDialog(t="Incorrect Input",
                               m="Your input is invalid, use default, modify it in Action Editor if you want.")
    elif stat == "Cancel":
        return

    cmds.confirmDialog(t="Finish",
                       m="Action setup completed, you can modify/setup it in Miarmy > Actions > Action Editor...")

    # rebuild last;
    mel.eval("McdSetAgentDataCmd;")


def McdSaveAction(pathD):                                               # From McdSaveAction.py
    dir = mel.eval("getenv MCD_ACTION_PATH;")
    if dir == "" or dir == None:
        dir = ""

    selObjs = cmds.ls(sl=True)
    if MIsBlank(selObjs):
        cmds.confirmDialog(t="Abort", m="Please select one or more action nodes")
        return

    fileName = pathD

    if fileName == None or fileName == "" or fileName == []:
        return

    exeObjs = []
    for i in range(len(selObjs)):
        if cmds.nodeType(selObjs[i]) == "McdAction":
            exeObjs.append(selObjs[i])

    dupObjs = cmds.duplicate(exeObjs, rr=True)
    if MIsBlank(dupObjs):
        cmds.confirmDialog(t="Abort", m="Cannot duplicate and save out.")
        return

    cmds.parent(w=True)

    # renaming selected:
    allRenameObj = cmds.ls(sl=True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        newName1 = allRenameObj[i].split('|')[-1]
        if newName1.find("_action_") > 0:
            newName = newName1.split("_action_")[0]
            cmds.rename(allRenameObj[i], newName)

    # renaming children:
    allRenameObj = cmds.ls(sl=True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        allCN = cmds.listRelatives(allRenameObj[i], ad=True, c=True, p=False, f=True)

        if not MIsBlank(allCN):
            for j in range(len(allCN)):
                newName1 = allCN[j].split('|')[-1]
                if newName1.find("_action_") > 0:
                    newName = newName1.split("_action_")[0]
                    cmds.rename(allCN[j], newName)

    cmds.file(fileName, exportSelected=True, type="mayaAscii")

    cmds.delete()

def McdLoadActions(actionName):
    dir = mel.eval("getenv MCD_ACTION_PATH;")
    if dir == "" or dir == None:
        dir = ""

    fileName = actionName

    if fileName == None or fileName == "" or fileName == []:
        return

    activeAgentName = McdGetActiveAgentName()
    if activeAgentName == "":
        cmds.confirmDialog(t="Abort", m="Please active agent type in Agent Manager.")
        return

    decRoot = cmds.ls("Action_" + activeAgentName)
    if MIsBlank(decRoot):
        stat = cmds.confirmDialog(t="Warning!", m="Cannot find the node root node:\n\n Action_" + activeAgentName,
                                  b=["Continue", "Cancel"])
        if stat == "Cancel":
            return

    cmds.file(fileName, i=True, type="mayaAscii", ra=True, rpr="MCDIMPORTACTION", lrd="all")

    allImpObjs = cmds.ls("MCDIMPORTACTION*", type="McdAction")
    if MIsBlank(allImpObjs):
        cmds.confirmDialog(t="Abort", m="Nothing imported.")
        return

    exeObjs = []
    for i in range(len(allImpObjs)):
        allParents = cmds.listRelatives(allImpObjs[i], c=0, p=1)
        if MIsBlank(allParents):
            exeObjs.append(allImpObjs[i])

    cmds.select(exeObjs)
    # renaming selected:
    allRenameObj = cmds.ls(sl=True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        nakedName = allRenameObj[i].split("|")[-1]
        newName = nakedName + "_action_" + activeAgentName
        cmds.rename(allRenameObj[i], newName)

    # renaming children:
    allRenameObj = cmds.ls(sl=True)
    if MIsBlank(allRenameObj):
        return
    for i in range(len(allRenameObj)):
        allCN = cmds.listRelatives(allRenameObj[i], ad=True, c=True, p=False, f=True)

        if not MIsBlank(allCN):
            for j in range(len(allCN)):
                nakedName = allCN[j].split("|")[-1]
                newName = nakedName + "_action_" + activeAgentName
                cmds.rename(allCN[j], newName)

    if not MIsBlank(decRoot):
        allRenameObj = cmds.ls(sl=True)
        cmds.parent(allRenameObj, decRoot[0])

    # ------------------------------------------
    try:
        # delete the suffix:
        allRenameObj = cmds.ls(sl=True)
        if MIsBlank(allRenameObj):
            return
        for i in range(len(allRenameObj)):
            nakedName = allRenameObj[i].split("|")[-1]
            newName = nakedName.split("MCDIMPORTACTION_")[-1]
            cmds.rename(allRenameObj[i], newName)

        # renaming children:
        allRenameObj = cmds.ls(sl=True)
        if MIsBlank(allRenameObj):
            return
        for i in range(len(allRenameObj)):
            allCN = cmds.listRelatives(allRenameObj[i], ad=True, c=True, p=False, f=True)

            if not MIsBlank(allCN):
                for j in range(len(allCN)):
                    nakedName = allCN[j].split("|")[-1]
                    newName = nakedName.split("MCDIMPORTACTION_")[-1]
                    cmds.rename(allCN[j], newName)
    except:
        cmds.confirmDialog(t="Abort", m="Cannot rename automatically, please rename manually.")