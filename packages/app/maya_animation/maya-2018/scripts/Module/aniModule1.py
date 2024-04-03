

#######################################################################################

# Script Name : 

# Author : som

# Last Updated : 2021.6.25

# Description : 

#######################################################################################
import maya.cmds as cmds




###############################################################################
def getNamespace():
    name_space = []
    listCtrls = cmds.ls(sl=1)

    if(listCtrls == []):
        cmds.error("you must select the controler or the object")
    else:
        sign =  listCtrls[0].find(':')
        if(sign == -1):
            return ''

        else:
            for ctrl in listCtrls:
                name_space.append(ctrl.split(":")[0])

            return name_space[0]






################################################################################
def findMinMaxKey(controlers, nameSpace):

    cmds.select(cl=1)

    for controler in controlers:
        #print nameSpace
        if(cmds.objExists(nameSpace + ":" + controler)):
            cmds.select(nameSpace + ":" + controler, add=1)

    sel_ctrls = cmds.ls(sl=1)
    minKeyTimes = []
    maxKeyTimes = []
    cmds.selectKey()

    for num in range(len(sel_ctrls)):
        keyPosition = cmds.keyframe(sel_ctrls[num], sl=1, q=1, tc=1)
        if keyPosition != None:
            minKeyTimes.append(min(keyPosition))
            maxKeyTimes.append(max(keyPosition))

    if (minKeyTimes == []):
        cmds.error("key does not exist on this controller(s)")

    minKeyPst = min(minKeyTimes)
    maxKeyPst = max(maxKeyTimes)

    return minKeyPst, maxKeyPst
    
    






################################################################################
def setKeyAll(nameSpace, controlers):

    cmds.select(cl=1)
    for controler in controlers:
        cmds.select(nameSpace + ":" + controler, add=1)

    cmds.setKeyframe(breakdown=0, hierarchy='none', controlPoints=0, shape=0)
    
    
    
   
    
    
    
    
    
################################################################################    
def makeInitPose(nameSpace, controlers, attributes):

    attList = attributes.keys()

    for controler in controlers:

        for att in attList:
            attExist = cmds.attributeQuery(att, n=nameSpace + ":" + controler, ex=1)

            if (attExist == 1):
                lockStatus = cmds.getAttr(nameSpace + ":" + controler + "." + att, l=1)
                # connectStatus = cmds.connectionInfo(nameSpace + ":" + controler + "." + att, ies=True)
                KeyStatus = cmds.keyframe(nameSpace + ":" + controler, attribute=att, tc=1, q=1)
                if (lockStatus == 0 and KeyStatus != None):
                    cmds.setAttr(nameSpace + ":" + controler + "." + att, attributes[att])    