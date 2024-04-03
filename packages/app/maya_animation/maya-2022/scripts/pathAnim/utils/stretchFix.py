
import maya.cmds as cmds


def stretchBtnProc(multiplier, stretchRange):
    stretchVal = stretchRange * multiplier
    fixStretchedRig(stretchVal)

def fixStretchedRig(blendVal):
    sel = cmds.ls(sl=True)
    ctrl = sel[0].split("_")[0] + "_PathAnim_Ctrl"
    attrNames = cmds.listAttr(ctrl, cb=True, sn=True)
    lowestVal = 100
    attrNameArray = list()

    for each in attrNames:
        if each != "pathLength":
            attrNameArray.append(each)
            curAttr = cmds.getAttr(ctrl + "." + each)
            if curAttr < lowestVal:
                lowestVal = curAttr

    for eachAttr in attrNameArray:
        thisAttr = cmds.getAttr(ctrl + "." + eachAttr)
        if thisAttr > lowestVal:
            attrDif = ((thisAttr - lowestVal) * blendVal)
            attrNewVal = thisAttr - attrDif
            cmds.setAttr(ctrl + "." + eachAttr, attrNewVal)