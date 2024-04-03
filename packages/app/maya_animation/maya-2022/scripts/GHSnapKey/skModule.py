# encoding:utf-8
import maya.cmds as cmds


def bakeLocators(
        nodes,
        minTime,
        maxTime,
        toAnother=False,
        newTarget=None,
        sampleBy=1,
        smart=False,
):
    LocatorList = []
    tempConstraintList = []

    for i in nodes:
        if toAnother and newTarget:
            locatorName = ":".join(i.split(":")[:-1] + [newTarget + "_GHsnapKey_LOC"])
        else:
            locatorName = i + "_GHsnapKey_LOC"
        LocatorList.append(str(cmds.spaceLocator(n=locatorName)[0]))
        tempConstraintList.append(cmds.parentConstraint(i, locatorName, mo=False, w=True, st="none", sr="none")[0])
    if smart:
        cmds.bakeResults(LocatorList,
                         simulation=True,
                         t=(minTime - 1, maxTime + 1),
                         smart=smart,
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True)
    else:
        cmds.bakeResults(LocatorList,
                         simulation=True,
                         sampleBy=sampleBy,
                         t=(minTime - 1, maxTime + 1),
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True)

    cmds.delete(tempConstraintList)


def loc2ctrl(
        minTime,
        maxTime,
        sampleBy=1,
        smart=False,
):
    tempConstList2 = []
    consList = []

    locList = cmds.ls(type="locator")
    bakedlocs = []

    for i in locList:
        if i.count("_GHsnapKey_LOC") != 0:
            bakedlocs.append(cmds.listRelatives(i, p=1)[0])

    for i in bakedlocs:
        try:
            consName = i.split("_GHsnapKey_LOC")[0]
            consList.append(consName)

            TlockStatus = cmds.getAttr(consName + ".tx", l=True)
            RlockStatus = cmds.getAttr(consName + ".rx", l=True)

            if not TlockStatus and not RlockStatus:
                tempConstList2.append(cmds.parentConstraint(i, consName, w=1, mo=False)[0])
            elif TlockStatus:
                tempConstList2.append(cmds.orientConstraint(i, consName, w=1, mo=False)[0])
            elif RlockStatus:
                tempConstList2.append(cmds.pointConstraint(i, consName, w=1, mo=False)[0])
        except:
            pass
    if smart:
        cmds.bakeResults(consList,
                         simulation=True,
                         t=(minTime, maxTime),
                         smart=smart,
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True)
    else:
        cmds.bakeResults(consList,
                         simulation=True,
                         sampleBy=sampleBy,
                         t=(minTime, maxTime),
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True)

    cmds.delete(tempConstList2)
    cmds.delete(bakedlocs)