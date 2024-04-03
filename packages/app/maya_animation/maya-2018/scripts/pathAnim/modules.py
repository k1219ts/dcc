import string
import maya.cmds as cmds


def lockAllChannels(object):
    cmds.setAttr(object + ".tx", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".ty", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".tz", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".rx", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".ry", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".rz", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".sx", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".sy", l=True, k=False, channelBox=False)
    cmds.setAttr(object + ".sz", l=True, k=False, channelBox=False)


def bakeProc(object, hideOption, startFrame, stopFrame):
    print "Baking... please wait"
    if hideOption:
        cmds.paneLayout('viewPanes', e=True, manage=False)

    cmds.bakeResults(object, simulation=True, time=(startFrame, stopFrame), pok=True)

    if hideOption:
        cmds.paneLayout('viewPanes', e=True, manage=True)


def hideAnimOffsetLocators(checkBox):
    state = not checkBox.isChecked()

    animOffsetLocators = cmds.ls("*_AnimOffset", r=True)
    for each in animOffsetLocators:
        offsetShape = cmds.listRelatives(each, s=True)
        cmds.setAttr(offsetShape[0] + ".visibility", state)


def restoreTheView():
    cmds.paneLayout('viewPanes', e=True, manage=True)


def hideCtrlPivotLocators(checkBox):
    state = not checkBox.isChecked()
    animOffsetLocators = cmds.ls("*_CtrlPivot", r=True)
    for each in animOffsetLocators:
        offsetShape = cmds.listRelatives(each, s=True)
        cmds.setAttr(offsetShape[0] + ".visibility", state)


def groundOffset(options):
    prefix = options['prefix']
    offset = float(options['groundOffset'])
    projectMeshNodes = cmds.ls(prefix + '_prjMesh*')

    for prjMesh in projectMeshNodes:
        if cmds.objExists(prjMesh):
            cmds.setAttr(prjMesh + '.offsetY', offset)


def deletePathAnim(prefix):
    curves = prefix + "_PathAnimCurves"
    hooks = prefix + "_PathAnimHooks"
    contach = prefix + "_curveF_attach_GRP"
    contrl = prefix + "_curveF_crtl_GRP"

    if cmds.objExists(hooks):
        cmds.delete(hooks)
    if cmds.objExists(curves):
        cmds.delete(curves)
    if cmds.objExists(contach):
        cmds.delete(contach)
    if cmds.objExists(contrl):
        cmds.delete(contrl)

class CreatePathCurves():
    def __init__(self, options):
        self.prefix = options['prefix']
        self.scale = options['scale']
        self.pathCtrl = None
        self.groundCurve = None
        self.bodyCurve = None
        self.groupName = self.prefix + '_PathAnimCurves'

    def create(self):
        if cmds.objExists(self.groupName):
            cmds.confirmDialog(
                title="Warning",
                message="{} already exsist".format(self.groupName),
                button="Close"
            )
            return

        worldUpOption = cmds.upAxis(q=True, axis=True)
        pathCurvesGrp = cmds.group(em=True, name=self.groupName)
        if worldUpOption == "y":
            self.pathCtrl = cmds.curve(
                d=1,
                p=[(-2, 10, -4),
                   (2, 10, -4),
                   (2, 10, 2),
                   (5, 10, 2),
                   (0, 10, 6),
                   (-5, 10, 2),
                   (-2, 10, 2),
                   (-2, 10, -4)],
                k=[0, 1, 2, 3, 4, 5, 6, 7],
                n=self.prefix + "_PathAnim_Ctrl"
            )
            cmds.setAttr(self.pathCtrl + ".tx", lock=True, keyable=False, channelBox=False)
            cmds.setAttr(self.pathCtrl + ".ty", keyable=False, channelBox=False)
            cmds.setAttr(self.pathCtrl + ".tz", lock=True, keyable=False, channelBox=False)
        if worldUpOption == "z":
            self.pathCtrl = cmds.curve(
                d=1,
                p=[(-2, 4, 10),
                   (2, 4, 10),
                   (2, -2, 10),
                   (5, -2, 10),
                   (0, -6, 10),
                   (-5, -2, 10),
                   (-2, -2, 10),
                   (-2, 4, 10)],
                k=[0, 1, 2, 3, 4, 5, 6, 7],
                n=self.prefix + "_PathAnim_Ctrl"
            )
            cmds.setAttr(self.pathCtrl + ".ty", lock=True, keyable=False, channelBox=False)
            cmds.setAttr(self.pathCtrl + ".tz", keyable=False, channelBox=False)
            cmds.setAttr(self.pathCtrl + ".tx", lock=True, keyable=False, channelBox=False)

        cmds.setAttr(self.pathCtrl + ".rx", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".ry", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".rz", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".sx", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".sy", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".sz", lock=True, keyable=False, channelBox=False)
        cmds.setAttr(self.pathCtrl + ".v", lock=True, keyable=False, channelBox=False)

        pathTravelAt = "pathTravel"
        cmds.addAttr(self.pathCtrl, ln=pathTravelAt, at="double", minValue=0, maxValue=100)
        cmds.setAttr(self.pathCtrl + "." + pathTravelAt, e=True, keyable=True)
        pathLenAt = "pathLength"
        cmds.addAttr(self.pathCtrl, ln=pathLenAt, at="double", minValue=0, maxValue=100000)
        cmds.setAttr(self.pathCtrl + "." + pathLenAt, e=True, keyable=False, cb=True)

        knots = [0, 0] + [i for i in range(0, 30)] + [29, 29]
        if worldUpOption == "y":
            groundPoints = [(0, 0, i) for i in range(-10, 310, 10)]
            self.groundCurve = cmds.curve(
                d=3,
                p=groundPoints,
                k=knots,
                name=self.prefix + "_groundCurve"
            )
            bodyPoints = [(0, 5, i) for i in range(-10, 310, 10)]
            self.bodyCurve = cmds.curve(
                d=3,
                p=bodyPoints,
                k=knots,
                name=self.prefix + "_bodyCurve"
            )
        if worldUpOption == "z":
            groundPoints = [(0, i, 0) for i in range(10, -310, -10)]
            self.groundCurve = cmds.curve(
                d=3,
                p=groundPoints,
                k=knots,
                name=self.prefix + "_groundCurve"
            )
            bodyPoints = [(0, i, 5) for i in range(10, -310, -10)]
            self.bodyCurve = cmds.curve(
                d=3,
                p=bodyPoints,
                k=knots,
                name=self.prefix + "_bodyCurve"
            )
        cmds.parent(self.pathCtrl, self.groundCurve, self.bodyCurve, pathCurvesGrp)
        lockAllChannels(self.groundCurve)
        lockAllChannels(self.bodyCurve)

        sCurve = self.groundCurve

        curveInfoNode = cmds.arclen(sCurve, ch=True)
        cmds.addAttr(sCurve, ln="lengthCur", at="double")
        cmds.setAttr(sCurve + ".lengthCur", e=True, keyable=False)
        cmds.addAttr(sCurve, ln="lengthDef", at="double")
        cmds.setAttr(sCurve + ".lengthDef", e=True, keyable=False)
        cmds.addAttr(sCurve, ln="origLen", at="double")
        cmds.setAttr(sCurve + ".origLen", e=True, keyable=False)
        cmds.connectAttr(curveInfoNode + ".arcLength", sCurve + ".lengthCur", f=True)
        defLength = cmds.getAttr(sCurve + ".lengthCur")
        cmds.setAttr(sCurve + ".lengthDef", defLength)
        cmds.setAttr(sCurve + ".origLen", defLength)
        lnCalc = cmds.shadingNode("multiplyDivide", asUtility=True, n=self.groundCurve + "LengthCalc")
        cmds.setAttr(lnCalc + ".operation", 2)
        worldMultNode = cmds.shadingNode("multiplyDivide", asUtility=True, name="worldMultNode")
        cmds.setAttr(worldMultNode + ".input2X", 1)

        scaleVal = self.scale
        cmds.setAttr(pathCurvesGrp + ".scaleX", scaleVal)
        cmds.setAttr(pathCurvesGrp + ".scaleY", scaleVal)
        cmds.setAttr(pathCurvesGrp + ".scaleZ", scaleVal)

        cmds.connectAttr(sCurve + ".lengthCur", lnCalc + ".input1X", f=True)
        cmds.connectAttr(sCurve + ".lengthDef", worldMultNode + ".input1X", f=True)
        cmds.connectAttr(pathCurvesGrp + ".scaleX", worldMultNode + ".input2X", f=True)
        cmds.connectAttr(worldMultNode + ".outputX", lnCalc + ".input2X", f=True)
        cmds.connectAttr(lnCalc + ".outputX", self.pathCtrl + "." + pathLenAt, f=True)

        cmds.setAttr(pathCurvesGrp + ".sx", keyable=False, channelBox=False)
        cmds.setAttr(pathCurvesGrp + ".sy", keyable=False, channelBox=False)
        cmds.setAttr(pathCurvesGrp + ".sz", keyable=False, channelBox=False)
        cmds.select(self.pathCtrl, r=True)
        gndShp = str(cmds.listRelatives(self.prefix + "_groundCurve", type="shape")[0])
        cmds.setAttr(gndShp + ".dispCV", 1)
        bodyShp = str(cmds.listRelatives(self.prefix + "_bodyCurve", type="shape")[0])
        cmds.setAttr(bodyShp + ".dispCV", 1)

class CreateHooks():
    def __init__(self, options):
        self.prefix = options['prefix']
        self.groundOrBody = options['groundOrBody']
        self.scale = options['scale']

    def attachLocatorsToPath(self, locs, pathCurve, ctrl):
        for eachLoc in locs:
            pathTravelAttr = ".pathTravel"
            rigCtrlname = eachLoc.replace(":", "_")
            pathOffsetAttr = rigCtrlname + "Offset"
            cmds.addAttr(ctrl, ln=pathOffsetAttr, at="double")
            cmds.setAttr(ctrl + "." + pathOffsetAttr, e=True, keyable=False, cb=True)
            cmds.select(eachLoc, pathCurve, r=True)

            curveShp = cmds.listRelatives(pathCurve, s=True)
            motionPathNode = cmds.createNode("motionPath")

            cmds.connectAttr(curveShp[0] + ".worldSpace[0]", motionPathNode + ".geometryPath", f=True)

            cmds.setAttr(motionPathNode + ".fractionMode", 1)
            worldUpOption = cmds.upAxis(q=True, axis=True)

            if worldUpOption == "y":
                cmds.setAttr(motionPathNode + ".worldUpType", 3)
                cmds.setAttr(motionPathNode + ".worldUpVectorX", 0)
                cmds.setAttr(motionPathNode + ".worldUpVectorY", 1)
                cmds.setAttr(motionPathNode + ".worldUpVectorZ", 0)
                cmds.setAttr(motionPathNode + ".follow", 1)
                cmds.setAttr(motionPathNode + ".frontAxis", 0)
                cmds.setAttr(motionPathNode + ".upAxis", 1)
            if worldUpOption == "z":
                cmds.setAttr(motionPathNode + ".worldUpType", 3)
                cmds.setAttr(motionPathNode + ".worldUpVectorX", 0)
                cmds.setAttr(motionPathNode + ".worldUpVectorY", 0)
                cmds.setAttr(motionPathNode + ".worldUpVectorZ", 1)

                cmds.setAttr(motionPathNode + ".follow", 1)
                cmds.setAttr(motionPathNode + ".frontAxis", 1)
                cmds.setAttr(motionPathNode + ".upAxis", 2)
                cmds.setAttr(motionPathNode + ".inverseFront", 1)

            cmds.connectAttr(motionPathNode + ".xCoordinate", eachLoc + ".translateX", f=True)
            cmds.connectAttr(motionPathNode + ".yCoordinate", eachLoc + ".translateY", f=True)
            cmds.connectAttr(motionPathNode + ".zCoordinate", eachLoc + ".translateZ", f=True)
            cmds.connectAttr(motionPathNode + ".rotateX", eachLoc + ".rotateX", f=True)
            cmds.connectAttr(motionPathNode + ".rotateY", eachLoc + ".rotateY", f=True)
            cmds.connectAttr(motionPathNode + ".rotateZ", eachLoc + ".rotateZ", f=True)
            cmds.connectAttr(motionPathNode + ".rotateOrder", eachLoc + ".rotateOrder", f=True)

            offsetNode = cmds.shadingNode("plusMinusAverage", asUtility=True)
            offsetMultNode = cmds.shadingNode('multiplyDivide', asUtility=True)
            cmds.setAttr(offsetMultNode + ".input2X", 0.01)

            cmds.connectAttr(offsetNode + ".output1D", offsetMultNode + ".input1X", f=True)

            cmds.connectAttr(offsetMultNode + ".outputX", motionPathNode + ".uValue", f=True)

            cmds.connectAttr(ctrl + pathTravelAttr, offsetNode + ".input1D[0]", f=True)
            cmds.connectAttr(ctrl + "." + pathOffsetAttr, offsetNode + ".input1D[1]", f=True)

    def snapToPosition(self, toSnap, target):
        t = cmds.xform(target, q=True, ws=True, t=True)
        cmds.xform(toSnap, ws=True, t=(t[0], t[1], t[2]))

    def calculateHookOffsets(self, ctrls, hookAttr, curveNames):
        worldUpOption = cmds.upAxis(q=True, axis=True)
        upD = str()
        forwardD = str()
        sideD = str()

        if worldUpOption == "y":
            upD = "Y"
            forwardD = "Z"
            sideD = "X"

        if worldUpOption == "z":
            upD = "Z"
            forwardD = "Y"
            sideD = "X"

        for cnt, each in enumerate(ctrls):
            distLocator = cmds.spaceLocator(n=each + "_tempDistanceLoc")
            tempPointCons = cmds.pointConstraint(each, distLocator[0])
            cmds.delete(tempPointCons)
            # cmds.setAttr(distLocator[0] + ".translate" + sideD, 0)
            hookToUse = str()

            if not curveNames[cnt].find("groundCurve") == -1:
                hookToUse = string.join([self.prefix, each, "GroundHook"], "_")
                cmds.setAttr(distLocator[0] + ".translate" + upD, 0)
            if not curveNames[cnt].find("bodyCurve") == -1:
                hookToUse = string.join([self.prefix, each, "BodyHook"], "_")
                yMult = cmds.getAttr(self.prefix + "_PathAnimCurves.scaleX")
                height = yMult * 5
                cmds.setAttr(distLocator[0] + ".translate" + upD, height)

            distanceShape = cmds.distanceDimension(sp=(-1.0, 4.0, 1.0), ep=(-1.0, -0.0, 1.0))
            distanceNodeLocs = cmds.listConnections(distanceShape)
            cmds.pointConstraint(distLocator[0], distanceNodeLocs[0])

            c = 0.0
            minDistance = 10000
            curOffset = 0

            while c < 40:
                cmds.setAttr(hookAttr[cnt], c)
                self.snapToPosition(distanceNodeLocs[1], hookToUse)
                curDistance = cmds.getAttr(distanceShape + ".distance")

                if curDistance < minDistance:
                    minDistance = curDistance
                    curOffset = c
                c = c + 0.05

            cmds.setAttr(hookAttr[cnt], curOffset)
            distanceTransforms = cmds.listRelatives(distanceShape, p=True)
            cmds.delete(distanceNodeLocs, distanceTransforms, distLocator[0])

        checkIt = None

        if len(ctrls) == 2:
            checkIt = cmds.confirmDialog(
                title="Confirm",
                message="Two controllers were selected - are these a matching pair of left/right controllers?",
                button=("Yes", "No"),
                defaultButton="Yes",
                cancelButton="No",
                dismissString="No"
            )

        if checkIt == "Yes":
            hookAt1 = cmds.getAttr(hookAttr[0])
            hookAt2 = cmds.getAttr(hookAttr[1])
            hookAverage = (hookAt1 + hookAt2) / 2
            cmds.setAttr(hookAttr[0], hookAverage)
            cmds.setAttr(hookAttr[1], hookAverage)
            print "Averaged out both hook offsets for this pair of controls"

    def createAHook(self):
        if not cmds.objExists(self.prefix + "_PathAnim_Ctrl"):
            raise Exception(
                "Please Create Path Curves first"
            )
        sel = cmds.ls(sl=True)
        if not cmds.objExists(self.prefix + "_PathAnimHooks"):
            hooksGrp = cmds.group(em=True, n=self.prefix + "_PathAnimHooks")
            lockAllChannels(hooksGrp)
        else:
            hooksGrp = self.prefix + "_PathAnimHooks"

        ctrl = list()
        hook = list()
        attrs = list()
        curveNames = list()

        for i, each in enumerate(sel):
            ng = self.prefix + "_" + each + self.groundOrBody
            curveToUse = str()
            if self.groundOrBody == "_GroundHook":
                curveToUse = self.prefix + "_groundCurve"
            if self.groundOrBody == "_BodyHook":
                curveToUse = self.prefix + "_bodyCurve"

            ctrl.append(each)
            hook.append(ng)
            curveNames.append(curveToUse)

            tempNull = cmds.spaceLocator(name=ng)
            cmds.parent(tempNull[0], hooksGrp)
            # cmds.select(each)
            # cmds.select(tempNull[0], add=True)
            cmds.setAttr(tempNull[0] + ".localScaleX", (5 * self.scale))
            cmds.setAttr(tempNull[0] + ".localScaleY", (5 * self.scale))
            cmds.setAttr(tempNull[0] + ".localScaleZ", (5 * self.scale))
            cmds.setAttr(tempNull[0] + ".overrideEnabled", 1)
            cmds.setAttr(tempNull[0] + ".overrideColor", 16)
            cmds.parentConstraint(each, tempNull[0], weight=1)
            findPtCns = cmds.listRelatives(tempNull[0], type="parentConstraint")
            cmds.delete(findPtCns)
            cmds.select(tempNull[0], r=True)
            self.attachLocatorsToPath(tempNull, curveToUse, self.prefix + "_PathAnim_Ctrl")

            rigCtrlname = each.replace(":", "_")
            pathOffsetAttr = self.prefix + "_" + rigCtrlname + self.groundOrBody + "Offset"
            attrs.append(self.prefix + "_PathAnim_Ctrl." + pathOffsetAttr)

            print "Created Body Hook for " + each

        self.calculateHookOffsets(ctrl, attrs, curveNames)


class AttachToHooks():
    def __init__(self, options):
        self.prefix = options['prefix']
        self.startFrame = options['loopStartFrame']
        self.stopFrame = options['loopEndFrame']
        self.scale = options['scale']
        self.hideOption = options['hideOption']
        self.groundMesh = options['ground']
        self.groundOffset = options['groundOffset']

    def getHookLocators(self):
        nameSpacer = str()

        while True:
            nameSpacer += "*:"
            hookLs = cmds.ls(self.prefix + nameSpacer + "*GroundHook", r=True)
            if hookLs:
                break

        nameSpacer = str()
        while True:
            nameSpacer += "*:"
            bodyHookLs = cmds.ls(self.prefix + nameSpacer + "*BodyHook", r=True)
            if bodyHookLs:
                hookLs += bodyHookLs
                break

        return hookLs

    def attach(self):
        worldUpOption = cmds.upAxis(q=True, axis=True)

        if not cmds.objExists(self.prefix + "_PathAnim_Ctrl"):
            raise Exception(
                "Please Create Path Curves first"
            )
        cmds.cycleCheck(e=False)
        isTravelKeyed = cmds.listConnections(self.prefix + "_PathAnim_Ctrl.pathTravel", d=False, type="animCurve")

        if isTravelKeyed:
            keyedOrNot = len(isTravelKeyed)
            if keyedOrNot == 1:
                cmds.currentTime(self.startFrame)
                cmds.mute(self.prefix + "_PathAnim_Ctrl.pathTravel")

        hookLocators = list()
        hookLs = self.getHookLocators()

        for each in hookLs:
            if each.find("_GroundHook") != -1:
                animOffsetGrp = each.replace("_GroundHook", "_AnimOffset")
            else:
                animOffsetGrp = each.replace("_BodyHook", "_AnimOffset")

            if not cmds.objExists(animOffsetGrp):
                hookLocators.append(each)

        for each in hookLocators:
            shapeNode = cmds.listRelatives(each, shapes=True)
            cmds.delete(shapeNode)
            if each.find("GroundHook") != -1:
                rigCtrl = each.replace("_GroundHook", "").replace(self.prefix + "_", "")
            else:
                rigCtrl = each.replace("_BodyHook", "").replace(self.prefix + "_", "")

            ng = self.prefix + "_" + rigCtrl + "_AnimOffset"

            tempNull2 = cmds.spaceLocator(name=ng)

            if worldUpOption == "y":
                cmds.setAttr(tempNull2[0] + ".localScaleX", (1 * self.scale))
                cmds.setAttr(tempNull2[0] + ".localScaleZ", (4 * self.scale))

            if worldUpOption == "z":
                cmds.setAttr(tempNull2[0] + ".localScaleX", (4 * self.scale))
                cmds.setAttr(tempNull2[0] + ".localScaleZ", (1 * self.scale))

            cmds.setAttr(tempNull2[0] + ".localScaleY", (1 * self.scale))

            cmds.setAttr(tempNull2[0] + ".overrideEnabled", 1)
            cmds.setAttr(tempNull2[0] + ".overrideColor", 13)
            cmds.parent(tempNull2[0], each)
            cmds.setAttr(tempNull2[0] + ".translateX", 0)
            cmds.setAttr(tempNull2[0] + ".translateY", 0)
            cmds.setAttr(tempNull2[0] + ".translateZ", 0)
            cmds.setAttr(tempNull2[0] + ".rotateX", 0)
            cmds.setAttr(tempNull2[0] + ".rotateY", 0)
            cmds.setAttr(tempNull2[0] + ".rotateZ", 0)

            ng = self.prefix + "_" + rigCtrl + "_AnimLoc"
            animLocNull = cmds.spaceLocator(name=ng)

            cmds.setAttr(animLocNull[0] + ".localScaleX", (1 * self.scale))
            cmds.setAttr(animLocNull[0] + ".localScaleY", (1 * self.scale))
            cmds.setAttr(animLocNull[0] + ".localScaleZ", (1 * self.scale))
            cmds.setAttr(animLocNull[0] + ".overrideEnabled", 1)
            cmds.setAttr(animLocNull[0] + ".overrideColor", 17)

            cmds.parent(animLocNull[0], tempNull2[0])
            cmds.select(rigCtrl)
            cmds.select(animLocNull[0], add=True)

            cmds.parentConstraint(weight=1)

            bakeProc(animLocNull[0], self.hideOption, self.startFrame, self.stopFrame)
            cmds.setInfinity(poi="cycle")

            findPtCns = cmds.listRelatives(animLocNull[0], type="parentConstraint")
            cmds.delete(findPtCns)

            shapeNode3 = cmds.listRelatives(animLocNull[0], shapes=True)
            cmds.delete(shapeNode3)
            ng = self.prefix + "_" + rigCtrl + "_CtrlPivot"
            tempNull3 = cmds.spaceLocator(name=ng)

            cmds.setAttr(tempNull3[0] + ".localScaleX", (2 * self.scale))
            cmds.setAttr(tempNull3[0] + ".localScaleY", (2 * self.scale))
            cmds.setAttr(tempNull3[0] + ".localScaleZ", (2 * self.scale))
            cmds.setAttr(tempNull3[0] + ".overrideEnabled", 1)
            cmds.setAttr(tempNull3[0] + ".overrideColor", 17)
            cmds.parent(tempNull3[0], animLocNull[0])
            cmds.setAttr(tempNull3[0] + ".translateX", 0)
            cmds.setAttr(tempNull3[0] + ".translateY", 0)
            cmds.setAttr(tempNull3[0] + ".translateZ", 0)
            cmds.setAttr(tempNull3[0] + ".rotateX", 0)
            cmds.setAttr(tempNull3[0] + ".rotateY", 0)
            cmds.setAttr(tempNull3[0] + ".rotateZ", 0)

            tLock = 0
            rLock = 0
            lockX = cmds.getAttr(rigCtrl + ".tx", l=True)

            if lockX:
                tLock = 1

            lockRX = cmds.getAttr(rigCtrl + ".rx", l=True)

            if lockRX:
                rLock = 1

            if not tLock:
                cmds.pointConstraint(tempNull3[0], rigCtrl, weight=1, n=self.prefix + "_pathAnim_PointConstraint")

            if rLock == 0:
                cmds.orientConstraint(tempNull3[0], rigCtrl, weight=1, n=self.prefix + "_pathAnim_OrientConstraint")

            if each.find("GroundHook") != -1 and self.groundMesh:
                ctrl = self.prefix + "_PathAnim_Ctrl"
                cmds.addAttr(ctrl, ln="groundOffset", at="double", dv=self.groundOffset)
                projectMeshNode = cmds.createNode("ghProjectMesh", n=self.prefix + '_prjMesh#')
                cmds.setAttr(projectMeshNode + ".WeightRotate", 0)
                #cmds.setAttr(projectMeshNode + ".offsetY", 0.5)
                cmds.connectAttr(ctrl + ".groundOffset", projectMeshNode + ".offsetY")
                groundShape = cmds.listRelatives(self.groundMesh, s=True, type="mesh")
                pairBlendNode = cmds.listConnections(rigCtrl + ".translateY")
                pmaNode = cmds.createNode("plusMinusAverage")
                cmds.connectAttr(pairBlendNode[0] + ".outTranslateY", pmaNode + ".input1D[0]")
                cmds.connectAttr(pmaNode + ".output1D", rigCtrl + ".translateY", f=True)
                cmds.connectAttr(animLocNull[0] + ".worldMatrix[0]", projectMeshNode + ".inputMatrix")
                cmds.connectAttr(rigCtrl + ".parentInverseMatrix[0]", projectMeshNode + ".inParentInverseMatrix")
                cmds.connectAttr(groundShape[0] + ".worldMesh[0]", projectMeshNode + ".inputMeshTarget")

                cmds.connectAttr(projectMeshNode + ".outputTranslateY", pmaNode + ".input1D[1]")

            print "Attached " + rigCtrl + " to Hook"

        muted = cmds.mute(self.prefix + "_PathAnim_Ctrl.pathTravel", q=True)
        if muted:
            cmds.mute(self.prefix + "_PathAnim_Ctrl.pathTravel", disable=True, force=True)

        cmds.select(self.prefix + "_PathAnim_Ctrl", r=True)
        print "Rig Attached To Hooks"


class bakePathAnimToRigProc():
    def __init__(self, options, parent=None):
        self.prefix = options['prefix']
        self.hideOption = options['hideOption']
        self.startFrame = options['bakeStartFrame']
        self.stopFrame = options['bakeEndFrame']
        self.parent = parent

    def getLocators(self, type):
        nameSpacer = str()

        while True:
            nameSpacer += "*:"
            locs = cmds.ls(self.prefix + nameSpacer + "*{}".format(type), r=True)
            if locs:
                break
        return locs

    def bake(self):
        rigCtrls = list()

        groundHookLs = self.getLocators("GroundHook")
        bodyHookLs = self.getLocators("BodyHook")

        for each in groundHookLs:
            rigCtrl = each.replace("_GroundHook", "").replace(self.prefix + "_", "")
            rigCtrls.append(rigCtrl)
        for each in bodyHookLs:
            rigCtrl = each.replace("_BodyHook", "").replace(self.prefix + "_", "")
            rigCtrls.append(rigCtrl)

        animOffsetLocs = self.getLocators("AnimOffset")

        if not animOffsetLocs:
            raise Exception(
                "No Rig Ctrls Attached to Path System.. Bake not attempted"
            )

        if self.hideOption:
            cmds.paneLayout('viewPanes', e=True, manage=False)

        cmds.select(rigCtrls, r=True)

        cmds.cycleCheck(e=False)
        print "Baking Path Animation.."
        frameCount = (self.stopFrame + 1) - self.startFrame
        cmds.currentTime(self.startFrame)
        cmds.setKeyframe(rigCtrls)

        stepVal = 100 / frameCount

        for f in range(int(self.startFrame), int(self.stopFrame + 1)):
            self.parent.progressBar.setValue(stepVal * f)
            cmds.currentTime(f)
            cmds.setKeyframe(rigCtrls)

        cmds.currentTime(self.startFrame)
        print "Path Animation Baked to Rig.. Now ready to Delete Path Anim System"

        if self.hideOption:
            cmds.paneLayout('viewPanes', e=True, manage=True)


