
import maya.cmds as cmds
import aniCommon


def lockAllChannels(object, state):
    cmds.setAttr(object + ".tx", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".ty", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".tz", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".rx", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".ry", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".rz", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".sx", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".sy", l=state, k=False, channelBox=False)
    cmds.setAttr(object + ".sz", l=state, k=False, channelBox=False)


def rebuildPathCurves(curve, newSpans):
    for c in curve:
        cmds.rebuildCurve(
            c,
            ch=1,
            rpo=1,
            rt=0,
            end=1,
            kr=0,
            kcp=0,
            kep=1,
            kt=0,
            s=newSpans,
            d=2,
            tol=0.0001
        )

    print "Path Curves Rebuilt with " + str(newSpans) + " spans"


def reverseCurve(curves):
    for c in curves:
        cmds.reverseCurve(c, ch=0, rpo=1)
    print "Path Curve Direction Reversed"


def projectPathCurves(sel):
    if sel and len(sel) != 2:
        raise Exception("Please select Mesh object and Curve")
    if sel:
        selShp = cmds.listRelatives(sel[0], s=True, fullPath=True)
        if not cmds.objectType(selShp) == 'mesh':
            raise Exception("Please select Mesh object first")

    nameSpace = aniCommon.getNameSpace(sel[1])

    if nameSpace:
        animCurve = sel[1].split(":")[-1]
        prefix = animCurve.split("_")[0]
        GC = nameSpace + ":" + prefix + "_groundCurve"
        BC = nameSpace + ":" + prefix + "_bodyCurve"
        curveGrp = nameSpace + ":" + prefix + "_PathAnimCurves"
    else:
        animCurve = sel[1]
        prefix = animCurve.split("_")[0]
        GC = prefix + "_groundCurve"
        BC = prefix + "_bodyCurve"
        curveGrp = prefix + "_PathAnimCurves"

    floorMesh = sel[0]

    worldUpOption = cmds.upAxis(q=True, axis=True)

    yOffset = float()
    zOffset = float()
    projCurve = list()

    groundCurveShp = cmds.listRelatives(GC, s=True)
    bodyCurveShp = cmds.listRelatives(BC, s=True)
    cvs = cmds.getAttr(groundCurveShp[0] + ".spans")
    deg = cmds.getAttr(groundCurveShp[0] + ".degree")
    if isinstance(cvs, list):
        cvs = int(list(set(cvs))[0])
    if isinstance(deg, list):
        deg = int(list(set(deg))[0])
    totalCurvePoints = cvs + deg
    rigScale = cmds.getAttr(curveGrp + ".scaleX")

    if worldUpOption == "y":
        yOffset = 5 * rigScale
        zOffset = 0
        projCurve = cmds.polyProjectCurve(
            GC, floorMesh,
            ch=True,
            direction=(0, 1, 0),
            pointsOnEdges=0,
            curveSamples=50,
            automatic=1
        )
    if worldUpOption == "z":
        yOffset = 0
        zOffset = 5 * rigScale
        projCurve = cmds.polyProjectCurve(
            GC, floorMesh,
            ch=True,
            direction=(0, 0, 1),
            pointsOnEdges=0,
            curveSamples=50,
            automatic=1
        )

    rebuiltProjectedCurve = cmds.rebuildCurve(
        projCurve[0],
        ch=1,
        rpo=1,
        rt=0,
        end=1,
        kr=0,
        kcp=0,
        kep=1,
        kt=0,
        s=cvs,
        d=deg,
        tol=0.0001
    )
    cmds.delete(projCurve)

    rebuiltProjCurveShp = cmds.listRelatives(rebuiltProjectedCurve[0]+"rebuiltCurve1", s=True)

    for c in range(totalCurvePoints):
        xPos = cmds.getAttr(rebuiltProjCurveShp[0]+".controlPoints[{0}].xValue".format(c))
        try:
            cmds.setAttr(groundCurveShp[0]+".controlPoints[{0}].xValue".format(c), (xPos / rigScale))
            cmds.setAttr(bodyCurveShp[0]+".controlPoints[{0}].xValue".format(c), (xPos / rigScale))
        except:
            cmds.setAttr(groundCurveShp[0] + ".controlPoints[{0}].xValue".format(c), (xPos / rigScale), (xPos / rigScale))
            cmds.setAttr(bodyCurveShp[0] + ".controlPoints[{0}].xValue".format(c), (xPos / rigScale), (xPos / rigScale))

        yPos = cmds.getAttr(rebuiltProjCurveShp[0]+".controlPoints[{0}].yValue".format(c))
        try:
            cmds.setAttr(groundCurveShp[0]+".controlPoints[{0}].yValue".format(c), (yPos / rigScale))
            cmds.setAttr(bodyCurveShp[0]+".controlPoints[{0}].yValue".format(c), ((yPos + yOffset) / rigScale))
        except:
            cmds.setAttr(groundCurveShp[0] + ".controlPoints[{0}].yValue".format(c), (yPos / rigScale), (yPos / rigScale))
            cmds.setAttr(bodyCurveShp[0] + ".controlPoints[{0}].yValue".format(c), ((yPos + yOffset) / rigScale), ((yPos + yOffset) / rigScale))

        zPos = cmds.getAttr(rebuiltProjCurveShp[0]+".controlPoints[{0}].zValue".format(c))
        try:
            cmds.setAttr(groundCurveShp[0]+".controlPoints[{0}].zValue".format(c), (zPos / rigScale))
            cmds.setAttr(bodyCurveShp[0]+".controlPoints[{0}].zValue".format(c), ((zPos + zOffset) / rigScale))
        except:
            cmds.setAttr(groundCurveShp[0] + ".controlPoints[{0}].zValue".format(c), (zPos / rigScale), (zPos / rigScale))
            cmds.setAttr(bodyCurveShp[0] + ".controlPoints[{0}].zValue".format(c), ((zPos + zOffset) / rigScale), ((zPos + zOffset) / rigScale))

    cmds.delete(rebuiltProjectedCurve[0]+"rebuiltCurve1")

    lockAllChannels(GC, False)
    lockAllChannels(BC, False)

    cmds.parent(GC, w=True)
    cmds.parent(BC, w=True)

    cmds.setAttr(GC + ".tx", 0)
    cmds.setAttr(GC + ".ty", 0)
    cmds.setAttr(GC + ".tz", 0)
    cmds.setAttr(GC + ".rx", 0)
    cmds.setAttr(GC + ".ry", 0)
    cmds.setAttr(GC + ".rz", 0)

    cmds.setAttr(BC + ".tx", 0)
    cmds.setAttr(BC + ".ty", 0)
    cmds.setAttr(BC + ".tz", 0)
    cmds.setAttr(BC + ".rx", 0)
    cmds.setAttr(BC + ".ry", 0)
    cmds.setAttr(BC + ".rz", 0)

    cmds.parent(GC, curveGrp)
    cmds.parent(BC, curveGrp)

    lockAllChannels(GC, True)
    lockAllChannels(BC, True)

    print "PathCurves Projected to " + floorMesh

def resetCurvePoints(sel):
    worldUpOption = cmds.upAxis(q=True, axis=True)

    up = str()
    forward = str()
    side = str()
    neg = int()

    if worldUpOption == "y":
        up = "y"
        forward = "z"
        side = "x"
        neg = 1

    if worldUpOption == "z":
        up = "z"
        forward = "y"
        side = "x"
        neg = -1

    selV = cmds.filterExpand(sm=28, ex=1)

    if not selV:
        raise Exception("Please Select CVs")

    curveName = sel[0].split(".")[0]
    if curveName.find("groundCurve") != -1:
        GC = curveName
    elif curveName.find("bodyCurve") != -1:
        GC = curveName.replace("bodyCurve", "groundCurve")
    else:
        raise Exception("Please Select Ground or Body curve's CVs")

    groundCurveShp = cmds.listRelatives(GC, s=True)
    cvs = cmds.getAttr(groundCurveShp[0] + ".spans")
    deg = cmds.getAttr(groundCurveShp[0] + ".degree")
    totalCurvePoints = cvs + deg

    for each in selV:
        cmds.setAttr(each + "." + side + "Value", 0)

        buffer = each.split(".")
        curveNameStart = buffer[0]

        curveNameE = buffer[1] + "cv"
        curvePointNumName1 = curveNameE.replace("cv[", "")
        curvePointNumName2 = curvePointNumName1.replace("]cv", "")
        curveZPoint = int(curvePointNumName2)

        yPos = 0
        if curveNameStart.find("bodyCurve") != -1:
            yPos = 5

        cmds.setAttr(each + "." + up + "Value", yPos)

        zPos = -10
        forwardOffset = 310 / (totalCurvePoints-1)
        zPos = (zPos + (forwardOffset * curveZPoint)) * neg
        cmds.setAttr(each + "." + forward + "Value", zPos)