from pxr import Usd, UsdGeom
import os
import hou
import json
import math
import _alembic_hom_extensions as abc

def readJson(jsonPath):
    file = open(jsonPath,'r')
    js = json.loads(file.read())
    file.close()
    return js

def camZoom(node,fps,axis):
    fileName = node.parm("file").eval().replace(".usd", ".abc")
    objectPath = "/"
    time = node.parm("frame").eval()/fps
    abcCamDict = abc.alembicGetCameraDict(fileName,objectPath,time)
    ap=[]

    ap.append(abcCamDict["aperture"]/25.4)
    ap.append(abcCamDict["aperture"]/abcCamDict["filmaspectratio"]/25.4)

    return ap[axis]

def setKey(tempJs, tempParm, fps, type, scale = 1.0):
    if (type == 0):
        tempParm.set(tempJs["value"] * scale)

    elif (type == 1):
        tempFrame = tempJs['frame']

        for j in range(len(tempFrame)):
            hou_keyframe = hou.Keyframe()
            time = hou.frameToTime(tempFrame[j])
            value = tempJs['value'][j] * scale
            hou_keyframe.setTime(time)
            hou_keyframe.setValue(value)
            tempParm.setKeyframe(hou_keyframe)
        keys = tempParm.keyframes()

        if tempJs.has_key('weight'):
            for j in range(len(keys)):
                hou_keyframe = keys[j]
                inAngle = tempJs['angle'][j * 2]
                outAngle = tempJs['angle'][j * 2 + 1]
                inWeight = tempJs['weight'][j * 2]
                outWeight = tempJs['weight'][j * 2 + 1]
                inSlope = fps * math.tan(math.radians(inAngle))
                outSlope = fps * math.tan(math.radians(outAngle))
                inAccel = math.sqrt(
                    math.pow(inWeight, 2) * (math.pow(inSlope, 2) + 1) / (math.pow(fps, 2) + math.pow(inSlope, 2)))
                outAccel = math.sqrt(
                    math.pow(outWeight, 2) * (math.pow(outSlope, 2) + 1) / (math.pow(fps, 2) + math.pow(outSlope, 2)))
                hou_keyframe.setSlope(outSlope)
                hou_keyframe.setInSlope(inSlope)
                hou_keyframe.setAccel(outAccel)
                hou_keyframe.setInAccel(inAccel)
                hou_keyframe.setExpression('bezier()', hou.exprLanguage.Hscript)
                tempParm.setKeyframe(hou_keyframe)

        else:
            for j in range(len(keys)):
                hou_keyframe = keys[j]
                inAngle = tempJs['angle'][j * 2]
                outAngle = tempJs['angle'][j * 2 + 1]
                inSlope = fps * math.tan(math.radians(inAngle))
                outSlope = fps * math.tan(math.radians(outAngle))
                hou_keyframe.setSlope(outSlope)
                hou_keyframe.setInSlope(inSlope)
                hou_keyframe.setExpression('cubic()', hou.exprLanguage.Hscript)
                tempParm.setKeyframe(hou_keyframe)

def makePointInstancerPackage(stage, layoutName, protoRelList, mainFile, primPath, geoNode):
    # protorelList
    # geoNode = hou.node(hou.pwd().path() + "/..")

    mergeNode = geoNode.createNode("merge")
    pointProtoNode = geoNode.createNode("usdinstanceprototypes")
    pointProtoNode.setInput(0, mergeNode)

    blastNode = geoNode.createNode("blast")
    blastNode.setInput(0, pointProtoNode)
    blastNode.parm("group").set('`stamp("../copy1", "protoindex", 0)`')
    blastNode.parm("negate").set(True)

    protoNodeList = []
    for index, protoRel in enumerate(protoRelList):
        protoPrim = stage.GetPrimAtPath(protoRel)
        protoNode = geoNode.createNode("usdimport", protoPrim.GetName())
        filepath = protoPrim.GetAssetInfo()["identifier"].resolvedPath
        protoNode.parm("import_file").set(filepath)
        protoNode.parm("import_primpath").set("/asset/" + os.path.basename(filepath).split('.')[0])

        if protoPrim.HasAttribute("xformOpOrder") and protoPrim.GetAttribute("xformOpOrder").Get() != None:
            xformNode = geoNode.createNode("xform", protoPrim.GetName() + "_xform")

            for OpOrder in protoPrim.GetAttribute("xformOpOrder").Get():
                attr = protoPrim.GetAttribute(OpOrder).Get()
                xyzOrder = "xyz"
                prefix = "t"

                if "translate" in OpOrder:
                    prefix = "t"
                    # xformNode.parm('tx').set(attr[0])
                    # xformNode.parm('ty').set(attr[1])
                    # xformNode.parm('tz').set(attr[2])
                elif "scale" in OpOrder:
                    prefix = "s"
                    # xformNode.parm("sx").set(attr[0])
                    # xformNode.parm("sy").set(attr[1])
                    # xformNode.parm("sz").set(attr[2])
                elif "rotate" in OpOrder:
                    xyzOrder = OpOrder[-3:]
                    xyzOrder = xyzOrder.lower()
                    prefix = "r"
                    # xformNode.parm("rOrd").set(rotateOrder[xyzOrder])
                else:
                    continue
                    # for index in range(3):
                    #     xformNode.parm("r%s" % xyzOrder[index]).set(attr[index])

                print prefix, xyzOrder, attr
                for index in range(3):
                    xformNode.parm("%s%s" % (prefix, xyzOrder[index])).set(attr[index])

            xformNode.setInput(0, protoNode)
            mergeNode.setInput(index, xformNode)
            protoNodeList.append(protoNode)
            protoNodeList.append(xformNode)
        else:
            mergeNode.setInput(index, protoNode)
            protoNodeList.append(protoNode)


    mainUsdFile = geoNode.createNode("usdimport")
    mainUsdFile.parm("import_file").set(mainFile)
    mainUsdFile.parm("import_primpath").set(str(primPath))

    unpackNode = geoNode.createNode("usdunpack")
    unpackNode.parm("unpack_traversal").set(4)
    unpackNode.parm("unpack_geomtype").set(1)
    unpackNode.parm("import_primvars").set("*")
    unpackNode.setInput(0, mainUsdFile)

    copyStampNode = geoNode.createNode("copy")
    copyStampNode.parm("param1").set("protoindex")
    copyStampNode.parm("val1").setExpression("@protoindex")
    copyStampNode.parm("stamp").set(True)
    copyStampNode.parm("cacheinput").set(True)
    copyStampNode.setInput(0, blastNode)
    copyStampNode.setInput(1, unpackNode)

    outNode = geoNode.createNode("null", "OUT")
    outNode.setFirstInput(copyStampNode)
    outNode.setDisplayFlag(1)
    outNode.setRenderFlag(True)

    divideCount = 5
    basePos = hou.Vector2(0, 0)
    for index, protoNode in enumerate(protoNodeList):
        protoNode.setPosition(hou.Vector2(basePos.x() + ((index / divideCount) * 2), basePos.y() - ((index % divideCount) + 1)))

    mergeNode.moveToGoodPosition()
    pointProtoNode.moveToGoodPosition()
    blastNode.moveToGoodPosition()
    mainUsdFile.moveToGoodPosition()
    unpackNode.moveToGoodPosition()
    copyStampNode.moveToGoodPosition()
    outNode.moveToGoodPosition()

    layoutSubnet = geoNode.collapseIntoSubnet(protoNodeList + [mergeNode, pointProtoNode, blastNode, mainUsdFile, unpackNode, copyStampNode, outNode], layoutName)

    return layoutSubnet

def rigSetup(shotNode, rigPrim):
    rigNetworkBox = shotNode.createNetworkBox("rigNetworkBox")
    rigNetworkBox.setComment("RIG")
    rigNetworkBox.setColor(hou.Color((0.765, 1, 0.576)))

    rigVariantList = ["aniVersion", "simVersion", "zennVersion"]
    makeNodeList = []

    mergeNode = shotNode.createNode("merge", "RIG_Merge")

    outNode = shotNode.createNode("null", "RIG_OUT")
    outNode.setFirstInput(mergeNode)

    for index, dataPrim in enumerate(rigPrim.GetChildren()):
        # /shot/rig/tiger{aniVersion = v023}{zennVersion = v029}
        primPath = str(dataPrim.GetPath())
        for variant in rigVariantList:
            if dataPrim.GetVariantSets().HasVariantSet(variant):
                selectVariant = dataPrim.GetVariantSets().GetVariantSet(variant).GetVariantSelection()
                primPath += "{%s=%s}" % (variant, selectVariant)
        rigNode = shotNode.createNode("usdimport", dataPrim.GetName())
        rigNode.parm("import_file").set(rigPrim.GetStage().GetRootLayer().realPath)
        rigNode.parm("import_primpath").set(primPath)
        rigNode.parm("import_time").setExpression("$FF")

        unpackNode = shotNode.createNode("usdunpack", "%s_unpack" % dataPrim.GetName())
        unpackNode.parm("unpack_traversal").set(4)
        unpackNode.parm("unpack_geomtype").set(1)
        unpackNode.parm("import_primvars").set("*")
        unpackNode.setInput(0, rigNode)

        makeNodeList.append(rigNode)
        makeNodeList.append(unpackNode)

        mergeNode.setInput(index, unpackNode)

    outNode.setDisplayFlag(1)
    outNode.setRenderFlag(True)

    for makeNode in makeNodeList:
        rigNetworkBox.addNode(makeNode)
    rigNetworkBox.addNode(mergeNode)
    rigNetworkBox.addNode(outNode)

    basePos = hou.Vector2(0, 0)
    divideCount = 2
    for index, makeNode in enumerate(makeNodeList):
        makeNode.setPosition(hou.Vector2(basePos.x() + ((index / divideCount) * 2), basePos.y() - ((index % divideCount) + 1)))
        # makeNode.moveToGoodPosition()

    mergeNode.moveToGoodPosition()
    outNode.moveToGoodPosition()

    rigNetworkBox.fitAroundContents()

    return rigNetworkBox

def camSetup(camPrim, camName):
    objNode = hou.node("/obj")
    camNode = objNode.createNode("pixar::usdcamera", camName)
    fps = hou.fps()

    mainCam = None
    for prim in camPrim.GetChildren():
        if prim.GetName() == "main_cam":
            mainCam = prim

    sdfPath = mainCam.GetAssetInfoByKey("identifier")
    usdCamPath = sdfPath.resolvedPath

    curCamDir = os.path.dirname(usdCamPath)

    camNode.parm("file").set("%s/camera.usd" % curCamDir)
    camNode.parm("primpath").set("/cameras/main_cam")

    # TODO: 2D PanZoom
    # panzoomFile = usdCamPath.replace(".usd", ".panzoom")
    # if os.path.exists(panzoomFile):
    #     pzjs = readJson(panzoomFile)
    #     camNode.parmTuple("win").deleteAllKeyframes()
    #     camNode.parmTuple("winsize").deleteAllKeyframes()
    #     scaleX = 1 / camZoom(camNode, fps, 0)
    #     scaleY = 1 / camZoom(camNode, fps, 1)
    #     setKey(pzjs["2DPanZoom"]["hpn"], camNode.parm("winx"), fps, (pzjs["2DPanZoom"]["hpn"].has_key("frame")) and 1 or 0, scaleX)
    #     setKey(pzjs["2DPanZoom"]["vpn"], camNode.parm("winy"), fps, (pzjs["2DPanZoom"]["vpn"].has_key("frame")) and 1 or 0, scaleY)
    #     setKey(pzjs["2DPanZoom"]["zom"], camNode.parm("winsizex"), fps, (pzjs["2DPanZoom"]["zom"].has_key("frame")) and 1 or 0)
    #     setKey(pzjs["2DPanZoom"]["zom"], camNode.parm("winsizey"), fps, (pzjs["2DPanZoom"]["zom"].has_key("frame")) and 1 or 0)


def setSetup(shotNode, stage):
    setNetworkBox = shotNode.createNetworkBox("setNetworkBox")
    setNetworkBox.setComment("SET")
    setNetworkBox.setColor(hou.Color((0.29, 0.565, 0.886)))

    layoutList = []
    for travPrim in stage.Traverse():
        if travPrim.GetTypeName() == "PointInstancer":
            protorelList = UsdGeom.PointInstancer(travPrim).GetPrototypesRel().GetTargets()
            layoutList.append(makePointInstancerPackage(stage, travPrim.GetParent().GetName(),
                                                        protorelList, stage.GetRootLayer().realPath,
                                                        travPrim.GetPath(), shotNode))

    mergeNode = shotNode.createNode("merge", "SET_Merge")
    outNode = shotNode.createNode("null", "SET_OUT")

    for index, node in enumerate(layoutList):
        node.setPosition(hou.Vector2((index / 5) * 2, (index % 5) + 1))
        mergeNode.setInput(index, node)
    outNode.setInput(0, mergeNode)

    mergeNode.moveToGoodPosition()
    outNode.moveToGoodPosition()
    outNode.setDisplayFlag(1)
    outNode.setRenderFlag(True)

    # layoutNode = shotNode.collapseIntoSubnet(layoutList + [mergeNode, outNode],
    #                                                                    os.path.basename(stage.GetRootLayer().realPath).split('.')[0])
    # layoutNode.moveToGoodPosition()
    for layoutNode in layoutList:
        setNetworkBox.addNode(layoutNode)
    setNetworkBox.addNode(mergeNode)
    setNetworkBox.addNode(outNode)
    setNetworkBox.fitAroundContents()

    return setNetworkBox

def sceneSetup():
    show = hou.expandString("$SHOW")
    seq = hou.expandString("$SEQ")
    shot = hou.expandString("$SHOT")
    # show = "ssr"
    # seq = "MET"
    # shot = "MET_0410"

    shotNode = hou.node("/obj/{SHOW}_{SHOT}".format(SHOW = show, SHOT = shot))

    if not shotNode:
        shotNode = hou.node("/obj").createNode("geo", "{SHOW}_{SHOT}".format(SHOW = show, SHOT = shot))

    stage = Usd.Stage.Open("/show/{SHOW}_pub/shot/{SEQ}/{SHOT}/{SHOT}.usd".format(SHOW = show, SHOT = shot, SEQ = seq))

    defPrim = stage.GetDefaultPrim()

    startFrame = stage.GetStartTimeCode()
    endFrame = stage.GetEndTimeCode()
    fps = stage.GetTimeCodesPerSecond()

    hou.playbar.setTimeRange((startFrame - 1) / fps, endFrame / fps)
    # fps = hou.fps()
    # tset = "tset {0} {1}".format((startFrame - 1) / fps, endFrame / fps)
    # hou.hscript(tset)
    hou.playbar.setPlaybackRange(startFrame, endFrame)
    hou.setFrame(1001)

    networkBoxList = []
    for partPrim in defPrim.GetChildren():
        if partPrim.GetName() == "cameras":
            camSetup(partPrim, "%s_%s_main_cam" % (show, shot))
        elif partPrim.GetName() == "rig":
            networkBoxList.append(rigSetup(shotNode, partPrim))
        elif partPrim.GetName() == "set":
            networkBoxList.append(setSetup(shotNode, stage))

    if len(networkBoxList) > 0:
        nodePosX = 0
        input1Pos = hou.Vector2(0, 0) # hou.pwd().indirectInputs()[0].position()
        for netNode in networkBoxList:
            print hou.Vector2(nodePosX, 0)
            netNode.setPosition(hou.Vector2(input1Pos.x() + nodePosX, input1Pos.y() - 2))
            nodePosX += netNode.size().x() + 0.5
