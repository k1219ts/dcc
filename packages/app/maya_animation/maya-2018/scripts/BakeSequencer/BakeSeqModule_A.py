__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.mel as mm

def bakeObject(object=list(), inTime=1, outTime=1, bakeToNewLayer = False):
    cmds.bakeResults(object, simulation=True, t=(900, outTime), sampleBy=1,
                     disableImplicitControl=True, preserveOutsideKeys=True,
                     sparseAnimCurveBake=False, removeBakedAttributeFromLayer=False,
                     removeBakedAnimFromLayer=False, bakeOnOverrideLayer = bakeToNewLayer,
                     minimizeRotation=True, controlPoints=False, shape=True)

def bakeAllKeys():
    animCrv = cmds.ls(type="animCurve")

    animObj = list()

    for crv in animCrv:
        objName = cmds.listConnections(crv)
        if objName:
            if not animObj or objName[0] not in animObj:
                animObj.append(objName[0])
            elif objName[0] in animObj:
                pass

    lowTime = None
    highTime = None
    for s in animCrv:
        isReferenced = cmds.referenceQuery(s, isNodeReferenced=1)
        if not isReferenced:
            stimeList = cmds.keyframe(s, q=1, tc=1)
            if stimeList:
                if lowTime == None:
                    lowTime = stimeList[0]
                elif stimeList[0] <= lowTime:
                    lowTime = stimeList[0]
                elif stimeList[0] > lowTime:
                    lowTime = lowTime

                if highTime == None:
                    highTime = stimeList[-1]
                elif stimeList[-1] >= highTime:
                    highTime = stimeList[-1]
                elif stimeList[-1] < highTime:
                    highTime = highTime
            else:
                cmds.confirmDialog("No KeyFrames")
                return

    bakeObject(animObj, lowTime, highTime)

    return animObj

def writeSeqData(ls_sequencer = list() ):
    seqData = dict()

    for ls_sequencer_i in ls_sequencer:
        seqData[ls_sequencer_i] = dict()
        ls_seqShot = cmds.listConnections('%s.shots' % ls_sequencer_i)

        for ls_seqShot_i in ls_seqShot:
            seqData[ls_sequencer_i][ls_seqShot_i] = dict()
            ori_startFrame = cmds.getAttr(ls_seqShot_i + ".startFrame")
            ori_endFrame = cmds.getAttr(ls_seqShot_i + ".endFrame")
            seqShot_startFrame = cmds.getAttr(ls_seqShot_i + ".sequenceStartFrame")
            seqShot_endFrame = cmds.getAttr(ls_seqShot_i + ".sequenceEndFrame")
            seqShot_scale = cmds.getAttr(ls_seqShot_i + ".scale")

            seqData[ls_sequencer_i][ls_seqShot_i]["startframe"] = ori_startFrame
            seqData[ls_sequencer_i][ls_seqShot_i]["endframe"] = ori_endFrame
            seqData[ls_sequencer_i][ls_seqShot_i]["sequenceStartFrame"] = seqShot_startFrame
            seqData[ls_sequencer_i][ls_seqShot_i]["sequenceEndFrame"] = seqShot_endFrame
            seqData[ls_sequencer_i][ls_seqShot_i]["scale"] = seqShot_scale

    return seqData

def writeseqDataFrameIndexed(seqData):
    seqDataFrameIndexed = dict()
    for seqName in seqData:
        for shotName in seqData[seqName]:
            shotStartFrame = seqData[seqName][shotName]["startframe"]
            seqStartFrame = seqData[seqName][shotName]["sequenceStartFrame"]
            seqEndFrame = seqData[seqName][shotName]["sequenceEndFrame"]
            seqScale = seqData[seqName][shotName]["scale"]

            seqDataFrameIndexed[shotStartFrame] = [seqStartFrame, seqEndFrame, seqScale]

    return seqDataFrameIndexed

def applyTimewarp(seqData, seqDataFrameIndexed, animObj):
    playbackStart = cmds.playbackOptions(q=1, min=1)
    playbackEnd = cmds.playbackOptions(q=1, max=1)
    startKeyFrameList = seqDataFrameIndexed.keys()
    startKeyFrameList.sort()

    seqStartFrameList = list()
    seqEndFrameList = list()
    for i in seqData.values():
        for a in i.values():
            seqStartFrameList.append(a["sequenceStartFrame"])
            seqEndFrameList.append( a["sequenceEndFrame"] )

    seqStartFrameList.sort()
    seqEndFrameList.sort()

    timewarpNode = cmds.createNode("animCurveTT", n = "timewarp")

    cmds.setKeyframe(timewarpNode,  v=playbackStart, time = playbackStart )
    cmds.setKeyframe(timewarpNode,  v=playbackEnd, time = playbackEnd )
    cmds.keyTangent(timewarpNode, itt = "linear", ott = "linear" )

    for oriTime in startKeyFrameList:
        if oriTime != playbackStart:
            cmds.setKeyframe(timewarpNode, insert = True, v = oriTime, time = oriTime)

    cmds.connectAttr("%s.apply" %timewarpNode, "time1.timewarpIn.timewarpIn_Raw")
    cmds.setAttr("time1.enableTimewarp", 1)
    cmds.keyframe(timewarpNode, e=1, option="over", absolute=True,
                  timeChange=seqEndFrameList[-1],
                  t=(playbackEnd, playbackEnd))

    for oriTime in startKeyFrameList:
        cmds.keyframe(timewarpNode, e=1, option="over", absolute=True,
                      timeChange=seqDataFrameIndexed[oriTime][0],
                      t=(oriTime, oriTime))

    cmds.keyTangent(timewarpNode, itt = "linear", ott = "linear" )

    bakeObject(animObj, seqStartFrameList[0], seqEndFrameList[-1], bakeToNewLayer = True)

    cmds.setAttr("time1.enableTimewarp", 0)
    #mm.eval('string $layers[] = {"BaseAnimation", "BakeResults"};layerEditorMergeAnimLayer( $layers, 0 )')
    #cmds.delete(timewarpNode)

"""
ALLOBJ = bakeAllKeys()
SEQ_DATA = writeSeqData()
SDFI = writeseqDataFrameIndexed(SEQ_DATA)
applyTimewarp(SDFI, ALLOBJ)
"""
"""
def findHightLow(object, state = "low"):
    lowValue = None
    highValue = None

    for s in object:
        if state == "low":
            if lowValue == None:
                lowValue = s
            elif s <= lowValue:
                lowValue = s
            elif s > lowValue:
                lowValue = lowValue
        elif state == "high":
            if highValue == None:
                highValue = s
            elif s >= highValue:
                highValue = s
            elif s < highValue:
                highValue = highValue
"""