import maya.cmds as cmds
import json
import os

"""
import Scripts_Dev.seqRetimeToKey as StoK
reload(StoK)

ls_sequencer = cmds.ls(type="sequencer")
keyDataFilename = "/show/kfyg/shot/POS/OPNpos_0010/ani/dev/data/RetimeKeyData.json"

if ls_sequencer:
	StoK.GetSeqData( ls_sequencer, keyDataFilename )
"""

def GetSeqData(ls_sequencer, keyDataFilename = None):
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

    WriteKeysFromDic(seqData, keyDataFilename)
    """
    maya_fileName = cmds.file(q=1, sn=1)
    rootDir = os.sep.join(maya_fileName.split(os.sep)[:-2])
    seq_data_fileName = os.sep.join([rootDir, "data/testSeq.json"])

    f = open(seq_data_fileName, 'w')
    json.dump(seqData, f, indent=4)
    f.close()
    """

# list start, end frames

def WriteKeysFromDic(seqData, keyDataFilename):

    keyDic = dict()
    seqOriStartFrameList = list()
    seqOriEndFrameList = list()
    seqStartFrameList = list()
    seqEndFrameList = list()
    seqScaleList = list()

    for i in seqData.values():
        for a in i.values():
            seqOriStartFrameList.append(a["startframe"])
            seqOriEndFrameList.append(a["endframe"])
            seqStartFrameList.append(a["sequenceStartFrame"])
            seqEndFrameList.append(a["sequenceEndFrame"])
            seqScaleList.append(a["scale"])

    seqOriStartFrameList.sort()
    seqOriEndFrameList.sort()
    seqStartFrameList.sort()
    seqEndFrameList.sort()
    seqScaleList.sort()

    keyDic["retime_Time"] = seqStartFrameList.append( seqEndFrameList[-1] )
    keyDic["retime_Value"] = seqOriStartFrameList.append( seqOriEndFrameList[-1] )
    keyDic["retime_Scale"] = 1/seqScaleList[-1]

    f = open(keyDataFilename, 'w')
    json.dump(keyDic, f, indent=4)
    f.close()


"""
import Scripts_Dev.seqRetimeToKey as StoK
reload(StoK)
KeyDataJson = "/show/kfyg/shot/POS/OPNpos_0010/ani/dev/data/RetimeKeyData.json"
StoK.SetTimewarpKey(KeyDataJson)
"""

def SetTimewarpKey(KeyDataJson):
    if not cmds.objExists("seq_timewarp"):
        timewarpNode = cmds.createNode("animCurveTT", n = "seq_timewarp")

    # keyData = json.loads(open(jsonFileName, 'r').read())
    with open(KeyDataJson, 'r') as f:
        keyData = json.loads(f.read())

    for i in range(len(keyData['retime_Time']))[:-2]:
        cmds.setKeyframe(timewarpNode, v=keyData["retime_Value"][i], time=keyData["retime_Time"][i])

    cmds.setKeyframe(timewarpNode, v=keyData["retime_Value"][-1] + 1, time=keyData["retime_Time"][-1] + 1)

    cmds.keyTangent(timewarpNode, itt="linear", ott="linear")

    cmds.connectAttr("%s.apply" % timewarpNode, "time1.timewarpIn.timewarpIn_Raw")
    cmds.setAttr("time1.enableTimewarp", 1)

    os.remove( KeyDataJson )


