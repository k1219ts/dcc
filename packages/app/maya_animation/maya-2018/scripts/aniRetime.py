# -*- coding: utf-8 -*-

import os
import logging
import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import maya.OpenMayaAnim as OpenMayaAnim
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def getPath(exportFile):
    path = dict()
    path['sceneDir'] = os.sep.join(exportFile.split(os.sep)[:-1])
    path['sceneName'] = os.path.splitext(exportFile.split(os.sep)[-1])[0]
    path['pubDataDir'] = os.sep.join(exportFile.split(os.sep)[:-2] + ["data"])
    path['retimeDir'] = os.sep.join([path['pubDataDir'], 'retime'])
    path['nukeScript'] = os.path.join(path['retimeDir'], path['sceneName'] + "_imageplane.nk")
    path['timewarpNodeMB'] = os.path.join(path['retimeDir'], path['sceneName'] + "_timewarp.mb")
    return path

def getImageplanes(cameras):
    """

    :param cameras: List of cameras
    :return: A list of imageplanes
    """
    impList = list()
    for que in cameras:
        if cmds.objectType(que) == "transform":
            shapeName = cmds.listRelatives(que, s=1)[0]
        else:
            shapeName = que
        logger.debug(u'Get imageplane of <{0}>'.format(shapeName))
        objType = cmds.objectType(shapeName)
        if objType == "camera":
            imp = cmds.listConnections(shapeName, type="imagePlane")
            if imp:
                impShape = cmds.listRelatives(imp, s=1)[0]
                impList.append(impShape)
        else:
            impList.append(cmds.listRelatives(que, s=1)[0])
    return impList

def getSequencerData( sequencers ):
    """Maya sequencer information to dictionary data

    :param sequencers: Mayq 'sequencer' list
    :return: A dictionary of sequencer data
    """
    seqData = dict()
    for ls_sequencer_i in sequencers:
        seqData[ls_sequencer_i] = dict()
        ls_seqShot = cmds.listConnections( '%s.shots' % ls_sequencer_i )
        if ls_seqShot:
            for ls_seqShot_i in ls_seqShot:
                seqData[ls_sequencer_i][ls_seqShot_i] = dict()
                ori_startFrame = cmds.getAttr( ls_seqShot_i + ".startFrame" )
                ori_endFrame = cmds.getAttr( ls_seqShot_i + ".endFrame" )
                seqShot_startFrame = cmds.getAttr( ls_seqShot_i + ".sequenceStartFrame" )
                seqShot_endFrame = cmds.getAttr( ls_seqShot_i + ".sequenceEndFrame" )
                seqShot_scale = cmds.getAttr( ls_seqShot_i + ".scale" )

                seqData[ls_sequencer_i][ls_seqShot_i]["startframe"] = ori_startFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["endframe"] = ori_endFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["sequenceStartFrame"] = seqShot_startFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["sequenceEndFrame"] = seqShot_endFrame
                seqData[ls_sequencer_i][ls_seqShot_i]["scale"] = seqShot_scale
        else:
            raise RuntimeError( "No 'shots' in 'sequencer'" )
    return seqData

def writeKeysFromDic( seqData ):
    """Sequencer data to keyframe data

    :param seqData: Maya 'sequencer' list
    :return: A dictionary of key data for timewarp
    """
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

    seqStartFrameList.append(seqEndFrameList[-1])
    seqOriStartFrameList.append(seqOriEndFrameList[-1])

    keyDic["retime_Time"] = seqStartFrameList
    keyDic["retime_Value"] = seqOriStartFrameList
    keyDic["retime_Scale"] = 1/seqScaleList[-1]

    return keyDic

def sequencerToTimewarp( sequencers, isRender=True, fitTimeLine=True ):
    """Add Maya scenetimewarp to scene

    :param sequencers: Maya 'sequencer' list
    :param isRender: If True, set timewarp enabled
    :return: A string of Maya sceneTimeWarp node name, A float of retime scale
    """
    sequencerData = getSequencerData( sequencers )
    keyData = writeKeysFromDic( sequencerData )

    # Delete timewarp node if exists
    if cmds.objExists( "seq_timewarp*" ):
        cmds.delete( "seq_timewarp*" )
    timewarpNode = cmds.createNode( "animCurveTT", n="seq_timewarp" )

    for i in range( len( keyData['retime_Time'] ) )[:-1]:
        cmds.setKeyframe( timewarpNode, v=keyData["retime_Value"][i],
                          time=keyData["retime_Time"][i] )
    cmds.setKeyframe( timewarpNode, v=keyData["retime_Value"][-1] + 1,
                      time=keyData["retime_Time"][-1] + 1 )
    timewarpNodeKeyList = cmds.keyframe( timewarpNode, q=1 )
    cmds.keyTangent( timewarpNode, itt="linear", ott="linear" )
    cmds.keyTangent( timewarpNode, e=1,
                     t=(timewarpNodeKeyList[0], timewarpNodeKeyList[0]),
                     itt="spline", ott="spline" )
    cmds.keyTangent( timewarpNode, e=1,
                     t=(timewarpNodeKeyList[-1], timewarpNodeKeyList[-1]),
                     itt="spline", ott="spline" )
    cmds.selectKey( timewarpNode, add=True, k=True )
    cmds.setInfinity( pri="linear", poi="linear" )
    cmds.connectAttr( "%s.apply" % timewarpNode, "time1.timewarpIn.timewarpIn_Raw" )

    # Set enable/disable timewarp
    if isRender:
        cmds.setAttr( "time1.enableTimewarp", 1 )
    else:
        cmds.setAttr("time1.enableTimewarp", 0)
    if fitTimeLine:
        cmds.playbackOptions( e=1, min=keyData["retime_Time"][0],
                              max=keyData["retime_Time"][-1] )
    return timewarpNode, keyData['retime_Scale']


def createReverseTimewarp(timewarp):
    animcrvNode = OpenMaya.MSelectionList()
    OpenMaya.MGlobal.getSelectionListByName(timewarp, animcrvNode)
    mobject = OpenMaya.MObject()
    animcrvNode.getDependNode(0, mobject)
    new_animcrvfn = OpenMayaAnim.MFnAnimCurve()
    new_animcrvfn.setObject(mobject)

    keySet = set()

    for index in range(new_animcrvfn.numKeys()):
        keyTime = new_animcrvfn.time(index)
        keySet.add(keyTime.value())

    keyList = list(keySet)
    keyList.sort()

    startFrame = keyList[0]
    endFrame = keyList[-1]

    keyDic = dict()

    for i in numpy.arange(startFrame, endFrame + 1):
        value = cmds.getAttr(timewarp + ".output", t=i)
        keyDic[i] = value
        # value = new_animcrvfn.evaluate(OpenMaya.MTime(i))
        # print i, value, value / 5880000.0

    reverseTWNode = cmds.createNode('animCurveTT', n='reverse_timewarp')

    for i in keyDic:
        cmds.setKeyframe(reverseTWNode, t=keyDic[i], value=i)

    return reverseTWNode


class ExportAniRetime():
    @staticmethod
    def createNukeScriptString(cameras, startFrame, endFrame):
        """

        :type cameras: list
        :type startFrame: int
        :type endFrame: int
        :rtype: str
        """
        imageplanes = getImageplanes(cameras)
        logger.debug(u'Imageplanes : {}'.format(imageplanes))

        nukeString = str()
        for ip in imageplanes:
            frameoffset = cmds.getAttr(str(ip) + ".frameOffset") * -1
            if frameoffset:
                nukeString = "TimeOffset {\n"
                nukeString += " time_offset {0}\n".format(str(frameoffset))
                nukeString += ' time ""\n'
                nukeString += ' name {}_timeOffset\n'.format(ip)
                nukeString += '}\n\n'
            nukeString += "TimeWarp {\n"
            nukeString += " lookup {{curve x{0}".format(startFrame)
            for frame in range(startFrame, endFrame + 1):
                retimedFrame = cmds.getAttr(str(ip) + ".frameExtension", time=frame)
                nukeString += " {0}".format(retimedFrame)
            nukeString += "}}\n"
            nukeString += " filter none\n"
            nukeString += " name {0}_timeWarp\n".format(ip)
            nukeString += "}\n\n"
        return nukeString

    @staticmethod
    def nukeExport(fileName, nukeString ):
        """Export imageplane's frame extention as Nuke script

        :param fileName : maya filename
        :param nukeString : nuke script text
        """
        paths = getPath(fileName)
        with open(paths['nukeScript'], "w") as f:
            logger.debug(u'Write nuke script < {} >'.format(paths['nukeScript']))
            f.write(nukeString)
            f.close()

    @staticmethod
    def mayaPreprocess(fileName, cameras, isRender):
        """

        :type fileName: str
        :type cameras: list
        :type isRender: bool
        :return: A float of frame step
        """
        logger.debug(u'Starting sequencer retime setup...')
        logger.debug(u'# To Render = {0}'.format(isRender))
        paths = getPath(fileName)
        sequencer = cmds.ls(type='sequencer')

        step = 1.0
        isRender = True

        if not sequencer: return step

        timewarpNode, step = sequencerToTimewarp(sequencers=sequencer,
                                                 isRender=isRender,
                                                 fitTimeLine=isRender)
        cmds.select(timewarpNode, add=False)

        if not os.path.exists(paths['retimeDir']):
            os.mkdir(paths['retimeDir'])

        cmds.file(paths['timewarpNodeMB'],
                  force=True,
                  options="v=0;",
                  typ="mayaBinary",
                  pr=True,
                  es=True)

        if isRender:
            step = 1.0
        else:
            cmds.delete(timewarpNode)
            startFrame = int(cmds.playbackOptions(q=True, min=True))
            endFrame = int(cmds.playbackOptions(q=True, max=True))
            nukeString = ExportAniRetime.createNukeScriptString(cameras, startFrame, endFrame)
            ExportAniRetime.nukeExport(fileName, nukeString)
        cmds.file(save=True)

        return step