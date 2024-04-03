# -*- coding: utf-8 -*-

import maya.cmds as cmds
import maya.mel as mel
import os
import json
import logging
import GH_sceneCleanup.modules as bariquant;reload(bariquant)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROJECT_CAM_INFO = "/show/{project}/asset/global/matchmove/camera/cameraInfo.json"

def createMetadataString(mayaFilePath, artistName):
    """

    :param mayaFilePath: A string of maya scene file path
    :param artistName: A string of artist name
    :return: A string of mov metadata
    """
    movMetadata = '\'{\"mayaFilePath\":\"%s\",\"artist\":\"%s\"}\'' % (mayaFilePath.strip(), artistName.strip())
    logger.debug(u'[Option] Metadata : {0}\n'.format(movMetadata))
    return movMetadata


def cleanupSequencer():
    unusedSequencer = bariquant.cleanupSequencer(parent=None)
    logger.debug(u'Unused sequencer <{0}> deleted\n'.format(unusedSequencer))

def getFrameRange(type, sequencer=False, cleanupSequencer=False):
    """

    :param type: 'start' or 'end'
    :param sequencer:
    :return: A float of playback time
    """
    if sequencer:
        if cleanupSequencer:
            cleanupSequencer()
        sequencer_list = cmds.ls(type='sequencer')
        if type == 'start':
            min = cmds.getAttr(sequencer_list[0] + ".minFrame")
            return min
        elif type == 'end':
            max = cmds.getAttr(sequencer_list[0] + '.maxFrame')
            return max
    if type == 'start':
        min = cmds.playbackOptions(q=True, min=True)
        return min
    elif type == 'end':
        max = cmds.playbackOptions(q=True, max=True)
        return max


def getPath(type='file', currentScene=True, filepath=None):
    """

    :param type:
    :param filepath: Get file name from path, if currentScene is False
    :return: A string of file name
    """
    # 현재 씬에서 파일명 또는 경로명을 받아올때.
    if currentScene and filepath == None:
        fullPath = cmds.file(q=True, sn=True)
        fileName = os.path.basename(fullPath)
        if type == 'file':
            # 저장이 안된 씬일 경우 Untitled 를 돌려줌.
            if not fileName:
                return 'Untitled'
            # 씬 이름이 있을경우 확장자 명을 뺀 파일명을 돌려줌
            fileNameOnly = os.path.splitext(fileName)[0]
            return fileNameOnly
        elif type == 'folder':
            if not fullPath:
                return "Select file path"
            pathSplit = fullPath.split(os.sep)
            # 경로에 show 나 shot 이 없을경우 씬경로를 돌려줌.
            if not 'show' in pathSplit and not 'shot' in pathSplit:
                folder = os.sep.join(pathSplit[:-1])
                return folder
            elif 'pub' in pathSplit and 'matchmove' in pathSplit:
                folder = os.sep.join(pathSplit[:-3] + ['preview'])
                return folder
            else:
                folder = os.sep.join(pathSplit[:-2] + ['preview'])
                return folder
    # 임의의 입력된 경로에서 파일명을 받아올때.
    elif not currentScene and filepath:
        baseName = os.path.basename(filepath)
        fileName = os.path.splitext(baseName)[0]
        return fileName


def getProjectInfo(project):
    """Get Camera Information From Json File

    :param project: Project name
    :return: A dictionary of camera information
    """
    infoFile = PROJECT_CAM_INFO.format(project=project)
    if not os.path.exists(infoFile):
        return  False
    with open(infoFile, 'r') as f:
        camInfo = json.loads(f.read())
    logger.debug(u'Project Camera Information Loaded from <{}>\n'.format(infoFile))
    return camInfo


def mayaBackgroundEdit(rgb, offimp):
    """

    :param rgb: A list of RGB color
    :param offimp: If true, change image plane display mode
    """
    state = 0
    if cmds.displayRGBColor('background', q=True) != rgb:
        mel.eval("cycleBackgroundColor;")
        mel.eval("cycleBackgroundColor;")
        cmds.displayRGBColor("background", rgb[0], rgb[1], rgb[2])
    else:
        mel.eval("cycleBackgroundColor;")
        mel.eval("cycleBackgroundColor;")
        cmds.displayRGBColor("background", 0.631, 0.631, 0.631)
        state = 3
    if offimp:
        cameras = cmds.ls(type='camera')
        for camera in cameras:
            camIP = cmds.listConnections(camera + ".imagePlane", d=True)
            if camIP:
                cmds.setAttr(camIP[0] + ".displayMode", state)
