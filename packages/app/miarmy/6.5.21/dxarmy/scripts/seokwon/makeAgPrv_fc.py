# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
import sys
import json

scrPath = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Scripts/Other/'
crwPath = '/opt/Basefount/Miarmy/maya/scripts/'
if not scrPath in sys.path:
    sys.path.append(scrPath)
if not crwPath in sys.path:
    sys.path.append(crwPath)
import abcLocatorPointsExportN
reload(abcLocatorPointsExportN)
import McdPlacementFunctions
reload(McdPlacementFunctions)
import McdMeshDriveSetup
reload(McdMeshDriveSetup)

def timelineCheck():
    minT = cmds.playbackOptions(q=1, minTime=1)
    maxT = cmds.playbackOptions(q=1, maxTime=1)
    return minT, maxT

def searchCam():
    camList = cmds.ls(type="camera")
    renCam = [str(cmds.listRelatives(str(i), p=True)[0]) for i in camList if cmds.getAttr(str(i) + ".renderable") == True]
    if len(renCam) == 0:
        cmds.error("Renderable Camera Error")
    elif len(renCam) == 1:
        pass
    else:
        cmds.error("Too many cameras")
    return renCam[0]

def dtypeCheck(dtype, minT):
    if dtype == "Crowd":
        minT = int(cmds.getAttr("McdBrain1.startTime"))
    else:
        pass
    return minT

def exportAlcDiv(mayaFile, mayaEx, minT, maxT, dtype, frs):
    cmds.file(mayaFile, f=True, options="v=0;", iv=True, o=True)
    minT = dtypeCheck(dtype,minT)
    cmds.playbackOptions(minTime=minT)
    cmds.playbackOptions(maxTime=maxT)
    # 군중 Re-Placement
    if dtype == "Crowd":
        if cmds.ls("MDGGrp_*"):
            McdMeshDriveSetup.McdMeshDrive2Clear()
        elif cmds.ls("McdAgent*"):
            McdPlacementFunctions.dePlacementAgent()
        else:
            pass
        McdPlacementFunctions.placementAgent()
    else:
        pass
    cmds.currentTime(minT)
    cam = searchCam()

    # 넘어온 데이터들 : [ 씬파일경로, 펍파일경로, 듀레이션, 펍타입, 뽑을 프레임 목록 ]

def exportAlc(mayaFile, mayaEx, minT, maxT, dtype):
    cmds.file(mayaFile, f=True, options="v=0;", iv=True, o=True)
    minT = dtypeCheck(dtype, minT)
    cmds.playbackOptions(minTime=minT)
    cmds.playbackOptions(maxTime=maxT)
    # 군중 Re-Placement
    if dtype == "Crowd":
        if cmds.ls("MDGGrp_*"):
            McdMeshDriveSetup.McdMeshDrive2Clear()
        elif cmds.ls("McdAgent*"):
            McdPlacementFunctions.dePlacementAgent()
        else:
            pass
        McdPlacementFunctions.placementAgent()
    else:
        pass
    cmds.currentTime(minT)
    cam = searchCam()

if __name__ == '__main__':
    mayaFile = sys.argv[1]
    mayaEx = sys.argv[2]
    from pymel.all import *
    if len(sys.argv) == 7:
        minTime = sys.argv[3]
        maxTime = sys.argv[4]
        dtype = sys.argv[5]
        frs = sys.argv[6]
        exportAlcDiv(mayaFile, mayaEx, minTime, maxTime, dtype, frs)
    else:
        minTime = sys.argv[3]
        maxTime = sys.argv[4]
        dtype = sys.argv[5]
        exportAlc(mayaFile, mayaEx, minTime, maxTime, dtype)
    os._exit(0)