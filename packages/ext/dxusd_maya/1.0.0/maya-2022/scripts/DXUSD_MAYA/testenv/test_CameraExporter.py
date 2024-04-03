#coding:utf-8
from __future__ import print_function
from pymel.all import *
import maya.cmds as cmds

try:
    cmds.loadPlugin('pxrUsd')
except:
    assert False, "not found pxrUsd plugin"

if not cmds.pluginInfo('pxrUsd', q=True, l=True):
    assert False, "Failed Load PxrUsd Plugin"

# Requires load DXUSD_Maya plugin
try:
    cmds.loadPlugin('DXUSD_Maya')
except:
    assert False, "not found DXUSD_Maya plugin"

import DXUSD_MAYA.Camera as Cam

# ALL EXPORT TEST
def allExport():
    cmds.file('/show/pipe/template/Camera/PKL_0290_main1_matchmove_v03_all.mb', o=True, f=True)
    # nodes = ['dxCamera1']
    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExport(nodes, version='v004')

# ONLY CAMERA EXPORT TEST
def onlyCameraExport():
    cmds.file('/show/pipe/template/Camera/PKL_0290_main1_matchmove_v03_cam.mb', o=True, f=True)
    # camNode = ['dxCamera1']
    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExport(nodes, version='v004')

def moreCamsExport():
    cmds.file('/show/pipe/template/Camera/PKL_0290_main1_matchmove_v03_moreCams.mb', o=True, f=True)
    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExport(nodes, version='v007')

def stereoCamsExport():
    cmds.file('/show/pipe/template/Camera/PKL_0290_main1_matchmove_v03_stereo.mb', o=True, f=True)
    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExport(nodes, version='v003')

def testExport():
    sceneFile = '/show/pipe/works/ANI/MNO/MNO_0350/scenes/MNO_0350_ani_v001.mb'
    cmds.file(sceneFile, o=True, f=True)

    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExport(nodes, show='pipe', version='v001', overwrite=True, process='both')

    # import DXUSD_MAYA.Rig as Rig
    # rigNode   = 'monorail:monorail_rig_GRP'
    # Rig.shotExport(node=rigNode, version='v001', process='both')

def assetCameraExport():
    cmds.file('/show/pipe/template/Camera/tree_model_v001_assetCamera.mb', o=True, f=True)
    nodes = cmds.ls(type='dxCamera')
    Cam.cameraExportAsset(nodes, version='v005')

# onlyCameraExport()
testExport()
# assetCameraExport()
