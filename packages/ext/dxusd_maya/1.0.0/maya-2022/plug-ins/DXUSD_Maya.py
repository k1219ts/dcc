#encoding=utf-8

#-------------------------------------------------------------------------------
#
#   Dexter OpenSource Plugin
#
#-------------------------------------------------------------------------------

import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds
import maya.mel as mel

# dxs_xBlock
from nodes.dxBlock import dxBlock
kBlockName = 'dxBlock'
kBlockID   = OpenMaya.MTypeId(0x10170010)
kBlockMID  = OpenMaya.MTypeId(0x10170011)

# dxRig
from nodes.dxRig import dxRig
kRigRootName    = "dxRig"
kRigRootID      = OpenMaya.MTypeId(0x79000)
kRigRootMID     = OpenMaya.MTypeId(0x79001)

# dxs_CamRoot
from nodes.dxCamera import dxCamera
kCamRootName    = "dxCamera"
kCamRootID      = OpenMaya.MTypeId(0x79002)
kCamRootMID     = OpenMaya.MTypeId(0x79003)

# dxTimeNode
from nodes.dxTime import dxTime
kTimeOffsetName = 'dxTimeOffset'
kTimeOffsetID   = OpenMaya.MTypeId(0x10170020)

# dxs menu
kPluginCmdName = 'DXUSD_Maya'
kVenderName    = 'DexterStudios OpenSource Project'
kPluginVersion = '1.0.0'

import nodes.DXUSD_CallBack as dxCallBack

# Command
class scriptedCommand(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        print('Welcome to DexterStudios OpenSource Project!')

# Creator
def cmdCreator():
    return OpenMayaMPx.asMPxPtr(scriptedCommand())

# Initialize the script plug-in
def initializePlugin(mobject):
    # initialize
    if not cmds.pluginInfo('pxrUsd', q=True, l=True):
        cmds.loadPlugin('pxrUsd')
        if not cmds.pluginInfo('pxrUsd', q=True, l=True):
            assert False, "Failed Load PxrUsd Plugin"

    if not cmds.pluginInfo('pxrUsdPreviewSurface', q=True, l=True):
        cmds.loadPlugin('pxrUsdPreviewSurface')
        if not cmds.pluginInfo('pxrUsdPreviewSurface', q=True, l=True):
            assert False, "Failed Load pxrUsdPreviewSurface Plugin"

    if not cmds.pluginInfo('AL_USDMayaPxrTranslators', q=True, l=True):
        cmds.loadPlugin('AL_USDMayaPxrTranslators')
        if not cmds.pluginInfo('AL_USDMayaPxrTranslators', q=True, l=True):
            assert False, "Failed Load pxrUsdTranslators Plugin"

    if not cmds.pluginInfo('backstageMenu', q=True, l=True):
        cmds.loadPlugin('backstageMenu')
        if not cmds.pluginInfo('backstageMenu', q=True, l=True):
            assert False, "Failed Load backstageMenu Plugin"

    mplugin = OpenMayaMPx.MFnPlugin(mobject, kVenderName, kPluginVersion, 'Any')

    try:
        mplugin.registerCommand(kPluginCmdName, cmdCreator)

        # dxBlock
        mplugin.registerTransform(kBlockName, kBlockID, dxBlock.dxBlock,
                                  dxBlock.dxBlock_Initializer, dxBlock.dxBlockMatrix_Creator,
                                  kBlockMID)

        # dxRig
        mplugin.registerTransform(kRigRootName, kRigRootID, dxRig.dxRig,
                                  dxRig.dxRig_Initializer, dxRig.dxRigMatrix_Creator,
                                  kRigRootMID)

        # dxs_CamRoot
        mplugin.registerTransform(kCamRootName, kCamRootID, dxCamera.dxCamera_Creator,
                                  dxCamera.dxCamera_Initializer,
                                  dxCamera.dxCameraMatrix_Creator,
                                  kCamRootMID)

        # dxTimeOffset
        mplugin.registerNode(kTimeOffsetName, kTimeOffsetID,
                             dxTime.dxTimeOffset_Creator,
                             dxTime.dxTimeOffset_Initialize)

    except:
        sys.stderr.write('Failed to register command: %s\n' % kPluginCmdName)
        raise

    # Test Case
    node = cmds.createNode("dxBlock")

    if not node and cmds.nodeType(node) != 'dxBlock':
        assert False, "Failed Create dxBlock"
    else:
        cmds.delete(node)

    node = cmds.createNode("dxRig")

    if not node and cmds.nodeType(node) != 'dxRig':
        assert False, "Failed Create dxRig"
    else:
        cmds.delete(node)

    node = cmds.createNode("dxCamera")

    if not node and cmds.nodeType(node) != 'dxCamera':
        assert False, "Failed Create dxCamera"
    else:
        cmds.delete(node)

    node = cmds.createNode("dxTimeOffset")

    if not node and cmds.nodeType(node) != 'dxTimeOffset':
        assert False, "Failed Create dxTimeOffset"
    else:
        cmds.delete(node)

    # Look Procedual Error Remove
    mel.eval('outlinerEditor -edit -selectCommand "" "outlinerPanel1";')

    if OpenMaya.MGlobal.mayaState() == OpenMaya.MGlobal.kInteractive:
        OpenMaya.MSceneMessage.addCallback(OpenMaya.MSceneMessage.kAfterOpen, dxCallBack.openCallback)
        OpenMaya.MSceneMessage.addCallback(OpenMaya.MSceneMessage.kAfterSave, dxCallBack.saveCallback)


    # Menu
    mplugin.registerUI(createDxsMenu, deleteDxsMenu)

    print("Success load DXUSD_Maya Plugin")


def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(kPluginCmdName)
        mplugin.deregisterNode(kBlockID)
        mplugin.deregisterNode(kRigRootID)
        mplugin.deregisterNode(kCamRootID)
    except:
        sys.stderr.write('Failed to unregister command: %s\n' % kPluginCmdName)


def createDxsMenu():
    OpenMaya.MGlobal.sourceFile('dxsInitUI.mel')
    OpenMaya.MGlobal.executeCommand('dxsMenu_Create')

def deleteDxsMenu():
    OpenMaya.MGlobal.executeCommand('dxsMenu_Delete')
