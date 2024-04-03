import sys

import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx


import mfdscripts.mfdNodes as mfdNodes
kAgentGroupName = 'AgentGroup'
kAgentGroupID   = OpenMaya.MTypeId(0x10190001)
kAgentGroupMID  = OpenMaya.MTypeId(0x10190002)


# Miarmy For Dexter
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject, 'Miarmy For Dexter', '1.0', 'Any')
    try:
        mplugin.registerTransform(
            kAgentGroupName, kAgentGroupID,
            mfdNodes.dAgentGroupNode_Creator,
            mfdNodes.dAgentGroupNode_Initializer,
            mfdNodes.dAgentGroupMatrix_Creator,
            kAgentGroupMID
        )
    except:
        sys.stderr.write('Failed to register MiarmyForDexter Plugin\n')
        raise
    mplugin.registerUI(createMenu, deleteMenu)


def uninitializePlugin(mobject):
    mplugin = MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(kAgentGroupID)
    except:
        sys.stderr.write('Failed to unregister MiarmyForDexter Plugin\n')


def createMenu():
    OpenMaya.MGlobal.sourceFile('mfdInitUI.mel')
    OpenMaya.MGlobal.executeCommand('MiarmyForDexter_CreateUI')

def deleteMenu():
    OpenMaya.MGlobal.executeCommand('MiarmyForDexter_DeleteUI')
