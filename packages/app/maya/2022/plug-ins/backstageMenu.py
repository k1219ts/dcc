#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#    Dexter CG Supervisor
#
#        sanghun.kim        rman.td@gmail.com
#
#   Pipe-line Plug-ins
#
#    rman.td 2017.01.25 $2
#-------------------------------------------------------------------------------

import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.mel as mel

# SceneGraph
import pysource.scenegraphNodes as sgNodes
kComponentName = 'dxComponent'
kComponentID   = OpenMaya.MTypeId( 0x10170001 )
kComponentMID  = OpenMaya.MTypeId( 0x10170002 )
kAssemblyName  = 'dxAssembly'
kAssemblyID    = OpenMaya.MTypeId( 0x10170003 )
kAssemblyMID   = OpenMaya.MTypeId( 0x10170004 )

#-----------------------------------------------------

# backstage menu
kPluginCmdName = "backstageMenu"
kVendorName = 'Dexter Digital'
kPluginVersion = '3.0'

# Command
class scriptedCommand(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt( self, argList ):
        print('Welcome to Backstage!')

# Creator
def cmdCreator():
    return OpenMayaMPx.asMPxPtr( scriptedCommand() )

# Initialize the script plug-in
def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin( mobject, kVendorName, kPluginVersion, 'Any' )
    try:
        mplugin.registerCommand( kPluginCmdName, cmdCreator )
        # dxComponent
        mplugin.registerTransform(
            kComponentName, kComponentID,
            sgNodes.ComponentNode_Creator, sgNodes.ComponentNode_Initializer, sgNodes.ComponentMatrix_Creator,
            kComponentMID
        )
        # dxAssembly
        mplugin.registerTransform(
            kAssemblyName, kAssemblyID,
            sgNodes.AssemblyNode_Creator, sgNodes.AssemblyNode_Initializer, sgNodes.AssemblyMatrix_Creator,
            kAssemblyMID
        )

        # fix outlinear bug (cannot find "look")
        mel.eval('outlinerEditor -edit -selectCommand "" "outlinerPanel1";')

    except:
        sys.stderr.write("Failed to register command: %s\n" % kPluginCmdName)
        raise

    if OpenMaya.MGlobal.mayaState() == OpenMaya.MGlobal.kInteractive:
        OpenMaya.MGlobal.sourceFile('sgUI.mel')
    mplugin.registerUI(createBackstageMenu, deleteBackstageMenu)

# Uninitialize the script plug-in
def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(kPluginCmdName)
        mplugin.deregisterNode(kComponentID)
        mplugin.deregisterNode(kAssemblyID)
    except:
        sys.stderr.write("Failed to unregister command: %s\n" % kPluginCmdName)

def createBackstageMenu():
    OpenMaya.MGlobal.sourceFile('backstageInitUI.mel')
    OpenMaya.MGlobal.executeCommand('backstage_CreateUI')

def deleteBackstageMenu():
    OpenMaya.MGlobal.executeCommand('backstage_DeleteUI')
