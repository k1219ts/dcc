#encoding=utf-8
#!/usr/bin/env python

"""
Dexter Pipe-line SceneGraph customNodes

LAST RELEASE:
- 2017.08.28 $8 : add jsonFile attribute. for ovrride attributes
- 2017.09.02 $9 : add curve render attributes. curveRoot, curveTip
                  add index of primitive. (primid)
- 2017.09.09 $2 : curve render bugfix. add dicehair, roundcurve
"""

import os, sys

import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx


class SceneGraphMatrix( OpenMayaMPx.MPxTransformationMatrix ):
    def __init__( self ):
        OpenMayaMPx.MPxTransformationMatrix.__init__( self )

    def asMatrix( self ):
        matrix = OpenMayaMPx.MPxTransformationMatrix.asMatrix( self )
        tm = OpenMaya.MTransformationMatrix( matrix )
        return tm.asMatrix()



class ComponentNode( OpenMayaMPx.MPxTransform ):
    """
    dxComponent transform node. for alembic archive
    """
    version		  = OpenMaya.MObject()
    action		  = OpenMaya.MObject()
    mode		  = OpenMaya.MObject()
    display		  = OpenMaya.MObject()
    abcFileName	  = OpenMaya.MObject()
    renderFile    = OpenMaya.MObject()
    worldFileName = OpenMaya.MObject()
    # alembic attributes
    dt				= OpenMaya.MObject()
    subdiv			= OpenMaya.MObject()
    objectid		= OpenMaya.MObject()
    groupid			= OpenMaya.MObject()
    primvar			= OpenMaya.MObject()
    txvariation		= OpenMaya.MObject()
    objectInstance	= OpenMaya.MObject()
    # visibility for RMan
    cameraVisibility		= OpenMaya.MObject()
    indirectVisibility		= OpenMaya.MObject()
    transmissionVisibility	= OpenMaya.MObject()

    jsonFile = OpenMaya.MObject()

    # curve render
    curveRoot = OpenMaya.MObject()
    curveTip  = OpenMaya.MObject()
    dicehair  = OpenMaya.MObject()
    roundcurve= OpenMaya.MObject()

    # index of primitive
    primid = OpenMaya.MObject()


    def __init__( self ):
        OpenMayaMPx.MPxTransform.__init__( self )

def ComponentMatrix_Creator():
    return OpenMayaMPx.asMPxPtr( SceneGraphMatrix() )

def ComponentNode_Creator():
    return OpenMayaMPx.asMPxPtr( ComponentNode() )

def ComponentNode_Initializer():
    numericAttr = OpenMaya.MFnNumericAttribute()
    enumAttr	= OpenMaya.MFnEnumAttribute()
    typedAttr	= OpenMaya.MFnTypedAttribute()

    ComponentNode.version = numericAttr.create(
            'version', 'version',
            OpenMaya.MFnNumericData.kInt, 0 )
    ComponentNode.addAttribute( ComponentNode.version )

    ComponentNode.action = enumAttr.create( 'action', 'action', 0 )
    enumAttr.addField( 'None', 0 )
    enumAttr.addField( 'Cache Export', 1 )
    enumAttr.addField( 'Layout Export', 2 )
    enumAttr.addField( 'Cache Copy', 3 )
    enumAttr.addField( 'RDB Export', 4 )
    ComponentNode.addAttribute( ComponentNode.action )

    ComponentNode.mode = enumAttr.create( 'mode', 'mode', 1 )
    enumAttr.addField( 'Mesh', 0 )
    enumAttr.addField( 'GPU', 1 )
    ComponentNode.addAttribute( ComponentNode.mode )

    ComponentNode.display = enumAttr.create( 'display', 'display', 1 )
    enumAttr.addField( 'BoundingBox', 0 )
    enumAttr.addField( 'Render', 1 )
    enumAttr.addField( 'Mid', 2 )
    enumAttr.addField( 'Low', 3 )
    enumAttr.addField( 'Simulation', 4 )
    ComponentNode.addAttribute( ComponentNode.display )

    ComponentNode.abcFileName = typedAttr.create(
            'abcFileName', 'abcFileName',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    ComponentNode.addAttribute( ComponentNode.abcFileName )

    ComponentNode.renderFile = typedAttr.create(
            'renderFile', 'renderFile',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    ComponentNode.addAttribute( ComponentNode.renderFile )

    ComponentNode.worldFileName = typedAttr.create(
            'worldFileName', 'worldFileName',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    ComponentNode.addAttribute( ComponentNode.worldFileName )

    # Alembic Attributes
    #
    # dt
    ComponentNode.dt = numericAttr.create(
            'dt', 'dt', OpenMaya.MFnNumericData.kFloat, 0.0 )
    ComponentNode.addAttribute( ComponentNode.dt )

    # subdiv
    ComponentNode.subdiv = numericAttr.create(
            'subdiv', 'subdiv', OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.subdiv )

    # objectid
    ComponentNode.objectid = numericAttr.create(
            'objectid', 'objectid', OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.objectid )

    # groupid
    ComponentNode.groupid = numericAttr.create(
            'groupid', 'groupid', OpenMaya.MFnNumericData.kBoolean, 0 )
    ComponentNode.addAttribute( ComponentNode.groupid )

    # primvar
    ComponentNode.primvar = numericAttr.create(
            'primvar', 'primvar', OpenMaya.MFnNumericData.kBoolean, 0 )
    ComponentNode.addAttribute( ComponentNode.primvar )

    # txvariation
    ComponentNode.txvariation = numericAttr.create(
            'txvariation', 'txvariation', OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.txvariation )

    # objectInstance
    ComponentNode.objectInstance = numericAttr.create(
            'objectInstance', 'objectInstance', OpenMaya.MFnNumericData.kBoolean, 0 )
    ComponentNode.addAttribute( ComponentNode.objectInstance )

    # visibility camera
    ComponentNode.cameraVisibility = numericAttr.create(
            'cameraVisibility', 'cameraVisibility',
            OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.cameraVisibility )

    # visibility indirect
    ComponentNode.indirectVisibility = numericAttr.create(
            'indirectVisibility', 'indirectVisibility',
            OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.indirectVisibility )

    # visibility transmission
    ComponentNode.transmissionVisibility = numericAttr.create(
            'transmissionVisibility', 'transmissionVisibility',
            OpenMaya.MFnNumericData.kBoolean, 1 )
    ComponentNode.addAttribute( ComponentNode.transmissionVisibility )

    # jsonFile
    ComponentNode.jsonFile = typedAttr.create(
        'jsonFile', 'jsonFile',
        OpenMaya.MFnData.kString,
        OpenMaya.MFnStringData().create('')
    )
    ComponentNode.addAttribute(ComponentNode.jsonFile)

    # curve render
    ComponentNode.curveRoot = numericAttr.create(
        'curveRoot', 'curveRoot', OpenMaya.MFnNumericData.kFloat, 1.0
    )
    ComponentNode.addAttribute(ComponentNode.curveRoot)
    ComponentNode.curveTip = numericAttr.create(
        'curveTip', 'curveTip', OpenMaya.MFnNumericData.kFloat, 1.0
    )
    ComponentNode.addAttribute(ComponentNode.curveTip)
    ComponentNode.dicehair = numericAttr.create(
        'dicehair', 'dicehair', OpenMaya.MFnNumericData.kBoolean, 0
    )
    ComponentNode.addAttribute(ComponentNode.dicehair)
    ComponentNode.roundcurve = numericAttr.create(
        'roundcurve', 'roundcurve', OpenMaya.MFnNumericData.kBoolean, 0
    )
    ComponentNode.addAttribute(ComponentNode.roundcurve)

    # index of primitive
    ComponentNode.primid = numericAttr.create(
        'primid', 'primid', OpenMaya.MFnNumericData.kInt, 0
    )
    ComponentNode.addAttribute(ComponentNode.primid)




#-------------------------------------------------------------------------------
#
#	dxAssembly
#
#-------------------------------------------------------------------------------
class AssemblyNode( OpenMayaMPx.MPxTransform ):
    version		= OpenMaya.MObject()
    action		= OpenMaya.MObject()
    mode		= OpenMaya.MObject()
    display		= OpenMaya.MObject()
    fileName	= OpenMaya.MObject()

    def __init__( self ):
        OpenMayaMPx.MPxTransform.__init__( self )

def AssemblyMatrix_Creator():
    return OpenMayaMPx.asMPxPtr( SceneGraphMatrix() )

def AssemblyNode_Creator():
    return OpenMayaMPx.asMPxPtr( AssemblyNode() )

def AssemblyNode_Initializer():
    numericAttr = OpenMaya.MFnNumericAttribute()
    enumAttr	= OpenMaya.MFnEnumAttribute()
    typedAttr	= OpenMaya.MFnTypedAttribute()

    AssemblyNode.version = numericAttr.create(
            'version', 'version',
            OpenMaya.MFnNumericData.kInt, 0 )
    AssemblyNode.addAttribute( AssemblyNode.version )

    AssemblyNode.action = enumAttr.create( 'action', 'action', 0 )
    enumAttr.addField( 'Reference', 0 )
    enumAttr.addField( 'Export', 1 )
    AssemblyNode.addAttribute( AssemblyNode.action )

    AssemblyNode.mode = enumAttr.create( 'mode', 'mode', 1 )
    enumAttr.addField( 'Mesh', 0 )
    enumAttr.addField( 'GPU', 1 )
    AssemblyNode.addAttribute( AssemblyNode.mode )

    AssemblyNode.display = enumAttr.create( 'display', 'display', 1 )
    enumAttr.addField( 'BoundingBox', 0 )
    enumAttr.addField( 'Render', 1 )
    enumAttr.addField( 'Mid', 2 )
    enumAttr.addField( 'Low', 3 )
    AssemblyNode.addAttribute( AssemblyNode.display )

    AssemblyNode.fileName = typedAttr.create(
            'fileName', 'fileName',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    AssemblyNode.addAttribute( AssemblyNode.fileName )
