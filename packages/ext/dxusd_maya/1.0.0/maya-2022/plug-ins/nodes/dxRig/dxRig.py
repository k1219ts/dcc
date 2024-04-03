#encoding=utf-8
'''
 @ author       : daeseok.chae@Dexter Studio
 @ last modify  : 2021.01.22
 @ change log
    - 2021.01.22 : Add objects subframe support
    - 2020.07.01 : first setup
'''

import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import maya.utils as Utils


class dxRigMatrix( OpenMayaMPx.MPxTransformationMatrix ):
    def __init__( self ):
        OpenMayaMPx.MPxTransformationMatrix.__init__( self )

    def asMatrix( self ):
        matrix = OpenMayaMPx.MPxTransformationMatrix.asMatrix( self )
        tm = OpenMaya.MTransformationMatrix( matrix )
        return tm.asMatrix()

class dxRig(OpenMayaMPx.MPxTransform):
    assetName = OpenMaya.MObject()
    version = OpenMaya.MObject()
    action = OpenMaya.MObject()
    rigType = OpenMaya.MObject()
    rigBake = OpenMaya.MObject()
    rootCon = OpenMaya.MObject()
    editable = OpenMaya.MObject()
    renderMeshes = OpenMaya.MObject()
    midMeshes = OpenMaya.MObject()
    lowMeshes = OpenMaya.MObject()
    simMeshes = OpenMaya.MObject()
    skinJoints = OpenMaya.MObject()
    controllers = OpenMaya.MObject()
    controllersData = OpenMaya.MObject()

    # for usd
    variant   = OpenMaya.MObject()
    exportUVs = OpenMaya.MObject()
    # for usd subframe export
    exportStep = OpenMaya.MObject()
    frameStepSize = OpenMaya.MObject()
    fileStepSize  = OpenMaya.MObject()
    stepMeshes = OpenMaya.MObject()

    # old attributes
    controlers = OpenMaya.MObject()
    controlersData = OpenMaya.MObject()

    scriptString = OpenMaya.MObject()

    output = OpenMaya.MObject()

    def __init__( self ):
        OpenMayaMPx.MPxTransform.__init__( self )

    def compute( self, plug, data ):
        if plug == dxRig.output:
            thisNode = self.thisMObject()
            dagFn    = OpenMaya.MFnDagNode( thisNode )
            pathName = dagFn.fullPathName()
#            print(pathName, type(pathName))
            assetName_plug = OpenMaya.MPlug( thisNode, self.assetName )
            assetName_gv   = assetName_plug.asString()
            script_plug    = OpenMaya.MPlug( thisNode, self.scriptString )
            script_gv      = script_plug.asString()
            if OpenMaya.MGlobal.mayaState() == OpenMaya.MGlobal.kInteractive:
                if assetName_gv and script_gv:
                    for i in script_gv.split(';'):
                        if i:
                            Utils.executeDeferred('''import %s; %s("%s")''' % (i.split('.')[0], i, pathName))
#                            exec( '''import %s; %s("%s")''' % (i.split('.')[0], i, pathName) )


def dxRigMatrix_Creator():
    return OpenMayaMPx.asMPxPtr(dxRigMatrix())

def dxRig_Creator():
    return OpenMayaMPx.asMPxPtr(dxRig())

def dxRig_Initializer():
    numericAttr = OpenMaya.MFnNumericAttribute()
    enumAttr    = OpenMaya.MFnEnumAttribute()
    typedAttr   = OpenMaya.MFnTypedAttribute()

    dxRig.assetName = typedAttr.create(
            'assetName', 'assetName',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create(''))
    dxRig.addAttribute(dxRig.assetName)

    dxRig.version = numericAttr.create(
            'version', 'version',
            OpenMaya.MFnNumericData.kInt, 0 )
    dxRig.addAttribute(dxRig.version)

    dxRig.action = enumAttr.create('action', 'action', 1)
    enumAttr.addField('None', 0)
    enumAttr.addField('Export', 1)
    dxRig.addAttribute(dxRig.action)

    dxRig.rigType = enumAttr.create('rigType', 'rigType')
    enumAttr.addField('Biped', 0)
    enumAttr.addField('Quad', 1)
    enumAttr.addField('Prop',  2)
    enumAttr.addField('Vehicle', 3)
    enumAttr.addField('Etc', 4)
    dxRig.addAttribute( dxRig.rigType )

    dxRig.rigBake = numericAttr.create('rigBake', 'rigBake', OpenMaya.MFnNumericData.kBoolean, 1)
    dxRig.addAttribute(dxRig.rigBake)

    dxRig.rootCon = typedAttr.create('rootCon', 'rootCon',
                                             OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxRig.addAttribute(dxRig.rootCon)

    dxRig.editable = numericAttr.create(
            'editable', 'editable',
            OpenMaya.MFnNumericData.kBoolean, 1 )
    dxRig.addAttribute(dxRig.editable)

    dxRig.renderMeshes = typedAttr.create(
            'renderMeshes', 'renderMeshes',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.renderMeshes)

    dxRig.midMeshes = typedAttr.create(
            'midMeshes', 'midMeshes',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.midMeshes)

    dxRig.lowMeshes = typedAttr.create(
            'lowMeshes', 'lowMeshes',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.lowMeshes)

    dxRig.simMeshes = typedAttr.create(
            'simMeshes', 'simMeshes',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.simMeshes)

    dxRig.skinJoints = typedAttr.create(
            'skinJoints', 'skinJoints',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.skinJoints)

    dxRig.controllers = typedAttr.create(
            'controllers', 'controllers',
            OpenMaya.MFnData.kStringArray,
            OpenMaya.MFnStringArrayData().create([]) )
    dxRig.addAttribute(dxRig.controllers)

    dxRig.controllersData = typedAttr.create(
            'controllersData', 'controllersData',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    dxRig.addAttribute(dxRig.controllersData)

    dxRig.variant = typedAttr.create('variant', 'variant',
                                     OpenMaya.MFnData.kString,
                                     OpenMaya.MFnStringData().create(''))
    dxRig.addAttribute(dxRig.variant)

    dxRig.exportUVs = numericAttr.create('exportUVs', 'exportUVs',
                                         OpenMaya.MFnNumericData.kBoolean, 0)
    dxRig.addAttribute(dxRig.exportUVs)

    dxRig.exportStep = numericAttr.create('exportStep', 'exportStep', OpenMaya.MFnNumericData.kBoolean, 0)
    dxRig.addAttribute(dxRig.exportStep)

    dxRig.frameStepSize = numericAttr.create('frameStepSize', 'frameStepSize', OpenMaya.MFnNumericData.kFloat, 0.1)
    dxRig.addAttribute(dxRig.frameStepSize)

    dxRig.fileStepSize = numericAttr.create('fileStepSize', 'fileStepSize', OpenMaya.MFnNumericData.kInt, 10)
    dxRig.addAttribute(dxRig.fileStepSize)

    dxRig.stepMeshes = typedAttr.create('stepMeshes', 'stepMeshes', OpenMaya.MFnData.kStringArray, OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.stepMeshes)


    # old attributes
    dxRig.controlers = typedAttr.create('controlers', 'controlers', OpenMaya.MFnData.kStringArray, OpenMaya.MFnStringArrayData().create([]))
    dxRig.addAttribute(dxRig.controlers)
    dxRig.controlersData = typedAttr.create('controlersData', 'controlersData', OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxRig.addAttribute(dxRig.controlersData)

    dxRig.scriptString = typedAttr.create(
            'scriptString', 'scriptString',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    dxRig.addAttribute(dxRig.scriptString)

    # output
    dxRig.output = numericAttr.create(
            'output', 'output',
            OpenMaya.MFnNumericData.kFloat )
    numericAttr.setHidden( True )
    numericAttr.setStorable( True )
    numericAttr.setWritable( True )
    dxRig.addAttribute(dxRig.output)

    # affects
    dxRig.attributeAffects(dxRig.assetName, dxRig.output)
