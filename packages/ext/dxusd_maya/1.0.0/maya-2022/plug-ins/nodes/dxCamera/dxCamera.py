#encoding=utf-8
'''
 @ author       : daeseok.chae@Dexter Studio
 @ last modify  : 2020.07.01
 @ change log
    - 2020.07.01 : first setup
'''

import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

class dxCameraMatrix( OpenMayaMPx.MPxTransformationMatrix ):
    def __init__( self ):
        OpenMayaMPx.MPxTransformationMatrix.__init__( self )

    def asMatrix( self ):
        matrix = OpenMayaMPx.MPxTransformationMatrix.asMatrix( self )
        tm = OpenMaya.MTransformationMatrix( matrix )
        return tm.asMatrix()


class dxCamera( OpenMayaMPx.MPxTransform ):
    camMode = OpenMaya.MObject()
    version = OpenMaya.MObject()
    action  = OpenMaya.MObject()
    fileName = OpenMaya.MObject()

    def __init__( self ):
        OpenMayaMPx.MPxTransform.__init__( self )


def dxCameraMatrix_Creator():
    return OpenMayaMPx.asMPxPtr(dxCameraMatrix())

def dxCamera_Creator():
    return OpenMayaMPx.asMPxPtr(dxCamera())

def dxCamera_Initializer():
    numericAttr = OpenMaya.MFnNumericAttribute()
    enumAttr    = OpenMaya.MFnEnumAttribute()
    typedAttr   = OpenMaya.MFnTypedAttribute()

    dxCamera.version = numericAttr.create(
            'version', 'version',
            OpenMaya.MFnNumericData.kInt, 0 )
    dxCamera.addAttribute(dxCamera.version)

    dxCamera.action = enumAttr.create('action', 'action', 1)
    enumAttr.addField( 'None', 0 )
    enumAttr.addField( 'Export', 1 )
    dxCamera.addAttribute(dxCamera.action)

    dxCamera.fileName = typedAttr.create(
            'fileName', 'fileName',
            OpenMaya.MFnData.kString,
            OpenMaya.MFnStringData().create('') )
    dxCamera.addAttribute(dxCamera.fileName)

