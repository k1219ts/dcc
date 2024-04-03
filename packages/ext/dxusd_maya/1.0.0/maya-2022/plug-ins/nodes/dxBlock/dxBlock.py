#encoding=utf-8
'''
 @ author       : daeseok.chae@Dexter Studio
 @ last modify  : 2020.07.01
 @ change log
    - 2020.07.01 : first setup
'''
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

class dxBlockMatrix(OpenMayaMPx.MPxTransformationMatrix):
    def __init__(self):
        OpenMayaMPx.MPxTransformationMatrix.__init__(self)

    def asMatrix(self):
        matrix = OpenMayaMPx.MPxTransformationMatrix.asMatrix(self)
        tm = OpenMaya.MTransformationMatrix(matrix)
        return tm.asMatrix()


class dxBlock(OpenMayaMPx.MPxTransform):
    m_type   = OpenMaya.MObject()
    m_action = OpenMaya.MObject()
    m_rootPrimPath = OpenMaya.MObject()
    m_nsLayer   = OpenMaya.MObject()
    m_imporFile = OpenMaya.MObject()
    m_mergeFile = OpenMaya.MObject()
    m_output    = OpenMaya.MObject()

    def __init__(self):
        OpenMayaMPx.MPxTransform.__init__(self)


def dxBlockMatrix_Creator():
    return OpenMayaMPx.asMPxPtr(dxBlockMatrix())

def dxBlock_Creator():
    return OpenMayaMPx.asMPxPtr(dxBlock())

def dxBlock_Initializer():
    numericAttr = OpenMaya.MFnNumericAttribute()
    enumAttr    = OpenMaya.MFnEnumAttribute()
    typedAttr   = OpenMaya.MFnTypedAttribute()
    msgAttr     = OpenMaya.MFnMessageAttribute()

    dxBlock.m_type = enumAttr.create('type', 'type', 0)
    enumAttr.addField('None', 0)
    enumAttr.addField('Model', 1)
    enumAttr.addField('PointInstancer', 2)
    enumAttr.addField('Simulation', 3)
    enumAttr.addField('Light', 4)
    enumAttr.setStorable(True)
    dxBlock.addAttribute(dxBlock.m_type)

    dxBlock.m_action = enumAttr.create('action', 'action', 0)
    enumAttr.addField('None', 0)
    enumAttr.addField('Export', 1)
    # enumAttr.addField('Reference', 2)
    enumAttr.setStorable(True)
    dxBlock.addAttribute(dxBlock.m_action)

    dxBlock.m_rootPrimPath = typedAttr.create('rootPrimPath', 'rootPrimPath', OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxBlock.addAttribute(dxBlock.m_rootPrimPath)

    dxBlock.m_nsLayer = typedAttr.create('nsLayer', 'nsLayer', OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxBlock.addAttribute(dxBlock.m_nsLayer)

    dxBlock.m_importFile = typedAttr.create('importFile', 'importFile', OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxBlock.addAttribute(dxBlock.m_importFile)

    dxBlock.m_mergeFile = typedAttr.create('mergeFile', 'mergeFile', OpenMaya.MFnData.kString, OpenMaya.MFnStringData().create(''))
    dxBlock.addAttribute(dxBlock.m_mergeFile)

    dxBlock.m_pxrStageNode = msgAttr.create('pxrStageNode', 'pxrStageNode')
    dxBlock.addAttribute(dxBlock.m_pxrStageNode)

    dxBlock.m_referencedXBlock = msgAttr.create('referencedXBlock', 'referencedXBlock')
    dxBlock.addAttribute(dxBlock.m_referencedXBlock)

    dxBlock.m_output = numericAttr.create('output', 'output', OpenMaya.MFnNumericData.kFloat)
    numericAttr.setHidden(True)
    numericAttr.setWritable(False)
    numericAttr.setStorable(False)
    dxBlock.addAttribute(dxBlock.m_output)

    dxBlock.attributeAffects(dxBlock.m_type, dxBlock.m_output)
    dxBlock.attributeAffects(dxBlock.m_action, dxBlock.m_output)