
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx

class dAgentGroupMatrix(OpenMayaMPx.MPxTransformationMatrix):
    def __init__(self):
        OpenMayaMPx.MPxTransformationMatrix.__init__(self)

    def asMatrix(self):
        matrix = OpenMayaMPx.MPxTransformationMatrix.asMatrix(self)
        tm = OpenMaya.MTransformationMatrix(matrix)
        return tm.asMatrix()

class dAgentGroupNode(OpenMayaMPx.MPxTransform):
    agentIds = OpenMaya.MObject()

    def __init__(self):
        OpenMayaMPx.MPxTransform.__init__(self)


def dAgentGroupMatrix_Creator():
    return OpenMayaMPx.asMPxPtr(dAgentGroupMatrix())

def dAgentGroupNode_Creator():
    return OpenMayaMPx.asMPxPtr(dAgentGroupNode())

def dAgentGroupNode_Initializer():
    typedAttr = OpenMaya.MFnTypedAttribute()
    dAgentGroupNode.agentIds = typedAttr.create(
        'agentIds', 'agentIds',
        OpenMaya.MFnData.kIntArray
    )
    dAgentGroupNode.addAttribute(dAgentGroupNode.agentIds)
