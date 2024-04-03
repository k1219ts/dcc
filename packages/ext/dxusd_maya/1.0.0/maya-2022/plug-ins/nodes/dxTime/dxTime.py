'''
    @ nodeName : dxTime
    @ author : Dexter Studio by daeseok.chae
    @ date : 2019.03.14
'''
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx


class dxTimeOffset(OpenMayaMPx.MPxNode):
    inputTime = OpenMaya.MObject()
    outputOffset = OpenMaya.MObject()

    addtiveOffset = OpenMaya.MObject()

    def __init__(self):
        OpenMayaMPx.MPxNode.__init__(self)

    def compute(self, plug, data):
        inputValue = data.inputValue(dxTimeOffset.inputTime).asTime()
        addtiveValue = data.inputValue(dxTimeOffset.addtiveOffset).asFloat()

        outputHandle = data.outputValue(dxTimeOffset.outputOffset)
        outputHandle.setMTime(inputValue + addtiveValue)

        data.setClean(plug)


def dxTimeOffset_Creator():
    return OpenMayaMPx.asMPxPtr(dxTimeOffset())


def dxTimeOffset_Initialize():
    # setup attr function
    nAttr = OpenMaya.MFnNumericAttribute()
    kFloat = OpenMaya.MFnNumericData.kFloat

    tAttr = OpenMaya.MFnUnitAttribute()
    kTime = OpenMaya.MFnUnitAttribute.kTime

    # output attribute
    dxTimeOffset.outputOffset = tAttr.create("outTime", "outTime", kTime)
    tAttr.setWritable(False)
    tAttr.setStorable(False)
    dxTimeOffset.addAttribute(dxTimeOffset.outputOffset)

    # input attribute
    dxTimeOffset.inputTime = tAttr.create("time", "time", kTime, 0.0)
    tAttr.setKeyable(True)
    dxTimeOffset.addAttribute(dxTimeOffset.inputTime)
    dxTimeOffset.attributeAffects(dxTimeOffset.inputTime, dxTimeOffset.outputOffset)

    # add attribute to node
    dxTimeOffset.addtiveOffset = nAttr.create('offset', 'offset', kFloat, 0.0)
    nAttr.hidden = False
    nAttr.keyable = True

    dxTimeOffset.addAttribute(dxTimeOffset.addtiveOffset)
    dxTimeOffset.attributeAffects(dxTimeOffset.addtiveOffset, dxTimeOffset.outputOffset)
