import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMPx
import math

nodeName = "RippleDeformer"
nodeId = OpenMaya.MTypeId(0x102fff)

class Ripple(OpenMayaMPx.MPxDeformerNode):
    '''
    Commands ------> MPxCommand
    Custom   ------> MPxNode
    Deformer ------> MPxDeformerNode
    '''

    mobj_Amplitude = OpenMaya.MObject()
    mobj_Displace = OpenMaya.MObject()

    def __init__(self):
        OpenMayaMPx.MPxDeformerNode.__init__(self)

    def deform(self, dataBlock, geoIterator, matrix, geometryIndex):
        input = OpenMayaMPx.cvar.MPxDeformerNode_input

        dataHandleInputArray = dataBlock.inputArrayValue(input)
        dataHandleInputArray.jumpToElement(geometryIndex)
        dataHandleInputElement = dataHandleInputArray.inputValue()

        inputGeom = OpenMayaMPx.cvar.MPxDeformerNode_inputGeom
        dataHandleInputGeom = dataHandleInputElement.child(inputGeom)
        inMesh = dataHandleInputGeom.asMesh()

        # Envelope
        envelope = OpenMayaMPx.cvar.MPxDeformerNode_envelope
        dataHandleEnvelope = dataBlock.inputValue(envelope)
        envelopeValue = dataHandleEnvelope.asFloat()

        # Amplitude
        dataHandleAmplitude = dataBlock.inputValue(Ripple.mobj_Amplitude)
        amplitudeValue = dataHandleAmplitude.asFloat()

        # Displace
        dataHandleDisplace = dataBlock.inputValue(Ripple.mobj_Displace)
        displaceValue = dataHandleDisplace.asFloat()

        mFloatVectorArray_normal = OpenMaya.MFloatVectorArray()
        mFnMesh = OpenMaya.MFnMesh(inMesh)
        mFnMesh.getVertexNormals(False, mFloatVectorArray_normal, OpenMaya.MSpace.kObject)

        mPointArray_meshVert = OpenMaya.MPointArray()
        while not geoIterator.isDone():
            pointPosition = geoIterator.position()
            weight = self.weightValue(dataBlock, geometryIndex, geoIterator.index())
            pointPosition.x = pointPosition.x + math.sin(geoIterator.index() + displaceValue) * amplitudeValue * \
                                                mFloatVectorArray_normal[geoIterator.index()].x * weight * envelopeValue
            pointPosition.y = pointPosition.y + math.sin(geoIterator.index() + displaceValue) * amplitudeValue * \
                                                mFloatVectorArray_normal[geoIterator.index()].y * weight * envelopeValue
            pointPosition.z = pointPosition.z + math.sin(geoIterator.index() + displaceValue) * amplitudeValue * \
                                                mFloatVectorArray_normal[geoIterator.index()].z * weight * envelopeValue
            mPointArray_meshVert.append(pointPosition)
            #geoIterator.setPosition(pointPosition)
            geoIterator.next()
        geoIterator.setAllPositions(mPointArray_meshVert)



def deformerCreator():
    nodePtr = OpenMayaMPx.asMPxPtr(Ripple())
    return nodePtr

def nodeInitializer():
    '''
    Create Attributes
    Attach Attributes
    Design Circuitry
    '''

    mFnAttr = OpenMaya.MFnNumericAttribute()
    Ripple.mobj_Amplitude = mFnAttr.create("AmplitudeValue", "AmpVal", OpenMaya.MFnNumericData.kFloat, 0.0)
    mFnAttr.setKeyable(1)
    mFnAttr.setMin(0.0)
    mFnAttr.setMax(1.0)

    Ripple.mobj_Displace = mFnAttr.create("DisplaceValue", "DisVal", OpenMaya.MFnNumericData.kFloat, 0.0)
    mFnAttr.setKeyable(1)
    mFnAttr.setMin(0.0)
    mFnAttr.setMax(10.0)

    mFnMatrixAttr = OpenMaya.MFnMatrixAttribute()


    Ripple.addAttribute(Ripple.mobj_Amplitude)
    Ripple.addAttribute(Ripple.mobj_Displace)

    outputGeom = OpenMayaMPx.cvar.MPxDeformerNode_outputGeom
    Ripple.attributeAffects( Ripple.mobj_Amplitude, outputGeom )
    Ripple.attributeAffects( Ripple.mobj_Displace, outputGeom )


def initializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject, "gyeongheon jeong", "1.0")
    try:
        mplugin.registerNode(nodeName, nodeId, deformerCreator, nodeInitializer, OpenMayaMPx.MPxNode.kDeformerNode)
    except:
        sys.stderr.write("Failed to register Node: {0}\n".format(nodeName))


def uninitializePlugin(mobject):
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(nodeId)
    except:
        sys.stderr.write("Failed to unregister Node: {0}\n".format(nodeName))
