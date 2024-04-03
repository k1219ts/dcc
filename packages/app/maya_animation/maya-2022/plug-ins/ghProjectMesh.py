
import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMpx

kPluginNodeTypeName = "ghProjectMesh"

kPluginNodeId = OpenMaya.MTypeId(0x87668)

class ProjectMeshNode(OpenMayaMpx.MPxNode):
    inputMatrix = OpenMaya.MObject()
    inputMeshTarget = OpenMaya.MObject()
    outputTranslateX = OpenMaya.MObject()
    outputTranslateY = OpenMaya.MObject()
    outputTranslateZ = OpenMaya.MObject()
    weight_rotate = OpenMaya.MObject()
    inputParentInverseMatrix = OpenMaya.MObject()
    offsetY = OpenMaya.MObject()

    def __init__(self):
        OpenMayaMpx.MPxNode.__init__(self)

    def compute(self, plug, data):
        dataHandle = data.inputValue(ProjectMeshNode.inputMatrix)
        inputParentInverseHandle = data.inputValue(ProjectMeshNode.inputParentInverseMatrix)
        inMeshTargetHandle = data.inputValue(ProjectMeshNode.inputMeshTarget)
        weight_rotateHandle = data.inputValue(ProjectMeshNode.weight_rotate)
        offsetYHandle = data.inputValue(ProjectMeshNode.offsetY)

        srcParentInverseMtx = inputParentInverseHandle.asMatrix()
        weight = weight_rotateHandle.asFloat()
        srcMatrix = dataHandle.asMatrix()
        offsety = offsetYHandle.asFloat()
        mTransformMtx = OpenMaya.MTransformationMatrix(srcMatrix)
        trans = mTransformMtx.getTranslation(OpenMaya.MSpace.kTransform)
        vec = OpenMaya.MVector.yAxis
        vec = (vec*srcMatrix).normal()
        meshTarget = inMeshTargetHandle.asMesh()
        mFnMeshTarget = OpenMaya.MFnMesh(meshTarget)
        raySource = OpenMaya.MFloatPoint(trans.x, trans.y, trans.z)

        rayDirection = OpenMaya.MFloatVector(vec.x * weight,
                                             vec.y,
                                             vec.z * weight)

        hitPoint = OpenMaya.MFloatPoint()
        hitFaceUtil = OpenMaya.MScriptUtil()
        hitFace = hitFaceUtil.asIntPtr()
        idsSorted = False
        testBothDirections = False
        faceIds = None
        triIds = None
        accelParams = None
        tolerance = 1.0
        hitRayParam = None
        hitTriangle = None
        hitBary1 = None
        hitBary2 = None
        maxParamPtr = 99999999

        hit = None
        for i in [1, -1]:
            hit = mFnMeshTarget.closestIntersection(
                raySource,
                rayDirection * i,
                faceIds,
                triIds,
                idsSorted,
                OpenMaya.MSpace.kWorld,
                maxParamPtr,
                testBothDirections,
                accelParams,
                hitPoint,
                hitRayParam,
                hitFace,
                hitTriangle,
                hitBary1,
                hitBary2,
                tolerance
            )
            if hit:
                break
        if hit:
            outputHandleX = data.outputValue(ProjectMeshNode.outputTranslateX)
            outputHandleY = data.outputValue(ProjectMeshNode.outputTranslateY)
            outputHandleZ = data.outputValue(ProjectMeshNode.outputTranslateZ)
            worldMatrixHandle = data.outputValue(ProjectMeshNode.worldMatrix)

            newWorldMatrix = OpenMaya.MMatrix()
            vpos = OpenMaya.MVector(hitPoint.x, hitPoint.y, hitPoint.z)
            matrixList = [0, 0, 0, 0,
                          0, 0, 0, 0,
                          0, 0, 0, 0,
                          vpos[0], vpos[1] + offsety, vpos[2], 1]
            OpenMaya.MScriptUtil.createMatrixFromList(matrixList, newWorldMatrix)
            localMatrix = newWorldMatrix * srcParentInverseMtx
            outTransformMtx = OpenMaya.MTransformationMatrix(localMatrix)
            outTranslate = outTransformMtx.getTranslation(OpenMaya.MSpace.kTransform)

            outputHandleX.setFloat(outTranslate.x)
            outputHandleY.setFloat(outTranslate.y + offsety)
            outputHandleZ.setFloat(outTranslate.z)
            worldMatrixHandle.setMMatrix(newWorldMatrix)
        data.setClean(plug)


def nodeCreator():
    nodePtr = OpenMayaMpx.asMPxPtr(ProjectMeshNode())
    return nodePtr

def nodeInitializer():
    typedAttr = OpenMaya.MFnTypedAttribute()
    # Setup the input attributes
    ProjectMeshNode.inputMeshTarget = typedAttr.create("inputMeshTarget",
                                                       "inMeshTrgt",
                                                       OpenMaya.MFnData.kMesh)

    inmatrixAttr = OpenMaya.MFnMatrixAttribute()
    ProjectMeshNode.inputMatrix = inmatrixAttr.create("inputMatrix",
                                                      "inMtx",
                                                      OpenMaya.MFnMatrixAttribute.kDouble)
    inmatrixAttr.setReadable(False)

    ProjectMeshNode.inputParentInverseMatrix = inmatrixAttr.create("inParentInverseMatrix",
                                                                   "inPInvsMtx",
                                                                   OpenMaya.MFnMatrixAttribute.kDouble)
    inmatrixAttr.setReadable(False)

    numedAttr = OpenMaya.MFnNumericAttribute()
    ProjectMeshNode.outputTranslateX = numedAttr.create("outputTranslateX",
                                                        "outTransX",
                                                        OpenMaya.MFnNumericData.kFloat)
    ProjectMeshNode.outputTranslateY = numedAttr.create("outputTranslateY",
                                                        "outTransY",
                                                        OpenMaya.MFnNumericData.kFloat)
    ProjectMeshNode.outputTranslateZ = numedAttr.create("outputTranslateZ",
                                                        "outTransZ",
                                                        OpenMaya.MFnNumericData.kFloat)
    ProjectMeshNode.outputTranslate = numedAttr.create("outputTranslate",
                                                       "outTrans",
                                                       ProjectMeshNode.outputTranslateX,
                                                       ProjectMeshNode.outputTranslateY,
                                                       ProjectMeshNode.outputTranslateZ)
    numedAttr.setWritable(False)

    ProjectMeshNode.weight_rotate = numedAttr.create("WeightRotate",
                                                     "wr",
                                                     OpenMaya.MFnNumericData.kFloat, 1.0)
    numedAttr.setKeyable(True)
    numedAttr.setMin(0.0)
    numedAttr.setMax(1.0)

    ProjectMeshNode.offsetY = numedAttr.create("offsetY",
                                               "osy",
                                               OpenMaya.MFnNumericData.kFloat, 0)
    numedAttr.setReadable(True)
    numedAttr.setKeyable(True)

    outmatrixAttr = OpenMaya.MFnMatrixAttribute()
    ProjectMeshNode.worldMatrix = outmatrixAttr.create("worldMatrix",
                                                       "wrdMtx",
                                                       OpenMaya.MFnMatrixAttribute.kDouble)
    outmatrixAttr.setWritable(False)

    ProjectMeshNode.addAttribute(ProjectMeshNode.weight_rotate)
    ProjectMeshNode.addAttribute(ProjectMeshNode.inputMatrix)
    ProjectMeshNode.addAttribute(ProjectMeshNode.inputParentInverseMatrix)
    ProjectMeshNode.addAttribute(ProjectMeshNode.inputMeshTarget)
    ProjectMeshNode.addAttribute(ProjectMeshNode.offsetY)
    ProjectMeshNode.addAttribute(ProjectMeshNode.outputTranslate)
    ProjectMeshNode.addAttribute(ProjectMeshNode.worldMatrix)

    ProjectMeshNode.attributeAffects(ProjectMeshNode.weight_rotate, ProjectMeshNode.outputTranslate)
    ProjectMeshNode.attributeAffects(ProjectMeshNode.inputMatrix, ProjectMeshNode.outputTranslate)
    ProjectMeshNode.attributeAffects(ProjectMeshNode.inputParentInverseMatrix, ProjectMeshNode.outputTranslate)
    ProjectMeshNode.attributeAffects(ProjectMeshNode.offsetY, ProjectMeshNode.outputTranslate)
    ProjectMeshNode.attributeAffects(ProjectMeshNode.weight_rotate, ProjectMeshNode.worldMatrix)
    ProjectMeshNode.attributeAffects(ProjectMeshNode.inputMatrix, ProjectMeshNode.worldMatrix)

def initializePlugin(mobject):
    mplugin = OpenMayaMpx.MFnPlugin(mobject, "gyeongheon.jeong", "1.0")
    try:
        mplugin.registerNode(kPluginNodeTypeName, kPluginNodeId, nodeCreator, nodeInitializer)
    except:
        sys.stderr.write("Failed to register node: {0}\n".format(kPluginNodeTypeName))


def uninitializePlugin(mobject):
    mplugin = OpenMayaMpx.MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(kPluginNodeId)
    except:
        sys.stderr.write("Failed to unregister node: {0}\n".format(kPluginNodeTypeName))
