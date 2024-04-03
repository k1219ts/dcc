import maya.cmds as cmds
import anchorUtils

__author__    = "Robert Joosten"
__version__   = "0.1.0"
__email__     = "rwm.joosten@gmail.com"

def anchorSelection(start, end):
    transforms = cmds.ls(sl=True, transforms=True) or []
    for transform in transforms:
        anchorTransform(transform, start, end)

def anchorTransform(transform, start, end):
    with anchorUtils.UndoChunkContext():
        rotOrder = cmds.getAttr("{0}.rotateOrder".format(transform))
        anchorMatrix = anchorUtils.getMatrix(transform, start, "worldMatrix")
        for i in range(start, end + 1):
            inverseMatrix = anchorUtils.getMatrix(transform,i,"parentInverseMatrix")
            localMatrix = anchorMatrix * inverseMatrix
            rotPivot = cmds.getAttr("{0}.rotatePivot".format(transform))[0]
            transformValues = anchorUtils.decomposeMatrix(localMatrix,rotOrder,rotPivot,)
            for attr, value in zip(anchorUtils.ATTRIBUTES, transformValues):
                for j, channel in enumerate(anchorUtils.CHANNELS):
                    node = "{0}.{1}{2}".format(transform, attr, channel)
                    tangents = {"inTangentType": "linear","outTangentType": "linear"}
                    animInputs = cmds.listConnections(node,type="animCurve",destination=False)
                    if animInputs and i == end:
                        tangents["outTangentType"] = anchorUtils.getOutTangent(animInputs[0],end)
                    elif animInputs and i == start:
                        tangents["inTangentType"] = anchorUtils.getInTangent(animInputs[0],start)
                    cmds.setKeyframe(node,t=i,v=value[j])