import maya.cmds as cmds


def findChild(node):
    cameraNode = None
    while True:
        child = cmds.listRelatives(node, c=True)
        if child:
            childShape = cmds.listRelatives(child, s=True)
            if childShape and cmds.objectType(childShape) == 'camera':
                cameraNode = child
                return cameraNode
            else:
                node = child
        else:
            break
    return cameraNode


def addCameraSpeed():
    exString = "$startTime = `playbackOptions -q -min`;\n"
    exString += "{handle}.translateZ = ({cameraName}.speed/10) * (frame - $startTime);"

    sel = cmds.ls(sl=True)
    if sel and cmds.nodeType(sel[0]) == "dxCamera":
        cameraNode = sel[0]
        camTransform = findChild(cameraNode)[0]
        transhandleNode = cmds.group(camTransform, n=cameraNode + '_handle_T')
        rotatehandleNode = cmds.group(transhandleNode, n=cameraNode + '_handle_R')
        # cmds.connectAttr(camTransform + '.translateX', rotatehandleNode + '.rotatePivotX')
        # cmds.connectAttr(camTransform + '.translateY', rotatehandleNode + '.rotatePivotY')
        # cmds.connectAttr(camTransform + '.translateZ', rotatehandleNode + '.rotatePivotZ')
        exString = exString.format(handle=transhandleNode, cameraName=cameraNode)
        attrs = cmds.listAttr(cameraNode, k=True)
        if 'speed' not in attrs:
            cmds.addAttr(cameraNode, ln="speed", at="double", dv=0)
            cmds.setAttr(cameraNode + ".speed", e=True, keyable=True)
        cmds.expression(s=exString, o=cameraNode, ae=1, uc="all")
    else:
        msg = cmds.confirmDialog(title='Warning!',
                                 message='Select dxCamera node',
                                 button=['close'],
                                 defaultButton='close')