import maya.cmds as cmds

def offsetKey():
    cam = cmds.ls(sl=True, dag=True)

    for i in cam:
        #focalLength
        if i.find('Shape') > 0:
            #unlock channels
            if cmds.getAttr(i + ".focalLength", lock=True) is True:
                cmds.setAttr(i + ".focalLength", lock = False)
            
            startOffsetKey(i, "focalLength", 1)
            endOffsetKey(i, "focalLength", 1)
            
            #lock channels
            if cmds.getAttr(i + ".focalLength", lock=True) is False:
                cmds.setAttr(i + ".focalLength", lock = True)

        else:
            #unlock channels
            if cmds.getAttr(i + ".translate", lock=True) is True:
                cmds.setAttr(i + ".translate", lock = False)
            if cmds.getAttr(i + ".rotate", lock=True) is True:
                cmds.setAttr(i + ".rotate", lock = False)

            for attr in ['tx','ty','tz','rx','ry','rz','sx','sy','sz']:
                startOffsetKey(i, attr, 1)
                endOffsetKey(i, attr, 1)

            #lock channels
            if cmds.getAttr(i + ".translate", lock=True) is False:
                cmds.setAttr(i + ".translate", lock = True)
            if cmds.getAttr(i + ".rotate", lock=True) is False:
                cmds.setAttr(i + ".rotate", lock = True)


def endOffsetKey(name, attr, offset):
    tmp = []
    nType = ""
    con = 0
    endFrame = []
    endV = 0.0
    offsetV = 0.0
    value = 0.0
    node = name + "." + attr

    con = cmds.connectionInfo(node, id=True)

    if con == 1:
        tmp = cmds.listConnections(node)
        nType = cmds.nodeType(tmp[0])

        if nType.count("anim") > 0:
            endFrame = cmds.keyframe(node, q=True, lsl=True)
            endV = cmds.getAttr(node, t=endFrame[0])
            offsetV = cmds.getAttr(node, t=endFrame[0]-offset)
            value = endV-offsetV
            #print endFrame[0], endV, offsetV, value
            cmds.setKeyframe(node, itt="spline", ott="spline", t=endFrame[0]+offset, at=attr, v=endV+value)



def startOffsetKey(name, attr, offset):
    tmp = []
    nType = ""
    con = 0
    startFrame = []
    startV = 0.0
    offsetV = 0.0
    value = 0.0
    node = name + "." + attr

    con = cmds.connectionInfo(node, id=True)

    if con == 1:
        tmp = cmds.listConnections(node)
        nType = cmds.nodeType(tmp[0])

        if nType.count("anim") > 0:
            startFrame = cmds.keyframe(node, q=True, a=True)
            startV = cmds.getAttr(node, t=startFrame[0])
            offsetV = cmds.getAttr(node, t=startFrame[0]+offset)
            value = offsetV-startV
            #print startFrame[0], startV, offsetV, value
            cmds.setKeyframe(node, itt="spline", ott="spline", t=startFrame[0]-offset, at=attr, v=startV-value)
