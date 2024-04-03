# -*- coding: utf-8 -*-
"""
author : gyeongheon.jeong

"""
import maya.cmds as cmds

def getObjectCenter(object):
    pivot = cmds.objectCenter(object, gl=True)
    return pivot

def rivet(objects, targetMesh):
    """Maya rivet module, using follicle node
    
    :param objects: A list of maya objects 
    :param targetMesh: A string of mesh object name
    :return:
    """
    targetMeshShape = cmds.listRelatives(targetMesh, s=True)[0]
    follicleList = list()

    if cmds.objectType(targetMeshShape) != "mesh":
        cmds.error("Select polymesh object")

    cmds.undoInfo(ock=True)
    print "\n# Target Mesh : {0}".format(targetMesh)
    for object in objects:
        print " > Object : " + object
        closest = cmds.createNode('closestPointOnMesh')
        cmds.connectAttr(targetMeshShape + ".outMesh", closest + ".inMesh")
        objectCenter = getObjectCenter(object)
        print " > Center : {0}\n".format(objectCenter)
#        objPivot = cmds.xform(object, q=True, piv=True)
#        cmds.xform(object, cp=True)
#        objectCenter = cmds.xform(object, rp=True, ws=True, q=True)
#        print "> translate : {0}\n".format(objectCenter)
        cmds.setAttr(closest + ".inPositionX", objectCenter[0])
        cmds.setAttr(closest + ".inPositionY", objectCenter[1])
        cmds.setAttr(closest + ".inPositionZ", objectCenter[2])
        follicle = cmds.createNode('follicle')
        #follicleList.append(follicle)
        follicleTrans = cmds.listRelatives(follicle, type='transform', p=True)
        cmds.connectAttr(follicle + '.outRotate', follicleTrans[0] + '.rotate')
        cmds.connectAttr(follicle + '.outTranslate', follicleTrans[0] + '.translate')
        cmds.connectAttr(targetMeshShape + '.worldMatrix', follicle + '.inputWorldMatrix')
        cmds.connectAttr(targetMeshShape + '.worldMesh', follicle + '.inputMesh')
        cmds.setAttr(follicle + '.simulationMethod', 0)
        u = cmds.getAttr(closest + '.u')
        v = cmds.getAttr(closest + '.v')
        cmds.setAttr(follicle + '.parameterU', u)
        cmds.setAttr(follicle + '.parameterV', v)
#        cmds.xform(object, piv=(objPivot[0], objPivot[1], objPivot[2]))
        cmds.parentConstraint(follicleTrans[0], object, mo=True)
        cmds.delete(closest)
        follicleTrans_renamed = cmds.rename(follicleTrans, "GhRivet_FLCL_#")
        follicleList.append(follicleTrans_renamed)
    cmds.group(follicleList, name="GhRivet_Follicles_GRP_#")
    cmds.undoInfo(cck=True)
