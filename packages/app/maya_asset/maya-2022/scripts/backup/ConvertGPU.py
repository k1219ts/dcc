import maya.cmds as cmds
import sgComponent as sgc;


def ConvertGpuCache(node):
    nodeName = node
    translate = cmds.getAttr('%s.translate' % nodeName)
    rotation = cmds.getAttr('%s.rotate' % nodeName)
    scale = cmds.getAttr('%s.scale' % nodeName)
    abcFileName = cmds.getAttr('%sArc.cacheFileName' % nodeName)
    parent = cmds.listRelatives(node, p=True, f=True)[0]
    print translate
    print rotation
    print scale
    print abcFileName
    print parent

    cmds.delete(nodeName)
    nodeName = cmds.createNode('dxComponent', n=nodeName, p=parent)
    cmds.setAttr('%s.translateX' % nodeName, translate[0][0])
    cmds.setAttr('%s.translateY' % nodeName, translate[0][1])
    cmds.setAttr('%s.translateZ' % nodeName, translate[0][2])

    cmds.setAttr('%s.rotateX' % nodeName, rotation[0][0])
    cmds.setAttr('%s.rotateY' % nodeName, rotation[0][1])
    cmds.setAttr('%s.rotateZ' % nodeName, rotation[0][2])

    cmds.setAttr('%s.scaleX' % nodeName, scale[0][0])
    cmds.setAttr('%s.scaleY' % nodeName, scale[0][1])
    cmds.setAttr('%s.scaleZ' % nodeName, scale[0][2])

    cmds.setAttr('%s.abcFileName' % nodeName, abcFileName, type='string')
    cmds.setAttr('%s.action' % nodeName, 2)
    cmds.setAttr('%s.mode' % nodeName, 1)
    cmds.setAttr('%s.display' % nodeName, 1)
    if '_low' in abcFileName:
        cmds.setAttr('%s.display' % nodeName, 3)
    sgc.componentReload("%s." % nodeName)


def RecursiveNode(node):
    if cmds.nodeType(node) == "dxAbcArchive":
        ConvertGpuCache(node)
        return

    for childNode in cmds.listRelatives(node, children=True):
        RecursiveNode(childNode)


def ChangeNode():
    for mainNode in cmds.ls(sl=True):
        RecursiveNode(mainNode)

    print "END"


