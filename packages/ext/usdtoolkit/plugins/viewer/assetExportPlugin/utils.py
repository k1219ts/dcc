#coding:utf-8

import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim
import maya.cmds as cmds
import os
import re
import string

import DXUSD.Utils as utl
import DXUSD_MAYA.Message as msg
from pxr import Gf
import dxBlockUtils


def getMayaPath(abname):
    mayafile = '/show/pipe/works/AST/temp/%s_model_v001.mb' % abname
    return mayafile

def importData(modeldir):
    # import Geom
    modelList = []
    for geomName in os.listdir(modeldir):
        if 'GRP.high_geom.usd' in geomName or 'GRP.low_geom.usd' in geomName or 'GRP.mid_geom.usd' in geomName:
            filepath = os.path.join( modeldir, geomName)
            modelList.append(filepath)

    nodes =[]
    for path in modelList:
        node = dxBlockUtils.UsdImport(path).importGeom(path)
        if not '_model_' in node:
            if 'low_geom' in path:
                node = cmds.rename(node, node + '_model_low_GRP')
            elif 'mid_geom' in path:
                node = cmds.rename(node, node + '_model_mid_GRP')
            else:
                node = cmds.rename(node, node + '_model_GRP')
        nodes.append(node)
        nodeName = node
        childnode = cmds.listRelatives(node, f=True, c=True, type='transform')
        grp= cmds.group(childnode, parent=node)
        cmds.parent(grp, w=1)
        cmds.delete(node)
        cmds.rename(grp, nodeName)
        print 'node:',node
    return nodes

def rename(nodes , basepath, abname):
    newlist = []
    for node in nodes:
        editAttr(node, basepath)
        newNode = renameNode(node, abname, '_model_')
        newlist.append(newNode)
    return newlist


def renameNode(node , nodename , splitText):
    old = node.split(splitText)[0]
    newNode = node.replace(old, nodename)
    newNode = cmds.rename(node, newNode)
    return newNode


def editAttr(node, txPath):
    for i in cmds.ls(node, dag=True, type='surfaceShape', ni=True):
        if not cmds.attributeQuery("txBasePath", n=i, exists=True):
            cmds.addAttr(i, ln="txBasePath", nn="txBasePath", dt="string")
        cmds.setAttr("%s.%s" % (i, "txBasePath"), txPath, type="string")


# ------------------------------------------------------------------------------
# Proxy
# ------------------------------------------------------------------------------
def proxy(self, nodes):
    if not len(nodes) > 1:
        if '_low_' in nodes[0] or '_mid_' in nodes[0]:
            dupnode = cmds.duplicate(nodes[0])
            newNode = '%s_model_GRP' % self.abname
        else:
            dupnode = cmds.duplicate(nodes[0])
            newNode = '%s_model_low_GRP' % self.abname
        newNode = cmds.rename(dupnode[0], newNode)
        nodes.append(newNode)



# ------------------------------------------------------------------------------
# Zenn
# ------------------------------------------------------------------------------

def bodyMeshAttr(txPath):
    meshList = []
    for i in cmds.ls(type='ZN_Import'):
        findMesh = cmds.listConnections(i, source=True, destination=False, type='mesh')[0]
        if not findMesh in meshList:
            meshList.append(findMesh)
    for node in meshList:
        node = cmds.listRelatives(node, shapes=True)[0]
        if not cmds.attributeQuery("txBasePath", n=node, exists=True):
            cmds.addAttr(node, ln="txBasePath", nn="txBasePath", dt="string")
        if not cmds.attributeQuery("modelVersion", n=node, exists=True):
            cmds.addAttr(node, ln="modelVersion", nn="modelVersion", dt="string")
        cmds.setAttr("%s.%s" % (node, "txBasePath"), txPath, type="string")
        cmds.setAttr("%s.%s" % (node, "modelVersion"), 'v001', type="string")

def groomExport(hairPath, newShow ,nodename, txPath ):
    import DXUSD_MAYA.Groom as Groom
    cmds.file(new=True, force=True)
    cmds.file(hairPath, open=True)

    hairTempFile = '/show/pipe/works/AST/groom/%s_groom_%s.mb' % (nodename, 'v001')
    cmds.file(rename=hairTempFile)
    cmds.file(save=True, type="mayaBinary")

    if not '_3d' in hairPath:
        groomNode = cmds.ls('*_ZN_GRP')[0]
        groomNode = renameNode(groomNode, nodename, '_ZN_')
        cmds.rename(groomNode, groomNode.replace('ZN','groom'))

        dxblock = cmds.ls('*_rig_GRP', type= 'dxBlock')[0]
        dxblock = renameNode(dxblock, nodename, '_rig_')
        dxblock = cmds.rename(dxblock, dxblock.replace('rig', 'ZN'))

    else:
        dxblock = cmds.ls('*_ZN_GRP', type= 'dxBlock')[0]
        dxblock = renameNode(dxblock, nodename, '_ZN_')

        groomNode = cmds.ls('*_groom_GRP')[0]
        renameNode(groomNode, nodename, '_groom_')


    orgSource = cmds.ls('*_model_GRP')
    editAttr(orgSource, txPath)
    bodyMeshAttr(txPath)

    if not cmds.objExists("MaterialSet"):
        cmds.sets(name="MaterialSet")
        material= 'fur'
        newMaterialSet = cmds.sets(name=material)
        cmds.sets(newMaterialSet, add='MaterialSet')
        for node in cmds.ls('*_ZN_Deform', type='ZN_Deform'):
            if not cmds.attributeQuery('MaterialSet', n=node, exists=True):
                cmds.addAttr(node, ln='MaterialSet', dt='string')
            cmds.sets(node, add=material)

        for node in cmds.ls(type='ZN_FeatherInstance'):
            mesh = cmds.listConnections(cmds.listConnections(node, s=True, type='ZN_FeatherImport')[0], s=True, type='mesh')[0]
            material = cmds.getAttr('%s.MaterialSet' % mesh)
            if material:
                if not cmds.objExists(material):
                    newMaterialSet = cmds.sets(name=material)
                    cmds.sets(newMaterialSet, add='MaterialSet')
                cmds.sets(node, add=material)
    # print('ZN node:', dxblock)
    Groom.assetExport(node=dxblock, show=newShow)
    if os.path.exists(hairTempFile):
        os.remove(hairTempFile)
