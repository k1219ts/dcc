import maya.cmds as cmds
import os

import string

materialList = ["bronze", "chrome", "gold", "metal", "silver", "fabric", "glass", "leather", "plastic", "rubber","feather",
                "paint", "wood", "leaf", "ice", "ocean", "mineral", "rock", "snow", "skin", "fur","eye","tile"]

#-------------------------------------------------------------------------------
#
# Add MaterialSet Attribute by ShadingGroup
#
#-------------------------------------------------------------------------------
def AddMaterialSetAttributeByShadingGroup():
    selected = cmds.ls(sl=True, dag=True, type='surfaceShape')
    if not selected:
        selected = cmds.ls(dag=True, type='surfaceShape')

    for shape in selected:
        sg = cmds.listConnections(shape, type='shadingEngine')
        if sg:
            if not 'initial' in sg[0]:
                mtlname = cmds.listConnections('%s.surfaceShader' % sg[0])
                if mtlname:
                    if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
                        cmds.addAttr(shape, ln='MaterialSet', dt='string')
                    cmds.setAttr('%s.MaterialSet' % shape, mtlname[0], type='string')


def AddMaterialSetAttribute():
    if not cmds.objExists('MaterialSet'):
        return
    # clear
    for shape in cmds.ls(type='surfaceShape', ni=True):
        if cmds.attributeQuery('MaterialSet', n=shape, ex=True):
            cmds.setAttr('%s.MaterialSet' % shape, '', type='string')

    for i in cmds.sets('MaterialSet', q=True):
        source = cmds.sets(i, q=True)
        memberShape = cmds.ls(source, dag=True, type='surfaceShape', ni=True)
        for shape in memberShape:
            if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
                cmds.addAttr(shape, ln='MaterialSet', dt='string')
            cval = cmds.getAttr('%s.MaterialSet' % shape)
            if not cval:
                cmds.setAttr('%s.MaterialSet' % shape, i, type='string')
            else:
                names = cval.split(',')
                if not i in names:
                    names.append(i)
                cmds.setAttr('%s.MaterialSet' % shape, string.join(names, ','), type='string')

def materialImport():

    dir="/dexter/Cache_DATA/ASSET/1.pipeline/Katana/material"
    filename = os.path.join(dir,"mayaMaterial.ma")
    cmds.file(filename, i=True)


def findMaterialSetAssign():
    materialList = ["bronze", "chrome", "gold", "metal", "silver", "fabric", "glass", "leather", "plastic", "rubber",
                    "feather",
                    "paint", "wood", "leaf", "ice", "ocean", "mineral", "rock", "snow", "skin", "fur", "eye", "tile"]

    selected = cmds.ls(sl=True, dag=True, type='surfaceShape', ni=True)
    if selected:
        objects = selected
    else:
        objects = cmds.ls(dag=True, type='surfaceShape', ni=True)
    for s in objects:
        if cmds.attributeQuery('MaterialSet', n=s, ex=True):
            if cmds.getAttr('%s.MaterialSet'%s) in materialList:
                materialName = cmds.getAttr('%s.MaterialSet' % s)
                cmds.sets(s, forceElement=materialName + "SG")
            else:
                cmds.sets(s, forceElement="error" + "SG")
                # continue
        else:
            cmds.sets(s, forceElement="error" + "SG")





def createMaterialSet():
    '''
    @ author : daeseok.chae in Dexter RND
    @ date   : 2018.05.04
    @ comment: Auto set for material naming convention
    @ detail
        2018.05.04 - Material Base in '/dexter/Cache_DATA/ASSET/1.asset/5.shader'
    :return: None
    '''
    if cmds.objExists('MaterialSet'):
        return

    cmds.select(cl = True)

    for material in materialList:
        cmds.sets(name=material)
    cmds.sets(materialList, name="MaterialSet")

def autoAddMaterialSet():
    '''
    @ author : moonseok.chae in Dexter Asset
    @ date   : 2018.05.04
    @ comment: Auto set for material naming convention
    :return: None
    '''
    for material in materialList:
        try:
            cmds.sets('*M%s*_PLY' % material, fe=material)
        except:
            pass

def removeMaterialSet():
    '''
    @ author : moonseok.chae in Dexter Asset
    @ date   : 2018.05.04
    @ comment: All Remove for material naming convention
    :return: None
    '''
    for material in materialList:
        try:
            cmds.sets('*M%s*_PLY' % material, rm=material)
        except:
            pass
