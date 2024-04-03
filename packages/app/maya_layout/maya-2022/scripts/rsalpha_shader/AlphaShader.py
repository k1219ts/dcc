# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
from maya import cmds
from maya import mel

def main():
    plug = ['redshift4maya', 'SLiB', 'backstageMenu']
    for p in plug:
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)
            cmds.pluginInfo(p, e=True, autoload=True)

    imagefile = cmds.ls(selection=True, type='file')
    effect = cmds.ls(selection=True, type='lambert')
    lambert = cmds.ls(selection=True, type='shadingEngine')

    texturename = cmds.getAttr(imagefile[0] + '.fileTextureName')

    rssprite = cmds.shadingNode('RedshiftSprite', asShader=True)
    rsshader = cmds.sets(name=rssprite, renderable=True, noSurfaceShader=True, empty=True)
    # rsshader = cmds.createNode('shadingEngine')
    cmds.connectAttr(rssprite + '.outColor', rsshader + '.surfaceShader', f=True)
    cmds.setAttr(rssprite + '.useFrameExtension', 1)
    cmds.setAttr(rssprite + '.mode', 1)
    cmds.setAttr(rssprite + '.tex0', texturename, type='string')

    cmds.connectAttr(effect[0] + '.outColor', rssprite + '.input', f=True)
    cmds.connectAttr(rssprite + '.outColor', lambert[0] + '.rsSurfaceShader', f=True)
    cmds.connectAttr(imagefile[0] + '.frameExtension', rssprite + '.frameExtension', f=True)
    cmds.connectAttr(imagefile[0] + '.frameOffset', rssprite + '.frameOffset', f=True)