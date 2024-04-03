#coding:utf-8
from __future__ import print_function
import maya.mel as mel
import maya.cmds as cmds
from functools import partial
from DXUSD.Structures import Arguments
import DXUSD.Utils as utl
import os
import glob

class PreviewAssign:
    def __init__(self, type='proxy'):
        self.type = type
        sceneFile = cmds.file(q=True, sn=True)

        arg = Arguments()
        arg.D.SetDecode(os.path.dirname(sceneFile),'SHOW')
        root = arg.D.ROOTS

        for shape in cmds.ls(sl=1, dag=1, type='surfaceShape', l=1):

            if not cmds.attributeQuery('txBasePath', n=shape, exists=True):
                continue
            if not cmds.attributeQuery('txLayerName', n=shape, exists=True):
                continue

            basepath = cmds.getAttr('%s.txBasePath' % shape)
            txlayer = cmds.getAttr('%s.txLayerName' % shape)

            if cmds.attributeQuery('txVersion', n=shape, exists=True):
                version = cmds.getAttr('%s.txVersion' % shape)
            else:
                version = self.GetTexVer(root, basepath)

            shadername = txlayer
            if arg.branch:
                shadername = arg.branch + '_' + txlayer

            if self.type == 'image':
                texuredir = os.path.join(root, basepath, 'images', version)

            else:
                shadername = 'proxy_' + shadername
                texuredir = os.path.join(root, basepath, 'proxy', version)

            filepath = self.GetImagePath(texuredir , txlayer)
            node = cmds.listRelatives(shape, p=True, f=True)[0]
            if filepath:
                self.AssignSG(txlayer,shadername, node, filepath)


    def GetTexVer(self, root, basepath):
        texVer = ''
        texusdfile = os.path.join(root, basepath, 'tex', 'tex.usd')
        texdir = os.path.join(root, basepath, 'tex')

        if os.path.exists(texusdfile):
            layer = utl.AsLayer(texusdfile)
            texVer = layer.subLayerPaths[0].split('/')[1]

        if not texVer:
            verList = []
            for file in os.listdir(os.path.join(texdir)):
                if 'v' in file:
                    verList.append(file)
            verList.sort()
            texVer = verList[-1]

        return texVer

    def GetImagePath(self, imgdir, txlayer):
        if os.path.exists(imgdir):
            for fileName in os.listdir(imgdir):
                if not "diffC" in fileName:
                    continue
                if not '_diffC_' in fileName:
                    if txlayer in fileName:
                        filepath = os.path.join(imgdir, fileName)
                        if os.path.exists(filepath):
                            return filepath

    def AssignSG(self, txlayer, shaderName, node, txFile):
        shading_group = cmds.ls(shaderName + '_SG', type='shadingEngine')
        if not shading_group:
            shader = cmds.shadingNode('lambert', asShader=True, name='%s_SHD' % shaderName)
            filenode = cmds.shadingNode('file', asTexture=True)
            shading_group = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shaderName + '_SG')
            cmds.connectAttr('%s.outColor' % shader, '%s.surfaceShader' % shading_group)
            cmds.connectAttr('%s.outColor' % filenode, '%s.color' % shader)
            cmds.setAttr('%s.fileTextureName' % filenode, txFile, type='string')
            os.path.basename(txFile).split('.')

            shape = cmds.listRelatives(node, c=True, f=True)[0]
            if cmds.attributeQuery('txmultiUV', n=shape, exists=True):
                if cmds.getAttr('%s.txmultiUV' % shape) == 1:
                    files = glob.glob('%s/%s_diffC*' %(os.path.dirname(txFile), txlayer))
                    if len(files) > 1:
                        cmds.setAttr('%s.fileTextureName' % filenode, files[1], type='string')
                        cmds.setAttr('%s.uvTilingMode' % filenode, 3)
                        mel.eval("generateUvTilePreview %s;" % filenode)
            cmds.sets(node, forceElement=shading_group)
        else:
            cmds.sets(node, forceElement=shading_group[0])


class Main(object):
    def __init__(self, name='BasicUI', title='Preview Texture'):
        self.title = title
        self.name = name
        self.__build__()

    def __build__(self):
        if cmds.window(self.name, exists=True): cmds.deleteUI(self.name)

        cmds.window(self.name, title=self.title)
        self.mainColumn = cmds.columnLayout(adjustableColumn=True)
        self.label = cmds.text(label='Select Texture Type')

        self.button = cmds.button(label='Proxy Texture', command=partial(self.ProxyShader, 'className_#'))
        self.button2 = cmds.button(label='Render Texture', command=partial(self.ImageShader, 'className_#'))

        cmds.showWindow(self.name)
        cmds.window(self.name, edit=True, widthHeight=(225, 75))

    def ProxyShader(self,*args):
        PreviewAssign(type='proxy')

    def ImageShader(self,*args):
        PreviewAssign(type='image')

