#coding:utf-8
from __future__ import print_function
import os, re, glob
import maya.cmds as cmds

import DXRulebook.Interface as rb

from DXUSD.Tweakers.Tweaker import Tweaker, ATweaker
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD_MAYA.Utils as mutl
import DXUSD_MAYA.Message as msg


class ASaveMayaScene(ATweaker):
    def __init__(self, **kwargs):
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        return var.SUCCESS

class SaveMayaScene(Tweaker):
    ARGCLASS = ASaveMayaScene

    def DoIt(self):
        if not self.arg.has_attr('pubscene'):
            return var.SUCCESS

        branch = ''
        if 'branch' in self.arg:
            branch = self.arg.branch

        cmds.undoInfo(openChunk=True, cn='assignSG')

        # Assign Maya Shaders
        for node in cmds.ls(dag=1, type="surfaceShape"):
            AssignShader(node = node,
                         assetdir = self.arg.D.ASSET,
                         pubscene = self.arg.pubscene,
                         asset = self.arg.asset,
                         branch = branch)

        # Save as Scene
        cmds.select(self.arg.nodes)

        if not os.path.exists(os.path.dirname(self.arg.pubscene)):
            os.mkdir(os.path.dirname(self.arg.pubscene))

        cmds.file(self.arg.pubscene, pr=True, typ='mayaBinary', options='v=0;',
                  es=True, f=True)
        msg.debug('> Save As maya scene\t:', self.arg.pubscene)

        cmds.select(cl=True)

        # Undo Assign Maya Shaders
        cmds.undoInfo(closeChunk=True, cn='assignSG')
        try:    cmds.undo()
        except: msg.debug('did not Undo!')

        return var.SUCCESS



class AssignShader:
    def __init__(self, **kwargs):
        node = kwargs['node']
        assetdir = kwargs['assetdir']
        self.pubscene = kwargs['pubscene']
        asset = kwargs['asset']
        branch = kwargs['branch']
        self.textureInfo = list()
        self.layeredShaderList = list()

        if cmds.attributeQuery('txLayerName', n=node, exists=True):
            txLayerName = cmds.getAttr('%s.txLayerName' % node)
        else:
            msg.debug('>>> %s ERROR:' % node, 'no txLayerName!')
            return

        if cmds.attributeQuery('txBasePath', n=node, exists=True):
            basepath = cmds.getAttr('%s.txBasePath' % node)
        else:
            msg.debug('>>> %s ERROR:' % node, 'no txBasePath!')
            return

        if cmds.attributeQuery('txVersion', n=node, exists=True):
            ver = cmds.getAttr('%s.txVersion' % node)
        else:
            msg.debug('>>> %s ERROR:' % node, 'no txVersion!')
            return

        texdir = ''
        if branch:
            texdir = os.path.join(assetdir, 'branch', branch, 'texture/images', ver)
            sname = '%s_%s' % (branch, txLayerName)
        elif '/branch/' in basepath:
            branch = basepath.split('/')[3]
            texdir = os.path.join(assetdir, 'branch', branch, 'texture/images', ver)
            sname = '%s_%s' % (branch, txLayerName)
        else:
            assetname = branch if branch else asset
            txBasePath = cmds.getAttr('%s.txBasePath' % node)
            if not txBasePath:
                msg.debug('>>> %s ERROR:' % node, 'no txBasePath!')
                return
            splitpath = txBasePath.split('/')
            an = splitpath[1]
            if len(splitpath)>3:
                an = splitpath[-2]

            if assetname == an or not 'branch' in txBasePath:
                texdir = os.path.join(assetdir, 'texture/images', ver)
                sname = txLayerName
            else:
                texdir = os.path.join(assetdir, 'branch', an, 'texture/images', ver)
                sname = '%s_%s' % (an, txLayerName)

        msg.debug('> TexDir\t:', texdir)
        if not os.path.exists(texdir):
            return

        self.checkLayeredShader(texdir, txLayerName)
        self.doit(node, texdir, txLayerName, sname)

    def checkLayeredShader(self, texdir,txLayerName ):
        for filename in os.listdir(texdir):
            if 'diffC' in filename:
                if not txLayerName in self.layeredShaderList:
                    self.layeredShaderList.append(txLayerName)

    def doit(self, node, texdir, txLayerName, sname):
        data = {txLayerName: {'attrs' : {'txmultiUV' : 0},
                              'channels' : []}
        }

        channelData = {
            'diffC': {'outColor': 'color'},
            'specG': {'outColor': 'specularColor'},
            'specR': {'outAlpha': 'eccentricity'},
            'norm': {'outNormal': 'normalCamera'},
            'Alpha': {'outColor': 'transparency'},
        }

        files = []
        for filename in os.listdir(texdir):
            if '%s_diffC' % txLayerName in filename:
                fullname = os.path.join(texdir, filename)
                files.append(fullname)
        if not files:
            return

        SG = cmds.ls(sname + '_SG', type='shadingEngine')
        if not SG:
            shd = cmds.shadingNode('blinn', asShader=True, name='%s_SHD' % sname)
            SG = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sname + '_SG')
            cmds.connectAttr('%s.outColor' % shd, '%s.surfaceShader' % SG)
            filename = files[0]

            for k, v in channelData.items():
                output = v.keys()[0]
                shdinput = v[output]
                filenode = self.channelConnect(node, filename, data[txLayerName],texdir,txLayerName, channel=k)
                if filenode:
                    msg.debug('>>> %s texFile:' % node, filename)
                    cmds.connectAttr('%s.%s' % (filenode, output), '%s.%s' % (shd, shdinput))
                    if not data in self.textureInfo:
                        self.textureInfo.append(data)
            cmds.sets(node, forceElement=SG)
        else:
            cmds.sets(node, forceElement=SG[0])

    def channelConnect(self, node, filename, data, texdir,txLayerName, channel='diffC'):
        if 'diffC' != channel:
            filename = filename.replace('diffC', channel)
            # if 'specG' != channel:
            #     filename = filename.replace('jpg', 'tif')

        filenode = cmds.shadingNode('file', asTexture=True)
        cmds.setAttr('%s.fileTextureName' % filenode, utl.GetRelPath(self.pubscene, filename), type='string')

        manifold = cmds.shadingNode('place2dTexture', asUtility=True)
        cmds.connectAttr('%s.outUvFilterSize' % manifold, '%s.uvFilterSize' % filenode)

        # multiUV
        if cmds.attributeQuery('txmultiUV', n=node, exists=True):
            if cmds.getAttr('%s.txmultiUV' % node):
                files = glob.glob('%s/%s_%s*' % (texdir,txLayerName,channel))
                if len(files) > 1:
                    cmds.setAttr('%s.fileTextureName' % filenode, utl.GetRelPath(self.pubscene, files[0]), type='string')
                    cmds.setAttr('%s.uvTilingMode' % filenode, 3)

        if channel == 'diffC' or channel == 'specG':
            cmds.setAttr('%s.colorSpace' % filenode, 'Utility - sRGB - Texture', type="string")
        elif channel == 'specR' or channel == 'norm' or channel == 'Alpha':
            cmds.setAttr('%s.colorSpace' % filenode,'Utility - Raw', type="string")
            if channel == 'specR':
                cmds.setAttr('%s.alphaIsLuminance' % filenode, 1)

            # norm
            if channel == 'norm':
                bumpnode = cmds.shadingNode('bump2d', asUtility=True)
                cmds.setAttr('%s.bumpInterp' % bumpnode, 1)
                cmds.connectAttr('%s.outAlpha' % filenode, '%s.bumpValue' % bumpnode)
                filenode = bumpnode

        return filenode
