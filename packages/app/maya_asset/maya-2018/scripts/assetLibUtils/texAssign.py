from __future__ import print_function
import os, glob
import maya.cmds as cmds

import DXUSD.Utils as utl
import DXUSD_MAYA.Message as msg

class AssignShaderlegacy:
    def __init__(self, node, texdir, scene):
        self.pubscene = scene
        self.textureInfo = list()
        self.layeredShaderList = list()

        if cmds.attributeQuery('rman__riattr__user_txLayerName', n=node, exists=True):
            txLayerName = cmds.getAttr('%s.rman__riattr__user_txLayerName' % node)
        elif cmds.attributeQuery('txLayerName', n=node, exists=True):
            txLayerName = cmds.getAttr('%s.txLayerName' % node)
        else:
            msg.debug('>>> %s ERROR:' % node, 'no txLayerName!')
            return

        sname = txLayerName

        if cmds.attributeQuery('rman__riattr__user_txAssetName', n=node, exists=True):
            basepath = cmds.getAttr('%s.rman__riattr__user_txAssetName' % node)
        elif cmds.attributeQuery('txBasePath', n=node, exists=True):
            basepath = cmds.getAttr('%s.txBasePath' % node)
        else:
            msg.debug('>>> %s ERROR:' % node, 'no txAssetName!')
            return

        if not os.path.exists(texdir):
            return

        self.checkLayeredShader(texdir, txLayerName)
        self.doit(node, texdir, txLayerName, sname)

    def checkLayeredShader(self, texdir, txLayerName):
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
            'diffG': {'outColor': 'illumColor'},
            'specG': {'outAlpha': 'reflectionGlossiness'},
            'bump': {'outColor': 'bumpMap'},
        }

        files = []
        for filename in os.listdir(texdir):
            if '%s_diffC' % txLayerName in filename:
                fullname = os.path.join(texdir, filename)
                files.append(fullname)
        if not files:
            msg.debug('>>> %s ERROR:' % node, 'no texFiles!')
            return

        SG = cmds.ls(sname + '_SG', type='shadingEngine')
        if not SG:
            shd = cmds.shadingNode('VRayMtl', asShader=True, name='%s_SHD' % sname)
            cmds.setAttr('%s.bumpMapType' % shd, 1)
            SG = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sname + '_SG')
            cmds.connectAttr('%s.outColor' % shd, '%s.surfaceShader' % SG)
            filename = files[0]

            for k, v in channelData.items():
                output = v.keys()[0]
                shdinput = v[output]
                filenode = self.channelConnect(node, filename, data[txLayerName],texdir,txLayerName, channel=k)
                if filenode:
                    msg.debug('>>> %s texFile:' % node, filename)
                    msg.debug(' %s connect:' % node, '%s.%s' % (filenode, output), '>>> %s.%s' % (shd, shdinput))
                    cmds.connectAttr('%s.%s' % (filenode, output), '%s.%s' % (shd, shdinput))
                    if not data in self.textureInfo:
                        self.textureInfo.append(data)
            cmds.sets(node, forceElement=SG)
        else:
            cmds.sets(node, forceElement=SG[0])

    def channelConnect(self, node, filename, data, texdir,txLayerName, channel='diffC'):
        if 'diffC' != channel:
            filename = filename.replace('diffC', channel)

        if not os.path.isfile(filename):
            msg.debug('>>> %s ERROR:' % node, 'no texFiles! %s' % filename)
            return None

        filenode = cmds.shadingNode('file', asTexture=True)
        # cmds.setAttr('%s.fileTextureName' % filenode, utl.GetRelPath(self.pubscene, filename), type='string')
        cmds.setAttr('%s.fileTextureName' % filenode, filename, type='string')

        manifold = cmds.shadingNode('place2dTexture', asUtility=True)
        cmds.connectAttr('%s.outUvFilterSize' % manifold, '%s.uvFilterSize' % filenode)

        # multiUV
        if cmds.attributeQuery('rman__riattr__user_txmultiUV', n=node, exists=True):
            if cmds.getAttr('%s.rman__riattr__user_txmultiUV' % node):
                files = glob.glob('%s/%s_%s*' % (texdir,txLayerName,channel))
                if len(files) > 1:
                    # cmds.setAttr('%s.fileTextureName' % filenode, utl.GetRelPath(self.pubscene, files[0]), type='string')
                    cmds.setAttr('%s.fileTextureName' % filenode, files[0], type='string')
                    cmds.setAttr('%s.uvTilingMode' % filenode, 3)

        if channel in ['diffC', 'diffG', 'specG']:
            cmds.setAttr('%s.colorSpace' % filenode, 'Utility - sRGB - Texture', type="string")
        elif channel in ['specR', 'norm', 'bump', 'Alpha']:
            cmds.setAttr('%s.colorSpace' % filenode,'Utility - Raw', type="string")
            if channel == 'specR':
                cmds.setAttr('%s.alphaIsLuminance' % filenode, 1)

        return filenode


class AssignShader:
    def __init__(self, node, asset, assetdir, branch, pubscene):
        self.pubscene = pubscene
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

    def checkLayeredShader(self, texdir, txLayerName):
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
            shd = cmds.shadingNode('VRayMtl', asShader=True, name='%s_SHD' % sname)
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
