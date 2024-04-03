#coding:utf-8
from __future__ import print_function

from DXUSD.Structures import Arguments

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
from pxr import Usd, Sdf
import PUtils as putl
import TUtils as tutl
import os, sys

scriptsDir = os.path.dirname(__file__)
sys.path.append(scriptsDir+'/..')
import dbutils

class APackage(Arguments):
    def __init__(self, **kwargs):

        self.dst= ''
        self.usdpath = ''
        self.dataType = ''
        self.categoryTask =''
        self.dataFormat = ''

        Arguments.__init__(self, **kwargs)

    def Treat(self):
        return var.SUCCESS


class Package(object):
    ARGCLASS = APackage
    def __init__(self, arg):

        self.__name__ = self.__class__.__name__
        if arg.__name__ == 'Arguments':
            self.arg = self.ARGCLASS(**arg)
        elif arg.__name__ == self.ARGCLASS.__name__:
            self.arg = arg
        else:
            msg.error('__init__@Package : Arguments Failed (%s)'%self.__name__)

        msg.debug()
        msg.debug('#'*70)
        msg.debug('#'*70)
        msg.debug('')
        msg.debug('\t\t\t[ Start %s ]'% self.__name__)
        msg.debug()
        msg.debug('#'*70)
        msg.debug('#'*70)
        msg.debug()

        utl.CheckRes('[ Start Treating ]', self.arg.Treat, 'Treating', 0)

        msg.debug()
        msg.debug('#'*70)
        msg.debug(self.arg)

        msg.debug()
        msg.debug('#'*70)
        utl.CheckRes('[ Start Packaging ]', self.Packaging, 'Packaging', 0)


        msg.debug()
        msg.debug('#'*70)
        utl.CheckRes('[ Start Completing ]', self.Completing, 'Completing', 0)


    def Packaging(self):
        msg.warning('Packaging@Package : Not overrided.', dev=True)

        return var.SUCCESS

    def Copying(self):
        return var.SUCCESS

    def Open(self, mayafile):
        print('##### Open Maya file #####', mayafile)
        import maya.cmds as cmds

        msg.debug('Maya File Open : %s' % mayafile)
        cmds.file(new=True, force=True)
        cmds.file(mayafile, open=True)

        return var.SUCCESS

    def ImportUsd(self, usdpath):
        print('##### Import pxrUsdProxy node #####')
        import maya.cmds as cmds
        refNode = cmds.createNode('pxrUsdProxyShape')
        refNode = cmds.listRelatives(refNode, p=True, f=True)[0]
        refNode = cmds.rename(refNode, usdpath.split('/')[-1].split('.')[0])
        cmds.setAttr('%s.filePath' % refNode, self.arg.usdpath, type='string')
        return var.SUCCESS

    def UsdToMesh(self):
        print('##### pxrUsdProxy node => Mesh  #####')
        import maya.cmds as cmds
        import extra

        nodes = cmds.ls(dag=True, type='pxrUsdProxyShape', l=1)
        if nodes:
            if self.arg.dataType == 'Locator':
                extra.USDProxyShapeToLocator(nodes).doit()
            else:
                extra.USDProxyShapeToMesh(nodes).doit()
                if cmds.objExists('|reference_sc_tmp'):
                    cmds.delete('|reference_sc_tmp')

        if cmds.ls('reference_sc_tmp_*', type='transform'):
            for loc in cmds.ls('reference_sc_tmp_*', type='transform'):
                new = loc.replace('reference_sc_tmp_', '')
                cmds.rename(loc, new)

        return var.SUCCESS


    def CopyCache(self, rigfile, scenedir, nslyr):
        print('##### Copy Rig Cachefile #####')

        import maya.cmds as cmds
        cachefileNodes = cmds.ls(type='cacheFile')
        if cachefileNodes:
            for node in cachefileNodes:
                cachepath = cmds.getAttr('%s.cachePath' % node)
                newcachepath = cachepath.replace(os.path.join(os.path.dirname(rigfile), nslyr), os.path.join(scenedir,nslyr))
                # print('newcachepath:',newcachepath)
                for sc in os.listdir(cachepath):
                    putl.MakeDir(newcachepath)
                    scpath = os.path.join(cachepath, sc)
                    targetpath = os.path.join(newcachepath, sc)
                    if not os.path.exists(targetpath):
                        putl.CopyFile(scpath, newcachepath)
                        dbutils.updatePackage(scpath, newcachepath, {'pkgType': 'mb'})

        return var.SUCCESS



    def AssignShader(self):
        print('##### Assign Shader #####')
        import maya.cmds as cmds

        textureInfo = list()
        layeredShaderList = list()

        for node in cmds.ls(dag=1, type="surfaceShape"):
            if self.arg.has_key('asset'):
                asset = self.arg.asset
                branch = self.arg.branch

            else:
                asset = self.arg.asset
                branch = ''

            tutl.AssignShader(node=node,
                              dstdir = self.arg.dstdir,
                              textureInfo = textureInfo ,
                              layeredShaderList = layeredShaderList,
                              asset= asset,
                              branch= branch ,
                              dataType= self.arg.dataType,
                              )

        tutl.WriteInfo(self.arg.dstdir, textureInfo,layeredShaderList)
        return var.SUCCESS


    def missingAssign(self,nodes):
        print('##### Find missing attributes #####')
        import maya.cmds as cmds
        for node in cmds.ls(type='surfaceShape', dag=1, l=1):
            if not 'Proxy' in node:
                if cmds.attributeQuery('txLayerName', n=node, exists=True):
                    pass
                    txLayerName = cmds.getAttr('%s.txLayerName' % node)
                elif 'HdImaging' in node:
                    pass
                else:
                    if cmds.objExists(node):
                        nodes.append(node)


    def Save(self, dstfile):
        print('##### Save file #####')
        import maya.cmds as cmds

        putl.MakeDir(os.path.dirname(dstfile))
        if self.arg.dataFormat == 'mb(abc)':
            pass
        else:
            cmds.file(rename=dstfile)
            cmds.file(save=True, type="mayaBinary")

        return var.SUCCESS


    def SaveGroom(self, dstfile):
        print('##### Save Groom file #####')
        import maya.cmds as cmds

        utl.MakeDir(os.path.dirname(dstfile))

        deletelist = []
        for node in cmds.ls(type='transform', l=True):
            if not '_groom_GRP' in node:
                deletelist.append(node)
            if 'Viewer' in node or 'viewer' in node:
                deletelist.append(node)

        for node in cmds.ls(type='objectSet', l=True):
            if 'MaterialSet' in node or 'ZN_ExportSet' in node:
                deletelist.append(node)

        cmds.delete(deletelist)

        node = cmds.ls('*_groom_GRP')[0]
        # node = '%s_groom_GRP' % self.arg.asset
        # if self.arg.branch:
        #     node = '%s_%s_groom_GRP' % (self.arg.asset,self.arg.branch)
        print('>>>>>>>>>>>>>>>>>node:',node)

        cmds.select(node)
        cmds.file(rename=dstfile)
        cmds.file(f=True, es=True, type="mayaBinary")

        return var.SUCCESS


    def Completing(self):
        return var.SUCCESS




# ------------------------------------------------------------------------------
# Rig
# ------------------------------------------------------------------------------

class ARigPack(APackage):
    def __init__(self, **kwargs):
        # input argument
        # self.type = ''
        # self.usdpath = ''

        # treat compute
        self.scenedir = ''
        self.assetname = ''
        self.mayafiles = list()
        self.dstfiles = list()

        # initialize
        APackage.__init__(self, **kwargs)

    def Treat(self):
        print('self.usdpath:', self.usdpath)
        self.dstdir = os.path.join(self.dst, self.asset)
        if self.categoryTask == 'shot':
            self.dstdir = os.path.join(self.dst, '_3d', 'asset', self.asset)

        self.assetname = self.asset
        if self.branch:
            self.assetname = self.branch

        self.scenedir = os.path.join(self.dstdir, 'scenes')
        if self.branch:
            self.scenedir = os.path.join(self.dstdir, 'scenes', 'branch',self.branch)

        print('>>> self.scenedir:', self.scenedir)
        rigfile = putl.GetRigSceneFile(self.usdpath)
        if not rigfile:
            return

        print('os.path.basename(rigfile):',os.path.basename(rigfile))
        nslyr = os.path.basename(rigfile).split('.')[0]
        print('nslyr:', nslyr)
        dir = os.path.dirname(rigfile)
        for f in os.listdir(dir):
            rigfilepath = os.path.join(dir, f)
            if os.path.isfile(rigfilepath):
                if nslyr in f or nslyr + '_low' in f or nslyr + '_mid' in f:
                    if not 'json' in f:
                        print('nslyr file:', f)
                        self.mayafiles.append(rigfilepath)
                        dstfile = os.path.join(self.scenedir, os.path.basename(f))
                        # if self.dataType == 'Locator':
                        #     name = os.path.basename(f).split('.')
                        #     dstfile =os.path.join(self.scenedir, '%s_locator.mb' % name )

                        self.dstfiles.append(dstfile)
        print('self.dstfiles:',self.dstfiles)
        return var.SUCCESS

class RigPack(Package):
    ARGCLASS = ARigPack

    def Packaging(self):
        for i in range(len(self.arg.dstfiles)):
            mayafile = self.arg.mayafiles[i]
            print('##### Rig Package #####',mayafile )
            self.Open(mayafile)
            self.UsdToMesh()
            self.CopyCache(mayafile, self.arg.scenedir, os.path.basename(mayafile).split('.')[0])
            self.AssignShader()
            self.Save(self.arg.dstfiles[i])
            dbutils.updatePackage(mayafile, self.arg.dstfiles[i], {'pkgType': 'mb'})


# ------------------------------------------------------------------------------
# Groom
# ------------------------------------------------------------------------------
class AGroomPack(APackage):
    def __init__(self, **kwargs):
        # input argument
        # self.usdpath = ''

        # treat compute
        self.scenedir = ''
        self.assetname = ''
        self.mayafile = ''
        self.dstfile = ''

        # initialize
        APackage.__init__(self, **kwargs)

    def Treat(self):
        self.dstdir = os.path.join(self.dst, self.asset)
        if self.categoryTask == 'shot':
            self.dstdir = os.path.join(self.dst, '_3d', 'asset', self.asset)

        self.assetname = self.asset
        self.scenedir = os.path.join(self.dstdir, 'scenes')

        if self.branch:
            self.assetname = self.branch
            self.scenedir = os.path.join(self.dstdir, 'scenes', 'branch',self.branch)

        self.mayafile = putl.GetGroomSceneFile(self.usdpath)
        print('self.mayafile:',self.mayafile)
        self.dstfile = os.path.join(self.scenedir, os.path.basename(self.mayafile))

        return var.SUCCESS


class GroomPack(Package):
    ARGCLASS = AGroomPack
    def Packaging(self):
        print('##### Groom Package #####')
        print('self.arg.mayafile:',self.arg.mayafile)
        self.Open(self.arg.mayafile)
        self.SaveGroom(self.arg.dstfile)
        dbutils.updatePackage(self.arg.mayafile, self.arg.dstfile, {'pkgType': 'mb'})


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class AModelPack(APackage):
    def __init__(self, **kwargs):
        # input argument
        # self.dst= ''
        # self.usdpath = ''
        # self.dataType = ''
        # self.categoryTask =''

        # treat compute
        self.scenedir = ''
        self.assetname = ''
        self.dstfile = ''
        self.modelType = ''

        # initialize
        APackage.__init__(self, **kwargs)

    def Treat(self):
        if self.has_key('asset'):

            self.assetname = self.branch if self.branch else self.asset
            self.dstdir = os.path.join(self.dst, self.asset)
            self.scenedir = os.path.join(self.dstdir, 'scenes')

            modelVer = putl.GetModelDir(self.usdpath) #/show/cdh1/_3d/asset/mundosuk/model/v004
            modelVer = modelVer.split('/')[-1]
            if self.branch:
                self.scenedir = os.path.join(self.dst, self.asset, 'scenes', 'branch', self.branch)
            self.dstfile = os.path.join(self.scenedir, '%s_model_%s.mb' % (self.assetname, modelVer))

            print('self.dstfile:',self.dstfile)


        # elif self.has_key('shot'):
        #     shotname = '%s_%s' % (self.seq, self.shot)
        #     self.assetname = shotname
        #     self.dstdir = os.path.join(self.dst, self.assetname)
        #     self.scenedir = os.path.join(self.dstdir, 'scenes')
        #     self.dstfile =os.path.join(self.scenedir, '%s_layout.mb' % self.assetname)
        #     print('self.dstfile:',self.dstfile)


        return var.SUCCESS


class ModelPack(Package):
    ARGCLASS = AModelPack

    def Packaging(self):
        print('##### Model Package #####')

        self.ImportUsd(self.arg.usdpath)
        self.UsdToMesh()
        self.AssignShader()
        self.Save(self.arg.dstfile)
        dbutils.updatePackage(self.arg.usdpath, self.arg.dstfile, {'pkgType': 'mb'})


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class ALidarPack(APackage):
    def __init__(self, **kwargs):
        # input argument
        # self.dst= ''
        # self.usdpath = ''
        # self.dataType = ''
        # self.categoryTask =''

        # treat compute
        self.scenedir = ''
        self.assetname = ''
        self.dstfile = ''
        self.modelType = ''

        # initialize
        APackage.__init__(self, **kwargs)

    def Treat(self):
        if self.has_key('asset'):

            self.assetname = self.branch if self.branch else self.asset
            self.dstdir = os.path.join(self.dst, self.asset)
            self.scenedir = os.path.join(self.dstdir, 'scenes')

            modelVer = putl.GetLidarDir(self.usdpath)
            modelVer = modelVer.split('/')[-1]
            if self.branch:
                self.scenedir = os.path.join(self.dst, self.asset, 'scenes', 'branch', self.branch)
            self.dstfile = os.path.join(self.scenedir, '%s_lidar_%s.mb' % (self.assetname, modelVer))

            print('self.dstfile:',self.dstfile)


        # elif self.has_key('shot'):
        #     shotname = '%s_%s' % (self.seq, self.shot)
        #     self.assetname = shotname
        #     self.dstdir = os.path.join(self.dst, self.assetname)
        #     self.scenedir = os.path.join(self.dstdir, 'scenes')
        #     self.dstfile =os.path.join(self.scenedir, '%s_layout.mb' % self.assetname)
        #     print('self.dstfile:',self.dstfile)

        return var.SUCCESS


class LidarPack(Package):
    ARGCLASS = ALidarPack

    def Packaging(self):
        print('##### Lidar Package #####')

        self.ImportUsd(self.arg.usdpath)
        self.UsdToMesh()
        self.Save(self.arg.dstfile)
        dbutils.updatePackage(self.arg.usdpath, self.arg.dstfile, {'pkgType': 'mb'})


# ------------------------------------------------------------------------------
# Texture
# ------------------------------------------------------------------------------
class ATexturePack(APackage):
    def __init__(self, **kwargs):
        # # input argument
        # self.usdpath = ''
        # self.categoryTask =''

        # treat compute
        self.texturedir  = ''
        self.textureType = ''

        # initialize
        APackage.__init__(self, **kwargs)

    def Treat(self):

        if self.has_key('asset'):
            self.assetname = self.branch if self.branch else self.asset
            self.dstdir = os.path.join(self.dst, self.asset)

        elif self.has_key('shot'):
            shotname = '%s_%s' % (self.seq, self.shot)
            self.assetname = shotname
            self.dstdir = os.path.join(self.dst, self.assetname)

        self.texturedir = os.path.join(self.dstdir, 'sourceimages')

        if self.branch:
            self.texturedir = os.path.join(self.dstdir, 'sourceimages', self.branch)


        return var.SUCCESS


class TexturePack(Package):
    ARGCLASS = ATexturePack

    def Packaging(self):
        print('##### Texture Package #####')
        filepathList = tutl.GetTexfileList(self.arg.usdpath, self.arg.texturedir, task='model', texFmt=self.arg.texFmt).doit()
        texturedir = self.arg.texturedir

        if filepathList:
            for f in filepathList:
                if not '(' in f or not ')' in f:
                    if self.arg.branch or not '/branch' in f or self.arg.dataType == 'Reference':
                        pass

                    else:
                        # impoted asset
                        bArg = Arguments()
                        bArg.D.SetDecode(f)
                        assetname = bArg.branch if bArg.branch else bArg.asset
                        branchTexuredir = os.path.join(self.arg.dstdir, 'sourceimages', assetname)
                        texturedir = branchTexuredir

                    # print('texturedir:', self.arg.texturedir)

                    # if self.arg.textureType == 'prevtex':
                    #     if 'diffC' in f:
                    #         if not 'ZN' in f:
                    #             putl.CopyFile(f, texturedir)
                    # else:
                    putl.CopyFile(f, texturedir)
                    dbutils.updatePackage(f, texturedir+'/'+os.path.basename(f), {'pkgType': 'mb'})


class ATextureRefPack(ATexturePack):
    def __init__(self, **kwargs):
        ATexturePack.__init__(self, **kwargs)

    def Treat(self):
        assetname = ''
        if self.has_key('asset'):
            assetname = self.branch if self.branch else self.asset

        elif self.has_key('shot'):
            shotname = '%s_%s' % (self.arg.seq, self.arg.shot)
            assetname = shotname

        self.dstdir = self.dst
        self.texturedir = os.path.join(self.dstdir, 'sourceimages', assetname)
        print('self.texturedir:TextureRefPack:',self.texturedir)

        return var.SUCCESS

class TextureRefPack(TexturePack):
    ARGCLASS = ATextureRefPack


# ------------------------------------------------------------------------------
# Model-Reference
# ------------------------------------------------------------------------------
class AModelRefPack(AModelPack):
    def __init__(self, **kwargs):
        # initialize
        AModelPack.__init__(self, **kwargs)

    def Treat(self):
        if self.has_key('asset'):
            self.assetname = self.branch if self.branch else self.asset

        elif self.has_key('shot'):
            shotname = '%s_%s' % (self.seq, self.shot)
            self.assetname = shotname


        self.dstdir = self.dst

        modelVer = putl.GetModelDir(self.usdpath) #/show/cdh1/_3d/asset/mundosuk/model/v004
        modelVer = modelVer.split('/')[-1]
        self.scenedir = os.path.join(self.dstdir, 'scenes','branch')
        self.dstfile = os.path.join(self.scenedir, '%s_model_%s.mb' % (self.assetname, modelVer))
        print('>>>>>self.dstfile:', self.dstfile)

        # write ref info
        infofile = os.path.join(self.dstdir, 'reference_info.json')
        if os.path.exists(infofile):
            data = putl.jsonRead(infofile)
        else:
            data = {}

        if not self.usdpath in  data["usdpath"]:
            data["usdpath"].append(self.usdpath)
            putl.jsonDump(infofile, data)

        return var.SUCCESS

class ModelRefPack(ModelPack):
    ARGCLASS = AModelRefPack
