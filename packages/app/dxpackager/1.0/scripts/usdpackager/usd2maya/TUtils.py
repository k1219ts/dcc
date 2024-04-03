#coding:utf-8
from __future__ import print_function
import glob
import os
import subprocess
import argparse

if not __name__ == '__main__':
    from pxr import Usd, UsdGeom
    import DXUSD.Message as msg
    import DXUSD.Utils as utl
    from DXUSD.Structures import Arguments
    import DXUSD.Vars as var
    import PUtils as putl

class GetTexfileList:
    def __init__(self, usdpath, dstdir, task='', texFmt=''):
        if not usdpath:
            return
        self.usdpath = usdpath
        self.dstdir = dstdir
        self.task = task
        self.texFmt = texFmt

    def doit(self):
        texList = []
        filepathList = []
        self.GetFilepathList(self.usdpath, filepathList, texList)

        filepaths = []
        self.FindReference(self.usdpath,filepaths)
        if filepaths:
            for path in filepaths:
                self.GetFilepathList(path, filepathList, texList)
        # print('filepaths:',filepaths)

        if texList:
            # Check
            for texpath in texList:
                srcimgdir = utl.DirName(texpath).replace('/tex/', '/images/')
                srcimgdir = srcimgdir + '_vendor'

                #putl.MakeDir(srcimgdir)

                imgname = utl.BaseName(texpath).split('.')

                if len(imgname) == 3:  # ['pirateShip_body_diffC', '1001', 'tif']
                    imgname = '.'.join(imgname[0:2])
                else:
                    imgname = imgname[0]

                format = ''
                if self.texFmt != 'texture':
                    format = self.texFmt

                else:
                    if 'disF' in imgname or 'mask' in imgname:
                        format = 'exr'

                    else:
                        format = 'jpg'
                #elif 'diffC' in imgname or 'specG' in imgname or 'specR' in imgname or 'norm' in imgname:
                
                if format:
                    filepath = os.path.join(srcimgdir, '%s.%s' % (imgname,format) )
                    if not os.path.exists(filepath):
                        self.textureConvert(texpath, filepath, format)
                    putl.AppendList(filepathList, filepath)

        return filepathList


    def textureConvert(self, source, target, format= 'jpg'):
        batchfile = '%s/TUtils.py' % os.path.dirname(__file__)
        cmd = '/backstage/dcc/DCC rez-env pylibs prmantoolkit dxusd'
        cmd += ' -- python %s ' % batchfile
        cmd += ' --source %s' % source
        cmd += ' --target %s' % target
        cmd += ' --format %s' % format

        if '/assetlib/_3d' in target:
            suCmd = 'echo dexter2019 | su render -c "%s"' % cmd
            subprocess.Popen(suCmd, shell=True).wait()
        else:
            subprocess.Popen(cmd, shell=True).wait()


    def FindReference(self, usdpath,filepaths):
        stage = Usd.Stage.Open(usdpath)
        dprim = stage.GetDefaultPrim()
        walk(stage,dprim, filepaths, usdpath)

    def GetTexVersion(self, texusd):
        ver= ''
        if os.path.exists(texusd):
            try:
                layer = utl.AsLayer(texusd)
                ver = layer.subLayerPaths[0].split('/')[1]
                return ver
            except:
                print('tex.usd : Not found subLayers.')
                return

    def GetFilepathList(self, path, filepathList, texList):

        arg = Arguments()
        arg.D.SetDecode(self.usdpath)
        arg.task = var.T.TEXTURE  # 'texture'
        arg.nslyr = var.T.TEX  # 'tex'

        if arg.show:
            orgshow = '/show/%s/_3d' % arg.show
        else:
            orgshow = arg.customdir

        print('>>>>>path:', path)
        lyr = utl.AsLayer(path)
        with utl.OpenStage(lyr) as stage:
            treeIter = iter(Usd.PrimRange.AllPrims(stage.GetPseudoRoot()))
            treeIter.next()
            for p in treeIter:
                basepath = p.GetAttribute('primvars:txBasePath').Get()  # asset/agaji/texture
                txLayer = p.GetAttribute('primvars:txLayerName').Get()
                ver = p.GetAttribute('primvars:ri:attributes:user:txVersion').Get()
                if basepath and txLayer:
                    if not ver:
                        texusd = os.path.join(orgshow, basepath, var.T.TEX, arg.F.MASTER)
                        ver = self.GetTexVersion(texusd)
                    if ver:
                        # Query All Tex
                        srcdir = os.path.join(orgshow, basepath, 'tex', ver)
                        for texpath in glob.glob('%s/%s_*.*' % (srcdir, txLayer)):
                            putl.AppendList(texList, texpath)



def walk(stage, prim, reflist, usdpath ):
    for p in prim.GetAllChildren():
        path = p.GetPath().pathString
        if 'Cam' in path:
            pass
        else:
            if p.GetTypeName() == 'PointInstancer':
                ptgeom = UsdGeom.PointInstancer(p)
                prototypes = ptgeom.GetPrototypesRel().GetTargets()
                for i in xrange(len(prototypes)):
                    prim = stage.GetPrimAtPath(prototypes[i])
                    stack = prim.GetPrimStack()
                    spec = stack[0]
                    if 'omtl' in spec.layer.identifier:
                        spec = stack[1]

                    refs = prim.GetMetadata('references')
                    if refs:
                        identifier = spec.layer.identifier
                        assetPath = spec.referenceList.prependedItems[0].assetPath
                        fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
                        if not os.path.exists(fullPath):
                            return
                        if not fullPath in reflist:
                            reflist.append(fullPath)

            elif p.GetTypeName() == 'Xform':
                if p.GetParent().GetName() == 'Layout' or p.GetParent().GetName() == 'World':
                    walk(stage, p, reflist, usdpath)

                else:
                    if p.HasAuthoredReferences():
                        stack = p.GetPrimStack()
                        spec = stack[0]
                        if 'omtl' in spec.layer.identifier:
                            spec = stack[1]
                        identifier = spec.layer.identifier
                        try:
                            assetPath = spec.referenceList.prependedItems[0].assetPath
                        except:
                            return
                        fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
                        if not os.path.exists(fullPath):
                            return

                        if not fullPath in reflist:
                            if not usdpath in fullPath or not 'collection.usd' in fullPath:
                                reflist.append(fullPath)

                    elif p.HasAuthoredSpecializes():
                        if p.IsInstanceable():
                            print('>', prim.GetPath().pathString, 'instanceable')
                            stack = p.GetPrimStack()
                            spec = stack[-1]
                            source = spec.nameChildren.get('source')
                            assetPath = source.referenceList.prependedItems[0].assetPath
                            identifier = source.layer.identifier
                            fullPath = os.path.abspath(os.path.join(utl.DirName(identifier), assetPath))
                            if not os.path.exists(fullPath):
                                return

                            if not fullPath in reflist:
                                reflist.append(fullPath)
                    else:
                        walk(stage, p, reflist, usdpath)


# ------------------------------------------------------------------------------
# Maya Assign
# ------------------------------------------------------------------------------

class AssignShader:
    def __init__(self, **kwargs):
        import maya.cmds as cmds
        self.node = kwargs['node']
        self.dstdir = kwargs['dstdir']
        self.textureInfo = kwargs['textureInfo']
        self.layeredShaderList = kwargs['layeredShaderList']
        self.asset = kwargs['asset']
        self.branch = kwargs['branch']
        self.type = kwargs['dataType']

        node = self.node
        dstdir = self.dstdir

        if cmds.attributeQuery('txLayerName', n=node, exists=True):
            txLayerName = cmds.getAttr('%s.txLayerName' % node)
        else:
            return

        if not cmds.attributeQuery('txBasePath', n=node, exists=True):
            return

        ##
        basepath = cmds.getAttr('%s.txBasePath' % node)
        ##

        texdir = ''
        if self.branch:
            texdir = os.path.join(dstdir, 'sourceimages', self.branch)
            sname = '%s_%s' % (self.branch, txLayerName)
            if self.type == 'Reference':
                sname = txLayerName

        elif '/branch/' in basepath:
            branch = basepath.split('/')[3]
            texdir = os.path.join(dstdir, 'sourceimages', branch)

            sname = '%s_%s' % (branch, txLayerName)
            if self.type == 'Reference':
                sname = txLayerName

        elif self.type == 'Reference':
            texdir = os.path.join(dstdir, 'sourceimages', self.asset)
            print('>>>Reference Texture dir:',texdir)
            sname = txLayerName

        else:
            assetname = self.branch if self.branch else self.asset
            txBasePath = cmds.getAttr('%s.txBasePath' % node)
            if not txBasePath:
                return
            splitpath = txBasePath.split('/')
            an = splitpath[1]
            if len(splitpath)>3:
                an = splitpath[-2]

            if assetname == an or not 'branch' in txBasePath:
                texdir = os.path.join(dstdir, 'sourceimages')
                sname = txLayerName

            else:
                texdir = os.path.join(dstdir, 'sourceimages', an)
                sname = '%s_%s' % (an, txLayerName)

        # print('>>>texdir:',texdir)

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
        import maya.cmds as cmds

        data = {txLayerName:{'attrs' : {'txmultiUV' : 0}, 'channels' : []
                              }
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
                    cmds.connectAttr('%s.%s' % (filenode, output), '%s.%s' % (shd, shdinput))
                    if not data in self.textureInfo:
                        self.textureInfo.append(data)

            cmds.sets(node, forceElement=SG)

        else:
            cmds.sets(node, forceElement=SG[0])



    def channelConnect(self, node, filename, data, texdir,txLayerName, channel='diffC'):

        import maya.cmds as cmds
        if 'diffC' != channel:
            filename = filename.replace('diffC', channel)
            # if 'specG' != channel:
            #     filename = filename.replace('jpg', 'tif')

        if os.path.exists(filename):
            data['channels'].append(channel)

        else:
            return

        filenode = cmds.shadingNode('file', asTexture=True)
        filename= '../sourceimages/%s' %filename.split('sourceimages/')[-1]
        cmds.setAttr('%s.fileTextureName' % filenode, filename, type='string')

        manifold = cmds.shadingNode('place2dTexture', asUtility=True)
        cmds.connectAttr('%s.outUvFilterSize' % manifold, '%s.uvFilterSize' % filenode)
        # multiUV
        if cmds.attributeQuery('txmultiUV', n=node, exists=True):
            if cmds.getAttr('%s.txmultiUV' % node):
                files = glob.glob('%s/%s_%s*' % (texdir,txLayerName,channel))
                if len(files) > 1:
                    cmds.setAttr('%s.fileTextureName' % filenode, files[0], type='string')
                    cmds.setAttr('%s.uvTilingMode' % filenode, 3)
                    data['attrs'] = {'txmultiUV': 1}

        if channel == 'diffC' or channel == 'specG':
            cmds.setAttr('%s.colorSpace' % filenode, 'Utility - sRGB - Texture',type="string")

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

# ------------------------------------------------------------------------------
# Maya Assign
# ------------------------------------------------------------------------------

def WriteInfo(dstdir, textureInfo, layeredShaderList, type= ''):
    if type == 'reference':
        return

    missingChannelList = []
    AlphaShaderList = []
    missingAttrMeshes = []

    for i in range(len(textureInfo)):
        data = textureInfo[i]
        value = data.values()[0]
        if len(value['channels']) < 4:
            if not data in missingChannelList:
                missingChannelList.append(data)
        if 'Alpha' in value['channels']:
            if not data in AlphaShaderList:
                AlphaShaderList.append(data)

    if missingChannelList:
        path = os.path.join(dstdir, 'no_full_channels.json')
        putl.jsonDump(path, missingChannelList)

    if AlphaShaderList:
        path = os.path.join(dstdir, 'AlphaShaderList.json')
        putl.jsonDump(path, AlphaShaderList)

    if layeredShaderList:
        path = os.path.join(dstdir, 'LayeredShaderList.json')
        putl.jsonDump(path, layeredShaderList)



def GetMissingAttr(nodes):
    import maya.cmds as cmds

    for node in cmds.ls(type='surfaceShape', dag=1, l=1):
        if cmds.objExists(node):
            if not 'Proxy' in node:
                if cmds.attributeQuery('txLayerName', n=node, exists=True) and cmds.attributeQuery('txBasePath', n=node, exists=True):
                    txLayerName = cmds.getAttr('%s.txLayerName' % node)
                    txBathPath = cmds.getAttr('%s.txBasePath' % node)
                    if txLayerName and txBathPath:
                        pass
                elif 'HdImaging' in node or 'Orig' in node:
                    pass
                else:
                    nodes.append(node)

# ------------------------------------------------------------------------------
# Maya Preview Assign
# ------------------------------------------------------------------------------

class AssignPreviewTexture(AssignShader):
    def __init__(self, **kwargs):
        AssignShader.__init__(self, **kwargs)

    def doit(self,node, texdir, txLayerName, sname ):
        self.assignSG(node, texdir, txLayerName, sname)

    def assignSG(self, node, texdir, txLayerName, sname):
        import maya.cmds as cmds

        data = {txLayerName:{'attrs' : {'txmultiUV' : 0}, 'channels' : []
                              }
                }

        channelData = {
            'diffC': {'outColor': 'color'}
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
            shd = cmds.shadingNode('lambert', asShader=True, name='%s_SHD' % sname)
            SG = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=sname + '_SG')
            cmds.connectAttr('%s.outColor' % shd, '%s.surfaceShader' % SG)
            filename = files[0]

            for k, v in channelData.items():
                output = v.keys()[0]
                shdinput = v[output]
                filenode = self.channelConnect(node, filename, data[txLayerName],texdir,txLayerName, channel=k)
                if filenode:
                    cmds.connectAttr('%s.%s' % (filenode, output), '%s.%s' % (shd, shdinput))

            cmds.sets(node, forceElement=SG)

        else:
            cmds.sets(node, forceElement=SG[0])


def SetAttr(shape, basepath, layername):
    import maya.cmds as cmds
    if not cmds.attributeQuery('txBasePath', n=shape, exists=True):
        cmds.addAttr(shape, ln='txBasePath', nn='txBasePath', dt='string')

    if not cmds.attributeQuery('txLayerName', n=shape, exists=True):
        cmds.addAttr(shape, ln='txLayerName', nn='txLayerName', dt='string')

    cmds.setAttr("%s.txBasePath" % shape, basepath, type='string')
    cmds.setAttr("%s.txLayerName" % shape, layername, type='string')

def SetMultiUV(shape):
    import maya.cmds as cmds
    if not cmds.attributeQuery('txmultiUV', n=shape, exists=True):
        cmds.addAttr(shape, ln='txmultiUV', nn='txmultiUV', at='long')
    cmds.setAttr("%s.txmultiUV" % shape, 1)


# ------------------------------------------------------------------------------
# Texture Convert
# ------------------------------------------------------------------------------
def exrConvert(source, target):
    cmd = '/backstage/dcc/DCC rez-env prmantoolkit'
    cmd += ' -- txmake'
    cmd += ' -envlatl -format openexr -compression zip -float'
    cmd += ' %s %s' % (source, target)
    subprocess.Popen(cmd, shell=True).wait()
    print('Convert exr: %s' % source, target)

def tifConvert(source, target):
    cmd = '/backstage/dcc/DCC rez-env prmantoolkit'
    cmd += ' -- txmake -t:8'
    cmd += ' -smode periodic -tmode periodic'
    cmd += ' -pattern diagonal'
    cmd += ' -format tiff'
    cmd += ' -compression LZW'
    cmd += ' -byte'
    cmd += ' -resize up-'
    cmd += ' %s %s' % (source, target)
    subprocess.Popen(cmd, shell=True).wait()
    print('Convert tif: %s' % source, target)


def jpgConvert(source, target):
    import ice
    loadImg = ice.Load(source)
    loadImg.Save(target, ice.constants.FMT_JPEG)
    print('Convert jpg: %s' % source, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source', type=str, required=True, help='texpath')
    parser.add_argument('--target', dest='target', type=str, required=True, help='output')
    parser.add_argument('--format', dest='format', type=str, required=True, help='format')

    args = parser.parse_args()
    source = args.source
    target = args.target
    format = args.format

    outputdir = os.path.dirname(target)
    if not os.path.exists(outputdir):
        os.system("mkdir -p %s" % outputdir)

    if format == 'jpg':
        jpgConvert(source, target)

    elif format == 'exr':
        exrConvert(source, target)

    elif format == 'tif':
        tifConvert(source, target)
