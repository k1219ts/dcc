# coding:utf-8
from __future__ import print_function
import argparse
import os, glob

import Package
import PUtils as putl

def previewAsset(dst, dstfiles):
    if len(dstfiles) < 1:
        return

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import assetpreview
    imgDir = dst.replace('/_3d/', '/preview/_3d/')
    try: os.makedirs(imgDir)
    except: pass

    for df in dstfiles:
        print('Make asset preview:', df, '->', imgDir)
        cmds.file(df, r=True, f=True)
        # fileList = cmds.ls(type='file')
        # texDir = os.path.dirname(df.replace('/scenes/', '/sourceimages/'))
        # if texDir.count('/branch/') > 0:
        #     texDir = os.path.dirname(df.replace('/scenes/branch/', '/sourceimages/'))
        # for fn in fileList:
        #     texPath = cmds.getAttr('%s.fileTextureName' % fn)
        #     texFn = os.path.basename(texPath)
        #     if not texFn.endswith('.jpg'):
        #         texFn = os.path.splitext(texFn)[0]+'.jpg'
        #     cmds.setAttr('%s.fileTextureName' % fn, texDir+'/'+texFn, type='string')
        assetpreview.createPreviewCameras()
        srcFn = os.path.basename(df)
        srcFnNoext = os.path.splitext(srcFn)[0]
        cmds.file(rename=imgDir+'/'+srcFnNoext+'-preview.mb')
        # cmds.file(s=True, f=True)
        prevDir = imgDir+'/'+os.path.splitext(srcFn)[0]
        assetpreview.makePreviews(prevDir, renderCams=['previewCamera02_02', 'previewCamera03_02'])

def modelPackage(dst, usdpath, dataType, categoryTask, format):
    arg = Package.AModelPack()
    arg.dst = dst
    arg.dataType = dataType
    arg.categoryTask = categoryTask
    arg.dataFormat = format
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.ModelPack(arg)

    # dstfiles = []
    # try: dstfiles.append(arg.dstfile)
    # except: pass
    # try: dstfiles = dstfiles+arg.dstfiles
    # except: pass
    # try: previewAsset(dst, dstfiles)
    # except: print('Model preview failed.')


def modelRefPackage(dst, usdpath, dataType, format):
    arg = Package.AModelRefPack()
    arg.dst = dst
    arg.dataType = dataType
    arg.dataFormat = format
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.ModelRefPack(arg)


def rigPackage(dst, usdpath, dataType, categoryTask, format):
    arg = Package.ARigPack()
    arg.dst = dst
    arg.dataType = dataType
    arg.dataFormat = format
    arg.categoryTask = categoryTask
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.RigPack(arg)


def groomPackage(dst, usdpath, categoryTask, format):
    arg = Package.AGroomPack()
    arg.dst = dst
    arg.categoryTask = categoryTask
    arg.dataFormat = format
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.GroomPack(arg)


def lidarPackage(dst, usdpath, categoryTask, format):
    arg = Package.ALidarPack()
    arg.dst = dst
    arg.categoryTask = categoryTask
    arg.dataFormat = format
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.LidarPack(arg)


def texturePackage(dst,usdpath,dataType,task,textureType,texFmt):
    arg = Package.ATexturePack()
    arg.dst = dst
    arg.dataType = dataType
    arg.task = task
    arg.textureType = textureType
    arg.usdpath = usdpath
    arg.texFmt = texFmt
    arg.D.SetDecode(usdpath)
    Package.TexturePack(arg)


def textureRefPackage(dst,usdpath):
    arg = Package.ATextureRefPack()
    arg.dst = dst
    arg.usdpath = usdpath
    arg.D.SetDecode(usdpath)
    Package.TextureRefPack(arg)


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', dest='src', default='', help='usdpath')
    parser.add_argument('--dst', dest='dst', default='', help='dst')
    parser.add_argument('--lyrtype', dest='lyrtype', default='', help='lyrtype')
    parser.add_argument('--dataType', dest='dataType', default='', help='dataType')
    parser.add_argument('--task', dest='task', default='', help='task')
    parser.add_argument('--format', dest='format', default='', help='format')
    parser.add_argument('--texFmt', dest='texFmt', default='', help='texFmt')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = getArgs()
    dst = args.dst
    usdpath = args.src
    lyrtype = args.lyrtype
    dataType = args.dataType
    task = args.task
    format = args.format
    texFmt = args.texFmt

    if lyrtype == 'texture':# or lyrtype == 'prevtex':
        textureType = ''
        if dataType == 'Reference':
            textureRefPackage(dst, usdpath)
        else:
            # if lyrtype == 'prevtex':
            #     textureType = 'prevtex'
            texturePackage(dst, usdpath, dataType, task, textureType, texFmt)

    else:
        from pymel.all import *
        import maya.standalone
        maya.standalone.initialize("Python")
        import maya.cmds as cmds
        plugins = ['backstageMenu', 'pxrUsd', 'DXUSD_Maya']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        if lyrtype == 'model':
            if dataType == 'Reference':
                modelRefPackage(dst, usdpath, dataType, format)
            else:
                modelPackage(dst, usdpath, dataType, task, format)

        if lyrtype == 'rig':
            rigPackage(dst, usdpath, dataType, task, format)
            if putl.istask(usdpath, 'groom'):
                groomPackage(dst, usdpath, task, format)

        if lyrtype == 'lidar':
            lidarPackage(dst, usdpath, task, format)
