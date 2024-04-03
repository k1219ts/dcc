import os
import string

import PathUtils


class CommonArgs:
    def __init__(self, **kwargs):
        self.usdformat = 'usda'

        self.comment = None
        self.clear  = False
        self.user   = ''

        # Frame
        self.fr  = (None, None) # (startframe, endframe)
        self.fps = 24.0
        self.step= 1.0

        self.CommonArgsDoIt(kwargs)

    def CommonArgsDoIt(self, kwargs):
        if kwargs.has_key('usdformat'):
            self.usdformat = kwargs['usdformat']

        if kwargs.has_key('comment'):
            self.comment = kwargs['comment']
        if kwargs.has_key('clear'):
            self.clear = kwargs['clear']
        if kwargs.has_key('user'):
            self.user = kwargs['user']

        if kwargs.has_key('fr'):
            self.fr = kwargs['fr']
        if kwargs.has_key('fps'):
            self.fps = kwargs['fps']
        if kwargs.has_key('step'):
            self.step= kwargs['step']


class MakeArgs(CommonArgs):
    def __init__(self, **kwargs):
        self.SdfPath = None
        self.addChild= False
        self.Kind = 'component' # firstRoot kind
        self.Name = None        # firstRoot name
        self.Type = None        # firstRoot geom type
        self.pKind= None        # SdfPath prim kind
        self.pName= None        # SdfPath prim name
        self.pType= None        # SdfPath prim geom type
        self.customLayerData = dict()
        self.customPrimData  = dict()

        CommonArgs.__init__(self, **kwargs)
        self.MakeArgsDoIt(kwargs)

    def MakeArgsDoIt(self, kwargs):
        if kwargs.has_key('SdfPath'):
            self.SdfPath = kwargs['SdfPath']
        if kwargs.has_key('addChild'):
            self.addChild = kwargs['addChild']

        if kwargs.has_key('Kind'):
            self.Kind = kwargs['Kind']
        if kwargs.has_key('Name'):
            self.Name = kwargs['Name']
        if kwargs.has_key('Type'):
            self.Type = kwargs['Type']

        if kwargs.has_key('pKind'):
            self.pKind = kwargs['pKind']
        if kwargs.has_key('pName'):
            self.pName = kwargs['pName']
        if kwargs.has_key('pType'):
            self.pType = kwargs['pType']

        if kwargs.has_key('customLayerData'):
            self.customLayerData = kwargs['customLayerData']
        if kwargs.has_key('customPrimData'):
            self.customPrimData = kwargs['customPrimData']


class AssetArgs(CommonArgs):
    '''
    Args:
        showDir :

        outDir  :

        asset   :
        version :
    '''
    def __init__(self, **kwargs):
        self.outDir  = None
        self.showName = None
        self.showDir  = None
        self.assetDir = None
        self.assetName= None
        self.version  = None

        CommonArgs.__init__(self, **kwargs)
        self.AssetArgsDoIt(kwargs)

    def AssetArgsDoIt(self, kwargs):
        if kwargs.has_key('version'):
            self.version = kwargs['version']

        if kwargs.has_key('showDir') and kwargs.has_key('asset'):
            self.showDir  = kwargs['showDir']
            self.showName = self.showDir.split('/')[-1]
            self.assetName= kwargs['asset']
            self.assetDir = '{DIR}/asset/{NAME}'.format(DIR=self.showDir, NAME=self.assetName)
        if kwargs.has_key('show') and kwargs.has_key('asset'):
            self.showDir, self.showName = PathUtils.GetRootPath('/show/' + kwargs['show'])
            self.assetName= kwargs['asset']
            self.assetDir = '{DIR}/asset/{NAME}'.format(DIR=self.showDir, NAME=self.assetName)

        if kwargs.has_key('outDir'):
            self.outDir = kwargs['outDir']
            splitPath = self.outDir.split('/')
            if 'show' in splitPath:
                index = splitPath.index('show')
                self.showDir = string.join(splitPath[:index+2], '/')
                self.showDir, self.showName = PathUtils.GetRootPath(self.showDir)
            if kwargs.has_key('asset'):
                self.assetName = kwargs['asset']
            elif 'asset' in splitPath:
                self.assetName = splitPath[splitPath.index('asset') + 1]

        if kwargs.has_key('assetDir'):
            self.assetDir = kwargs['assetDir']
            splitPath = self.assetDir.split('/')
            if 'show' in splitPath:
                index = splitPath.index('show')
                self.showDir = string.join(splitPath[:index + 2], '/')
                self.showDir, self.showName = PathUtils.GetRootPath(self.showDir)
            if kwargs.has_key('asset'):
                self.assetName = kwargs['asset']
            elif 'asset' in splitPath:
                self.assetName = splitPath[splitPath.index('asset') + 1]

    def computeVersion(self):
        if not self.outDir:
            print '# [Error] Arguments.AssetArgs.computeVersion : Not found outDir.'
            return
        if self.version:
            return
        if not os.path.exists(self.outDir):
            self.version = 'v001'
            return
        self.version = PathUtils.GetVersion(self.outDir)


class ShotArgs(CommonArgs):
    '''
    Args:
        showDir (str): show dir
        seq  (str): sequence name
        shot (str): shot name
        outDir  (str): output dir
        version (str):
    '''
    def __init__(self, **kwargs):
        self.outDir  = None
        self.showDir = None
        self.showName= None
        self.seqName = None
        self.shotName= None
        self.shotDir = None
        self.version = None

        CommonArgs.__init__(self, **kwargs)
        self.ShotArgsDoIt(kwargs)

    def ShotArgsDoIt(self, kwargs):
        if kwargs.has_key('version'):
            self.version = kwargs['version']

        if kwargs.has_key('showDir') and kwargs.has_key('shot'):
            self.showDir = kwargs['showDir']
            self.showName= self.showDir.split('/')[-1]
            self.shotName= kwargs['shot']
            if kwargs.has_key('seq'):
                self.seqName = kwargs['seq']
            else:
                self.seqName = self.shotName.split('_')[0]

        if kwargs.has_key('outDir'):
            self.outDir = kwargs['outDir']
            splitPath = self.outDir.split('/')
            if 'show' in splitPath:
                index = splitPath.index('show')
                self.showDir = string.join(splitPath[:index+2], '/')
                self.showName= splitPath[index+1]
            if 'shot' in splitPath:
                self.seqName = splitPath[splitPath.index('shot') + 1]
                self.shotName= splitPath[splitPath.index('shot') + 2]

        if self.showDir and self.seqName and self.shotName:
            self.shotDir = '{DIR}/shot/{SEQ}/{SHOT}'.format(DIR=self.showDir, SEQ=self.seqName, SHOT=self.shotName)

        if self.showName:
            self.showName = self.showName.replace('_pub', '')

    def computeVersion(self):
        assert self.outDir, '[Arguments.ShotArgs.computeVersion] - not found outDir.'
        if self.version:
            return
        if not os.path.exists(self.outDir):
            self.version = 'v001'
            return
        self.version = PathUtils.GetVersion(self.outDir)
