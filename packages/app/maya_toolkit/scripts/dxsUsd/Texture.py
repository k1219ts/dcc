'''
USD Texture Version Metadata Export

1. if not modelVersion,
    find last version by model.payload.usd

Script Command
> dxsUsd.TextureExport(showDir=$showDir, asset=$assetName, version=$ver).doIt()
> dxsUsd.TextureExport(modelVersion=$ver, showDir=$showDir, asset=$assetName).doIt()
> dxsUsd.TextureExport(elementName=$EL, showDir=$showDir, asset=$assetName).doIt()

Override Script Command
> dxsUsd.AssetTextureOverride(refAssetFile=$filename, showDir=$showDir, asset=$assetName).doIt()
'''
import os, re

from pxr import Usd, UsdGeom, UsdShade, Sdf

import dxsMsg
import Arguments
import PathUtils
import SessionUtils
import dxsUsdUtils


class TextureExport(Arguments.AssetArgs):
    '''
    Texture Attribute Export
    Args:
        elementName (str):
        modelVersion (str):
    '''
    def __init__(self, modelVersion=None, elementName=None, overWrite=True, **kwargs):
        Arguments.AssetArgs.__init__(self, **kwargs)
        self.modelVersion = modelVersion
        self.elementName  = elementName
        self.overWrite    = overWrite

        self.rootName = self.assetName
        self.modelFile= None

        if self.outDir:
            self.rootName = os.path.basename(self.outDir)
        if not self.outDir and self.assetDir:
            self.outDir   = '{DIR}/texture'.format(DIR=self.assetDir)
            self.modelFile= '{DIR}/model/model.payload.usd'.format(DIR=self.assetDir)
            if self.elementName:
                self.outDir   = '{DIR}/element/{NAME}/texture'.format(DIR=self.assetDir, NAME=self.elementName)
                self.modelFile= '{DIR}/element/{NAME}/model/model.usd'.format(DIR=self.assetDir, NAME=self.elementName)
                self.rootName = self.elementName
        self.getTexVersion()

    def getTexVersion(self):
        if self.version:
            return
        self.version = PathUtils.GetVersion(self.outDir + '/tex')

    def getModelVersion(self):
        if self.modelVersion or not self.modelFile:
            return
        rootLayer = Sdf.Layer.FindOrOpen(self.modelFile)
        if not rootLayer:
            dxsMsg.Print('warning', "[Texture.TextureExport.getModelVersion] - not found model file")
            self.modelVersion = 'v001'
            return
        stage = Usd.Stage.Open(rootLayer)
        prim = stage.GetDefaultPrim()
        vset = prim.GetVariantSets().GetVariantSet('modelVersion')
        values = vset.GetVariantNames()
        self.modelVersion = values[-1]


    def doIt(self):
        verfile = '{DIR}/tex/{VER}/tex.version.usd'.format(DIR=self.outDir, VER=self.version)
        if os.path.exists(verfile) and not self.overWrite:
            # print '# Debug : not write texture!'
            return
        self.getModelVersion()
        geomMasterFile = self.makeTexVersionGeom(verfile)
        texMasterFile  = self.makePackage(geomMasterFile)
        return texMasterFile


    def makeTexVersionGeom(self, filename):
        '''
        Texture raw version information file
        '''
        stage    = SessionUtils.MakeInitialStage(filename, clear=True)
        rootPrim = stage.DefinePrim('/' + self.rootName)
        # primvars:ri:attributes:user:txVersion
        riattr = UsdGeom.PrimvarsAPI(rootPrim).CreatePrimvar('ri:attributes:user:txVersion', Sdf.ValueTypeNames.String)
        riattr.Set(self.version)
        stage.SetDefaultPrim(rootPrim)
        stage.GetRootLayer().Save()

        masterfile = '{DIR}/tex/{VER}/tex.payload.usd'.format(DIR=self.outDir, VER=self.version)
        SdfPath = '/' + self.rootName
        if self.modelVersion:
            SdfPath += '{modelVersion=%s}' % self.modelVersion
        SessionUtils.MakeReferenceStage(masterfile, [(filename, None)], SdfPath=SdfPath, Type='', clear=False)
        return masterfile


    def makePackage(self, sourceFile):
        # Collect versions
        texFile = '{DIR}/tex/tex.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(texFile, [sourceFile])

        textureFile = '{DIR}/texture.usd'.format(DIR=self.outDir)
        SessionUtils.MakeReferenceStage(textureFile, [(texFile, None)], SdfPath='/' + self.rootName, Type='', clear=True)

        if self.assetDir:
            if self.elementName:
                outFile = '{DIR}/element/{NAME}/{NAME}.usd'.format(DIR=self.assetDir, NAME=self.elementName)
            else:
                outFile = '{DIR}/{NAME}.usd'.format(DIR=self.assetDir, NAME=self.assetName)
            self.overrideModelVersion(outFile)
        return textureFile

    def overrideModelVersion(self, sourceFile):
        outLayer = Sdf.Layer.FindOrOpen(sourceFile)
        if not outLayer:
            return
        stage = Usd.Stage.Open(sourceFile, load=Usd.Stage.LoadNone)
        prim  = stage.OverridePrim('/' + self.rootName)
        if prim:
            vset = prim.GetVariantSets().GetVariantSet('modelVersion')
            vset.SetVariantSelection(self.modelVersion)
        stage.GetRootLayer().Save()



class AssetTextureOverride(Arguments.AssetArgs):
    '''
    Texture Attribute Override.
        must input refAsset or refAssetFile
    Args:
        refAsset (str): composite archive source assetName
        refAssetFile (str): composite archive source asset filename
    '''
    def __init__(self, refAsset=None, refAssetFile=None, **kwargs):
        Arguments.AssetArgs.__init__(self, **kwargs)
        self.refAsset    = refAsset
        self.refAssetFile= refAssetFile
        if not refAsset and refAssetFile:
            self.refAsset = os.path.basename(refAssetFile).split('.')[0]
        self.outDir      = '{DIR}/texture/{NAME}'.format(DIR=self.assetDir, NAME=self.refAsset)
        self.modelFile   = '{DIR}/model/model.payload.usd'.format(DIR=self.assetDir)

    def doIt(self):
        # Export Texture
        TextureExport(outDir=self.outDir, asset=self.assetName, version=self.version).doIt()

        # Make Override Attributes
        if self.refAsset:
            if not self.refAssetFile:
                self.GetRefAssetFile()
            overrideAttrFile = self.MakeOverrideAttributes()

            texOverrideFile = '{DIR}/texture/texture.override.usd'.format(DIR=self.assetDir)
            SessionUtils.MakeSubLayerStage(texOverrideFile, [overrideAttrFile], defaultPrimSet=False)


    def MakeOverrideAttributes(self):
        if not self.refAssetFile:
            print '# Error : not found RefAssetFile!'
            return
        txBasePath = 'asset/{ASSET}/texture/{NAME}'.format(ASSET=self.assetName, NAME=self.refAsset)

        geomStage = Usd.Stage.Open(self.refAssetFile)
        geomDefaultPrim = geomStage.GetDefaultPrim()
        geomDefaultPath = geomDefaultPrim.GetPath()

        # Create attribute stage
        attrFile = '{DIR}/texture.override.usd'.format(DIR=self.outDir)
        newStage = SessionUtils.MakeInitialStage(attrFile, clear=True)

        txVerPath = Sdf.Path('/_{NAME}_OverTxVer'.format(NAME=self.refAsset))
        txVerPrim = newStage.CreateClassPrim(txVerPath)
        txVerPrim.SetPayload(Sdf.Payload('./texture.usd'))

        RootPath = Sdf.Path('/_{NAME}_OverAttr'.format(NAME=self.refAsset))
        RootPrim = newStage.CreateClassPrim(RootPath)

        treeIter = iter(Usd.PrimRange.AllPrims(geomDefaultPrim))
        treeIter.next()
        for p in treeIter:
            txAttr = p.GetAttribute('primvars:txBasePath')
            if txAttr:
                newPath = p.GetPath().ReplacePrefix(geomDefaultPath.pathString, RootPath)
                prim = newStage.DefinePrim(newPath)
                prim.GetInherits().AddInherit(txVerPath)
                primvarApi = UsdGeom.PrimvarsAPI(prim)
                primvar = primvarApi.CreatePrimvar('txBasePath', Sdf.ValueTypeNames.String)
                primvar.SetInterpolation(UsdGeom.Tokens.constant)
                primvar.Set(txBasePath)

        newStage.GetRootLayer().Save()
        print "# Override TextureAttribute file '%s'" % attrFile
        return attrFile


    def GetRefAssetFile(self):
        if not os.path.exists(self.modelFile):
            return
        stage = Usd.Stage.Open(self.modelFile)
        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            refs = p.GetMetadata('references')
            if refs:
                for r in refs.GetAddedOrExplicitItems():
                    name = os.path.basename(r.assetPath).split('.')[0]
                    if name == self.refAsset:
                        self.refAssetFile = r.assetPath
                        return


#-------------------------------------------------------------------------------
#
#
#
#-------------------------------------------------------------------------------
def GetTxFiles(dirpath, ext='tex'):
    files = list()
    if os.path.exists(dirpath):
        for n in os.listdir(dirpath):
            if not n.startswith('.') and n.split('.')[-1] == ext:
                files.append(n)
    files.sort()
    return files

def GetLayerInfoByFiles(files): # files: file basename list
    data = dict()
    for n in files:
        udim = False
        if len(n.split('.')) > 2:
            udim = True

        splitStr = n.split('.')[0].split('_')

        chindex = -1
        if re.match('\d+', splitStr[-1]):
            chindex = -2
        channel = splitStr[chindex]
        txlayer = '_'.join(splitStr[:chindex])

        # ZN texture
        if '_ZN_' in n:
            if not txlayer.endswith('_ZN'):
                tmpsplit = txlayer.split('_ZN_')
                txlayer  = tmpsplit[0] + '_ZN'
                channel  = tmpsplit[-1] + '_' + channel

        if not data.has_key(txlayer):
            data[txlayer] = {'channels': list(), 'udim': udim}
        if not channel in data[txlayer]['channels']:
            data[txlayer]['channels'].append(channel)
    return data


class MakeTexAttr:
    '''
    Make Texture Attribute.
        txPath or txAttr
    Args
        txPath (str): texture version path
        txAttr (dict): {$txBasePath: [($txLayerName, $UDIM), ...]} # txBasePath(abspath)
    '''
    def __init__(self, txPath=None, txAttr=dict(), **kwargs):
        self.txPath = txPath
        self.txAttr = txAttr

        # Member Variables
        self.modelVersion = None
        # - txVersion, txBasePath
        # - txLayerInfo {$txLayerName: {'channels': ['diffC', ...], 'udim': True}, ...}

        self.parseArgs(kwargs)

    def parseArgs(self, kwargs):
        if kwargs.has_key('modelVersion'):
            self.modelVersion = kwargs['modelVersion']


    def doIt(self):
        if self.txPath:
            print '# MakeTexAttr by texture version path'
            self.txPathProc()
        elif self.txAttr:
            print '# MakeTexAttr by txAttr data'
            self.txAttrProc()

    def txPathProc(self):
        self.txVersion = os.path.basename(self.txPath)

        splitPath = self.txPath.split('/')
        rootDir   = '/'.join(splitPath[:splitPath.index('texture')])
        if self.txPath.startswith('/assetlib'):
            self.txBasePath = rootDir + '/texture'
        else:
            self.txBasePath = '/'.join(splitPath[splitPath.index('asset'):splitPath.index('texture')+1])

        modelFile = os.path.join(rootDir, 'model', 'model.usd')
        self.getModelVersion(modelFile)

        txFiles = GetTxFiles(self.txPath)
        self.txLayerInfo = GetLayerInfoByFiles(txFiles)

        attrfile = os.path.join(self.txPath, 'tex_attr.usd')
        stage = SessionUtils.MakeInitialStage(attrfile, clear=False)
        for name in self.txLayerInfo:
            cprim = stage.CreateClassPrim(Sdf.Path('/' + name))
            self.makeLayerGeom(cprim, name, self.txLayerInfo[name]['udim'])

        stage.GetRootLayer().Save()
        del stage
        print '# result: TexAttr ->', attrfile

        self.makePackage(attrfile)

    def txAttrProc(self):
        for dir in self.txAttr: # dir: txBasePath(abspath)
            self.txBasePath = dir
            if not dir.startswith('/assetlib'):
                splitPath = dir.split('/')
                self.txBasePath = '/'.join(splitPath[splitPath.index('asset'):])

            modelFile = os.path.join(os.path.dirname(dir), 'model', 'model.usd')
            self.getModelVersion(modelFile)

            outdir  = os.path.join(dir, 'tex')
            self.txVersion = PathUtils.GetLastVersion(outdir)
            txFiles = GetTxFiles(os.path.join(outdir, self.txVersion))
            self.txLayerInfo = GetLayerInfoByFiles(txFiles)

            attrfile= os.path.join(outdir, self.txVersion, 'tex_attr.usd')
            mstfile = os.path.join(outdir, 'tex.attr.usd')

            rootLayer= Sdf.Layer.FindOrOpen(mstfile)
            if rootLayer:
                mstage = Usd.Stage.Open(rootLayer)
                _modified = 0
                for name, udim in self.txAttr[dir]:
                    if not mstage.GetPrimAtPath('/' + name):
                        _modified += 1
                        stage = SessionUtils.MakeInitialStage(attrfile, clear=False)
                        cprim = stage.CreateClassPrim(Sdf.Path('/' + name))
                        self.makeLayerGeom(cprim, name, udim)
                        stage.GetRootLayer().Save()
                        del stage
                if _modified > 0:
                    print '# result: Update TexAttr ->', attrfile
                del rootLayer, mstage
            else:
                stage = SessionUtils.MakeInitialStage(attrfile, clear=False)
                for name, udim in self.txAttr[dir]:
                    cprim = stage.CreateClassPrim(Sdf.Path('/' + name))
                    self.makeLayerGeom(cprim, name, udim)
                stage.GetRootLayer().Save()
                del stage

            # Package
            self.makePackage(attrfile)


    def getModelVersion(self, modelFile):
        if self.modelVersion:
            return
        rootLayer = Sdf.Layer.FindOrOpen(modelFile)
        if not rootLayer:
            self.modelVersion = 'v001'
            return
        stage = Usd.Stage.Open(rootLayer)
        dprim = stage.GetDefaultPrim()
        vset  = dprim.GetVariantSets().GetVariantSet('modelVersion')
        values= vset.GetVariantNames()
        del stage
        self.modelVersion = values[-1]

    def makeLayerGeom(self, prim, name, udim):
        vset = dxsUsdUtils.VariantSelection(prim, 'modelVersion', self.modelVersion)
        with vset.GetVariantEditContext():
            dxsUsdUtils.CreateRiAttribute(prim, 'user:txVersion', self.txVersion, Sdf.ValueTypeNames.String)

        dxsUsdUtils.CreateConstPrimvar(prim, 'txBasePath', self.txBasePath, Sdf.ValueTypeNames.String)
        dxsUsdUtils.CreateConstPrimvar(prim, 'txLayerName', name, Sdf.ValueTypeNames.String)
        if udim:
            dxsUsdUtils.CreateConstPrimvar(prim, 'txmultiUV', 1, Sdf.ValueTypeNames.Int)

        # from self.txLayerInfo
        if self.txLayerInfo and self.txLayerInfo.has_key(name):
            for ch in self.txLayerInfo[name]['channels']:
                dxsUsdUtils.CreateUserProperty(prim, 'Texture:channels:' + ch, 1, Sdf.ValueTypeNames.Int)
                if 'dis' in ch:
                    dxsUsdUtils.CreateRiAttribute(prim, 'displacementbound:sphere', 0.1, Sdf.ValueTypeNames.Float)

    def makePackage(self, srcfile):
        outfile = os.path.join(os.path.dirname(os.path.dirname(srcfile)), 'tex.attr.usd')
        stage = SessionUtils.MakeInitialStage(outfile, clear=False)
        outLayer = stage.GetRootLayer()
        SessionUtils.InsertSubLayer(outLayer, PathUtils.GetRelPath(outfile, srcfile))
        outLayer.Save()



class MakePrvMtl:
    '''
    Make Preview Materials.
        txPath or txAttr
        If already texture published, create by last version.
    Args
        txPath (str): proxy texture version path
        txAttr (dict):
    '''
    def __init__(self, txPath=None, txAttr=dict()):
        self.txPath = txPath
        self.txAttr = txAttr

        # Member Variables
        self.prvMtlFile = '/assetlib/3D/material/usd/preview/Materials.usd'
        # - rootName


    def doIt(self):
        if self.txPath:
            print '# MakePrvMtl by proxy texture version path'
            self.txPathProc()
        elif self.txAttr:
            print '# MakePrvMtl by txAttr data'
            self.txAttrProc()
        else:
            print '# Error: MakePrvMtl arguments error'


    def txPathProc(self):
        splitPath = self.txPath.split('/')
        self.rootName = splitPath[splitPath.index('texture') - 1]

        txFiles = GetTxFiles(self.txPath, ext='jpg')
        self.txLayerInfo = GetLayerInfoByFiles(txFiles)

        mtlfile= os.path.join(self.txPath, 'prv_mtl.usd')
        stage  = SessionUtils.MakeInitialStage(mtlfile, clear=False)

        for name in self.txLayerInfo:
            self.makeMaterial(stage, name, self.txLayerInfo[name]['udim'])

        stage.GetRootLayer().Save()
        del stage
        print '# result PreviewMaterial ->', mtlfile
        # Package
        self.makePackage(mtlfile)


    def txAttrProc(self):
        for dir in self.txAttr: # dir: txBasePath(abspath)
            self.rootName = os.path.basename(os.path.dirname(dir))

            outdir  = os.path.join(dir, 'proxy')
            version = PathUtils.GetLastVersion(outdir)
            mtlfile = os.path.join(outdir, version, 'prv_mtl.usd')
            mstfile = os.path.join(outdir, 'proxy.mtl.usd')

            txFiles = GetTxFiles(os.path.join(outdir, version), ext='jpg')
            self.txLayerInfo = GetLayerInfoByFiles(txFiles)

            rootLayer = Sdf.Layer.FindOrOpen(mstfile)
            if rootLayer:
                mstage = Usd.Stage.Open(rootLayer)
                _modified = 0
                for name, udim in self.txAttr[dir]:
                    mtlpath = '/' + self.rootName + '_' + name
                    if not mstage.GetPrimAtPath(mtlpath):
                        _modified += 1
                        stage = SessionUtils.MakeInitialStage(mtlfile, clear=False)
                        self.makeMaterial(stage, name, udim)
                        stage.GetRootLayer().Save()
                        del stage
                if _modified > 0:
                    print '# result: Update PrvMtl ->', os.path.join(outdir, version, 'prv_mtl.usd')
                del rootLayer, mstage
            else:
                stage = SessionUtils.MakeInitialStage(mtlfile, clear=False)
                for name, udim in self.txAttr[dir]:
                    self.makeMaterial(stage, name, udim)
                stage.GetRootLayer().Save()
                del stage
                print '# result: PrvMtl ->', mtlfile

            # Package
            self.makePackage(mtlfile)


    def makeMaterial(self, stage, txLayerName, udim):
        targetMaterial = '/Materials/DefaultMaterial'
        if txLayerName.endswith('_ZN'):
            targetMaterial = '/Materials/ZennMaterial'

        materialName = self.rootName + '_' + txLayerName    # {ASSET}_{LAYER}
        mtlpath = '/' + materialName

        material = UsdShade.Material.Define(stage, Sdf.Path(mtlpath))
        prim = material.GetPrim()
        prim.SetPayload(Sdf.Payload(self.prvMtlFile, targetMaterial))
        overprim = stage.OverridePrim(Sdf.Path(mtlpath + '/pbsSurface/diffC_Tex'))
        shader   = UsdShade.Shader(overprim)
        fileAttr = shader.GetInput('file')

        jpgfile  = txLayerName + '_diffC'
        if self.txLayerInfo and self.txLayerInfo.has_key(txLayerName):
            if self.txLayerInfo[txLayerName]['channels']:
                jpgfile = txLayerName + '_' + self.txLayerInfo[txLayerName]['channels'][0]
        value = './' + jpgfile
        if udim:
            value += '.<UDIM>'
        value += '.jpg'
        fileAttr.Set(value)


    def makePackage(self, srcfile):
        outfile = os.path.join(os.path.dirname(os.path.dirname(srcfile)), 'proxy.mtl.usd')
        stage   = SessionUtils.MakeInitialStage(outfile, clear=False)
        outLayer= stage.GetRootLayer()
        SessionUtils.InsertSubLayer(outLayer, PathUtils.GetRelPath(outfile, srcfile))
        outLayer.Save()
