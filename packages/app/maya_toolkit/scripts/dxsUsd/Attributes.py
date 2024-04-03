import os
import string
import re
import shutil

from pxr import Usd, UsdGeom, Sdf, Kind

import SessionUtils
import PathUtils
import Texture


#-------------------------------------------------------------------------------
#
#
#
#-------------------------------------------------------------------------------
class ExtractAttr:
    '''
    Create Attribute File by Geometry File.
        root prim is Class. This class inherit to geom root.
        support attributes:
            txBasePath, txLayerName, txmultiUV, st
    Args
        geomFile (str): geometry file
    '''
    def __init__(self, geomFile, attrFile=None, **kwargs):
        self.geomFile = geomFile
        self.attrFile = attrFile
        if not self.attrFile:
            self.attrFile = self.geomFile.replace('geom.usd', 'attr.usd')

        # Member Variables
        # - self.showDir, self.modelVersion
        self.usdformat = 'usda'
        self.expST = False
        global __MaterialSet
        __MaterialSet = 'userProperties:MaterialSet'

        self.parseArgs(kwargs)
        self.parsePath()

    def parseArgs(self, kwargs):
        if kwargs.has_key('st'):
            self.expST = kwargs['st']
        if self.expST:
            self.usdformat = 'usdc'

    def parsePath(self):
        if self.geomFile.startswith('/assetlib'):
            self.showDir = '/assetlib/3D'
        else:
            self.showDir, self.showName = PathUtils.GetRootPath(self.geomFile)

        print self.geomFile
        try:
            splitPath = self.geomFile.split('/')
            rootIndex = splitPath.index('asset')
            if 'element' in splitPath:
                rootIndex = splitPath.index('element')
            rootDir = '/'.join(splitPath[:rootIndex+2]) # assetDir or elementDir
            modelPath = os.path.join(rootDir, 'model')
            self.modelVersion = PathUtils.GetLastVersion(modelPath)
        except:
            self.modelVersion = "v000"


    @staticmethod
    def GetPrim(stage, primPath):
        prim = stage.GetPrimAtPath(primPath)
        if not prim:
            prim = stage.DefinePrim(primPath)
        return prim


    def CheckupTexturePub(self, prim):  # prim : geomfile root prim
        txAttrMap = dict()  # txBasePath(abspath) based ...
        # { txBasePath: [(txLayerName, udim), ...] }
        for p in iter(Usd.PrimRange.AllPrims(prim)):
            names = p.GetAuthoredPropertyNames()
            if 'primvars:txBasePath' in names and 'primvars:txLayerName' in names:
                txBasePath = p.GetAttribute('primvars:txBasePath').Get()
                txLayerName= p.GetAttribute('primvars:txLayerName').Get()
                udim = False
                if 'primvars:txmultiUV' in names:
                    udim = p.GetAttribute('primvars:txmultiUV')
                if not txBasePath.startswith('/'):
                    txBasePath = self.showDir + '/' + txBasePath

                # update txAttrMap
                if not txAttrMap.has_key(txBasePath):
                    txAttrMap[txBasePath] = list()
                item = (txLayerName, udim)
                if not item in txAttrMap[txBasePath]:
                    txAttrMap[txBasePath].append(item)

        # create tex.attr.usd, proxy.mtl.usd
        Texture.MakeTexAttr(txAttr=txAttrMap, modelVersion=self.modelVersion).doIt()
        Texture.MakePrvMtl(txAttr=txAttrMap).doIt()


    def doIt(self):
        self._ifever = -1
        isUpdate = False

        rootLayer = Sdf.Layer.FindOrOpen(self.geomFile)
        if not rootLayer:
            return

        for lyr in rootLayer.subLayerPaths:
            if lyr.find('_attr.usd') > -1:
                isUpdate = True
                self.attrFile = self.attrFile.replace('.usd', '_update.usd')

        self.stage = SessionUtils.MakeInitialStage(self.attrFile, usdformat=self.usdformat, clear=True)

        # read stage
        if isUpdate:
            geomStage = Usd.Stage.Open(rootLayer)
        else:
            geomStage = Usd.Stage.Open(rootLayer, load=Usd.Stage.LoadNone)

        gdprim = geomStage.GetDefaultPrim()
        gdpath = gdprim.GetPath()

        # check texture status
        self.CheckupTexturePub(gdprim)

        # new
        rootPath = Sdf.Path('/_{NAME}_Attr'.format(NAME=gdpath.name))
        rootPrim = self.stage.CreateClassPrim(rootPath)

        treeIter = iter(Usd.PrimRange.AllPrims(gdprim))
        treeIter.next()
        for p in treeIter:
            ptyp = p.GetTypeName()
            if ptyp == 'Mesh':
                newPath = p.GetPath().ReplacePrefix(gdpath.pathString, rootPath)

                # export ST
                if self.expST:
                    stAttr = p.GetAttribute('primvars:st')
                    if stAttr:
                        nprim = ExtractAttr.GetPrim(self.stage, newPath)
                        self.CopyPrimvars([stAttr.GetBaseName()], p, nprim)

                self.textureAttribute(p, newPath)

                # modelVersion
                modelVersionAttr = p.GetAttribute('primvars:modelVersion')
                if modelVersionAttr:
                    nprim = ExtractAttr.GetPrim(self.stage, newPath)
                    self.CopyPrimvars([modelVersionAttr.GetBaseName()], p, nprim)

                # MaterialSet
                mtlAttr = p.GetAttribute(__MaterialSet)
                if mtlAttr:
                    self._ifever += 1
                    nprim = ExtractAttr.GetPrim(self.stage, newPath)
                    mtlval= mtlAttr.Get()
                    if mtlval:
                        nprim.CreateAttribute(mtlAttr.GetName(), mtlAttr.GetTypeName()).Set(mtlval)
                        p.RemoveProperty(mtlAttr.GetName())

                # kind - component
                nprim = ExtractAttr.GetPrim(self.stage, newPath)
                Usd.ModelAPI(nprim).SetKind(Kind.Tokens.component)


        if self._ifever == 0:
            if os.path.exists(self.attrFile):
                os.remove(self.attrFile)
            return

        self.stage.GetRootLayer().Save()
        del self.stage

        if isUpdate:
            attrFile = self.attrFile.replace('_update.usd', '.usd')
            shutil.copy2(self.attrFile, attrFile)
            os.remove(self.attrFile)
            self.attrFile = attrFile

        geomLayer = geomStage.GetRootLayer()
        SessionUtils.InsertSubLayer(geomLayer, PathUtils.GetRelPath(self.geomFile, self.attrFile))
        gdprim.GetInherits().AddInherit(rootPath)
        geomLayer.Save()
        return self.attrFile


    def textureAttribute(self, prim, newPath):
        '''
        Create Class prim by txLayerName. ( /_{ASSET}_{txLayerName}_txAttr )
        Update txLayerPathMap for Collection
        '''
        names = prim.GetAuthoredPropertyNames()
        if not ('primvars:txBasePath' in names and 'primvars:txLayerName' in names):
            return

        txBasePath = prim.GetAttribute('primvars:txBasePath').Get()
        txLayerName= prim.GetAttribute('primvars:txLayerName').Get()

        modelVersion = self.modelVersion
        if 'primvars:modelVersion' in names:
            modelVersion = prim.GetAttribute('primvars:modelVersion').Get()

        self._ifever += 1

        splitPath = txBasePath.split('/')
        rootIndex = splitPath.index('asset')
        if 'element' in splitPath:
            rootIndex = splitPath.index('element')
        rootName = splitPath[rootIndex + 1]

        sdfpath = Sdf.Path('/_{NAME}_{LAYER}_txAttr'.format(NAME=rootName, LAYER=txLayerName))
        # Create Class Prim
        if not self.stage.GetPrimAtPath(sdfpath):
            txAttrFile = '{PATH}/tex/tex.attr.usd'.format(PATH=txBasePath)
            if not txAttrFile.startswith('/'):
                txAttrFile = self.showDir + '/' + txAttrFile

            cprim = self.stage.CreateClassPrim(sdfpath)
            cprim.SetPayload(Sdf.Payload(PathUtils.GetRelPath(self.geomFile, txAttrFile), Sdf.Path('/'+txLayerName)))
            vset = cprim.GetVariantSets().GetVariantSet('modelVersion')
            vset.SetVariantSelection(modelVersion)
        nprim = ExtractAttr.GetPrim(self.stage, newPath)
        nprim.GetInherits().AddInherit(sdfpath)

        # remove geomfile texture primvars
        prvapi = UsdGeom.PrimvarsAPI(prim)
        for prv in prvapi.GetPrimvars():
            basename = prv.GetBaseName()
            if basename.startswith('tx') and basename != 'txVarNum':
                prim.RemoveProperty(prv.GetName())


    def CopyPrimvars(self, attrs, sourcePrim, targetPrim):
        rpvClass = UsdGeom.PrimvarsAPI(sourcePrim)
        newClass = UsdGeom.PrimvarsAPI(targetPrim)
        for name in attrs:
            rpv = rpvClass.GetPrimvar(name)
            # Create
            npv = newClass.CreatePrimvar(rpv.GetName(), rpv.GetTypeName())
            npv.SetInterpolation(rpv.GetInterpolation())
            if rpv.IsIndexed():
                npv.SetIndices(rpv.GetIndices())
                # Delete
                sourcePrim.RemoveProperty(rpv.GetIndicesAttr().GetName())
            npv.Set(rpv.Get())
            # Delete
            sourcePrim.RemoveProperty(rpv.GetName())



# 19.07.12 -- mute. don't know, where using it.
# class RemoveAttributes:
