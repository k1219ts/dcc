import os

from pxr import Usd, UsdGeom, UsdShade, Sdf

import dxsUsdUtils
import PathUtils
import SessionUtils


#-------------------------------------------------------------------------------
def AddCollectionTarget(prim, name, source):
    col = Usd.CollectionAPI.GetCollection(prim, name)
    if col:
        rel = col.GetIncludesRel()
        rel.AddTarget(source)
    else:
        col = Usd.CollectionAPI.ApplyCollection(prim, name, Usd.Tokens.expandPrims)
        rel = col.CreateIncludesRel()
        rel.AddTarget(source)



def CompositeCollection(outfile, colfile, SdfPath=None):
    if not (os.path.exists(outfile) and os.path.exists(colfile)):
        return
    stage = Usd.Stage.Open(outfile, load=Usd.Stage.LoadNone)
    if SdfPath:
        prim = stage.GetPrimAtPath(SdfPath)
    else:
        prim = stage.GetDefaultPrim()
    prim.GetReferences().AddReference(Sdf.Reference(PathUtils.GetRelPath(outfile, colfile)))
    stage.GetRootLayer().Save()


#-------------------------------------------------------------------------------
class Main:
    '''
    '''
    def __init__(self, filename):
        # Member Variables
        fileDir   = os.path.dirname(filename)
        splitPath = filename.split('/')
        if "show" in splitPath:
            self.rootDir = '/'.join(splitPath[:splitPath.index('show') + 2])
        elif "assetlib" in splitPath:
            self.rootDir = '/'.join(splitPath[:splitPath.index('assetlib') + 2])
        else:
            assert False, "not found show or assetlib => %s" % filename
        self.outfile = '{DIR}/collection.usd'.format(DIR=fileDir)
        self.prvNameMap = dict()
        self.mtlNames   = list()

        self.istage = Usd.Stage.Open(filename)
        self.idprim = self.istage.GetDefaultPrim()

        self.ostage = SessionUtils.MakeInitialStage(self.outfile, clear=False)
        self.odprim = self.ostage.OverridePrim(self.idprim.GetPath())
        self.ostage.SetDefaultPrim(self.odprim)

        self.createCollection()

        mtlfile = '{DIR}/materials.usd'.format(DIR=fileDir)
        CreateMaterials(mtlfile, self.prvNameMap, self.mtlNames)
        BindingMaterials(self.ostage, self.odprim)

        self.ostage.GetRootLayer().Save()

        CompositeCollection(filename, self.outfile)


    def createCollection(self):
        vsets = self.idprim.GetVariantSets()
        if vsets.HasVariantSet('lodVariant'):
            lodvset = vsets.GetVariantSet('lodVariant')
            selected= lodvset.GetVariantSelection()

            ovset = self.odprim.GetVariantSets().AddVariantSet('lodVariant')
            for name in lodvset.GetVariantNames():
                lodvset.SetVariantSelection(name)
                ovset.AddVariant(name)
                ovset.SetVariantSelection(name)
                with ovset.GetVariantEditContext():
                    self.iterPrim(lodvset.GetPrim())

            lodvset.SetVariantSelection(selected)
            ovset.SetVariantSelection(selected)

        else:
            self.iterPrim(self.idprim)

    def iterPrim(self, parent):
        for p in iter(Usd.PrimRange.AllPrims(parent)):
            names = p.GetAuthoredPropertyNames()
            if 'primvars:txBasePath' in names and 'primvars:txLayerName' in names:
                txBasePath = p.GetAttribute('primvars:txBasePath').Get()
                txLayerName= p.GetAttribute('primvars:txLayerName').Get()

                splitPath  = txBasePath.split('/')
                rid = splitPath.index('asset')
                if 'element' in splitPath:
                    rid = splitPath.index('element')
                rootName = splitPath[rid + 1]

                baseName = '{ASSET}_{LAYER}'.format(ASSET=rootName, LAYER=txLayerName)

                # for preview
                colname = 'pb_' + baseName
                AddCollectionTarget(self.odprim, colname, p.GetPath())
                # update prvNameMap
                self.update_prvNameMap(p, baseName, txBasePath)

            if 'userProperties:MaterialSet' in names:
                # for hdPrman
                mtlset = p.GetAttribute('userProperties:MaterialSet').Get()
                if mtlset:
                    if not mtlset in self.mtlNames:
                        self.mtlNames.append(mtlset)
                    colname = 'fb_' + mtlset
                    AddCollectionTarget(self.odprim, colname, p.GetPath())

    def update_prvNameMap(self, prim, name, txBasePath):
        if not self.prvNameMap.has_key(name):
            if not txBasePath.startswith('/'):
                stackfile = dxsUsdUtils.GetPrimStackFilePath(prim, 'tex_attr.usd')
                if stackfile:
                    splitPath = stackfile.split('/')
                    rootDir   = '/'.join(splitPath[:splitPath.index('show') + 2])
                else:
                    rootDir = self.rootDir
                txBasePath  = os.path.join(rootDir, txBasePath)
            self.prvNameMap[name] = txBasePath


class CreateMaterials:
    def __init__(self, filename, prvNameMap, mtlNames):
        self.filename = filename

        stage = SessionUtils.MakeInitialStage(filename, clear=True)
        dprim = stage.DefinePrim('/MaterialRoot', 'Xform')
        stage.SetDefaultPrim(dprim)

        prim = stage.DefinePrim(dprim.GetPath().AppendChild('Materials'), 'Scope')

        if prvNameMap:
            self.previewMtl(stage, prim, prvNameMap)

        if mtlNames:
            self.prmanMtl(stage, prim, mtlNames)

        stage.GetRootLayer().Save()
        del stage
        print '# result: Geom Material ->', filename


    def previewMtl(self, stage, parent, prvNameMap):
        root = stage.DefinePrim(parent.GetPath().AppendChild('preview'), 'Scope')
        for name in prvNameMap:
            mpath = '/MaterialRoot/Materials/preview/' + name
            prim  = stage.OverridePrim(Sdf.Path(mpath))
            fn = '{DIR}/proxy/proxy.mtl.usd'.format(DIR=prvNameMap[name])
            ref= Sdf.Reference(PathUtils.GetRelPath(self.filename, fn), Sdf.Path('/' + name))
            prim.GetReferences().AddReference(ref)


    def prmanMtl(self, stage, parent, mtlNames):
        prmanMtlFile = '/assetlib/3D/material/usd/shaders/Materials.usd'
        mstage = Usd.Stage.Open(prmanMtlFile, load=Usd.Stage.LoadNone)
        prmans = list()
        for c in mstage.GetDefaultPrim().GetAllChildren():
            prmans.append(c.GetName())
        del mstage

        root = stage.DefinePrim(parent.GetPath().AppendChild('prman'), 'Scope')
        for name in mtlNames:
            setname = name
            if not name in prmans:
                setname = 'default'
            mpath = '/MaterialRoot/Materials/prman/' + setname
            prim  = stage.OverridePrim(Sdf.Path(mpath))
            ref = Sdf.Reference(PathUtils.GetRelPath(self.filename, prmanMtlFile), Sdf.Path('/Materials/' + setname))
            prim.GetReferences().AddReference(ref)



class BindingMaterials:
    def __init__(self, stage, parent):  # parent : Usd.Prim
        stagefile = stage.GetRootLayer().identifier

        mtlfile = os.path.join(os.path.dirname(stagefile), 'materials.usd')
        if not os.path.exists(mtlfile):
            print '# [ Error - BindingMaterials ] - not found %s' % mtlfile
            return

        prvset = dxsUsdUtils.VariantSelection(parent, 'preview', 'on')
        prvset.AddVariant('off')
        with prvset.GetVariantEditContext():
            prvset.GetPrim().GetReferences().AddReference('./materials.usd')

        # Selection 'lodVariant'
        vsets = parent.GetVariantSets()
        if vsets.HasVariantSet('lodVariant'):
            lodvset = vsets.GetVariantSet('lodVariant')
            selected= lodvset.GetVariantSelection()

            for name in lodvset.GetVariantNames():
                lodvset.SetVariantSelection(name)
                with prvset.GetVariantEditContext():
                    vset = dxsUsdUtils.VariantSelection(prvset.GetPrim(), 'lodVariant', name)
                    with vset.GetVariantEditContext():
                        self.makeBind(stage, vset.GetPrim())
                    vset.SetVariantSelection(selected)

            lodvset.SetVariantSelection(selected)
        else:
            with prvset.GetVariantEditContext():
                self.makeBind(stage, parent)


    def makeBind(self, stage, prim):
        for col in Usd.CollectionAPI.GetAllCollections(prim):
            name = col.GetName()
            if name.startswith('pb_'):  # for preview
                mtlpath = prim.GetPath().AppendPath('Materials/preview/' + name.replace('pb_', ''))
                mtlprim = stage.GetPrimAtPath(mtlpath)
                if mtlprim:
                    self.bind(mtlprim, prim, col, UsdShade.Tokens.preview)
            elif name.startswith('fb_'):
                mtlpath = prim.GetPath().AppendPath('Materials/prman/' + name.replace('fb_', ''))
                mtlprim = stage.GetPrimAtPath(mtlpath)
                if not mtlprim:
                    mtlpath = prim.GetPath().AppendPath('Materials/prman/default')
                    mtlprim = stage.GetPrimAtPath(mtlpath)
                if mtlprim:
                    self.bind(mtlprim, prim, col, UsdShade.Tokens.full)

    def bind(self, mtlprim, prim, bindcol, purpose):
        mtlgeom = UsdShade.Material(mtlprim)
        bindAPI = UsdShade.MaterialBindingAPI(prim)
        bindAPI.Bind(bindcol, mtlgeom, bindingName=bindcol.GetName(), materialPurpose=purpose)
