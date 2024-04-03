
import os
import glob

from fnpxr import Sdf

import DXRulebook.Interface as rb
import DXUSD_KAT.Utils as utl

_ASSETLIBPRMAN = '/assetlib/_3d/asset/_global/material/prman/prman.usd'

def ShaderMaster(out, src, vsets=list()):
    def getVariantSpec(parent, name, value):
        vsetSpec = utl.GetVariantSetSpec(parent, name)
        vspec    = utl.GetVariantSpec(vsetSpec, value)
        parent.variantSelections.update({name: value})
        return vspec.primSpec

    srclyr   = utl.AsLayer(src)
    relpath  = utl.GetRelPath(out, src)
    primPath = Sdf.Path('/' + srclyr.defaultPrim)

    outlyr = utl.AsLayer(out, create=True)
    outlyr.defaultPrim = srclyr.defaultPrim
    root = utl.GetPrimSpec(outlyr, primPath, specifier='over')

    parent = root
    for n, v in vsets:
        parent = getVariantSpec(parent, n, v)

    utl.ReferenceAppend(parent, relpath)

    outlyr.Save()
    del outlyr


class MaterialComposite:
    def __init__(self, input):  # input : material filename
        self.arg = utl.Arguments(utl.DirName(input))
        self.prman  = self.arg.prmanMaster
        self.master = self.arg.materialMaster

        self.src = input

        # base material
        tmp_fn = utl.SJoin(self.arg.D.ROOTS, 'asset', '_global', 'material', 'prman', 'prman.usd')
        if os.path.exists(tmp_fn):
            self.BASEPRMAN = tmp_fn
        else:
            self.BASEPRMAN = _ASSETLIBPRMAN
        self.BASEPRMAN_LAYER = utl.AsLayer(self.BASEPRMAN)


    def IsPrim(self, primPath):
        if self.BASEPRMAN_LAYER.GetPrimAtPath(primPath):
            return True

    def modifyPrimSpec(self, spec):
        if len(spec.referenceList.prependedItems) == 0:
            spec.specifier = Sdf.SpecifierDef
            spec.typeName  = 'Material'
        else:
            spec.specifier = Sdf.SpecifierOver
            spec.typeName  = ''


    def GlobalPrmanMaster(self):
        outlyr = utl.AsLayer(self.prman, create=True)
        outlyr.defaultPrim = 'prman'

        if self.arg.show and not self.arg.seq:
            utl.SubLayersAppend(outlyr, _ASSETLIBPRMAN)

        root = utl.GetPrimSpec(outlyr, '/prman', type='Scope')

        spec = utl.GetPrimSpec(outlyr, root.path.AppendChild(self.arg.nslyr), specifier='over')
        utl.ReferenceAppend(spec, utl.GetRelPath(self.prman, self.src), clear=True)

        outlyr.Save()
        del outlyr

    def PrmanMaster(self):
        outlyr = utl.AsLayer(self.prman, create=True)
        outlyr.defaultPrim = 'prman'

        root = utl.GetPrimSpec(outlyr, '/prman', type='Scope')

        spec = utl.GetPrimSpec(outlyr, root.path.AppendChild(self.arg.nslyr), specifier='over')
        utl.ReferenceAppend(spec, utl.GetRelPath(self.prman, self.src))

        # cleanup global materials
        if not self.arg.subdir:
            for r in spec.referenceList.prependedItems:
                if r.assetPath.endswith('_global/material/prman/prman.usd'):
                    spec.referenceList.prependedItems.remove(r)
            if self.IsPrim(spec.path):
                ref = Sdf.Reference(utl.GetRelPath(self.prman, self.BASEPRMAN), primPath=spec.path)
                if spec.referenceList.prependedItems.index(ref) == -1:
                    spec.referenceList.prependedItems.append(ref)

        # prim modify
        self.modifyPrimSpec(spec)

        outlyr.Save()
        del outlyr


    def DoIt(self):
        print('#### Composite Material ####')
        print('# Material Name\t\t:', self.arg.nslyr)
        # print(self.arg)

        # collect version
        print('# Material Master\t:', self.master)
        print('# COMP VER [%s]\t:' % self.arg.nsver, self.src)
        print('')
        ShaderMaster(self.master, self.src, [('entity', self.arg.entity), ('mtlVer', self.arg.nsver)])
        self.src = self.master

        # prman.usd
        print('# Prman Master\t\t:', self.prman)
        print('')
        if self.arg.asset == '_global':
            self.GlobalPrmanMaster()
        else:
            self.PrmanMaster()

        del self.BASEPRMAN_LAYER
