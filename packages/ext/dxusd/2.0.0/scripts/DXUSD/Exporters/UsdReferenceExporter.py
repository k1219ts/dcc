#coding:utf-8
from __future__ import print_function

import os, glob

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

from DXUSD.Structures import Arguments
from DXUSD.Exporters.Export import Export, AExport
import DXUSD.Tweakers as twk
import DXUSD.Compositor as cmp
from pxr import Sdf, Usd ,UsdGeom


class AUsdReferenceExporter(AExport):
    def __init__(self, **kwargs):

        # input argument
        self.input = ''
        self.show = ''
        self.overwrite = True
        self.ver = 'v001'
        self.ovr_asset =''
        
        # treat compute
        self.master = ''
        self.nodes =[]
        self.geomfiles =[]

        # initialize
        AExport.__init__(self, **kwargs)

        # attributes
        self.task = 'model'
        self.taskProduct = 'TASKV'

    def Treat(self):
        tmpnode = self.getNode(self.input) + '_model_GRP'
        if self.ovr_asset:
            tmpnode = self.ovr_asset + '_model_GRP'

        self.N.model.SetDecode(tmpnode)
        self.dstdir = self.D[self.taskProduct]
        self.N.model.SetDecode(tmpnode)
        self.master = utl.SJoin(self.dstdir, self.F.MASTER)

        layer = utl.AsLayer(self.input)
        with utl.OpenStage(layer) as stage:
            for prim in stage.GetDefaultPrim().GetChildren():
                node = prim.GetName()
                if '_' in node:
                    node = utl.renameAsset(node)
                self.N.model.SetDecode(node)
                if not self.lod:
                    self.lod = var.T.HIGH
                self.desc += '_' + node
                ofile = utl.SJoin(self.dstdir, self.F.GEOM)
                self.geomfiles.append(ofile)

        return var.SUCCESS

    def getNode(self,path):
        splitPath = path.split('/')
        asset = splitPath[4]
        if '_' in asset:
            asset = utl.renameAsset(asset)
        if 'element' in splitPath:
            branch = splitPath[6]
            if '_' in branch:
                branch = utl.renameAsset(branch)
            asset = asset + '_' + branch
        tmpnode = asset + '_model_GRP'
        return asset
    

class UsdReferenceExporter(Export):
    ARGCLASS = AUsdReferenceExporter
    def Exporting(self):
        print('input:',self.arg.input)

        refList= self.GetUsdData(self.arg.input)

        # print('refData:',refList)
        for i in range(len(refList)):
            key = refList[i].keys()[0]
            value = refList[i].values()[0]
            ofile = self.arg.geomfiles[i]
            self.CreateGeom(ofile , key, value)

    def CreateGeom(self,ofile, key, refData):
        print(refData)
        dstlyr = utl.AsLayer(ofile, create=True, clear=True)
        dstlyr.defaultPrim = "World"
        world = Sdf.PrimSpec(dstlyr, "World", Sdf.SpecifierDef, 'Xform')
        with utl.OpenStage(dstlyr) as stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        primName =key
        if '_' in key:
            primName = utl.renameAsset(key)
        # print('primName:',primName)

        root = Sdf.PrimSpec(world, primName, Sdf.SpecifierDef, 'Xform')
        pitSpec = utl.GetPrimSpec(dstlyr, root.path.AppendChild('scatter'), type='PointInstancer')
        relspec = Sdf.RelationshipSpec(pitSpec, 'prototypes', False, Sdf.VariabilityUniform)

        for i in refData['pt']['prototypes']:
            refName = i.keys()[0]
            filepath = i.values()[0]
            filepath = self.changeNewPath(self.arg.show, filepath)
            if not 'lidarPillarB' in refName:
                refName = os.path.splitext(os.path.basename(filepath))[0]
            print('refName:',refName)
            print('filepath:',filepath)
            lyr = pitSpec.layer
            rootSpec = utl.GetPrimSpec(lyr, pitSpec.path.AppendChild('Prototypes'))
            spec = utl.GetPrimSpec(lyr, rootSpec.path.AppendChild(refName), specifier='over')
            relpath = utl.GetRelPath(lyr.identifier, filepath)
            utl.ReferenceAppend(spec, relpath)
            print('spec.path:',spec.path)
            relspec.targetPathList.explicitItems.append(spec.path)

        ids = refData['pt']['ids']
        scales = refData['pt']['scales']
        orients = refData['pt']['orientations']
        positions = refData['pt']['positions']
        indices = refData['pt']['indices']
        utl.GetAttributeSpec(pitSpec, 'ids', ids, Sdf.ValueTypeNames.Int64Array)
        utl.GetAttributeSpec(pitSpec, 'scales', scales, Sdf.ValueTypeNames.Float3Array)
        utl.GetAttributeSpec(pitSpec, 'orientations', orients, Sdf.ValueTypeNames.QuathArray)
        utl.GetAttributeSpec(pitSpec, 'positions', positions, Sdf.ValueTypeNames.Point3fArray)
        utl.GetAttributeSpec(pitSpec, 'protoIndices', indices, Sdf.ValueTypeNames.IntArray)

        if refData['layout']:
            for i in range(len(refData['layout'])):
                refName = refData['layout'][i].keys()[0]
                filepath = refData['layout'][i][refName]['filepath']

                filepath = self.changeNewPath(self.arg.show, filepath)
                lspec = utl.GetPrimSpec(dstlyr, root.path.AppendChild(refName))
                relpath = utl.GetRelPath(ofile, filepath)
                utl.ReferenceAppend(lspec, relpath)
                if refData['layout'][i][refName]['matrix']:
                    mtx = refData['layout'][i][refName]['matrix']
                    utl.GetAttributeSpec(lspec, 'xformOp:transform', Gf.Matrix4d(*mtx),
                                         Sdf.ValueTypeNames.Matrix4d)
                    utl.GetAttributeSpec(lspec, 'xformOpOrder', ['xformOp:transform'],
                                         Sdf.ValueTypeNames.TokenArray)
        dstlyr.Save()
        del dstlyr

    def changeNewPath(self, showName, orgpath):
        splitPath = orgpath.split('/')
        asset = splitPath[4]
        branch = ''
        if '_' in asset:
            asset = utl.renameAsset(asset)
        if 'element' in splitPath:
            branch = splitPath[6]
            if '_' in branch:
                branch = utl.renameAsset(branch)
        usdpath = os.path.join('/show', showName, '_3d', 'asset', asset, asset + '.usd')
        if branch:
            usdpath = os.path.join('/show', showName, '_3d', 'asset', asset, 'branch', branch, branch + '.usd')
        return usdpath

    def GetUsdData(self, setpath):
        '''
        :param setpath:
        :return:
            [{primName :
                {'pt': {'indices': '', 'positions': '', 'ids': '', 'orientations': '', 'scales': '' ,'prototypes':[{refName:'filePath'}]},
                 'layout': [{refName: {'filePath': '', 'matrix': ''} }]
                 }
                 }]
        '''
        layer = utl.AsLayer(setpath)
        setList = []
        with utl.OpenStage(layer) as stage:
            dprim = stage.GetDefaultPrim()  # Usd.Prim(</hatchBridge_set>)
            for prim in dprim.GetChildren():
                primName = prim.GetName()  # hatchBridge_set
                setData = {}
                data = {'pt': '',
                        'layout': []}
                for set in prim.GetChildren():
                    if set.GetName() == 'scatter':
                        geom = UsdGeom.PointInstancer(set)
                        prototypes = geom.GetPrototypesRel().GetTargets()
                        data['pt'] = {'ids': geom.GetIdsAttr().Get(),
                                      'indices': geom.GetProtoIndicesAttr().Get(),
                                      'positions': geom.GetPositionsAttr().Get(),
                                      'scales': geom.GetScalesAttr().Get(),
                                      'orientations': geom.GetOrientationsAttr().Get(),
                                      'prototypes': []
                                      }
                        for i in xrange(len(prototypes)):
                            prim = stage.GetPrimAtPath(prototypes[i])
                            refs = prim.GetMetadata('references')
                            if refs:
                                filePath = refs.GetAddedOrExplicitItems()[0].assetPath
                                refName = os.path.basename(filePath).split('.')[0]
                                if '/mach/' in filePath:
                                    filePath = filePath.split('/mach')[-1]
                                data['pt']['prototypes'].append({refName: filePath})
                    else:
                        refName = set.GetName()
                        refs = set.GetMetadata('references')
                        if refs:
                            filePath = refs.GetAddedOrExplicitItems()[0].assetPath
                            matrix = set.GetAttribute('xformOp:transform').Get()
                            data['layout'].append({refName: {'filepath': filePath, 'matrix': matrix}})
                setData[primName] = data
                setList.append(setData)

        return setList

    def Arguing(self):
        return var.SUCCESS

    def Tweaking(self):
        twks = twk.Tweak()
        twks << twk.MasterModelPack(self.arg)
        twks.DoIt()
        return var.SUCCESS

    def Compositing(self):
        cmp.Composite(self.arg.master).DoIt()
        return var.SUCCESS

