'''
USD Environment Set Export

Outputs
asset / $ASSET / model / model.paylod.usd
                       / model.usd
                       / $VER / $ASSET.payload.usd
                              / $ASSET.usd
                              / $NODE.geom.usd

shot / $SEQ / $SHOT / set / set.payload.usd
                          / set.usd
                          / $VER / setscene.payload.usd
                                 / setscene.usd
                                 / $NODE.geom.usd
'''
import os, string

from pxr import Sdf, Usd, UsdGeom, Kind, Vt, Gf

print "Original"

import maya.cmds as cmds

import dxsMsg
import Arguments
import dxsMayaUtils
import dxsXformUtils
import dxsUsdUtils
import PathUtils
import SessionUtils
import PackageUtils
import Texture
import GeomMain


def GetSetNodes(selected=None):
    if selected:
        nodes = cmds.ls(selected, l=True)
    else:
        nodes = cmds.ls('*_set*', l=True, r=True)
    result = list()

    for n in nodes:
        if n.find('_set') > -1 and dxsMayaUtils.GetViz(n):
            ntype = cmds.nodeType(n)
            if ntype == 'TN_TaneTransform':
                result.append(n)
            if ntype == 'dxAssembly':
                result.append(n)
            if ntype == 'xBlock':
                btype = cmds.getAttr('%s.type' % n)
                if btype == 1 or btype == 4:
                    result.append(n)

    for n in nodes:
        if n.find('_set') > -1 and dxsMayaUtils.GetViz(n):
            ntype = cmds.nodeType(n)
            if ntype == 'pxrUsdReferenceAssembly':
                _isnot = 0
                splitStr = n.split('|')
                for i in range(1, len(splitStr)+1):
                    npath = string.join(splitStr[:i], '|')
                    if npath in result:
                        _isnot += 1
                if not _isnot:
                    result.append(n)

    result.sort()
    return result


class PointInstancerBlock:
    def __init__(self):
        self.ids = list()
        self.positions = list()
        self.scales    = list()
        self.orients   = list()
        self.tintColor = list()     # [(0, 0, 0), (1, 0, 0) ....]
        self.protoIndices= list()
        self.protoTypes  = list()   # [{'name': nodeName, 'reference': Sdf.Reference, 'vsets': {}, 'mtx': []}, ...]
        self.layoutTypes = list()   # [{'name': nodeName, 'reference': Sdf.Reference, 'vsets': {}, 'mtx': [], 'instance': bool}, ...]


class GetTane:
    '''
    Args:
        node (str): TN_TaneTransform node name
        data (PointInstancerBlock)
    '''
    def __init__(self, node, data):
        self.node = cmds.ls(node, dag=True, type='TN_Tane', l=True)[0]
        self.count= cmds.TN_GetPoint(nn=self.node, tg='count')
        self.data = data

    def doIt(self):
        if self.count == 0:
            print '# msg: tane count is zero!'
            return

        TN_pos = cmds.TN_GetPoint(nn=self.node, tg='pos')
        TN_scl = cmds.TN_GetPoint(nn=self.node, tg='scl')
        TN_rot = cmds.TN_GetPoint(nn=self.node, tg='rot')
        for i in xrange(self.count):
            self.data.ids.append(i)
            pos = TN_pos[i*3:i*3+3]
            scl = TN_scl[i*3:i*3+3]
            rot = TN_rot[i*3:i*3+3]
            pos, scl, rot = dxsXformUtils.GetGfXform(transform=(pos, scl, rot, 0)).Get()
            self.data.positions.append(pos)
            self.data.scales.append(scl)
            self.data.orients.append(rot)
        self.data.protoIndices = cmds.TN_GetPoint(nn=self.node, tg='sid')
        self.getPrototype()

    def getPrototype(self):
        envNode   = cmds.listConnections(self.node, type='TN_Environment', s=True, d=False)[0]
        sourceIDs = cmds.TN_QuerySources(nn=self.node, tg='sourceIDs')
        self.data.protoTypes = [None] * len(sourceIDs)
        for index in sourceIDs:
            proto = dict()
            srcNode = cmds.listConnections('%s.inSource[%s]' % (envNode, index), s=True, d=False, sh=True)[0]
            proto['name'] = srcNode.split('|')[-1].split(':')[-1]
            if cmds.nodeType(srcNode) == 'TN_UsdProxy':
                filename = cmds.getAttr('%s.renderFile' % srcNode)
                offset = 0
                if cmds.getAttr('%s.animCycle' % srcNode) == 1:
                    offset = cmds.getAttr('%s.frameOffset' % srcNode)
                ref = Sdf.Reference(filename, layerOffset=Sdf.LayerOffset(offset))
                proto['reference'] = ref
                # variant selection
                vsets = cmds.listAttr(srcNode, ud=True, st='usdVariantSet*')
                if vsets:
                    proto['vsets'] = {}
                    for ln in vsets:
                        name = ln.replace('usdVariantSet_', '')
                        value= cmds.getAttr(srcNode + '.' + ln)
                        if value:
                            proto['vsets'][name] = value
                # matrix
                mtx = cmds.xform(cmds.listRelatives(srcNode, p=True)[0], q=True, m=True)    # TODO: add space option
                proto['mtx'] = mtx
            self.data.protoTypes[index] = proto


class GetDxAssembly:
    def __init__(self, node, data, filePath):
        self.node = node
        self.data = data
        self.geomFile = filePath

    @staticmethod
    def GetFile(node):
        nt = cmds.nodeType(node)
        if nt == 'pxrUsdReferenceAssembly' or nt == 'pxrUsdProxyShape':
            fn = cmds.getAttr('%s.filePath' % node)
            return fn
        else:   # dxComponent
            fn = cmds.getAttr('%s.renderFile' % node)
            if not fn:
                fn = cmds.getAttr('%s.abcFileName' % node)
            return fn

    @staticmethod
    def GetRefName(filename, vsets, offset):
        '''
        Args:
            filename (str) - reference filename
            vsets (list)   - variant selection list ['A=B', 'C=D']
        '''
        # change log : name - filename based -> root prim name @20190425
        rootLayer = Sdf.Layer.FindOrOpen(filename)
        if rootLayer.rootPrims:
            name = rootLayer.rootPrims[0].name
        else:
            name = rootLayer.defaultPrim

        if "/element/" in filename:
            splitFileName = filename.split('/')
            if 'asset' in splitFileName:
                name = '%s_%s' % (splitFileName[splitFileName.index('asset') + 1], name)

        if vsets:
            name += '_' + string.join(vsets.values(), '_')
        if offset:
            name += '_' + str(int(offset))
        return name

    @staticmethod
    def GetVariantSelection(node):
        vsets = dict()
        nt = cmds.nodeType(node)
        if nt == 'pxrUsdReferenceAssembly':
            vsetAttrs = cmds.listAttr(node, ud=True, st='usdVariantSet_*')
            if vsetAttrs:
                for ln in vsetAttrs:
                    name = ln.replace('usdVariantSet_', '')
                    value= cmds.getAttr(node + '.' + ln)
                    if value:
                        vsets[name] = value
        return vsets

    @staticmethod
    def GetTimeOffset(node):
        offset = 0
        timeNode = cmds.listConnections('%s.time' % node, d=False, s=True, type='dxTimeOffset')
        if timeNode:
            offset = cmds.getAttr('%s.offset' % timeNode[0])
        return offset

    @staticmethod
    def GetVariation(node):
        varnum = 0
        if cmds.attributeQuery("txVarNum", n = node, exists = True):
            varnum = cmds.getAttr('%s.txVarNum' % node)

        return varnum

    @staticmethod
    def GetSource(node):    # node is dxAssembly node
        # pxrUsdReferenceAssembly, pxrUsdProxyShape
        pxrnodes = list()
        for n in cmds.ls(node, dag=True, l=True, type=['pxrUsdReferenceAssembly', 'pxrUsdProxyShape']):
            if cmds.nodeType(n) == 'pxrUsdProxyShape':
                transNode = cmds.listRelatives(n, p=True, f=True)[0]
                if cmds.nodeType(transNode) != 'pxrUsdReferenceAssembly':
                    pxrnodes.append(n)
            else:
                pxrnodes.append(n)

        pxrUsdRefMap= dict()
        for n in pxrnodes:
            if dxsMayaUtils.GetViz(n):
                filename = GetDxAssembly.GetFile(n)
                basename = os.path.splitext(os.path.basename(filename))[0]
                vsets = GetDxAssembly.GetVariantSelection(n)
                offset= GetDxAssembly.GetTimeOffset(n)
                name  = GetDxAssembly.GetRefName(filename, vsets, offset)
                if not pxrUsdRefMap.has_key(name):
                    pxrUsdRefMap[name] = {'filename': filename, 'nodes': list()}
                    if vsets:
                        pxrUsdRefMap[name]['vsets'] = vsets
                    if offset:
                        pxrUsdRefMap[name]['offset'] = offset
                nn = n
                if cmds.nodeType(n) == 'pxrUsdProxyShape':
                    nn = cmds.listRelatives(n, p=True, f=True)[0]
                pxrUsdRefMap[name]['nodes'].append(nn)
        return pxrUsdRefMap


    def doIt(self):
        refMap = GetDxAssembly.GetSource(self.node)
        protoIndex   = 0
        scatterIndex = 0
        for name in refMap:
            nodes = refMap[name]['nodes']
            if len(nodes) >= 2:
                # scatter
                prototype = dict()
                prototype['name'] = name
                if refMap[name].has_key('vsets') and refMap[name]['vsets']:
                    prototype['vsets'] = refMap[name]['vsets']
                offset = 0
                if refMap[name].has_key('offset') and refMap[name]['offset']:
                    offset = refMap[name]['offset']
                relPath = PathUtils.GetRelPath(self.geomFile, refMap[name]['filename'])
                print relPath
                prototype['reference'] = Sdf.Reference(relPath, layerOffset=Sdf.LayerOffset(offset * -1)) # refMap[name]['filename']
                self.data.protoTypes.append(prototype)

                for i in xrange(len(nodes)):
                    self.data.ids.append(i + scatterIndex)
                    node = nodes[i]
                    self.data.protoIndices.append(protoIndex)
                    if cmds.nodeType(node) == 'pxrUsdProxyShape':
                        node = cmds.listRelatives(node, p=True, f=True)[0]
                    pos, scl, rot = dxsXformUtils.GetGfXform(matrix=cmds.xform(node, q=True, m=True, ws=True)).Get()
                    self.data.positions.append(pos)
                    self.data.scales.append(scl)
                    self.data.orients.append(rot)
                    tintColor = cmds.getAttr("%s.tintColor" % node)[0]
                    self.data.tintColor.append(tintColor)

                protoIndex += 1
                scatterIndex += len(nodes)

            else:
                # layout
                layout = dict()
                layout['name'] = name
                layout['mtx']  = cmds.xform(nodes[0], q=True, m=True, ws=True)
                if refMap[name].has_key('vsets') and refMap[name]['vsets']:
                    layout['vsets'] = refMap[name]['vsets']
                offset = 0
                if refMap[name].has_key('offset') and refMap[name]['offset']:
                    offset = refMap[name]['offset']
                layout['reference'] = Sdf.Reference(refMap[name]['filename'], layerOffset=Sdf.LayerOffset(offset * -1))
                layout['instance']  = False
                self.data.layoutTypes.append(layout)




class EnvSetGeom:
    def __init__(self, node, data, filename):
        self.node = node
        self.data = data
        self.filename = filename

        self.comment = None

    def doIt(self):
        if not self.data.protoTypes and not self.data.layoutTypes:
            print '# msg : not found data'
            return
        name  = self.node.split('|')[-1].split(':')[-1]
        stage = SessionUtils.MakeInitialStage(self.filename, usdformat='usdc', clear=True, comment=self.comment)
        dprim = stage.DefinePrim('/' + name, 'Xform')
        stage.SetDefaultPrim(dprim)
        dxsUsdUtils.SetModelAPI(dprim, name=name, kind='component')

        if self.data.ids:
            self.makePointInstancer(stage, dprim)
        if self.data.layoutTypes:
            self.makeReferenceLayout(stage, dprim)

        stage.GetRootLayer().Save()
        print "# Export usd file '%s'" % self.filename
        return True


    def GetPrim(self, stage, primPath):
        prim = stage.GetPrimAtPath(primPath)
        if not prim:
            prim = stage.DefinePrim(primPath, 'Xform')
        return prim


    def makePointInstancer(self, stage, parent):
        istPrim = stage.DefinePrim(parent.GetPath().AppendChild('scatter'), 'PointInstancer')
        istGeom = UsdGeom.PointInstancer(istPrim)
        istGeom.CreateIdsAttr(self.data.ids)
        istGeom.CreatePositionsAttr(Vt.Vec3fArray(self.data.positions))
        istGeom.CreateScalesAttr(Vt.Vec3fArray(self.data.scales))
        istGeom.CreateOrientationsAttr(Vt.QuathArray(self.data.orients))
        istGeom.CreateProtoIndicesAttr(Vt.IntArray(self.data.protoIndices))
        dxsUsdUtils.AddPrimvar(istGeom, "uids", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex, self.data.ids)

        protorel = self.makePrototypes(stage, istPrim)
        istGeom.GetPrototypesRel().SetTargets(protorel)

    def makePrototypes(self, stage, parent):
        protoprim = stage.DefinePrim(parent.GetPath().AppendChild('Prototypes'))
        protorel  = list()
        for proto in self.data.protoTypes:
            prim = stage.DefinePrim(protoprim.GetPath().AppendChild(proto['name']))
            if proto.has_key('reference'):
                prim.GetReferences().AddReference(proto['reference'])
                # Add protoXform
                if proto.has_key('mtx'):
                    dxsXformUtils.AddXformOp(prim, matrix=proto['mtx'])
                # variant selection
                if proto.has_key('vsets'):
                    for name in proto['vsets']:
                        vset = prim.GetVariantSets().GetVariantSet(name)
                        vset.SetVariantSelection(proto['vsets'][name])
            protorel.append(prim.GetPath())
        return protorel


    def makeReferenceLayout(self, stage, parent):
        for layout in self.data.layoutTypes:
            prim = self.GetPrim(stage, parent.GetPath().AppendChild(layout['name']))
            prim.SetInstanceable(layout['instance'])
            prim.GetReferences().AddReference(layout['reference'])
            dxsXformUtils.AddXformOp(prim, matrix=layout['mtx'])
            # variant selection
            if layout.has_key('vsets'):
                for name in layout['vsets']:
                    vset = prim.GetVariantSets().GetVariantSet(name)
                    vset.SetVariantSelection(layout['vsets'][name])



#-------------------------------------------------------------------------------
#
#   ASSET
#
#-------------------------------------------------------------------------------
class SetAssetExport(Arguments.AssetArgs):
    '''
    '''
    def __init__(self, node=None, isElement=False, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])
        self.isElement = isElement

        self.node = None
        self.elementName = None

        Arguments.AssetArgs.__init__(self, **kwargs)
        if not self.outDir and self.assetDir:
            if self.isElement:
                self.elementName = self.getElementName(node)
                self.outDir = '{DIR}/element/{NAME}/model'.format(DIR=self.assetDir, NAME=self.elementName)
            else:
                self.outDir = '{DIR}/model'.format(DIR=self.assetDir)
        self.computeVersion()

        self.node = GetSetNodes(node)
        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        # Member Variables
        self.geomFile = None    # using by MtoK

    def getElementName(self, node):
        name = node.split('_set')[0]
        name = name.replace('%s_' % self.assetName.split('_set')[0], '') + '_set'
        return str(name)

    def initTransform(self, node):
        mtx = cmds.xform(node, q=True, m=True, ws=True)
        cmds.setAttr('%s.translate' % node, 0, 0, 0, type='double3')
        cmds.setAttr('%s.rotate' % node, 0, 0, 0, type='double3')
        cmds.setAttr('%s.scale' % node, 1, 1, 1, type='double3')
        return mtx

    def setDxAssemblyFile(self, filename):
        if self.showDir and self.assetDir and cmds.nodeType(self.node) == 'dxAssembly':
            cmds.setAttr('%s.fileName' % self.node, filename, type='string')

    def doIt(self):
        if not self.node:
            return
        self.node = self.node[0]
        if not dxsMayaUtils.GetViz(self.node):
            print '# msg : viz off'
            return
        ntype = cmds.nodeType(self.node)

        name = self.node.split('|')[-1].split(':')[-1]
        geomFile = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=name)
        geomMtx  = None
        _ifInstancerGeom = False

        if ntype == 'TN_TaneTransform':
            cmds.TN_ExportTane(fn=geomFile, nn=cmds.ls(self.node, dag=True, type='TN_Tane')[0])
            self.geomFile = geomFile
            print "# Export usd file '%s'" % geomFile
        elif ntype == 'xBlock':
            btype = cmds.getAttr('%s.type' % self.node)
            if btype == 1:      # Model
                GeomMain.Export(geomFile, [self.node]).doIt()
                self.geomFile = geomFile
            elif btype == 4:    # Layout
                _ifInstancerGeom = True
        elif ntype == 'dxAssembly':
            _ifInstancerGeom = True

        if _ifInstancerGeom:
            geomMtx = self.initTransform(self.node)
            Data = PointInstancerBlock()
            GetDxAssembly(self.node, Data, geomFile).doIt()
            geomClass = EnvSetGeom(self.node, Data, geomFile)
            geomClass.comment = self.comment
            if geomClass.doIt():
                self.geomFile = geomFile

        if geomMtx:
            cmds.xform(self.node, m=geomMtx, ws=True)

        if not self.geomFile:
            return

        if self.showDir and self.assetDir:
            texOverrideFile = '{DIR}/asset/{ASSET}/texture/texture.override.usd'.format(DIR=self.showDir, ASSET=self.assetName)
            if os.path.exists(texOverrideFile):
                SetAssetExport.AddOverrideInherit(geomFile, texOverrideFile)

        geomMasterFile = self.makeGeomPackage(geomFile)

        # Packaging
        if self.elementName:
            taskPayloadFile = self.makeElementPackage(geomMasterFile)
        else:
            taskPayloadFile = self.makeModelPackage(geomMasterFile)

        if self.showDir and self.assetDir:
            PackageUtils.AssetPackage(self.showDir, self.assetName, taskPayloadFile)


    def makeModelPackage(self, sourceFile):
        taskFile = '{DIR}/model.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{taskVariant=model}' % self.assetName
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath=SdfPath, clear=True)
        return taskPayloadFile

    def makeElementPackage(self, sourceFile):
        modelFile = '{DIR}/model.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(modelFile, [sourceFile])

        elmFile = '{DIR}/element/{NAME}/{NAME}.usd'.format(DIR=self.assetDir, NAME=self.elementName)
        SessionUtils.MakeReferenceStage(elmFile, [(modelFile, None)], SdfPath='/' + self.elementName, composite='reference')
        elmPayloadFile = elmFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(elmPayloadFile, [(elmFile, None)], SdfPath='/%s{elementVariant=%s}' % (self.assetName, self.elementName), clear=True)
        # dxAssembly set filename
        self.setDxAssemblyFile(elmFile)

        taskFile = '{DIR}/element/element.usd'.format(DIR=self.assetDir)
        SessionUtils.MakeSubLayerStage(taskFile, [elmPayloadFile])
        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/%s{taskVariant=element}' % self.assetName, clear=True)
        return taskPayloadFile

    def makeGeomPackage(self, geomFile):
        masterFile = '{DIR}/{VER}/{NAME}.usd'.format(DIR=self.outDir, VER=self.version, NAME=self.assetName)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/'+self.assetName, composite='reference', addChild=True, comment=self.comment)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath = '/%s{modelVersion=%s}' % (self.assetName, self.version)
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, clear=True, comment=self.comment)
        return masterPayloadFile


    @staticmethod
    def AddOverrideInherit(geomFile, texOverrideFile):
        '''
        For TextureOverride
        '''
        texRootDir = os.path.dirname(texOverrideFile)
        stage   = Usd.Stage.Open(geomFile, load=Usd.Stage.LoadNone)
        outLayer= stage.GetRootLayer()
        SessionUtils.InsertSubLayer(outLayer, PathUtils.GetRelPath(geomFile, texOverrideFile))

        refAssetList = list()
        texLayer = Sdf.Layer.FindOrOpen(texOverrideFile)
        for lyr in texLayer.subLayerPaths:
            if os.path.basename(lyr) == 'texture.override.usd':
                refAssetList.append(os.path.basename(os.path.dirname(lyr)))

        for p in iter(Usd.PrimRange.AllPrims(stage.GetDefaultPrim())):
            refs = p.GetMetadata('references')
            if refs:
                for r in refs.GetAddedOrExplicitItems():
                    refAsset = os.path.basename(r.assetPath).split('.')[0]
                    if refAsset in refAssetList:
                        p.GetInherits().AddInherit(Sdf.Path('/_{NAME}_OverAttr'.format(NAME=refAsset)))

        outLayer.Save()



class SetAssetImport:
    def __init__(self, filename):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])
        self.filename = filename

    def doIt(self):
        rootLayer = Sdf.Layer.FindOrOpen(self.filename)
        if not rootLayer:
            return

        self.stage = Usd.Stage.Open(rootLayer)
        dprim = self.stage.GetDefaultPrim()

        instancerPrims = list()
        for p in iter(Usd.PrimRange.AllPrims(self.stage.GetPseudoRoot())):
            if p.GetTypeName() == 'PointInstancer':
                instancerPrims.append(p)

        if not instancerPrims:
            return

        rootNode = cmds.createNode('dxAssembly', n=dprim.GetName())

        for p in instancerPrims:
            self.expandPointInstancer(p, rootNode)

        for p in self.stage.GetDefaultPrim().GetChildren():
            if p.GetName() != 'scatter':
                refs = p.GetMetadata('references')
                if refs:
                    r = refs.GetAddedOrExplicitItems()[0]
                    node= self.makeReferenceGeom(r.assetPath)
                    mtx = UsdGeom.Xform(p).MakeMatrixXform().Get()
                    if mtx:
                        mtx = self.getMatrixList(mtx)
                        cmds.xform(node, m=mtx, ws=True)
                    cmds.parent(node, rootNode)
        cmds.select(rootNode)
        return rootNode


    def expandPointInstancer(self, prim, rootNode):
        geom = UsdGeom.PointInstancer(prim)

        # prototypes info
        protoFiles = self.getPrototypesInfo(geom)

        indices  = geom.GetProtoIndicesAttr().Get()
        positions= geom.GetPositionsAttr().Get()
        orients  = geom.GetOrientationsAttr().Get()
        scales   = geom.GetScalesAttr().Get()

        for i in xrange(len(indices)):
            mtx   = dxsXformUtils.GetMatrixByGf(positions[i], orients[i], scales[i])
            index = indices[i]
            node  = self.makeReferenceGeom(protoFiles[index])
            cmds.xform(node, m=mtx, ws=True)
            cmds.parent(node, rootNode)


    def getPrototypesInfo(self, geom):
        protofiles = list()
        prototypes = geom.GetPrototypesRel().GetTargets()
        for i in xrange(len(prototypes)):
            prim = self.stage.GetPrimAtPath(prototypes[i])
            name = prim.GetName()
            refs = prim.GetMetadata('references')
            if refs:
                r = refs.GetAddedOrExplicitItems()[0]
                protofiles.append(r.assetPath)
        return protofiles

    def makeReferenceGeom(self, filename):
        nodeName = filename.split("/")[-1].split(".")[0]
        node = cmds.createNode('pxrUsdProxyShape', n = "%sShape" % nodeName)
        filename = os.path.join(os.path.dirname(self.filename), filename)
        cmds.setAttr('%s.filePath' % node, filename, type='string')
        return cmds.listRelatives(node, p=True, f = True)[0]

    def getMatrixList(self, mtx):
        result = list()
        for i in mtx:
            result += list(i)
        return result


#-------------------------------------------------------------------------------
#
#   SHOT
#
#-------------------------------------------------------------------------------
class SetShotExport(Arguments.ShotArgs):
    '''
    '''
    def __init__(self, node=None, **kwargs):
        dxsMayaUtils.PluginSetup(['backstageMenu', 'pxrUsd'])

        Arguments.ShotArgs.__init__(self, **kwargs)
        if not self.outDir and self.shotDir:
            self.outDir = self.shotDir + '/set'
        self.computeVersion()

        self.fps = dxsMayaUtils.GetFPS()
        # Frame Range
        if not self.fr[0] or not self.fr[1]:
            self.fr = dxsMayaUtils.GetFrameRange()
        self.expfr = [None, None]
        if self.fr[0] != self.fr[1]:
            self.expfr = (self.fr[0]-1, self.fr[1]+1)

        self.mayafile = dxsMayaUtils.GetMayaFilename()
        if self.mayafile:
            self.comment = 'Generated with %s' % self.mayafile

        self.expNodes, self.refNodes = SetShotExport.GetNodes(node)

    @staticmethod
    def GetNodes(selected=None):
        exportNodes   = list()
        referenceNodes= list()

        for n in GetSetNodes(selected):
            ntype = cmds.nodeType(n)
            if ntype == 'pxrUsdReferenceAssembly':
                referenceNodes.append(n)
            elif ntype == 'xBlock':
                btype = cmds.getAttr('%s.type' % n)      # Block Type 1: Model, 4: Layout
                action= cmds.getAttr('%s.action' % n)    # 1: export, 2: reference
                if action == 1:
                    exportNodes.append(n)
                elif action == 2:
                    referenceNodes.append(n)
            elif ntype == 'dxAssembly':
                action = cmds.getAttr('%s.action' % n)
                if action == 1:
                    exportNodes.append(n)
                else:
                    referenceNodes.append(n)
        return exportNodes, referenceNodes

    @staticmethod
    def GetUsdRootName(filename):
        rootLayer = Sdf.Layer.FindOrOpen(filename)
        if rootLayer:
            return rootLayer.rootPrims[0].name


    def doIt(self):
        if self.expNodes:
            for n in self.expNodes:
                geomFile = self.exportGeom(n)
                self.makePackage(geomFile)
        if self.refNodes:
            for n in self.refNodes:
                geomFile = self.referenceGeom(n)
                self.makePackage(geomFile)

    def makePackage(self, geomFile):
        if not geomFile:
            return
        geomMasterFile = self.makeGeomPackage(geomFile)
        self.makeTaskPackage(geomMasterFile)

    def makeGeomPackage(self, geomFile):
        assetName = os.path.basename(os.path.dirname(geomFile))
        masterFile = '{DIR}/{VER}/setscene.usd'.format(DIR=self.outDir, VER=self.version)
        SessionUtils.MakeReferenceStage(masterFile, [(geomFile, None)], SdfPath='/set', composite='reference', addChild=True, fr=self.fr, fps=self.fps, comment=self.comment)

        masterPayloadFile = masterFile.replace('.usd', '.payload.usd')
        SdfPath = '/set{setVersion=%s}' % self.version
        SessionUtils.MakeReferenceStage(masterPayloadFile, [(masterFile, None)], SdfPath=SdfPath, fr=self.fr, fps=self.fps, comment=self.comment, clear=True)
        return masterPayloadFile

    def makeTaskPackage(self, sourceFile):
        taskFile = '{DIR}/set.usd'.format(DIR=self.outDir)
        SessionUtils.MakeSubLayerStage(taskFile, [sourceFile])

        taskPayloadFile = taskFile.replace('.usd', '.payload.usd')
        SessionUtils.MakeReferenceStage(taskPayloadFile, [(taskFile, None)], SdfPath='/shot/set', Name=self.shotName, Kind='assembly', clear=True)

        PackageUtils.ShotPackage(self.showDir, self.seqName, self.shotName, taskPayloadFile, fr=self.fr, fps=self.fps)


    def exportGeom(self, node):
        nodeName = node.split('|')[-1].split(':')[-1]
        geomFile = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=nodeName)
        ntype = cmds.nodeType(node)
        _ifInstancerGeom = False
        if ntype == 'xBlock':
            btype = cmds.getAttr('%s.type' % node)
            if btype == 1:      # Model
                GeomMain.Export(geomFile, [node], userAttr=True, mtlAttr=True, fr=self.expfr).doIt()
                return geomFile
            elif btype == 4:    # Layout
                _ifInstancerGeom = True

        elif ntype == 'dxAssembly':
            _ifInstancerGeom = True

        if _ifInstancerGeom:
            Data = PointInstancerBlock()
            GetDxAssembly(node, Data, geomFile).doIt()
            geomClass = EnvSetGeom(node, Data, geomFile)
            geomClass.comment = self.comment
            if geomClass.doIt():
                return geomFile


    def referenceGeom(self, node):
        ntype = cmds.nodeType(node)
        if ntype == 'pxrUsdReferenceAssembly':
            # node info
            fileName = cmds.getAttr('%s.filePath' % node)
            primPath = cmds.getAttr('%s.primPath' % node)
            geomFile = self.MakeReferenceGeom(fileName, primPath, node)
            return geomFile
        elif ntype == 'xBlock':
            # node info
            fileName = cmds.getAttr('%s.importFile' % node)
            geomFile = self.MakeReferenceGeom(fileName, None, node)
            return geomFile
        elif ntype == 'dxAssembly':
            # node info
            fileName = cmds.getAttr('%s.fileName' % node)
            geomFile = self.MakeReferenceGeom(fileName, None, node)
            return geomFile
        else:
            return None

    def MakeReferenceGeom(self, srcFile, srcPrim, node):
        '''
        Args:
            node (str): transform
        '''
        nodeName = node.split('|')[-1].split(':')[-1]
        outFile  = '{DIR}/{VER}/{NAME}.geom.usd'.format(DIR=self.outDir, VER=self.version, NAME=nodeName)
        stage = SessionUtils.MakeInitialStage(outFile, clear=True, comment=self.comment, fr=self.fr)
        dprim = stage.DefinePrim('/' + nodeName, 'Xform')
        stage.SetDefaultPrim(dprim)
        relpath = PathUtils.GetRelPath(outFile, srcFile)
        payload = Sdf.Payload(relpath) if not srcPrim else Sdf.Payload(relpath, Sdf.Path(srcPrim))
        dxsUsdUtils.SetPayload(dprim, payload)
        matrixs, frames = dxsXformUtils.GetMatrix(node, fr=self.fr, step=self.step).doIt()
        dxsXformUtils.AddXformOp(dprim, matrix=matrixs, frames=frames)
        stage.GetRootLayer().Save()
        print "# Export usd file '%s'" % outFile
        return outFile



#-------------------------------------------------------------------------------
#
#   Texture Override
#
#-------------------------------------------------------------------------------
class TaneSourceTextureOverride:
    '''
    Selected TaneSource Texture Override
    '''
    def __init__(self, showDir=None):
        self.showDir = showDir
        self.sourceShapes = cmds.ls(sl=True, dag=True, type='TN_UsdProxy')

    def doIt(self):
        if not self.sourceShapes:
            dxsMsg.Print('dialog', ["Select 'TN_UsdProxy' node"])

        for s in self.sourceShapes:
            envNode = cmds.listConnections(s, d=True, s=False, type='TN_Environment')
            if envNode:
                rootNode = cmds.listConnections(envNode[0], d=True, s=False, type='TN_Tane')
                if rootNode:
                    rootNode = rootNode[0]
                    if len(rootNode.split('_set')) > 1:
                        assetName = rootNode.split('_set')[0] + '_set'
                        refAssetFile = cmds.getAttr('%s.renderFile' % s)
                        Texture.AssetTextureOverride(refAssetFile=refAssetFile, showDir=self.showDir, asset=assetName, version='v001').doIt()
