from __future__ import print_function
import os, re, gc

from Katana import FnGeolibServices, FnAttribute, NodegraphAPI, Nodes3DAPI
import UI4

from fnpxr import Sdf, Usd, Gf, Vt

import DXRulebook.Interface as rb
import DXUSD_KAT.Vars as var

#-------------------------------------------------------------------------------
#
# KATANA
#
#-------------------------------------------------------------------------------
def GetShaderFnAttr(shaderType):
    fnAttr = FnGeolibServices.AttributeFunctionUtil.Run(
        'PRManGetShaderInfo', FnAttribute.StringAttribute(shaderType)
    )
    return fnAttr


def importXML(filenames):
    parentNode    = UI4.App.Tabs.FindTopTab('Node Graph').getEnteredGroupNode()
    selectedNodes = NodegraphAPI.GetAllSelectedNodes()

    pos = NodegraphAPI.GetViewPortPosition(parentNode)[0]
    if selectedNodes and selectedNodes[-1] != parentNode:
        pos = NodegraphAPI.GetNodePosition(selectedNodes[-1])

    idx = 1
    for f in filenames:
        xmlTree, versionUp = NodegraphAPI.LoadElementsFromFile(f)
        nodes = NodegraphAPI.ParseNodesXmlIO(xmlTree)
        for n in nodes:
            n.setParent(parentNode)
            NodegraphAPI.SetNodePosition(n, (pos[0] + 200 * idx, pos[1] + 100))
            idx += 1

def importXML_by_data(data):
    parentNode    = UI4.App.Tabs.FindTopTab('Node Graph').getEnteredGroupNode()
    selectedNodes = NodegraphAPI.GetAllSelectedNodes()

    pos = NodegraphAPI.GetViewPortPosition(parentNode)[0]
    if selectedNodes and selectedNodes[-1] != parentNode:
        pos = NodegraphAPI.GetNodePosition(selectedNodes[-1])

    idx = 1
    for f, d in data.items():
        xmlTree, versionUp = NodegraphAPI.LoadElementsFromFile(f)
        nodes = NodegraphAPI.ParseNodesXmlIO(xmlTree)
        for n in nodes:
            n.setParent(parentNode)
            NodegraphAPI.SetNodePosition(n, (pos[0] + 150 * idx, pos[1] + 100))
            idx += 1
            # set params
            if n.getType() == 'NetworkMaterialCreate':
                mtln = n.getParameter('__node_networkMaterial').getValue(0)
                mtln = NodegraphAPI.GetNode(mtln)
                mtln.getParameter('name').setValue(d['name'], 0)
                mtln.getParameter('namespace').setValue(d['namespace'], 0)


def xmlImportDialog():
    fn = UI4.Util.AssetId.BrowseForAsset('', 'Import XML', True, {'fileTypes': 'xml'})
    if not fn:
        return
    importXML([fn])

def xmlImportScenegraph():
    viewNode = NodegraphAPI.GetViewNode()
    if not viewNode:
        assert False, '[ERROR] - set view node.'

    sg = Nodes3DAPI.ScenegraphManager.getActiveScenegraph()
    locations = sg.getSelectedLocations()
    if not locations:
        assert False, '[ERROR] - select locations in SceneGraph.'

    data = dict()   # {filename: {name: material name, namespace: material namespace}}
    root = Nodes3DAPI.GetGeometryProducer(node=viewNode)
    for loc in locations:
        prod = root.getProducerByPath(loc)
        attr = prod.getAttribute('usd.layerPath')
        if attr:
            filename = attr.getValue().replace('.usd', '.xml')
            src  = loc.split('/')
            name = src[-1]
            namespace = '/'.join(src[src.index('Looks')+1:-1])
            data[filename] = {'name': name, 'namespace': namespace}
    importXML_by_data(data)


#-------------------------------------------------------------------------------
#
# COMMON
#
#-------------------------------------------------------------------------------
SJoin = lambda *args: var.SEP.join(args)

DirName  = lambda *args: os.path.dirname(args[0])
BaseName = lambda *args: os.path.basename(args[0])

# Ver      = lambda *args: 'v%03d'%(args[0])
IsVer    = lambda *args: rb.MatchFlag('ver', args[0])
# VerAsInt = lambda *args: int(re.search('\d{3}', args[0]).group())

def Ver(*args):
    ver = 'v%s'%(str(args[0]).zfill(VerDigit()))
    if not IsVer(ver):
        ver = ver.upper()
    return ver

def VerAsInt(*args):
    ver = re.search('\d{%s}' % VerDigit(), args[0])
    if ver:
        return int(ver.group())
    else:
        return int(re.search('\d{3}', args[0]).group())

def VerDigit():
    coder = var.rb.Coder()
    pattern = coder.Rulebook().flag['ver'].pattern
    return int(re.search(r'(?<={)\d+(?=})', pattern).group())

def GetVersions(dir):
    vers = list()
    if os.path.exists(dir):
        for d in os.listdir(dir):
            if IsVer(d):
                vers.append(d)
    vers.sort()
    return vers

def GetLastVersion(dir, default=1):
    versions = GetVersions(dir)
    if versions:
        return versions[-1]
    else:
        return Ver(default)

def GetNextVersion(dir):
    ver = VerAsInt(GetLastVersion(dir, default=0))
    return Ver(ver + 1)

def GetRelPath(current, target):
    '''
    [Arguments]
    current (str) : file or directory
    target (str)  : file
    '''
    comp = os.path.commonprefix([current, target])
    # if comp is '' or '/' or '/show/'
    if comp == '' or comp == var.SEP or comp == '/%s/'%var.T.SHOW:
        return target

    def isfile(path):
        return '.' in os.path.basename(path)

    curdir = os.path.dirname(current) if isfile(current) else current
    tardir = os.path.dirname(target)  if isfile(target)  else target

    if curdir == tardir:
        if isfile(target):
            return var._SEP+os.path.basename(target)
        else:
            return var._SEP
    else:
        rel = os.path.relpath(target, start=curdir)
        if rel[0] != '.':
            rel = var._SEP+rel
        return rel


#-------------------------------------------------------------------------------
#
# Rulebook
#
#-------------------------------------------------------------------------------
class Arguments(rb.Flags):
    def __init__(self, input):  # input is directory
        rb.Flags.__init__(self, 'USD')
        self.entity = 'assetlib'

        # modify input
        src = input.split('/')
        if 'material' in src and not 'prman' in src:
            input = os.path.abspath(os.path.join(input, 'prman'))

        self.D.SetDecode(input)
        self.task   = 'material'
        self.render = 'prman'

        if self.show: self.pub = '_3d'

        # shot
        if self.shot:
            self.entity = 'shot'
        # seq or asset
        else:
            if not self.asset: self.asset = '_global'

        if self.show and self.asset == '_global':
            if self.seq:
                self.entity = 'seq'
            else:
                self.entity = 'show'

        if self.asset and self.asset != '_global':
            self.entity = self.asset
            if self.branch:
                self.entity = self.branch

        self.rootDir = self.D.TASKR
        if self.subdir:
            self.rootDir = self.D.TASKSR

    def setSubdir(self, name):
        if self.subdir:
            return
        if self.assetName != name:
            self.subdir  = name
            self.rootDir = self.D.TASKSR

    def setName(self, name):
        self.nslyr = name
        if not self.nsver:
            self.nsver = GetNextVersion(SJoin(self.rootDir, 'shaders', self.nslyr))

    def setVersion(self):
        self.nsver = GetNextVersion(SJoin(self.rootDir, 'shaders', self.nslyr))

    @property
    def assetName(self):
        if self.branch:
            return self.branch
        else:
            return self.asset

    @property
    def outDir(self):
        return SJoin(self.rootDir, 'shaders', self.nslyr, self.nsver)

    @property
    def prmanMaster(self):
        return SJoin(self.rootDir, 'prman.usd')

    @property
    def materialMaster(self):
        return SJoin(self.rootDir, 'shaders', self.nslyr, self.nslyr + '.usd')




#-------------------------------------------------------------------------------
#
# USD
#
#-------------------------------------------------------------------------------
def AsLayer(path, create=False, clear=False):
    layer = None
    if isinstance(path, (str, unicode)):
        layer = Sdf.Layer.FindOrOpen(path)
        if not layer and create:
            layer = Sdf.Layer.CreateNew(path, args={'format': 'usda'})
            layer.customLayerData = {'dxusd': "2.0.0"}
    elif isinstance(path, Sdf.Layer):
        layer = path
    elif isinstance(path, Usd.Stage):
        layer = path.GetRootLayer()

    if not layer:
        return

    if clear:
        layer.Clear()
        layer.customLayerData = {'dxusd': "2.0.0"}
    return layer


class OpenStage:
    def __init__(self, layer, loadAll=True):
        _load = Usd.Stage.LoadAll if loadAll else Usd.Stage.LoadNone
        self.stage = Usd.Stage.Open(layer, load=_load)

    def __enter__(self):
        return self.stage

    def __exit__(self, type, value, traceback):
        self.free(self.stage)

    def free(self, obj):
        for ref in gc.get_referrers(obj):
            if isinstance(ref, dict):
                for key, value in ref.items():
                    if value is obj:
                        ref[key] = None
        del obj
        gc.collect()


def GetPrimSpec(layer, path, specifier='def', type='Xform'):
    spec = layer.GetPrimAtPath(path)
    if not spec:
        spec = Sdf.CreatePrimInLayer(layer, path)
        if specifier == 'def':
            spec.specifier = Sdf.SpecifierDef
            spec.typeName  = type
        elif specifier == 'over':
            spec.specifier = Sdf.SpecifierOver
        elif specifier == 'class':
            spec.specifier = Sdf.SpecifierClass
        else:
            assert False, '# Error : specifier.'
    return spec


def GetAttributeSpec(spec, name, value, type, variability=Sdf.VariabilityVarying, custom=False, info=dict()):
    '''
    type
        Sdf.ValueTypeNames.Token, ...
    '''
    attrSpec = spec.properties.get(name)
    if not attrSpec:
        attrSpec = Sdf.AttributeSpec(spec, name, type, variability, declaresCustom=custom)
    if value != None:
        attrSpec.default = value
    if info:
        for k, v in info.items():
            attrSpec.SetInfo(k, v)
    return attrSpec


def GetRelationshipSpec(spec, name, value, variability=Sdf.VariabilityUniform):
    attrSpec = spec.properties.get(name)
    if not attrSpec:
        attrSpec = Sdf.RelationshipSpec(spec, name, False, variability)
    if value != None:
        if isinstance(value, list):
            for v in value:
                attrSpec.targetPathList.explicitItems.append(v)
        else:
            attrSpec.targetPathList.explicitItems.append(value)
    return attrSpec


def SubLayersAppend(layer, filename):
    if layer.subLayerPaths.index(filename) == -1:
        layer.subLayerPaths.insert(0, filename)


def GetVariantSetSpec(spec, name):
    vset = spec.variantSets.get(name)
    if not vset:
        vset = Sdf.VariantSetSpec(spec, name)
        spec.variantSetNameList.prependedItems.append(name)
    return vset

def GetVariantSpec(vsetSpec, value):
    vspec = vsetSpec.variants.get(value)
    if not vspec:
        vspec = Sdf.VariantSpec(vsetSpec, value)
    return vspec


def ReferenceAppend(spec, filename, path=Sdf.Path.emptyPath, clear=False):
    '''
    [Arguments]
    spec (Sdf.PrimSpec)
    filename (str)
    path (str, Sdf.Path) : reference target prim path
    clear (bool)
    '''
    ref   = Sdf.Reference(filename, primPath=path)
    items = spec.referenceList.prependedItems
    if clear:
        items.clear()
    if items.index(ref) == -1:
        # items.append(ref)
        items.insert(0, ref)

def GetReferenceAssetPath(spec):
    items = spec.referenceList.prependedItems
    if len(items) > 0:
        src = items[-1].assetPath
        if src.startswith('/'):
            return src
        else:
            dirname = DirName(spec.layer.identifier)
            return os.path.abspath(SJoin(dirname, src))

def XXXGetRefAssetPath(spec):
    items = spec.referenceList.prependedItems
    if not items:
        return
    assetPath = items[0].assetPath
    primPath  = items[0].primPath
    filename  = os.path.abspath(SJoin(DirName(spec.layer.identifier), assetPath))

    result = None
    with OpenStage(filename) as stage:
        if primPath:
            prim = stage.GetPrimAtPath(primPath)
        else:
            prim = stage.GetDefaultPrim()
        result = prim.GetPrimStack()[-1].layer.identifier

    return result

def GetRefAssetPath(spec):
    result = None
    with OpenStage(spec.layer) as stage:
        prim = stage.GetPrimAtPath(spec.path)
        result = prim.GetPrimStack()[-1].layer.identifier
    return result


def CreateLightSchemas(spec):
    spec.SetInfo('apiSchemas', Sdf.TokenListOp.Create(prependedItems=['ShapingAPI', 'ShadowAPI']))

def SetApiSchemas(spec, values):
    spec.SetInfo('apiSchemas', Sdf.TokenListOp.Create(prependedItems=values))


def GetVec3fArray(value):
    result = list()
    for i in range(int(len(value) / 3)):
        result.append(Gf.Vec3f(*value[i*3:i*3+3]))
    return result
