import os
import sys
import re
import glob
import pprint

from Katana import NodegraphAPI, Nodes3DAPI, FnGeolibServices, FnAttribute, RenderingAPI
import UI4

from fnpxr import Usd, UsdGeom, UsdShade, Sdf, UsdRi, Gf, Kind

import SessionUtils
import predefined as pdf


def printParams_PrmanShadingNode(shaderType):
    shaderInfoAttr = FnGeolibServices.AttributeFunctionUtil.Run('PRManGetShaderInfo', FnAttribute.StringAttribute(shaderType))

    shaderParams = shaderInfoAttr.getChildByName('params')
    for name, param in shaderParams.childList():
        paramType = shaderParams.getChildByName(name + '.type').getValue()
        usdType = None
        if pdf._USD_TYPE_MAP_.has_key(paramType):
            usdType = pdf._USD_TYPE_MAP_[paramType]

        print '>> {NAME} : {KTYPE} -> {UTYPE}'.format(NAME=name, KTYPE=pdf._KATANA_RENDER_MAP_[paramType], UTYPE=usdType)

def GetShaderInfoAttr(shaderType):
    return FnGeolibServices.AttributeFunctionUtil.Run('PRManGetShaderInfo', FnAttribute.StringAttribute(shaderType))


def ExportMaterialDialog():
    dir = UI4.Util.AssetId.BrowseForAsset('', 'USD Export Material (select directory)', True, {'acceptDir': True})
    if dir:
        ExportMaterials(dir)


class ExportMaterials:
    def __init__(self, assetDir):
        if not assetDir:
            assert False, '[ERROR] - select output directory.'

        viewNode = NodegraphAPI.GetViewNode()
        if not viewNode:
            assert False, '[ERROR] - set view node.'

        sg = Nodes3DAPI.ScenegraphManager.getActiveScenegraph()
        locations = sg.getSelectedLocations()
        if not locations:
            assert False, '[ERROR] - select locations in SceneGraph.'

        rootDir = os.path.join(assetDir, 'material', 'prman')
        if not os.path.exists(rootDir):
            os.makedirs(rootDir)

        for loc in locations:
            splitPath = loc.split('/')
            # material name
            name = self.getMaterialName(splitPath[-1])
            shaderDir = os.path.join(rootDir, 'shaders', name)
            # version file
            outfile = self.getOutfile(shaderDir, splitPath[-1])
            CreateUsdMaterial(viewNode, loc, outfile).doIt()
            # version collect pack
            packfile = os.path.join(shaderDir, name + '.usd')
            self.collectVersion(outfile, packfile, loc.split('/')[-1])
            # material collect pack
            masterfile = os.path.join(rootDir, 'prman.usd')
            self.collectMaterial(packfile, masterfile, name)


    def getMaterialName(self, name):
        if name.endswith('_NM'):
            return name.replace('_NM', '')
        else:
            return name

    def getOutfile(self, baseDir, name):
        source  = glob.glob(baseDir + '/v*')
        if not source:
            fn = '{DIR}/v001/{NAME}.usd'.format(DIR=baseDir, NAME=name)
            return fn
        source.sort()
        version = source[-1].split('/')[-1]
        version = int(version[1:])
        version = 'v%03d' % (version + 1)
        fn = '{DIR}/{VERSION}/{NAME}.usd'.format(DIR=baseDir, VERSION=version, NAME=name)
        return fn

    def collectVersion(self, infile, outfile, default):
        outLayer = Sdf.Layer.FindOrOpen(outfile)
        if not outLayer:
            outLayer = Sdf.Layer.CreateNew(outfile, args={'format': 'usda'})
            outLayer.defaultPrim = default
        relpath = './' + os.path.relpath(infile, start=os.path.dirname(outfile))
        if not relpath in outLayer.subLayerPaths:
            outLayer.subLayerPaths.insert(0, relpath)
        outLayer.Save()

    def collectMaterial(self, infile, outfile, name):
        outLayer = Sdf.Layer.FindOrOpen(outfile)
        if not outLayer:
            outLayer = Sdf.Layer.CreateNew(outfile, args={'format': 'usda'})
            spec = Sdf.CreatePrimInLayer(outLayer, '/prman')
            spec.specifier = Sdf.SpecifierDef
            spec.typeName  = 'Scope'
            outLayer.defaultPrim = 'prman'

        mtlpath = '/prman/' + name
        spec = outLayer.GetPrimAtPath(mtlpath)
        if not spec:
            spec = Sdf.CreatePrimInLayer(outLayer, mtlpath)
            spec.specifier = Sdf.SpecifierOver
        relpath = './' + os.path.relpath(infile, start=os.path.dirname(outfile))
        ref = Sdf.Reference(relpath)

        spec.ClearReferenceList()
        spec.referenceList.prependedItems.append(ref)

        outLayer.Save()


class CreateUsdMaterial:
    def __init__(self, evalNode, location, filename):
        root = Nodes3DAPI.GetGeometryProducer(node=evalNode)
        self.prod = root.getProducerByPath(location)
        self.mtlName  = location.split('/')[-1]
        self.filename = filename

    def doIt(self):
        self.nodesAttr = self.prod.getAttribute('material.nodes')
        if not self.nodesAttr:
            return

        # create stage
        self.stage = SessionUtils.MakeInitialStage(self.filename, clear=True)
        material = UsdShade.Material.Define(self.stage, '/' + self.mtlName)

        self.mtlprim = material.GetPrim()
        Usd.ModelAPI(self.mtlprim).SetKind(Kind.Tokens.subcomponent)
        self.stage.SetDefaultPrim(self.mtlprim)

        version = os.path.basename(os.path.dirname(self.filename))
        if re.match('v\d+', version):
            vset = self.mtlprim.GetVariantSets().AddVariantSet('materialVersion')
            vset.AddVariant(version)
            vset.SetVariantSelection(version)
            with vset.GetVariantEditContext():
                self.create(material)
        else:
            self.create(material)

        self.stage.GetRootLayer().Save()
        del self.stage
        print '->', self.filename

    def create(self, material):
        rimaterial = UsdRi.MaterialAPI(material)

        terminalsAttr = self.prod.getAttribute('material.terminals')
        terminals = terminalsAttr.childNames()
        # bxdf
        if 'prmanBxdf' in terminals:
            bxdfName = terminalsAttr.getChildByName('prmanBxdf').getValue()
            outputs  = self.getShade(bxdfName, [('out', Sdf.ValueTypeNames.Token)])

            risurface  = rimaterial.CreateSurfaceAttr()
            UsdShade.ConnectableAPI.ConnectToSource(risurface, outputs[0])
        # displace
        if 'prmanDisplacement' in terminals:
            dispName = terminalsAttr.getChildByName('prmanDisplacement').getValue()
            outputs  = self.getShade(dispName, [('out', Sdf.ValueTypeNames.Token)])

            ridisp = rimaterial.CreateDisplacementAttr()
            UsdShade.ConnectableAPI.ConnectToSource(ridisp, outputs[0])

    def getShade(self, name, ports):
        '''
        name (str): shader name
        ports (list): [(out port name, sdftype), ...]
        parent : parent prim
        '''
        nodeAttr = self.nodesAttr.getChildByName(name)

        ntype  = nodeAttr.getChildByName('type').getValue()
        srcname= nodeAttr.getChildByName('srcName').getValue()
        shader = UsdShade.Shader.Define(self.stage, self.mtlprim.GetPath().AppendChild(srcname))
        shader.SetShaderId(ntype)

        shaderInfoAttr  = GetShaderInfoAttr(ntype)
        shaderTypeTags  = shaderInfoAttr.getChildByName('shaderTypeTags').getValue()
        shaderParamsAttr= shaderInfoAttr.getChildByName('params')

        outputs = list()
        if shaderTypeTags == 'pattern':
            for n, t in ports:
                out = shader.CreateOutput(n, t)
                outputs.append(out)
        else:
            out = shader.CreateOutput(ports[0][0], ports[0][1])
            outputs.append(out)

        childnames = nodeAttr.childNames()

        if 'parameters' in childnames:
            self.setParameters(shader, nodeAttr, shaderParamsAttr)

        if 'connections' in childnames:
            self.setConnections(shader, nodeAttr, shaderParamsAttr)

        return outputs


    def setParameters(self, Shader, nodeAttr, shaderParamsAttr):
        paramsAttr = nodeAttr.getChildByName('parameters')
        for n in paramsAttr.childNames():
            parm   = paramsAttr.getChildByName(n)
            ptype  = shaderParamsAttr.getChildByName(n + '.type').getValue()
            usdAttr= Shader.CreateInput(n, pdf._USD_TYPE_MAP_[ptype])
            if parm.getNumberOfValues() > 1:
                usdAttr.Set(Gf.Vec3f(list(parm.getData())))
            else:
                usdAttr.Set(parm.getValue())

    def setConnections(self, Shader, nodeAttr, shaderParamsAttr):
        connectsAttr = nodeAttr.getChildByName('connections')

        # connected node data
        dstData = dict()    # {node: [(out port name, SdfType), ...]}
        # current node data
        srcData = dict()    # {node: [parm name, ...]}

        for n in connectsAttr.childNames():
            dst, slname = connectsAttr.getChildByName(n).getValue().split('@')
            if not dstData.has_key(slname):
                dstData[slname] = list()
            if not srcData.has_key(slname):
                srcData[slname] = list()

            ptype= shaderParamsAttr.getChildByName(n + '.type').getValue()
            src  = (dst, pdf._USD_TYPE_MAP_[ptype])
            dstData[slname].append(src)
            srcData[slname].append(n)

        if dstData and srcData:
            for n in dstData:
                outputs = self.getShade(n, dstData[n])
                for i in range(len(srcData[n])):
                    pname = srcData[n][i]
                    ptype = dstData[n][i][1]
                    usdAttr = Shader.CreateInput(pname, ptype)
                    UsdShade.ConnectableAPI.ConnectToSource(usdAttr, outputs[i])



class GetMaterialInfo:
    def __init__(self, evalNode, location, filename):
        root = Nodes3DAPI.GetGeometryProducer(node=evalNode)
        self.prod = root.getProducerByPath(location)
        self.mtlName  = location.split('/')[-1]
        self.filename = filename

    def doIt(self):
        self.nodesAttr = self.prod.getAttribute('material.nodes')
        if not self.nodesAttr:
            return

        # create stage
        self.stage = SessionUtils.MakeInitialStage(self.filename, clear=True)
        material = UsdShade.Material.Define(self.stage, '/' + self.mtlName)

        rimaterial = UsdRi.MaterialAPI(material)

        self.mtlprim = material.GetPrim()
        Usd.ModelAPI(self.mtlprim).SetKind(Kind.Tokens.subcomponent)
        self.stage.SetDefaultPrim(self.mtlprim)

        terminalsAttr = self.prod.getAttribute('material.terminals')
        terminals = terminalsAttr.childNames()
        # bxdf
        if 'prmanBxdf' in terminals:
            bxdfName = terminalsAttr.getChildByName('prmanBxdf').getValue()
            outputs  = self.getShade(bxdfName, [('out', Sdf.ValueTypeNames.Token)])

            risurface  = rimaterial.CreateSurfaceAttr()
            UsdShade.ConnectableAPI.ConnectToSource(risurface, outputs[0])
        # displace
        if 'prmanDisplacement' in terminals:
            dispName = terminalsAttr.getChildByName('prmanDisplacement').getValue()
            outputs  = self.getShade(dispName, [('out', Sdf.ValueTypeNames.Token)])

            ridisp = rimaterial.CreateDisplacementAttr()
            UsdShade.ConnectableAPI.ConnectToSource(ridisp, outputs[0])

        self.stage.GetRootLayer().Save()
        del self.stage
        print '->', self.filename


    def getShade(self, name, ports):
        '''
        name (str): shader name
        ports (list): [(out port name, sdftype), ...]
        parent : parent prim
        '''
        nodeAttr = self.nodesAttr.getChildByName(name)

        ntype  = nodeAttr.getChildByName('type').getValue()
        srcname= nodeAttr.getChildByName('srcName').getValue()
        shader = UsdShade.Shader.Define(self.stage, self.mtlprim.GetPath().AppendChild(srcname))
        shader.SetShaderId(ntype)

        shaderInfoAttr  = GetShaderInfoAttr(ntype)
        shaderTypeTags  = shaderInfoAttr.getChildByName('shaderTypeTags').getValue()
        shaderParamsAttr= shaderInfoAttr.getChildByName('params')

        outputs = list()
        if shaderTypeTags == 'pattern':
            for n, t in ports:
                out = shader.CreateOutput(n, t)
                outputs.append(out)
        else:
            out = shader.CreateOutput(ports[0][0], ports[0][1])
            outputs.append(out)

        childnames = nodeAttr.childNames()

        if 'parameters' in childnames:
            self.setParameters(shader, nodeAttr, shaderParamsAttr)

        if 'connections' in childnames:
            self.setConnections(shader, nodeAttr, shaderParamsAttr)

        return outputs


    def setParameters(self, Shader, nodeAttr, shaderParamsAttr):
        paramsAttr = nodeAttr.getChildByName('parameters')
        for n in paramsAttr.childNames():
            parm   = paramsAttr.getChildByName(n)
            ptype  = shaderParamsAttr.getChildByName(n + '.type').getValue()
            usdAttr= Shader.CreateInput(n, pdf._USD_TYPE_MAP_[ptype])
            if parm.getNumberOfValues() > 1:
                usdAttr.Set(Gf.Vec3f(list(parm.getData())))
            else:
                usdAttr.Set(parm.getValue())

    def setConnections(self, Shader, nodeAttr, shaderParamsAttr):
        connectsAttr = nodeAttr.getChildByName('connections')

        # connected node data
        dstData = dict()    # {node: [(out port name, SdfType), ...]}
        # current node data
        srcData = dict()    # {node: [parm name, ...]}

        for n in connectsAttr.childNames():
            dst, slname = connectsAttr.getChildByName(n).getValue().split('@')
            if not dstData.has_key(slname):
                dstData[slname] = list()
            if not srcData.has_key(slname):
                srcData[slname] = list()

            ptype= shaderParamsAttr.getChildByName(n + '.type').getValue()
            src  = (dst, pdf._USD_TYPE_MAP_[ptype])
            dstData[slname].append(src)
            srcData[slname].append(n)

        if dstData and srcData:
            for n in dstData:
                outputs = self.getShade(n, dstData[n])
                for i in range(len(srcData[n])):
                    pname = srcData[n][i]
                    ptype = dstData[n][i][1]
                    usdAttr = Shader.CreateInput(pname, ptype)
                    UsdShade.ConnectableAPI.ConnectToSource(usdAttr, outputs[i])
