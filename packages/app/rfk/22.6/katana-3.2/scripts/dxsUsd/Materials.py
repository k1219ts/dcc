import os, sys
import pprint

from Katana import NodegraphAPI, Nodes3DAPI, FnGeolibServices, FnAttribute, RenderingAPI
import UI4

from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdRi, Gf, Kind

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


def ExportMaterials(outdir):
    if not outdir:
        assert False, '[ERROR] - select output directory.'

    # selected = NodegraphAPI.GetAllSelectedNodes()
    # if not selected:
    #     assert False, '[ERROR] - select compute node.'
    viewNode = NodegraphAPI.GetViewNode()
    if not viewNode:
        assert False, '[ERROR] - set view node.'

    sg = Nodes3DAPI.ScenegraphManager.getActiveScenegraph()
    locations = sg.getSelectedLocations()
    if not locations:
        assert False, '[ERROR] - select locations in SceneGraph.'

    matdir = os.path.join(outdir, 'materials')
    if not os.path.exists(matdir):
        os.makedirs(matdir)

    for loc in locations:
        name = loc.split('/')[-1]
        outfile = os.path.join(matdir, name + '.usd')
        GetMaterialInfo(viewNode, loc, outfile).doIt()

    mtlfile = os.path.join(outdir, 'Materials.usd')
    stage = SessionUtils.MakeInitialStage(mtlfile)
    dprim = stage.DefinePrim('/Materials', 'Scope')
    stage.SetDefaultPrim(dprim)

    for loc in locations:
        name = loc.split('/')[-1]
        mtln = name.replace('_NM', '').lower()
        mtl  = UsdShade.Material.Define(stage, dprim.GetPath().AppendChild(mtln))
        Usd.ModelAPI(mtl.GetPrim()).SetKind(Kind.Tokens.subcomponent)
        mtl.GetPrim().SetPayload(Sdf.Payload('./materials/%s.usd' % name))

    stage.GetRootLayer().Save()


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
        self.stage.SetDefaultPrim(self.mtlprim)

        terminalsAttr = self.prod.getAttribute('material.terminals')
        terminals = terminalsAttr.childNames()
        # bxdf
        if 'prmanBxdf' in terminals:
            bxdfName = terminalsAttr.getChildByName('prmanBxdf').getValue()
            bxdfPort = terminalsAttr.getChildByName('prmanBxdfPort').getValue()
            outputs  = self.getShade(bxdfName, [(bxdfPort, Sdf.ValueTypeNames.Token)])

            risurface  = rimaterial.CreateSurfaceAttr()
            UsdShade.ConnectableAPI.ConnectToSource(risurface, outputs[0])
        # displace
        if 'prmanDisplacement' in terminals:
            dispName = terminalsAttr.getChildByName('prmanDisplacement').getValue()
            dispPort = terminalsAttr.getChildByName('prmanDisplacementPort').getValue()
            outputs  = self.getShade(dispName, [(dispPort, Sdf.ValueTypeNames.Token)])

            ridisp = rimaterial.CreateDisplacementAttr()
            UsdShade.ConnectableAPI.ConnectToSource(ridisp, outputs[0])

        self.stage.GetRootLayer().Save()
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
