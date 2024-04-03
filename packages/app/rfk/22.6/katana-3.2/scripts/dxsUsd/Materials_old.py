import os, sys
import pprint

from Katana import NodegraphAPI, Nodes3DAPI, FnGeolibServices, FnAttribute, RenderingAPI

from pxr import Usd, UsdGeom, UsdShade, Sdf, UsdRi, Gf

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


def GetPrmanShadingOutputPort(port):
    node = port.getNode()
    if node.getType() == 'PrmanShadingNode':
        return port
    return node.getInputPorts()[0].getConnectedPorts()[0]

def GetShaderInfoAttr(shaderType):
    return FnGeolibServices.AttributeFunctionUtil.Run('PRManGetShaderInfo', FnAttribute.StringAttribute(shaderType))

def GetSdfValueTypeByPort(port, shaderParams=None):
    if not shaderParams:
        node = port.getNode()
        shaderType = node.getParameter('nodeType').getValue(0)
        shaderInfoAttr = GetShaderInfoAttr(shaderType)
        shaderParams   = shaderInfoAttr.getChildByName('params')

    paramName = port.getName()
    return GetSdfValueTypeByName(paramName, shaderParams)

def GetSdfValueTypeByName(name, shaderParams):
    paramType = shaderParams.getChildByName(name + '.type').getValue()
    return pdf._USD_TYPE_MAP_[paramType]


def ExportMaterials(outdir):
    if not outdir:
        assert False, '[ERROR] - select output directory.'

    nodes = list()
    selected = NodegraphAPI.GetAllSelectedNodes()
    if selected:
        for n in selected:
            ntype = n.getType()
            if ntype == 'NetworkMaterial':
                nodes.append(n)
            elif ntype == 'NetworkMaterialCreate':
                nodes.append(n.getChildByIndex(0))

    if not nodes:
        assert False, '[ERROR] - not found NetworkMaterial'

    matdir = os.path.join(outdir, 'materials')
    if not os.path.exists(matdir):
        os.makedirs(matdir)

    # nodes = list()
    # netmats = NodegraphAPI.GetAllNodesByType('NetworkMaterial')
    # for n in netmats:
    #     if NodegraphAPI.IsNodeSelected(n):
    #         nodes.append(n)
    # if not nodes:
    #     nodes = netmats
    #
    # if not nodes:
    #     assert False, '[ERROR] - not found NetworkMaterial'
    #
    for n in nodes:
        name = n.getName()
        print '[INFO]', name
        outfile = os.path.join(matdir, name + '.usd')
        NetworkMaterialToUSD(n, outfile)
    #
    # mtlfile = os.path.join(outdir, 'Materials.usd')
    # stage = SessionUtils.MakeInitialStage(mtlfile)
    # dprim = stage.DefinePrim('/Materials', 'Scope')
    # stage.SetDefaultPrim(dprim)
    #
    # for n in nodes:
    #     nodeName = n.getName()
    #     name = nodeName.replace('_NM', '').lower()
    #     mtl = UsdShade.Material.Define(stage, dprim.GetPath().AppendChild(name))
    #     mtl.GetPrim().SetPayload(Sdf.Payload('./materials/%s.usd' % nodeName))
    #
    # stage.GetRootLayer().Save()


class NetworkMaterialToUSD:
    def __init__(self, node, filename):
        nodeName = node.getName()

        # create stage
        self.stage = SessionUtils.MakeInitialStage(filename)

        material = UsdShade.Material.Define(self.stage, '/' + nodeName)
        self.stage.SetDefaultPrim(material.GetPrim())

        # bxdf
        prmanBxdfPort = node.getInputPort('prmanBxdf')
        if prmanBxdfPort:
            connectedPorts = prmanBxdfPort.getConnectedPorts()
            if connectedPorts:
                shaderoutput = self.makeShade(connectedPorts[0], parent=material.GetPrim())

                rimaterial = UsdRi.MaterialAPI(material)
                risurface  = rimaterial.CreateSurfaceAttr()

                UsdShade.ConnectableAPI.ConnectToSource(risurface, shaderoutput)

        self.stage.GetRootLayer().Save()
        print '->', filename


    def makeShade(self, outport, inport=None, parent=None):
        '''
        Args
            outport : connected node output port
            inport  : current node input port
            parent  : parent prim
        '''
        outport = GetPrmanShadingOutputPort(outport)

        node = outport.getNode()
        nodeName = node.getName()
        portName = outport.getName()
        shaderType = node.getParameter('nodeType').getValue(0)

        shader = UsdShade.Shader.Define(self.stage, parent.GetPath().AppendChild(nodeName))
        shader.SetShaderId(shaderType)

        shaderInfoAttr = GetShaderInfoAttr(shaderType)
        shaderTypeTags = shaderInfoAttr.getChildByName('shaderTypeTags').getValue()
        shaderParams   = shaderInfoAttr.getChildByName('params')

        if shaderTypeTags == 'pattern':
            sdftype = GetSdfValueTypeByPort(inport)
        else:
            sdftype = Sdf.ValueTypeNames.Token

        output = shader.CreateOutput(portName, sdftype)

        # parameters
        self.getShaderParameters(node, shader)
        # connected parameters
        for p in node.getInputPorts():
            for op in p.getConnectedPorts():
                attr = shader.CreateInput(p.getName(), GetSdfValueTypeByPort(p, shaderParams))
                sout = self.makeShade(op, inport=p, parent=parent)
                UsdShade.ConnectableAPI.ConnectToSource(attr, sout)

        return output


    def getShaderParameters(self, node, shader):
        '''
        Args
            node : katana node
            shader : UsdShade.Shader
        '''
        shaderType = node.getParameter('nodeType').getValue(0)
        shaderInfoAttr = GetShaderInfoAttr(shaderType)
        shaderParams   = shaderInfoAttr.getChildByName('params')
        nodeParams = node.getParameter('parameters')

        for i in range(0, shaderParams.getNumberOfChildren()):
            paramName = shaderParams.getChildName(i)

            param = nodeParams.getChild(paramName)
            if not param:
                continue

            enable = param.getChild('enable')
            if enable and enable.getValue(0):
                paramPort = node.getInputPort(paramName)
                if paramPort and paramPort.getConnectedPorts():
                    continue

                usdAttr = shader.CreateInput(paramName, GetSdfValueTypeByName(paramName, shaderParams=shaderParams))

                paramVal = param.getChild('value')
                if paramVal.getChildren():
                    value = list()
                    for p in paramVal.getChildren():
                        value.append(p.getValue(0))
                    if len(value) == 3:
                        usdAttr.Set(Gf.Vec3f(value))
                else:
                    usdAttr.Set(paramVal.getValue(0))


class GetMaterialInfo:
    def __init__(self, evalNode, location, filename):
        root = Nodes3DAPI.GetGeometryProducer(node=evalNode)
        self.prod = root.getProducerByPath(location)
        self.mtlName  = location.split('/')[-1]
        self.filename = filename

    def doIt(self):
        self.nodes_ga = self.prod.getAttribute('material.nodes')
        self.nodesAttr = self.prod.getAttribute('material.nodes')
        if not self.nodes_ga:
            return

        # create stage
        self.stage = SessionUtils.MakeInitialStage(self.filename)
        material = UsdShade.Material.Define(self.stage, '/' + self.mtlName)
        self.mtlprim = material.GetPrim()
        self.stage.SetDefaultPrim(self.mtlprim)

        terminals_ga = self.prod.getAttribute('material.terminals')
        terminals_names = terminals_ga.childNames()
        # bxdf
        if 'prmanBxdf' in terminals_names:
            bxdfName = terminals_ga.getChildByName('prmanBxdf').getValue()
            bxdfPort = terminals_ga.getChildByName('prmanBxdfPort').getValue()
            outputs  = self.getShade(bxdfName, [(bxdfPort, Sdf.ValueTypeNames.Token)])

            rimaterial = UsdRi.MaterialAPI(material)
            risurface  = rimaterial.CreateSurfaceAttr()
            UsdShade.ConnectableAPI.ConnectToSource(risurface, outputs[0])

        self.stage.GetRootLayer().Save()
        print '->', self.filename

    def XXgetShade(self, name, ports, parent): # ports : output port list, parent : parent prim
        '''
        name : shader name
        ports (list) : [(name, sdftype)]
        parent : parent prim
        '''
        ga = self.nodes_ga.getChildByName(name)

        ntype  = ga.getChildByName('type').getValue()
        shader = UsdShade.Shader.Define(self.stage, parent.GetPath().AppendChild(name))
        shader.SetShaderId(ntype)

        shaderInfoAttr = GetShaderInfoAttr(ntype)
        shaderTypeTags = shaderInfoAttr.getChildByName('shaderTypeTags').getValue()
        shaderParams   = shaderInfoAttr.getChildByName('params')

        outputs = list()
        if shaderTypeTags == 'pattern':
            for n, t in ports:
                out = shader.CreateOutput(n, t)
                outputs.append(out)
        else:
            out = shader.CreateOutput(ports[0][0], ports[0][1])
            outputs.append(out)

        child_names = ga.childNames()

        if 'parameters' in child_names:
            # print '# parameters :', ga.getChildByName('parameters').childNames()
            params = ga.getChildByName('parameters')
            for n in params.childNames():
                parm = params.getChildByName(n)

                ptyp = shaderParams.getChildByName(n + '.type').getValue()
                usdAttr = shader.CreateInput(n, pdf._USD_TYPE_MAP_[ptyp])

                if parm.getNumberOfValues() > 1:
                    usdAttr.Set(Gf.Vec3f(list(parm.getData())))
                else:
                    usdAttr.Set(parm.getValue())

        if 'connections' in child_names:
            connectionsAttr = ga.getChildByName('connections')

            # connected node data
            dstData = dict() # {node: [(out port name, sdftype), ...]}
            # current node data
            srcData = dict() # {node: [parm name, ...]}

            for n in connectionsAttr.childNames():
                dst, slname = connectionsAttr.getChildByName(n).getValue().split('@')
                if not dstData.has_key(slname):
                    dstData[slname] = list()
                if not srcData.has_key(slname):
                    srcData[slname] = list()
                ptype= shaderParams.getChildByName(n+'.type').getValue()
                src  = (dst, pdf._USD_TYPE_MAP_[ptype])
                dstData[slname].append(src)
                srcData[slname].append(n)

            pprint.pprint(dstData, indent=4)
            pprint.pprint(srcData, indent=4)

            if dstData and srcData:
                for n in dstData:
                    outputs = self.getShade(n, dstData[n], parent)
                    for i in range(len(srcData[n])):
                        pname = srcData[n][i]
                        ptype = dstData[n][i][1]
                        usdAttr = shader.CreateInput(pname, ptype)
                        UsdShade.ConnectableAPI.ConnectToSource(usdAttr, outputs[i])

        return outputs


    def getShade(self, name, ports):
        '''
        name (str): shader name
        ports (list): [(out port name, sdftype), ...]
        parent : parent prim
        '''
        nodeAttr = self.nodesAttr.getChildByName(name)

        ntype  = nodeAttr.getChildByName('type').getValue()
        shader = UsdShade.Shader.Define(self.stage, self.mtlprim.GetPath().AppendChild(name))
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

        dstData = dict()
        srcData = dict()

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
