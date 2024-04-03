from __future__ import print_function
import os, sys

import Katana
from Katana import DrawingModule, KatanaFile, NodegraphAPI, Nodes3DAPI, RenderingAPI
import UI4
from fnpxr import Sdf, Gf, Vt

import DXRulebook.Interface as rb

import DXUSD_KAT.Utils as utl
import DXUSD_KAT.Compositor as cmp

_TNAMES = Sdf.ValueTypeNames

_KAT_TO_SDF_TYPE = {
    RenderingAPI.RendererInfo.kRendererObjectValueTypeString:  _TNAMES.String,
    RenderingAPI.RendererInfo.kRendererObjectValueTypePoint3:  _TNAMES.Float3,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeInt:     _TNAMES.Int,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeFloat:   _TNAMES.Float,
    RenderingAPI.RendererInfo.kRendererObjectValueTypePointer: _TNAMES.String,
    # RenderingAPI.RendererInfo.kRendererObjectValueTypeMatrix
    RenderingAPI.RendererInfo.kRendererObjectValueTypeVector3: _TNAMES.Float3,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeColor3:  _TNAMES.Color3f,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeNormal:  _TNAMES.Float3,
    # RenderingAPI.RendererInfo.kRendererObjectValueTypeShader
}


class CreateMaterial:
    def __init__(self, evalNode, location, output):
        root = Nodes3DAPI.GetGeometryProducer(node=evalNode)
        self.prod = root.getProducerByPath(location)
        self.name = location.split('/')[-1]
        self.output = output

    def doIt(self):
        self.nodesAttr = self.prod.getAttribute('material.nodes')
        if not self.nodesAttr:
            return

        self.outlyr = utl.AsLayer(self.output, create=True, clear=True)
        customLayerData = self.outlyr.customLayerData

        # current project filename
        katfile = NodegraphAPI.NodegraphGlobals.GetProjectFile()
        customData = {'sceneFile': katfile}
        customLayerData.update(customData)
        self.outlyr.customLayerData = customLayerData

        self.outlyr.defaultPrim = self.name

        self.root = utl.GetPrimSpec(self.outlyr, '/' + self.name, type='Material')
        self.root.SetInfo('kind', 'subcomponent')

        terminalsAttr = self.prod.getAttribute('material.terminals')
        terminals = terminalsAttr.childNames()
        # bxdf
        if 'prmanBxdf' in terminals:
            bxdfName = terminalsAttr.getChildByName('prmanBxdf').getValue()
            target   = self.walkNode(bxdfName, ('out', _TNAMES.Token))
            attrSpec = utl.GetAttributeSpec(self.root, 'outputs:ri:surface', None, _TNAMES.Token)
            attrSpec.connectionPathList.explicitItems.append(target.pathString + '.outputs:out')
        # displacement
        if 'prmanDisplacement' in terminals:
            dispName = terminalsAttr.getChildByName('prmanDisplacement').getValue()
            target   = self.walkNode(dispName, ('out', _TNAMES.Token))
            attrSpec = utl.GetAttributeSpec(self.root, 'outputs:ri:displacement', None, _TNAMES.Token)
            attrSpec.connectionPathList.explicitItems.append(target.pathString + '.outputs:out')

        self.outlyr.Save()
        del self.outlyr


    def walkNode(self, node, output=None):   # node name, output (tagName, Sdf.ValueTypeName)
        print('>>> walkNode :', node)
        nodeAttr = self.nodesAttr.getChildByName(node)

        slnm = nodeAttr.getChildByName('srcName').getValue()
        print('>>> srcName :', slnm)
        spec = utl.GetPrimSpec(self.outlyr, self.root.path.AppendChild(slnm), type='Shader')

        id = nodeAttr.getChildByName('type').getValue()
        utl.GetAttributeSpec(spec, 'info:id', id, _TNAMES.Token, variability=Sdf.VariabilityUniform)

        if output:
            utl.GetAttributeSpec(spec, 'outputs:%s' % output[0], None, output[1])

        slAttr = utl.GetShaderFnAttr(id)

        # connections
        connectsAttr = nodeAttr.getChildByName('connections')
        if connectsAttr:
            for n in connectsAttr.childNames():
                print('\t> connections :', n)
                idx  = slAttr.getChildByName('params.%s.type' % n).getValue()
                utyp = _KAT_TO_SDF_TYPE[idx]
                attrSpec = utl.GetAttributeSpec(spec, 'inputs:%s' % n, None, utyp)

                output, nodename = connectsAttr.getChildByName(n).getValue().split('@')
                target = self.walkNode(nodename, (output, utyp))

                items = attrSpec.connectionPathList.explicitItems
                items.clear()
                items.append(target.pathString + '.outputs:%s' % output)

        # set parameters
        paramsAttr = nodeAttr.getChildByName('parameters')
        if paramsAttr:
            for n in paramsAttr.childNames():
                print('\t> parameters :', n)
                idx  = slAttr.getChildByName('params.%s.type' % n).getValue()
                utyp = _KAT_TO_SDF_TYPE[idx]
                parm = paramsAttr.getChildByName(n)

                # special case
                if n == 'colorRamp_Knots':
                    utyp  = _TNAMES.FloatArray
                    value = Vt.FloatArray(list(parm.getData()))
                elif n == 'colorRamp_Colors':
                    utyp  = _TNAMES.Color3fArray
                    value = Vt.Vec3fArray(utl.GetVec3fArray(parm.getData()))

                # general case
                else:
                    if parm.getNumberOfValues() > 1:
                        value = Gf.Vec3f(list(parm.getData()))
                    else:
                        value = parm.getValue()
                attrSpec = utl.GetAttributeSpec(spec, 'inputs:%s' % n, value, utyp)

        return spec.path



def doIt(dir):
    print('>', dir)
    arg = utl.Arguments(dir)

    message = list()
    for node in NodegraphAPI.GetAllSelectedNodes():
        if node.getType() != 'NetworkMaterialCreate':
            continue

        print('#### Export Material ####')
        print('>>> Material Node\t:', node.getName())

        if Katana.version[0] >= 4:
            name = node.getNetworkMaterials()[0].getName()
            location = node.getParameter("rootLocation").getValue(0)
            location += '/' + name
        else:
            mtln = node.getParameter('__node_networkMaterial').getValue(0)  # string
            mtln = NodegraphAPI.GetNode(mtln)
            name = mtln.getParameter('name').getValue(0)
            namespace = mtln.getParameter('namespace').getValue(0)
            # scenegraph location
            location = '/root/materials'
            if namespace:
                location += '/' + namespace
                arg.setSubdir(namespace)
            location += '/' + name

        arg.setName(name)

        dstdir = arg.outDir
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        arg.pop('nsver')

        # usd material
        output = utl.SJoin(dstdir, node.getName() + '.usd')
        print('>>> USD File\t\t:', output)
        CreateMaterial(node, location, output).doIt()
        # composite
        cmp.MaterialComposite(output).DoIt()

        # xml material
        output = output.replace('.usd', '.xml')
        print('>>> XML File\t\t:', output)
        xmlTree = NodegraphAPI.BuildNodesXmlIO([node])
        xmlTree.write(output)
        print('')

        msg = '%s : %s' % (location, dstdir)
        message.append(msg)

    if message:
        UI4.Widgets.MessageBox.Information('Material Export Result', '\n'.join(message))


def exportDialog():
    nodes = list()
    for n in NodegraphAPI.GetAllSelectedNodes():
        if n.getType() == 'NetworkMaterialCreate':
            nodes.append(n)
    # ERROR Message
    if not nodes:
        UI4.Widgets.MessageBox.Critical('Material Selection Error', '\tSelect NetworkMaterialCreate Nodes!\t')
        return

    dir = UI4.Util.AssetId.BrowseForAsset('', 'USD Export Material (select directory)', True, {'acceptDir': True})
    if dir:
        doIt(dir)
