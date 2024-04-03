import os, sys, re

from Katana import NodegraphAPI
from Katana import Utils
from Katana import RenderingAPI


def _Initialize():
    if 'RMANTREE' in os.environ:
        rmantree = os.environ['RMANTREE']
        rmanpath = os.path.join(rmantree, 'bin')
        if rmanpath not in sys.path:
            sys.path.append(rmanpath)
    else:
        return False


def set_node_name_param(node, name):
    node.getParameter('name').setValue(name, 0)
    Utils.EventModule.ProcessAllEvents()

    uniqueName = node.getName()
    node.getParameter('name').setValue(uniqueName, 0)
    Utils.EventModule.ProcessAllEvents()

    return uniqueName


def create_nodes(Asset):
    '''
    Args:
        Asset (RmanAsset)

    Returns:

    '''
    root = NodegraphAPI.GetRootNode()

    label = Asset.label()
    label = re.sub(r'[\s:]+', '_', label)

    assetGroupNode = NodegraphAPI.CreateNode('Group', root)
    assetGroupNode.setName(str(label))

    shadingNetworkGroupNode = NodegraphAPI.CreateNode('Group', assetGroupNode)
    shadingNetworkGroupNode.setName(str(label) + '_ShadingNetworkGroup')

    networkMtlNode = NodegraphAPI.CreateNode('NetworkMaterial', assetGroupNode)
    set_node_name_param(networkMtlNode, str(label) + '_NetworkMaterial')

    nodeList = list()
    nodeDict = dict()

    for node in Asset.nodeList():
        nodeId    = node.name()
        nodeType  = node.type()
        nodeClass = node.nodeClass()

        nodeLabel = re.sub(r'[\s:]+', '_', nodeId)

        if nodeType == 'coordinateSystem':
            continue
        elif nodeType == 'shadingEngine':
            continue
        elif nodeClass == 'bxdf':
            bxdfNode = NodegraphAPI.CreateNode('PrmanShadingNode', shadingNetworkGroupNode)
            set_node_name_param(bxdfNode, str(nodeLabel))
            bxdfNode.getParameter('nodeType').setValue(str(nodeType), 0)

            bxdfInput = networkMtlNode.addInputPort('prmanBxdf')
            bxdfNode.getOutputPortByIndex(0).connect(bxdfInput)

            nodeList.append(bxdfNode)
            nodeDict[nodeId] = bxdfNode


def BuildNodeType(nodeType):
    if nodeType == 'DxTexture':
        return 'DxTexFile'



class CreateNodes:
    '''
    Args:
        Asset (RmanAsset)
    '''
    def __init__(self, Asset, parent):
        self.Asset  = Asset
        self.parent = parent

        rendererInfo= RenderingAPI.RenderPlugins.GetInfoPlugin('prman')
        shaderType  = RenderingAPI.RendererInfo.kRendererObjectTypeShader
        self.validShaders = rendererInfo.getRendererObjectNames(shaderType)

        # Member Variables
        self.nodeList = list()
        self.nodeDict = dict()

    def doIt(self):
        self.create()
        self.connect_nodes()


    def create(self):
        for node in self.Asset.nodeList():
            nodeId    = node.name()
            nodeType  = node.type()
            nodeClass = node.nodeClass()
            # print nodeId, nodeType, nodeClass
            nodeLabel = re.sub(r'[\s:]+', '_', nodeId)

            if not nodeType in self.validShaders:
                nodeType = BuildNodeType(nodeType)

            if nodeType == 'coordinateSystem':
                continue
            elif nodeClass == 'root':
                networkMtlNode = NodegraphAPI.CreateNode('NetworkMaterial', self.parent)
                set_node_name_param(networkMtlNode, str(nodeLabel))
                self.nodeList.append(networkMtlNode)
                self.nodeDict[nodeId] = networkMtlNode
            else:
                prmanNode = NodegraphAPI.CreateNode('PrmanShadingNode', self.parent)
                set_node_name_param(prmanNode, str(nodeLabel))
                prmanNode.getParameter('nodeType').setValue(str(nodeType), 0)
                prmanNode.checkDynamicParameters()
                self.nodeList.append(prmanNode)
                self.nodeDict[nodeId] = prmanNode


    def connect_nodes(self):
        for con in self.Asset.connectionList():
            if (con.srcNode() not in self.nodeDict) or (con.dstNode() not in self.nodeDict):
                continue
            srcNode = self.nodeDict[con.srcNode()]
            dstNode = self.nodeDict[con.dstNode()]
            srcAttr = con.srcParam()
            dstAttr = con.dstParam()
            # print 'dst : %s --> src : %s' % (dstAttr, srcAttr)
            if dstAttr == 'surfaceShader':
                dstPort = dstNode.addInputPort('prmanBxdf')
                srcPort = srcNode.getOutputPort('out')
                # dstNode.getInputPort('prmanBxdf').connect(srcNode.getOutputPort('out'))
            elif dstAttr == 'displacementShader':
                dstPort = dstNode.addInputPort('prmanDisplacement')
                srcPort = srcNode.getOutputPort('out')
                # dstNode.getInputPort('prmanDisplacement').connect(srcNode.getOutputPort('out'))
            else:
                dstPort = dstNode.getInputPort(str(dstAttr))
                srcPort = srcNode.getOutputPort(str(srcAttr))

            if srcPort.isConnected(dstPort) is False:
                srcPort.connect(dstPort)





def import_asset(filename):
    if _Initialize() == False:
        return
    from rmanAssets.core import RmanAsset

    Asset = RmanAsset()
    Asset.load(filename, localizeFilePaths=True)
    assetType = Asset.type()

    if assetType == 'nodeGraph':
        # create_nodes(Asset)
        CreateNodes(Asset, NodegraphAPI.GetRootNode()).doIt()
        # return Asset.connectionList()
