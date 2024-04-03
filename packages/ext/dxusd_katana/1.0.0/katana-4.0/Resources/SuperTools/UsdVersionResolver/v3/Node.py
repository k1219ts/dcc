
from Katana import NodegraphAPI, Nodes3DAPI, Utils, UniqueName

import ScriptActions as SA
import logging
log = logging.getLogger('UsdVersionResolver.Node')

class UsdVersionResolverNode(NodegraphAPI.SuperTool):
    def __init__(self):
        self.hideNodegraphGroupControls()
        self.addInputPort('in')
        self.addOutputPort('out')

        self.__buildDefaultNetwork()

    def __buildDefaultNetwork(self):
        varswithNode = NodegraphAPI.CreateNode('VariableSwitch', self)
        SA.AddNodeReferenceParam(self, 'node_varswith', varswithNode)

        varswithNode.getParameter('variableName').setValue('shotVariant', 0)
        varswithNode.addInputPort('i0').connect(self.getSendPort('in'))
        self.getReturnPort(self.getOutputPortByIndex(0).getName()).connect(varswithNode.getOutputPortByIndex(0))


    def upgrade(self):
        pass
