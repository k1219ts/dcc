from Katana import NodegraphAPI

#-------------------------------------------------------------------------------
#
#   Macro Work
#
#-------------------------------------------------------------------------------
def GetNode(node=None):
    if node:
        if type(node).__name__ == 'str':
            return NodegraphAPI.GetNode(node)
        else:
            return node
    else:
        nodes = NodegraphAPI.GetAllSelectedNodes()
        if nodes:
            return nodes[0]

def GetMacroInfo(node=None):
    if node:
        nodes = [node]
    else:
        nodes = NodegraphAPI.GetAllSelectedNodes()
    for n in nodes:
        macroParam = n.getParameter('user.macroName')
        verParam   = n.getParameter('user.version')
        if macroParam and verParam:
            print '# Node :', n.getName()
            print '  - macroName :', macroParam.getValue(0)
            print '  - version   :', verParam.getValue(0)

def SetMacroInfo(node=None, name=None, version=None):
    node = GetNode(node)
    macroParam = node.getParameter('user.macroName')
    verParam = node.getParameter('user.version')

    if not macroParam and not verParam:
        CreateMacroParams(node, name, version)
        return

    if macroParam and name:
        macroParam.setValue(name, 0)
    if verParam and version:
        verParam.setValue(version, 0)

def MacroInfoWidget(node=None, hide=True):
    if node:
        nodes = [node]
    else:
        nodes = NodegraphAPI.GetAllSelectedNodes()
    for n in nodes:
        macroParam = n.getParameter('user.macroName')
        verParam   = n.getParameter('user.version')
        if macroParam and verParam:
            macroParam.setHintString(repr({'hide': hide}))
            verParam.setHintString(repr({'hide': hide}))

def CreateMacroParams(node=None, name='', version=1.0):
    node = GetNode(node)
    if node:
        param = node.getParameter('user')
        if not param:
            param = node.getParameters().createChildGroup('user')

        param.createChildString('macroName', name)
        param.createChildNumber('version', version)

        MacroInfoWidget(node)

#-------------------------------------------------------------------------------
def getInputConnectedNodes(node):
    result = list()
    for p in node.getInputPorts():
        for cp in p.getConnectedPorts():
            cn = cp.getNode()
            if not cn in result:
                result.append(cn)
    return result

def getAllConnectedNodes(inputNodes):
    result = list()

    nodes = inputNodes
    chk = 0
    while chk < 400:
        new = list()
        for n in nodes:
            ic_nodes = getInputConnectedNodes(n)
            if ic_nodes:
                result += ic_nodes
                new += ic_nodes
        if not new:
            chk = 400
        nodes = list(new)
        chk += 1

    return result

def getAllConnectedNodesBySelected():
    return getAllConnectedNodes(NodegraphAPI.GetAllSelectedNodes())

def getNodesType(nodes, typename):
    result = list()
    for n in nodes:
        if n.getType() == typename:
            result.append(n)
    return result


#-------------------------------------------------------------------------------
class ConnectedNodes:
    def __init__(self, input):
        self.allNodes = list()

        if isinstance(input, list):
            for n in input:
                self.getInputConnectedNode(n)
        else:
            self.getInputConnectedNode(input)

    def getInputConnectedNode(self, node):
        self.allNodes.append(node)
        for p in node.getInputPorts():
            for cp in p.getConnectedPorts():
                cn = cp.getNode()
                if not cn in self.allNodes:
                    self.getInputConnectedNode(cn)

    def getType(self, typeName):
        result = list()
        for n in self.allNodes:
            if n.getType() == typeName:
                result.append(n)
        return result
