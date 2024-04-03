from Katana import NodegraphAPI
from Katana import Nodes3DAPI

# __TargetVariants = ['camVersion', 'setVersion', 'aniVersion', 'zennVersion', 'simVersion']

def GetShotName():
    variablesGroup = NodegraphAPI.GetRootNode().getParameter('variables')
    variableParam  = variablesGroup.getChild('shotVariant')
    if variableParam:
        return variableParam.getChild('value').getValue(0)

def GetRefNode(gnode, key):
    p = gnode.getParameter('node_'+key)
    if not p:
        return None
    return NodegraphAPI.GetNode(p.getValue(0))

def AddNodeReferenceParam(destNode, paramName, node):
    param = destNode.getParameter(paramName)
    if not param:
        param = destNode.getParameters().createChildString(paramName, '')
    param.setExpression('getNode(%r).getNodeName()' % node.getName())

def GetVersionGroup(gnode):
    shotName = GetShotName()
    if shotName:
        return GetRefNode(gnode, '%s_VersionCtrlGroup' % shotName)



class GetVersionTree:
    '''
    :result
        [
            (location, [(varname, varvalue)]),
            (location, [(varname, varvalue), (varname, varvalue), ...]),
            ...
        ]
    '''
    def __init__(self, gnode):
        self.rootLocation = '/root/world/geo/shot'
        self._node     = gnode
        self._Producer = None
        self.inputNode = None
        connectedPort  = gnode.getInputPort('in').getConnectedPort(0)
        if not connectedPort:
            return

        self.inputNode= connectedPort.getNode()
        self._Producer = Nodes3DAPI.GetGeometryProducer(self.inputNode)

    def GetRootProducer(self, node=None):
        if node:
            return Nodes3DAPI.GetGeometryProducer(node)
        else:
            return self._Producer

    def GetChildrenProducer(self, location):
        result = list()
        root = self._Producer.getProducerByPath(location)
        if not root:
            return result
        for c in root.iterChildren():
            result.append(c)
        return result

    def GetLocation(self, location):
        if location[0] == '/':
            return location
        else:
            return self.rootLocation + '/' + location


    def doIt(self):
        if not self._Producer:
            return

        producers = self.GetChildrenProducer(self.rootLocation)

        result = list()
        for c in producers:
            name = c.getFullName()
            vers = self.versionVariant(c)
            if vers:
                result.append((name, vers))

            for s in self.GetChildrenProducer(c.getFullName()):
                name = s.getFullName()
                vers = self.versionVariant(s)
                if vers:
                    result.append((name, vers))

        self.result = result
        return result


    def versionVariant(self, producer):
        __TargetVariants = ['camVersion', 'setVersion', 'aniVersion', 'zennVersion', 'simVersion']
        groupAttr = producer.getAttribute('info.usd.selectedVariants')
        if not groupAttr:
            return

        data = list()
        for i in range(groupAttr.getNumberOfChildren()):
            name = groupAttr.getChildName(i)
            if name in __TargetVariants:
                attr = producer.getAttribute('info.usd.selectedVariants.%s' % name)
                val  = attr.getValue()
                data.append((name, val))
        return data

    def variantValues(self, location, varname, node=None):
        root = self.GetRootProducer(node)
        prod = root.getProducerByPath(self.GetLocation(location))
        if prod:
            variantAttr = prod.getAttribute('info.usd.variants.%s' % varname)
            if variantAttr:
                selAttr = prod.getAttribute('info.usd.selectedVariants.%s' % varname)
                return variantAttr.getData(), selAttr.getValue()


    def computeTreeItem(self):
        main = list(); layerMap = dict(); variantMap = dict()
        for i in self.result:
            location= i[0]
            relpath = location.split('/shot/')[-1]

            variantMap[relpath] = list()
            for x in i[1]:
                variantMap[relpath].append(x[0])

            src = relpath.split('/')
            if not src[0] in main:
                main.append(src[0])
                layerMap[src[0]] = list()
            if len(src) > 1:
                layerMap[src[0]].append(src[1])

        self.variantMap = variantMap
        return main, layerMap


#-------------------------------------------------------------------------------
#
#   Create Node
#
#-------------------------------------------------------------------------------
def GetVariableSwitchInputPort(switcher, value):
    '''
    Create InputPort or Find Port, Set pattern value
    :param
        - switcher : VariableSwitch Node
        - value : pattern value. is shotName.
    '''
    inputPort = None
    for port in switcher.getInputPorts():
        patternValue = switcher.getParameter('patterns.%s' % port.getName()).getValue(0)
        if patternValue == value:
            inputPort = port

    if not inputPort:
        inputPort = switcher.addInputPort('i0')
        switcher.getParameter('patterns.%s' % inputPort.getName()).setValue(value, 0)

    return inputPort


# Main Group
def CreateVersionCtrlGroup(gnode):
    shotName = GetShotName()
    data = GetVersionTree(gnode).doIt()
    refName = '%s_VersionCtrlGroup' % shotName

    groupNode = NodegraphAPI.CreateNode('Group', gnode)
    groupNode.setName(refName)
    AddNodeReferenceParam(gnode, 'node_%s' % refName, groupNode)

    groupNode.addInputPort('in')
    groupNode.addOutputPort('out')

    groupNode.getInputPort('in').connect(gnode.getSendPort('in'))
    groupNode.getReturnPort('out').connect(groupNode.getSendPort('in'))

    varswithNode = GetRefNode(gnode, 'varswith')
    inputPort = GetVariableSwitchInputPort(varswithNode, shotName)
    inputPort.connect(groupNode.getOutputPort('out'))

    CreateVersionControler(groupNode, data)
    return groupNode


def CreateVersionControler(gnode, data):
    '''
    :param
        - gnode : $SHOT_VersionCtrlGroup node
        - data : GetVersionTree result
    '''
    last = gnode.getSendPort('in')

    for loc, vars in data:
        if loc.find('/shot/rig/') > -1:
            for n in ['aniVersion', 'simVersion', 'zennVersion']:
                node = CreatePxrUsdInVariantSelect(gnode, loc, n, last)
                last = node.getOutputPort('out')
        else:
            for n, v in vars:
                node = CreatePxrUsdInVariantSelect(gnode, loc, n, last)
                last = node.getOutputPort('out')

    gnode.getReturnPort('out').connect(last)
    pxrVarNodePlace(gnode)
    ctrlGroupPlace(gnode)


def CreatePxrUsdInVariantSelect(parent, location, variantName, last):
    name = location.split('/')[-1] + '_' + variantName

    node = NodegraphAPI.CreateNode('PxrUsdInVariantSelect', parent)
    AddNodeReferenceParam(parent, 'node_%s' % name, node)
    node.setName(name)

    node.getParameter('location').setValue(location, 0)

    node.getParameter('args.variantSetName.enable').setValue(1, 0)
    node.getParameter('args.variantSetName.value').setValue(variantName, 0)

    node.getInputPort('in').connect(last)
    return node



def pxrVarNodePlace(gnode):
    v_count = 0
    for n in gnode.getChildren():
        v_count += 1
        NodegraphAPI.SetNodePosition(n, (0, -80 * v_count))

def ctrlGroupPlace(gnode):
    h_count = 0
    for n in gnode.getParent().getChildren():
        ntype = n.getType()
        if ntype == 'Group':
            NodegraphAPI.SetNodePosition(n, (300 * h_count, 80))
            h_count += 1
