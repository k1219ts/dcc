import os
from Katana import NodegraphAPI, Nodes3DAPI

import dfk

_GetScenegraph__TargetVariants = [
    'camVer', 'layoutVer', 'aniVer', 'groomVer', 'simVer'
]

def GetShotName():
    variablesGroup = NodegraphAPI.GetRootNode().getParameter('variables')
    variableParam  = variablesGroup.getChild('shot')
    if variableParam:
        return variableParam.getChild('value').getValue(0)


def GetRefNode(parent, name):
    parm = parent.getParameter('node_' + name)
    if not parm:
        return None
    return NodegraphAPI.GetNode(parm.getValue(0))


def GetVersionGroup(parent):
    shotName = GetShotName()
    if shotName:
        return GetRefNode(parent, '%s_VersionCtrlGroup' % shotName)


def AddNodeReferenceParam(destNode, paramName, node):
    parm = destNode.getParameter(paramName)
    if not parm:
        parm = destNode.getParameters().createChildString(paramName, '')
    parm.setExpression('getNode(%r).getNodeName()' % node.getName())


def GetVariantValues(node, location, varname):
    root = Nodes3DAPI.GetGeometryProducer(node)
    prod = root.getProducerByPath(location)
    if prod:
        variantAttr = prod.getAttribute('info.usd.variants.%s' % varname)
        if variantAttr:
            selAttr = prod.getAttribute('info.usd.selectedVariants.%s' % varname)
            return variantAttr.getData(), selAttr.getValue()


class GetScenegraph:
    def __init__(self, node):
        # Member variable
        self.node = node
        self.Producer  = None
        self.inputNode = None       # first input connection node
        self.info      = list()
        self.variantMap= dict()
        self.maxDepth  = 9

        CNODES = dfk.ConnectedNodes(node)
        if not CNODES.allNodes:
            return

        usdInNode = CNODES.getType('UsdIn')[0]
        self.rootLocation = usdInNode.getParameter('location').getValue(0) + '/World'

        self.inputNode = CNODES.allNodes[1]
        self.RootProd  = Nodes3DAPI.GetGeometryProducer(self.inputNode)

    def doIt(self):
        if not self.RootProd:
            return
        self.getChildProducer(self.rootLocation)

        if self.info:
            for loc, vers in self.info:
                relpath = loc.split('/World/')[-1]
                self.variantMap[relpath] = list()
                for x in vers:
                    self.variantMap[relpath].append(x[0])

    def getChildProducer(self, location):
        prod = self.RootProd.getProducerByPath(location)
        for c in prod.iterChildren():
            name = c.getFullName()
            data = self.versionVariant(c)
            if data:
                self.info.append((name, data))
            if len(name.split('/')) < self.maxDepth:
                self.getChildProducer(name)

    def versionVariant(self, producer):
        groupAttr = producer.getAttribute('info.usd.selectedVariants')
        if not groupAttr:
            return
        result = list()
        for i in range(groupAttr.getNumberOfChildren()):
            name = groupAttr.getChildName(i)
            if name in __TargetVariants:
                attr = producer.getAttribute('info.usd.selectedVariants.%s' % name)
                val  = attr.getValue()
                result.append((name, val))
        return result

    def getLocation(self, location):
        if location[0] == '/':
            return location
        else:
            return self.rootLocation + '/' + location


class CreateVersionCtrlGroup:
    def __init__(self, parent):
        self.mainNode = parent
        self.shotName = GetShotName()

    def doIt(self):
        SG = GetScenegraph(self.mainNode)
        SG.doIt()
        refName = '%s_VersionCtrlGroup' % self.shotName

        rootNode = NodegraphAPI.CreateNode('Group', self.mainNode)
        rootNode.setName(refName)
        AddNodeReferenceParam(self.mainNode, 'node_%s' % refName, rootNode)

        self.connect(rootNode)

        swithNode = GetRefNode(self.mainNode, 'varswith')
        inputPort = self.getVariableSwitchInputPort(swithNode)
        inputPort.connect(rootNode.getOutputPort('out'))

        self.makeControler(rootNode, SG.info)
        return rootNode

    def connect(self, node):
        node.addInputPort('in')
        node.addOutputPort('out')
        node.getInputPort('in').connect(self.mainNode.getSendPort('in'))
        node.getReturnPort('out').connect(node.getSendPort('in'))

    def getVariableSwitchInputPort(self, switcher):
        inputPort = None
        for port in switcher.getInputPorts():
            patternValue = switcher.getParameter('patterns.%s' % port.getName()).getValue(0)
            if patternValue == self.shotName:
                inputPort = port
        if not inputPort:
            inputPort = switcher.addInputPort('i0')
            switcher.getParameter('patterns.%s' % inputPort.getName()).setValue(self.shotName, 0)
        return inputPort

    def makeControler(self, parent, data):
        last = parent.getSendPort('in')

        for loc, vars in data:
            if loc.find('/World/Rig/') > -1:
                for n in ['aniVer', 'simVer', 'groomVer']:
                    node = self.createUsdInVariantSelect(parent, loc, n, last)
                    last = node.getOutputPort('out')
            else:
                for n, v in vars:
                    node = self.createUsdInVariantSelect(parent, loc, n, last)
                    last = node.getOutputPort('out')

        parent.getReturnPort('out').connect(last)
        self.nodePlace(parent)

    def createUsdInVariantSelect(self, parent, location, variantName, last):
        name = location.split('/')[-1] + '_' + variantName
        node = NodegraphAPI.CreateNode('UsdInVariantSelect', parent)
        AddNodeReferenceParam(parent, 'node_%s' % name, node)
        node.setName(name)
        node.getParameter('location').setValue(location, 0)
        node.getParameter('args.variantSetName.enable').setValue(1, 0)
        node.getParameter('args.variantSetName.value').setValue(variantName, 0)
        node.getInputPort('in').connect(last)
        return node

    def nodePlace(self, parent):
        v_count = 0
        for n in parent.getChildren():
            v_count += 1
            NodegraphAPI.SetNodePosition(n, (0, -80 * v_count))

        h_count = 0
        for n in parent.getParent().getChildren():
            ntype = n.getType()
            if ntype == 'Group':
                NodegraphAPI.SetNodePosition(n, (300 * h_count, 80))
                h_count += 1
