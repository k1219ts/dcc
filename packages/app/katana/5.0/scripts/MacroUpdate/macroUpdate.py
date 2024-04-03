import string, re

import NodegraphAPI


class Connections:
    def __init__(self, node):
        self.inputPortMap  = dict()
        self.outputPortMap = dict()

        iports = node.getInputPorts()
        if iports:
            for i in iports:
                name = i.getName()
                self.inputPortMap[name] = i.getConnectedPorts()

        oports = node.getOutputPorts()
        if oports:
            for o in oports:
                name = o.getName()
                self.outputPortMap[name] = o.getConnectedPorts()

    def setPorts(self, node):
        for name in self.inputPortMap:
            ports = self.inputPortMap[name]
            if ports:
                for p in ports:
                    ip = node.getInputPort(name)
                    if ip:
                        ip.connect(p)

        for name in self.outputPortMap:
            ports = self.outputPortMap[name]
            if ports:
                for p in ports:
                    op = node.getOutputPort(name)
                    if op:
                        op.connect(p)


class NodeUpdate:
    def __init__(self, node):
        self.node = node
        self.inum = 0

        # Member Variable
        self.macroName    = None
        self.macroVersion = None
        nameParam = self.node.getParameter('user.macroName')
        if nameParam:
            self.macroName = nameParam.getValue(0)
        versionParam = self.node.getParameter('user.version')
        if versionParam:
            self.macroVersion = versionParam.getValue(0)

    def doIt(self):
        if not self.macroName or not self.macroVersion:
            print('# WARNING: not support this node -> %s', self.node)
            return

        print('# Debug : name - %s, version - %s' % (self.macroName, self.macroVersion))
        self.newNode = NodegraphAPI.CreateNode(self.macroName)
        newVersion   = self.newNode.getParameter('user.version').getValue(0)
        if newVersion <= self.macroVersion:
            self.newNode.delete()
            print('# Result: this is final version!')
            return

        PortConnections = Connections(self.node)

        param = self.GetParameter(self.node, 'user')
        self.iterParams(param)

        name   = self.node.getName()
        self.newNode.setParent(self.node.getParent())
        NodegraphAPI.SetNodePosition(self.newNode, NodegraphAPI.GetNodePosition(self.node))
        self.node.delete()
        self.newNode.setName(name)

        PortConnections.setPorts(self.newNode)


    def iterParams(self, param):
        self.inum += 1
        children = param.getChildren()
        if children:
            # print('# Debug :', param.getNode())
            for p in children:
                # print('# %s:' % self.inum, p.getName())
                ptype   = p.getType()
                if ptype == 'group':
                    self.iterParams(p)
                    continue
                hintstr = p.getHintString()
                if hintstr:
                    hint = eval(hintstr)
                    if hint.has_key('widget'):
                        if hint['widget'] == 'teleparam':
                            tmp  = p.getValue(0).split('.')
                            node = NodegraphAPI.GetNode(tmp[0])
                            if node:
                                pstr = string.join(tmp[1:], '.')
                                self.iterParams(self.GetParameter(node, pstr))
                        elif hint['widget'] == 'scriptButton':
                            pass
                        else:
                            self.GetParamValue(p)
                    else:
                        # print(hint)
                        self.GetParamValue(p)
                else: # get value
                    self.GetParamValue(p)


    def GetParameter(self, node, paramstr):
        return node.getParameter(paramstr)


    def GetParamValue(self, param):
        node     = param.getNode()
        nodeName = node.getName()
        if nodeName == self.node.getName():
            targetNode = self.newNode
        else:
            targetNode = self.getTargetNode(nodeName)

        name  = param.getName()
        ptype = param.getType()

        hintstr = param.getHintString()
        if hintstr:
            hint = eval(hintstr)
            if hint.has_key('readOnly') and hint['readOnly']:
                # print('# Debug : %s - readOnly' % name)
                return

        if ptype.find('Array') > -1:
            values = list()
            for i in param.getChildren():
                self.setParameter(i)
                values.append(i.getValue(0))
            # print('# Result : name - %s, value - %s' % (name, values), nodeName, targetNode)
        else:
            self.setParameter(param)
            value = param.getValue(0)
            # print('# Result : name - %s, value - %s' % (name, value), nodeName, targetNode)


    def setParameter(self, param):
        nodeName = param.getNode().getName()
        if nodeName == self.node.getName():
            targetNode = self.newNode
        else:
            targetNode = self.getTargetNode(nodeName)
        if not targetNode:
            # print('# Debug - setParameter: not found node!')
            return

        fullName = param.getFullName()
        if fullName == nodeName + '.user.version':
            return
        paramstr = fullName.replace('%s.' % nodeName, '')

        newParam = targetNode.getParameter(paramstr)
        if newParam:
            # is expression
            if param.isExpression():
                newParam.setExpression(param.getExpression())
            else:
                newParam.setValue(param.getValue(0), 0)


    def getTargetNode(self, nodeName):
        name = nodeName
        rule = re.compile('\d+').findall(name)
        if rule:
            name = nodeName.replace(rule[-1], '')

        for child in self.newNode.getChildren():
            if child.getName().find(name) > -1:
                return child
