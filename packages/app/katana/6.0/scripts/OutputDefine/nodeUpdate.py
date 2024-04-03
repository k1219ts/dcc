import os
from Katana import NodegraphAPI, Nodes3DAPI, UI4
from PyQt5 import QtGui, QtCore, QtWidgets


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

        # Directory Path
        self.rootNode = self.node.getParent()

        getType = self.node.getParameter('user.Location.locationType').getValue(0)
        getShow = self.node.getParameter('user.Location.shotsetup.i0').getValue(0)
        getSeq = self.node.getParameter('user.Location.shotsetup.i1').getValue(0)
        getShot = self.node.getParameter('user.Location.shotsetup.i2').getValue(0)
        getTask = self.node.getParameter('user.Location.Task').getValue(0)
        self.getLayer = self.node.getParameter('user.Location.Layer').getValue(0)
        self.getVersion = self.node.getParameter('user.Location.Version').getValue(0)
        getImageName = self.node.getParameter('user.Location.ImageName').getValue(0)

        self.dirPath = (getShow + '/_2d/shot/' + getSeq + '/' + getShot + '/' +
                        getTask + '/' + 'images/' + self.getLayer + '/' + self.getVersion)


    def doIt(self):
        if not os.path.exists(self.dirPath):
            print('# WARNING: The folder does not exist!')
            return
        if self.getLayer == '' or self.getVersion == '':
            print('# WARNING: Check the Layer, Version!')
            return

        output = os.path.join(self.dirPath, 'outputDefine.xml')
        if not os.path.exists(output):
            print('# WARNING: The XML file does not exist!')
            return
        print('>>> Loading')
        print('>>> XML File\t\t:', output)

        result = UI4.Widgets.MessageBox.Question("Update", "Do you want to continue?\n", acceptText="Load", cancelText="Cancel")
        if result == QtWidgets.QMessageBox.AcceptRole:
            xmlTree, versionUp = NodegraphAPI.LoadElementsFromFile(output)
            self.newNode = NodegraphAPI.ParseNodesXmlIO(xmlTree)[0]

            PortConnections = Connections(self.rootNode)

            name = self.rootNode.getName()
            self.newNode.setParent(self.rootNode.getParent())
            NodegraphAPI.SetNodePosition(self.newNode, NodegraphAPI.GetNodePosition(self.rootNode))
            self.rootNode.delete()
            self.newNode.setName(name)

            PortConnections.setPorts(self.newNode)


def nodeWrite(node):
    lastNode = node.getReturnPort('diskrender').getConnectedPort(0).getNode()
    root = Nodes3DAPI.GetGeometryProducer(lastNode)
    root = root.getProducerByPath('/root')

    location = root.getAttribute('renderSettings.outputs.primary.locationSettings.renderLocation')
    dirPath = os.path.dirname(location.getValue())
    for i in node.getChildren():
        getName = i.getName()
        if getName.startswith('RenderLocationSetupScript'):
            rlssNode = i

    getLayer = rlssNode.getParameter('user.Location.Layer').getValue(0)
    getVersion = rlssNode.getParameter('user.Location.Version').getValue(0)

    if not os.path.exists(dirPath):
        UI4.Widgets.MessageBox.Critical('Error', 'The folder does not exist!')
        print('# WARNING: The folder does not exist!')
        return
    if getLayer == '' or getVersion == '':
        UI4.Widgets.MessageBox.Critical('Error', 'Check the Layer, Version!')
        print('# WARNING: Check the Layer, Version!')
        return

    output = os.path.join(dirPath, 'outputDefine.xml')
    print('>>> XML File\t\t:', output)
    node.getParameter('user.Info.Location.i0').setValue(output, 0)

    if os.path.exists(output):
        result = UI4.Widgets.MessageBox.Question("Save As", "The file already exists.\n"
                "Do you want to continue?\n", acceptText="Overwrite", cancelText="Cancel")
        if result == QtWidgets.QMessageBox.AcceptRole:
            xmlTree = NodegraphAPI.BuildNodesXmlIO([node])
            xmlTree.write(output)
    else:
            xmlTree = NodegraphAPI.BuildNodesXmlIO([node])
            xmlTree.write(output)
