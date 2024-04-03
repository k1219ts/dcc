"""
NAME: NetworkMaterialGroup -> LiveGroup Update
ICON: /backstage/share/icons/katana/LiveGroupChildEditable16.png
DROP_TYPES:
SCOPE:

Selected NetworkMaterialGroups Update LiveGroup

"""

import string
from Katana import KatanaFile, NodegraphAPI, UI4


class UpdateLiveGroup:
    def __init__(self, liveNode, mtlNodes):
        self.liveNode = liveNode
        self.mtlNodes = mtlNodes

    def doIt(self):
        newMaterialNames = self.GetNewMaterials()

        liveMaterialNames, liveMaterialMap = self.GetLiveMaterials()

        intersects = list(set(liveMaterialNames).intersection(set(newMaterialNames)))
        if intersects:
            msg = UI4.Widgets.MessageBox.Question(
                'Overwrite Materials',
                "Do you want to overwrite '%s'?" % string.join(intersects, ', '),
                acceptText='Ok', cancelText='Cancel'
            )
            if msg:
                return

        self.liveNode.load()
        self.liveNode.makeEditable()

        self.inTeleport = self.liveNode.getSendPort('in').getConnectedPorts()[0].getNode()
        self.outTeleport= self.liveNode.getReturnPort('out').getConnectedPorts()[0].getNode()
        last = self.outTeleport.getInputPortByIndex(0).getConnectedPorts()[0].getNode()

        for n in self.mtlNodes:
            mtlName = self.getMaterialName(n)
            xml = NodegraphAPI.BuildNodesXmlIO([n])
            # delete new material
            n.delete()

            if mtlName in intersects:
                ln = liveMaterialMap[mtlName]
                last = ln.getInputPortByIndex(0).getConnectedPorts()[0].getNode()
                # delete live material
                self.deleteNode(ln)

            createNodes = KatanaFile.Paste(xml, self.liveNode)
            for cn in createNodes:
                self.connectNode(cn, last)
                last = cn

        self.nodePlace()

        # liveGroup publish
        self.liveNode.publishAssetAndFinishEditingContents()
        self.liveNode.load()


    def deleteNode(self, node):
        inports  = node.getInputPortByIndex(0).getConnectedPorts()
        outports = node.getOutputPortByIndex(0).getConnectedPorts()
        node.delete()
        if inports and outports:
            inports[0].connect(outports[0])

    def connectNode(self, node, last):
        lastConnected = last.getOutputPortByIndex(0).getConnectedPorts()[0].getNode()
        last.getOutputPortByIndex(0).connect(node.getInputPortByIndex(0))
        node.getOutputPortByIndex(0).connect(lastConnected.getInputPortByIndex(0))


    def getMaterialName(self, node):
        '''
        Args:
            node - NetworkMaterialGroup
        '''
        all = list()
        for n in node.getChildren():
            if n.getType() == 'Group':
                all += n.getChildren()
            else:
                all.append(n)

        for n in all:
            if n.getType() == 'NetworkMaterial':
                name = n.getParameter('name').getValue(0)
                if n.getOutputPortByIndex(0).getNumConnectedPorts() > 0:
                    return name


    def GetNewMaterials(self):
        result = list()
        for n in self.mtlNodes:
            name = self.getMaterialName(n)
            if name:
                result.append(name)
        return result

    def GetLiveMaterials(self):
        names = list()
        data  = dict()
        for n in self.liveNode.getChildren():
            if n.getType() == 'Group':
                param = n.getParameter('user.macroName')
                if param:
                    if param.getValue(0) == 'NetworkMaterialGroup':
                        name = self.getMaterialName(n)
                        if name:
                            names.append(name)
                            data[name] = n
        return names, data


    def nodePlace(self):
        NodegraphAPI.SetNodePosition(self.inTeleport, (0, 0))
        v_count = 1
        current = self.inTeleport
        for i in range(len(self.liveNode.getChildren())-1):
            next = self.getNext(current)
            if next:
                current = next
                NodegraphAPI.SetNodePosition(current, (0, -80 * v_count))
                v_count += 1

        NodegraphAPI.SetNodePosition(self.outTeleport, (0, -80 * v_count))

    def getNext(self, node):
        outputs = node.getOutputPortByIndex(0).getConnectedPorts()
        if outputs:
            next = outputs[0].getNode()
            if next.getType() == 'Group':
                return next


liveNode = None
mtlNodes = list()

selectedNodes = NodegraphAPI.GetAllSelectedNodes()
for n in selectedNodes:
    ntyp = n.getType()
    if ntyp == 'LiveGroup':
        liveNode = n
    elif ntyp == 'Group':
        macroParam = n.getParameter('user.macroName')
        if macroParam:
            if macroParam.getValue(0) == 'NetworkMaterialGroup':
                mtlNodes.append(n)

if liveNode and mtlNodes:
    UpdateLiveGroup(liveNode, mtlNodes).doIt()
