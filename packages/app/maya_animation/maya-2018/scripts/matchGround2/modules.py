
import string
import maya.OpenMaya as OpenMaya
import maya.cmds as cmds
import aniCommon


MG_NODES = [
    "_MGAnimLOC",
    "_MGPrjMesh",
    "_MGOffsetNUL",
    "_MGOffsetLOC",
    "MGOffset_GRP"
]

LAYER_NAME = "mgAnimLayer"

def initPlugin():
    if not cmds.pluginInfo("ghProjectMesh", q=True, l=True):
        cmds.loadPlugin("ghProjectMesh")

def getSelection():
    sel = cmds.ls(sl=True) or []
    return sel

class MatchGround:
    @staticmethod
    def deleteNodes(selection):
        if not selection:
            raise Exception("Select Object")

        for sel in selection:
            splitName = sel.split(":")
            nameSpace = string.join(splitName[:-1], ":")
            if nameSpace:
                nameSpace += ":*"
            else:
                nameSpace = "*"
            nodes = map(lambda node: nameSpace + node + "*", MG_NODES)
            for i in nodes:
                if cmds.ls(i): cmds.delete(i)

    def __init__(self):
        self._groundMesh = str()
        self._controlersInfo = dict()
        self._newLayer = False
        self.offsetNulls = list()

    @property
    def groundMesh(self):
        return self._groundMesh

    @groundMesh.setter
    def groundMesh(self, value):
        self._groundMesh = value

    @property
    def controlersInfo(self):
        return self._controlersInfo

    @controlersInfo.setter
    def controlersInfo(self, value):
        self._controlersInfo = value

    @property
    def newLayer(self):
        return self._newLayer

    @newLayer.setter
    def newLayer(self, status):
        self._newLayer = status

    def createAnimLayer(self):
        baseAnimLayer = "Base_" + LAYER_NAME
        animLayerName = baseAnimLayer
        kwargs = {"e": True}

        if self.newLayer and cmds.objExists(baseAnimLayer):
            animLayerName = LAYER_NAME
        elif self.newLayer and not cmds.objExists(baseAnimLayer):
            kwargs["e"] = False
        elif not self.newLayer and not cmds.objExists(baseAnimLayer):
            kwargs["e"] = False

        animLayerName = cmds.animLayer(animLayerName)
        return animLayerName, kwargs

    def addToAnimLayer(self, layerName, **kwargs):
        animLyr = cmds.animLayer(layerName, **kwargs)
        return animLyr

    def createAnimNull(self, controler):
        animNullNode = cmds.createNode('transform', n=controler + MG_NODES[0])
        parentNode = cmds.listRelatives(controler, p=True, type='transform')
        if parentNode:
            cmds.parent(animNullNode, parentNode[0])

        animCurves = cmds.listConnections(controler, scn=True, d=False, type='animCurve')

        attrs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        for attr in attrs:
            try:
                if animCurves:
                    cmds.copyKey(controler, at=attr)
                    cmds.pasteKey(animNullNode, option="replace", copies=True, at=attr)
                else:
                    cmds.copyAttr(controler, animNullNode, values=True, at=attr)
            except:
                pass
        return animNullNode

    def createOffsetLocator(self, controler, prjNode, offset):
        """

        :param controler:
        :param prjNode:
        :type offset: MVector
        :return:
        """
        prjNode_offset = cmds.createNode('ghProjectMesh', n=controler + MG_NODES[1])
        cmds.setAttr(prjNode_offset + ".WeightRotate", 0)
        offsetNul = cmds.createNode('transform', n=controler + MG_NODES[2])
        self.offsetNulls.append(offsetNul)
        offsetLoc = cmds.spaceLocator(n=controler + MG_NODES[3])[0]
        cmds.parent(offsetLoc, offsetNul)

        cmds.connectAttr(prjNode_offset + ".outputTranslate", offsetNul + ".translate", f=True)
        cmds.connectAttr(offsetLoc + ".translateY", prjNode + ".offsetY", f=True)

        cmds.setAttr(offsetLoc + ".ty", offset.y)

        return prjNode_offset


    def createPRJMNode(self, controler):
        prjNode = cmds.createNode('ghProjectMesh', n=controler + MG_NODES[1])
        cmds.setAttr(prjNode + '.offsetY', self.controlersInfo[controler])
        cmds.setAttr(prjNode + '.WeightRotate', 0)

        return prjNode

    def getWorldTranslate(self, matrixList):
        newWorldMatrix = OpenMaya.MMatrix()
        OpenMaya.MScriptUtil.createMatrixFromList(matrixList, newWorldMatrix)
        mTransformMtx = OpenMaya.MTransformationMatrix(newWorldMatrix)
        trans = mTransformMtx.translation(OpenMaya.MSpace.kWorld)
        return trans

    def prjConnection(self, prjNode, prjNode_offset, animNullNode):
        mmxNode = cmds.createNode('multMatrix')
        dcmNode = cmds.createNode('decomposeMatrix')
        pmaNode = cmds.createNode('plusMinusAverage')
        cmds.setAttr(pmaNode + ".operation", 2)

        cmds.connectAttr(animNullNode + ".worldMatrix", prjNode + ".inputMatrix", f=True)
        cmds.connectAttr(animNullNode + ".worldMatrix", prjNode_offset + ".inputMatrix", f=True)
        cmds.connectAttr(prjNode + ".worldMatrix", mmxNode + ".matrixIn[0]", f=True)
        cmds.connectAttr(animNullNode + ".parentInverseMatrix", mmxNode + ".matrixIn[1]", f=True)

        cmds.connectAttr(mmxNode + ".matrixSum", dcmNode + ".inputMatrix", f=True)
        cmds.connectAttr(dcmNode + ".outputTranslateX", pmaNode + ".input2D[0].input2Dx")
        cmds.connectAttr(dcmNode + ".outputTranslateZ", pmaNode + ".input2D[0].input2Dy")

        cmds.connectAttr(animNullNode + ".translateX", pmaNode + ".input2D[1].input2Dx")
        cmds.connectAttr(animNullNode + ".translateZ", pmaNode + ".input2D[1].input2Dy")

        return {'pma': pmaNode, 'dcm': dcmNode}



    def connectNodes(self, projectMeshNode, projectMeshNode_offset, inputBNodes, controler):
        cmds.connectAttr(self.groundMesh + ".worldMesh",
                         projectMeshNode + ".inputMeshTarget",
                         f=True)
        cmds.connectAttr(self.groundMesh + ".worldMesh",
                         projectMeshNode_offset + ".inputMeshTarget",
                         f=True)
        animBlendNodeX = cmds.listConnections(controler + ".tx", d=False)[0]
        animBlendNodeY = cmds.listConnections(controler + ".ty", d=False)[0]
        animBlendNodeZ = cmds.listConnections(controler + ".tz", d=False)[0]

        cmds.connectAttr(inputBNodes['pma'] + ".output2Dx",
                         animBlendNodeX + ".inputB",
                         f=True)
        cmds.connectAttr(inputBNodes['dcm'] + ".outputTranslateY",
                         animBlendNodeY + ".inputB",
                         f=True)
        cmds.connectAttr(inputBNodes['pma'] + ".output2Dy",
                         animBlendNodeZ + ".inputB",
                         f=True)

    @aniCommon.undo
    def attach(self):
        offsetNulNodeGRP = cmds.createNode('transform', n=MG_NODES[4] + "#")

        animLayerName, kwargs = self.createAnimLayer()
        for con in self.controlersInfo:
            prjNode = self.createPRJMNode(con)
            animNullNode = self.createAnimNull(con)
            offset = self.getWorldTranslate(cmds.getAttr(animNullNode + ".worldMatrix"))
            prjNode_offset = self.createOffsetLocator(con, prjNode, offset)

            inputBNodes = self.prjConnection(prjNode, prjNode_offset, animNullNode)
            kwargs["at"] = (con + '.tx', con + '.ty', con + '.tz')
            self.addToAnimLayer(animLayerName, **kwargs)
            self.connectNodes(prjNode, prjNode_offset, inputBNodes, con)

        if self.offsetNulls:
            cmds.parent(self.offsetNulls, offsetNulNodeGRP)