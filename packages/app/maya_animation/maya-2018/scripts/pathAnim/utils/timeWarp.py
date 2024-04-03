
import logging
import maya.cmds as cmds
import aniCommon
from timeWarper import timeWarper
reload(timeWarper)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PATimeWarp(timeWarper.TimeWarp):
    @staticmethod
    def getPrefix(node):
        prefix = node.split(":")[-1].split("_")[0]
        return prefix

    @ staticmethod
    def getAnimLocators(prefix):
        nameSpacer = str()
        locs = list()

        for i in range(5):
            locs = cmds.ls(prefix + nameSpacer + "*_AnimLoc", r=True)
            if locs:
                break
            nameSpacer += "*:"
        return locs

    def getPathAnimNodes(self):
        nameSpacer = str()
        nodes = list()

        for i in range(5):
            nodes = cmds.ls(nameSpacer + "*PathAnimCurves", r=True)
            if nodes:
                break
            nameSpacer += "*:"
        return nodes

    def getConnectedTimewarp(self, paNode):
        prefix = paNode.split("_")[0]
        pathCtrl = prefix + "_PathAnim_Ctrl"
        animCurves = cmds.listConnections(pathCtrl, scn=True, d=False, type='animCurve')
        timeCurve = None

        if animCurves:
            conditionNode = cmds.listConnections(animCurves[0], scn=True, d=False, type='condition')
            if conditionNode:
                timeCurve = cmds.listConnections(conditionNode[0], scn=True, d=False, type='animCurveTT')
        return timeCurve

    def selectTimeCurve(self, paNode):
        timeCurve = self.getConnectedTimewarp(paNode)
        if not timeCurve:
            logger.error('Create Timewarp first')
        else:
            cmds.select(timeCurve)


    def selectAppliedObject(self, paNode):
        timeCurve = self.getConnectedTimewarp(paNode)
        objects = list()

        if not timeCurve:
            return
        conditionNode = self.getConditionNode(timeCurve)
        animCurves = cmds.listConnections(conditionNode, scn=True, s=False)
        for ac in animCurves:
            node = cmds.listConnections(ac, scn=True, s=False)
            if node and node not in objects:
                objects.append(node[0])

        cmds.select(objects, r=True)
        logger.debug('Select {0}'.format(objects))


    @aniCommon.undo
    def PA_createNodes(self):
        min = cmds.playbackOptions(q=True, min=True)
        max = cmds.playbackOptions(q=True, max=True)
        timeCurveName = 'PA_TimeCurve#'
        timeCurve = cmds.createNode('animCurveTT', n=timeCurveName)

        cmds.setAttr(timeCurve + '.preInfinity', 1)
        cmds.setAttr(timeCurve + '.postInfinity', 1)
        cmds.setKeyframe(timeCurve, t=min, v=min, itt='spline', ott='spline')
        cmds.setKeyframe(timeCurve, t=max, v=max, itt='spline', ott='spline')

        conditionNode = cmds.createNode('condition', n='PA_EnableWarp#')
        cmds.setAttr(conditionNode + '.firstTerm', 1)
        cmds.setAttr(conditionNode + '.secondTerm', 1)
        cmds.connectAttr(timeCurve + '.output', conditionNode + '.colorIfTrueR')
        cmds.connectAttr('time1.outTime', conditionNode + '.colorIfFalseR')
        return timeCurve


