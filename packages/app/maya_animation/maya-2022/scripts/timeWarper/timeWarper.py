
import logging
import maya.cmds as cmds
import aniCommon

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TimeWarp():
    @staticmethod
    def remove(timeCurve, selection):
        conditionNode = TimeWarp().getConditionNode(timeCurve)
        if not conditionNode:
            raise Exception('No Condition Node Exists')
        for sel in selection:
            animCurves = cmds.listConnections(sel, scn=True, d=False, type='animCurve')
            if not animCurves:
                continue
            for ac in animCurves:
                cmds.disconnectAttr(conditionNode[0] + '.outColorR', ac + '.input')

    @staticmethod
    def delete(timeCurve):
        conditionNode = TimeWarp().getConditionNode(timeCurve)
        cmds.delete([timeCurve, conditionNode[0]])

    def __init__(self):
        self._timeCurve = str()
        self._selection = list()

    @property
    def timeCurve(self):
        return self._timeCurve

    @timeCurve.setter
    def timeCurve(self, value):
        self._timeCurve = value

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, value):
        self._selection = value

    @aniCommon.undo
    def createNodes(self, antiName):
        min = cmds.playbackOptions(q=True, min=True)
        max = cmds.playbackOptions(q=True, max=True)
        timeCurveName = antiName or 'DD_TimeCurve#'
        timeCurve = cmds.createNode('animCurveTT', n=timeCurveName)

        if antiName:
            for i in range(int(min), int(max)+1):
                time1Time = cmds.getAttr('time1.outTime')
                cmds.setKeyframe(timeCurve, t=time1Time, v=i)
        else:
            cmds.setAttr(timeCurve + '.preInfinity', 1)
            cmds.setAttr(timeCurve + '.postInfinity', 1)
            cmds.setKeyframe(timeCurve, t=min, v=min, itt='spline', ott='spline')
            cmds.setKeyframe(timeCurve, t=max, v=max, itt='spline', ott='spline')

        conditionNode = cmds.createNode('condition', n='DD_EnableWarp#')
        cmds.setAttr(conditionNode + '.firstTerm', 1)
        cmds.setAttr(conditionNode + '.secondTerm', 1)
        cmds.connectAttr(timeCurve + '.output', conditionNode + '.colorIfTrueR')
        cmds.connectAttr('time1.outTime', conditionNode + '.colorIfFalseR')
        return timeCurve

    def getConditionNode(self, timeCurve):
        conditionNode = cmds.listConnections(timeCurve, scn=True)
        return conditionNode

    def getConnectedCurves(self):
        conditionNode = self.getConditionNode(self.timeCurve)
        animCurves = cmds.listConnections(conditionNode, scn=True, s=False)
        return animCurves

    def getTimeInfo(self):
        warpedTime = cmds.currentTime(q=True)
        conditionNode = self.getConditionNode(self.timeCurve)
        timeInfo = {'warpedTime': warpedTime, 'unWarpedTime': warpedTime}
        if conditionNode:
            if cmds.nodeType(conditionNode[0]) != 'condition':
                return timeInfo
            if cmds.getAttr(conditionNode[0] + '.firstTerm'):
                timeInfo['unWarpedTime'] = cmds.getAttr(self.timeCurve + '.output')

        return timeInfo


    def connectAttrs(self):
        pass

    def enable(self, state):
        logger.debug('{0} Enabled : {1}'.format(self.timeCurve, state))
        conditionNode = self.getConditionNode(self.timeCurve)
        cmds.setAttr(conditionNode[0] + '.firstTerm', state)

    def selectApplied(self):
        pass

    @aniCommon.undo
    def apply(self):
        logger.debug('Apply {0} to Selection'.format(self.timeCurve))
        conditionNode = self.getConditionNode(self.timeCurve)
        print ("conditionNode : " + str(conditionNode))
        print ("timeCurve : " + str(self.timeCurve))
        for sel in self.selection:
            animCurves = cmds.listConnections(sel, scn=True, d=False, type='animCurve')
            if not animCurves:
                continue
            for ac in animCurves:
                print ac
                cmds.connectAttr(conditionNode[0] + '.outColorR', ac + '.input')

