# encoding:utf-8
"""
 * DD_timeWarper.mel
 *
 * A script to make and control independent time warps in a 
 * scene that can be applied to select objects
 *
 * Copyright (c) 2011 Robin Scher, Superfad. All Rights Reserved
"""

import os
import maya.cmds as cmds
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/timeWarpPy.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()

    def connectSignal(self):
        self.pushButton.clicked.connect(self.doit)

    def DD_tw_GetConditionNode(self, curveNode):
        if "tw_TimeCurve1" in curveNode:
            result = curveNode.replace("tw_TimeCurve1", "tw_EnableWarp")
        else:
            result = curveNode
        return result

    def DD_tw_ConnectToAnimCurves(self, connect, node):
        result = 0
        type = cmds.nodeType(node)
        if type == "animCurveTU" or type == "animCurveTA" or type == "animCurveTL":
            try:
                cmd = cmds.connectAttr(connect, node + ".input", f=True)
            except:
                pass
            if cmd:
                result = 1
        return result

    def DD_tw_ConnectToImagePlanes(self, connect, node):
        result = 0
        shapeNode = cmds.ls(node, shapes=True, dag=True)
        if len(shapeNode) == 0:
            return result
        shapeNodeFE = shapeNode[0] + ".frameExtension"
        type = cmds.nodeType(shapeNode)
        if type == "imagePlane":
            try:
                cmd = cmds.connectAttr(connect, shapeNodeFE, f=True)
            except:
                pass
            if cmd:
                result = 1
        return result

    def doit(self):
        tw_ActiveNode = "tw_TimeCurve1"
        count = 0
        ipcount = 0
        connect = self.DD_tw_GetConditionNode(tw_ActiveNode) + ".outColorR"
        sel = self.DD_tw_GetSelection()
        all = list()
        for i in sel:
            empty = list()
            all = self.DD_tw_GetInputs(i, empty, all) + all
        print "A"
        tempL = list()
        for j in all:
            if str(j) not in tempL:
                tempL.append(str(j))
        for k in tempL:
            count += self.DD_tw_ConnectToAnimCurves(connect, k)
            ipcount += self.DD_tw_ConnectToImagePlanes(connect, k)
        print "B"

    def DD_tw_GetSelection(self):
        return cmds.ls(sl=True) + cmds.ls(sl=True, dag=True, l=True)

    def DD_tw_GetInputs(self, node, recursion, current):
        result = list()
        if node in current:
            return result
        if node in recursion:
            return result
        cur = [node]
        newRecursion = recursion + cur
        result = [node]
        inputs = list()
        try:
            for i in cmds.listConnections(node, s=True, d=False, scn=True, type="geometryFilter"):
                if str(i) not in inputs:
                    inputs.append(str(i))
            tempL = list()
            for j in cmds.listConnections(node, s=True, d=False, scn=True, type="animCurve"):
                if str(j) not in tempL:
                    tempL.append(str(j))
            inputs = inputs + tempL
            tempL = list()
            for k in cmds.listConnections(node, s=True, d=False, scn=True, type="imagePlane"):
                if str(k) not in tempL:
                    tempL.append(str(k))
            inputs = inputs + tempL
            for s in inputs:
                result = self.DD_tw_GetInputs(s, newRecursion, current) + result
        except:
            pass
        return result

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()