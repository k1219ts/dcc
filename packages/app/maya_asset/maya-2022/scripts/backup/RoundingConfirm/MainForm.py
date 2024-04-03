from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

import maya.OpenMayaUI as mui
import maya.cmds as cmds
import maya.mel as mel

import shiboken2 as shiboken
import os
import json

from MainFormUI import Ui_Form

currentScriptPath = os.path.abspath(__file__)
srcPath = os.path.dirname(currentScriptPath)

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent=getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent=parent)

        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("what is tool name?")

        if cmds.objExists("RC_man_group"):
            pass
        else:
            if not cmds.pluginInfo('AbcImport', l=True, q=True):
                cmds.loadPlugin('AbcImport')
            mel.eval('AbcImport -mode import "%s/dummy.abc";' % srcPath)

        self.itemFont = QtGui.QFont()
        self.itemFont.setPointSize(13)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.baseModelEdit.setFont(self.itemFont)
        self.ui.execBtn.setFont(self.itemFont)

        self.ui.baseModelEdit.setText("RC_man_group")

        childNodeList = cmds.listRelatives(self.ui.baseModelEdit.text(), fullPath = True)
        for childNode in childNodeList:
            if cmds.getAttr("%s.visibility" % childNode):
                customListWidgetItem(self.ui.childTargetList, childNode)

        self.ui.execBtn.clicked.connect(self.execBtnClicked)

        f = open(os.path.join(os.path.dirname(__file__), "camera.json"), "r")
        camData = json.load(f)
        for attr in camData.keys():
            try:
                cmds.setAttr("persp.%s" % attr, camData[attr])
            except:
                print attr, camData[attr]
        f.close()

        f = open(os.path.join(os.path.dirname(__file__), "renderGlobal.json"), "r")
        renderGlobal = json.load(f)
        for attr in renderGlobal.keys():
            try:
                cmds.setAttr("defaultRenderGlobals.%s" % attr, renderGlobal[attr])
            except:
                print attr, renderGlobal[attr]
        f.close()

        cmds.setAttr('perspShape.renderable', 1)
        cmds.setAttr('defaultRenderGlobals.currentRenderer', 'renderManRIS', type='string')

    def execBtnClicked(self):
        baseName = self.ui.baseModelEdit.text()

        targetNodes = []
        for index in range(self.ui.childTargetList.count()):
            item = self.ui.childTargetList.item(index)

            if item.checkBox.isChecked():
                targetNodes.append(item.nodeName)

        nodeOffset = 0
        angleIndex = 0
        maxFrame = len(targetNodes) * 9
        for frame in range(len(targetNodes) * 9):
            cmds.currentTime(frame + 1)

            targetName = "%s" % targetNodes[nodeOffset]

            cmds.select(targetName)
            cmds.setAttr("persp.rotateX", 0)
            cmds.setAttr("persp.rotateY", 60 * angleIndex)
            if angleIndex >= 6:
                cmds.setAttr("persp.rotateX", -30)
                cmds.setAttr("persp.rotateY", 120 * (angleIndex % 3))
            cmds.viewFit(["persp"], f = 0.5)
            cmds.setKeyframe("persp", at=["tx", "ty", "tz", "rx", "ry", "rz"], time = frame + 1)

            angleIndex += 1
            if angleIndex == 9:
                nodeOffset += 1
                angleIndex = 0

        angleIndex = 0
        targetName = "%s" % baseName
        cmds.select(targetName)
        for frame in range(maxFrame, maxFrame + 3):
            cmds.currentTime(frame + 1)

            cmds.setAttr("persp.rotateX", 0)
            cmds.setAttr("persp.rotateY", 120 * angleIndex)
            cmds.viewFit(["persp"], f = 0.8)
            cmds.setKeyframe("persp", at=["tx", "ty", "tz", "rx", "ry", "rz"], time = frame + 1)

            angleIndex += 1

        maxFrame += 3

        cmds.hide(baseName)

        # cmds.setAttr('defaultRenderGlobals.animation', True)
        # cmds.setAttr('defaultRenderGlobals.outFormatControl', 0)
        cmds.setAttr("defaultRenderGlobals.startFrame", 1)
        cmds.setAttr("defaultRenderGlobals.endFrame", maxFrame)
        cmds.setAttr("renderManRISGlobals.rman__riopt__Hider_maxsamples", 64)
        mel.eval("BatchRender;")



class customListWidgetItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent = None, nodeName = ""):
        QtWidgets.QListWidgetItem.__init__(self, parent)

        self.nodeName = nodeName

        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)

        self.checkBox = QtWidgets.QCheckBox()
        self.checkBox.setText(nodeName.split("|")[-1])
        self.checkBox.setFont(itemFont)
        # self.checkBox.setChecked(True)
        parent.setItemWidget(self, self.checkBox)

        self.setSizeHint(QtCore.QSize(self.sizeHint().width(), 20))