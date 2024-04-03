#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2019.01.03'
__comment__ = 'ZENN Node Controller'
__windowName__ = "ZENN Controller"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds

from .ZENNControlerUI import Ui_Form

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

def getMayaWindow():
    '''
    get Maya Window Process
    :return: Maya window Process
    '''
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QMainWindow)
    except:
        return None

class ZENNController(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):

        # Load dependency plugin
        plugins = ['ZENNForMaya']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.ui.znImportUpdateMeshBtn.clicked.connect(self.znImportUpdateMeshBtnClicked)
        self.ui.znImportUpdateCurveBtn.clicked.connect(self.znImportUpdateCurveBtnClicked)
        self.ui.znGenerateUpdateBtn.clicked.connect(self.znGenerateUpdateBtnClicked)
        self.ui.ratioSlider.sliderMoved.connect(self.ratioSliderMoved)
        self.ui.ratioEdit.returnPressed.connect(self.ratioEditReturnPressed)

    def znImportUpdateMeshBtnClicked(self):
        print "znImportUpdateMeshBtnClicked"
        nodes = cmds.ls(type = "ZN_Import")
        for node in nodes:
            cmds.setAttr("%s.updateMesh" % node, 1)

    def znImportUpdateCurveBtnClicked(self):
        print "znImportUpdateCurveBtnClicked"
        nodes = cmds.ls(type="ZN_Import")
        for node in nodes:
            cmds.setAttr("%s.updateCurves" % node, 1)

    def znGenerateUpdateBtnClicked(self):
        print "znGenerateUpdateBtnClicked"
        nodes = cmds.ls(type="ZN_Generate")
        for node in nodes:
            cmds.setAttr("%s.update" % node, 1)

    def ratioSliderMoved(self, value):
        print "ratioSliderMoved"
        nodes = cmds.ls(type="ZN_StrandsViewer")
        self.ui.ratioEdit.setText(str(value / 1000.0))
        for node in nodes:
            cmds.setAttr("%s.displayRatio" % node, float(self.ui.ratioEdit.text()))

    def ratioEditReturnPressed(self):
        print "ratioEditReturnPressed"
        nodes = cmds.ls(type="ZN_StrandsViewer")
        value = float(self.ui.ratioEdit.text()) * 1000

        if value >= 1000:
            value = 1000
            self.ui.ratioEdit.setText("1.000")
        elif value <= 0:
            value = 0
            self.ui.ratioEdit.setText("0.000")
        self.ui.ratioSlider.setValue(value)

        for node in nodes:
            cmds.setAttr("%s.displayRatio" % node, float(self.ui.ratioEdit.text()))

def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = ZENNController()
    # app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    window.setObjectName(__windowName__)
    window.show()
