#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2018.12.06'
__comment__ = 'import asset for usd'
__windowName__ = "OffsetSetupMain"
##########################################

import os

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds

from .toolUI import Ui_Form

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

import dxsUsd

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

class OffsetSetupMain(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):

        # Load dependency plugin
        plugins = ['backstageMenu', 'pxrUsd']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.ui.pushButton.clicked.connect(self.setupOffset)

    def setupOffset(self):
        if not self.ui.lineEdit.text() or not self.ui.lineEdit_2.text():
            return

        minOffset = float(self.ui.lineEdit.text())
        maxOffset = float(self.ui.lineEdit_2.text())
        step = 1.0
        if self.ui.lineEdit_3.text():
            step = float(self.ui.lineEdit_3.text())

        nodes = cmds.ls(sl = True)

        dxsUsd.dxsMayaUtils.RandomizeOffsetByDxTimeOffset(nodes, minOffset, maxOffset, step)

def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = OffsetSetupMain()
    window.setObjectName(__windowName__)
    window.show()