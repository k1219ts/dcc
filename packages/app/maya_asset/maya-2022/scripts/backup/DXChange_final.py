import sys
import os

import shiboken2
import maya.OpenMayaUI as mui
import maya.cmds as cmds
import maya.mel as mel

from PySide2 import QtWidgets
from PySide2 import QtCore

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget) # (1)

#-------------------------------------------------------------------------------------------------------- start for maya

from DxChange import Ui_DxTextureChange_bymoonseok

class Widget(QtWidgets.QWidget ):# (1)
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)
        
        self.setWindowFlags(QtCore.Qt.Window)

#-------------------------------------------------------------------------------------------------------- in maya window

        self.ui = Ui_DxTextureChange_bymoonseok()
        self.ui.setupUi(self)

#-------------------------------------------------------------------------------------------------------- maya window mapping


        # QPushButton clicked signal connect to self.testBtnClicked
        self.ui.Change.clicked.connect(self.changeVersion)
        self.ui.Dev.clicked.connect(self.changeDev)
        self.ui.Pub.clicked.connect(self.changePub)

#-------------------------------------------------------------------------------------------------------- cord click

    # it is slot
    def changeVersion(self):
        version = self.ui.Version.text()
        for cms in cmds.ls(sl = True, type = 'DxTexture'):
            cmds.setAttr( '%s.txversion' % cms, version, type = "string")
            getTXversion = cmds.getAttr( '%s.txversion' % cms)

            print getTXversion

    def changeDev(self):
        for cms in cmds.ls(sl = True, type = 'DxTexture'):
            cmds.setAttr( '%s.txpath' % cms, 'dev', type = "string")
            getTXversion = cmds.getAttr( '%s.txpath' % cms)
            print getTXversion
    
    def changePub(self):
        for cms in cmds.ls(sl = True, type = 'DxTexture'):
            cmds.setAttr( '%s.txpath' % cms, 'pub', type = "string")
            getTXversion = cmds.getAttr( '%s.txpath' % cms)
            print getTXversion


#-------------------------------------------------------------------------------------------------------- cord



def main():
    mainVar = Widget(getMayaWindow())
    mainVar.show()

if __name__ == "__main__":
    main()

'''
import Widget
Widget.main()
'''



#-------------------------------------------------------------------------------------------------------- end
