#------------------------------------------------------------------------------ 
# Miarmy Support Customized Tool
# creating action, basic setting
# author : Sanghee Chung @ Dexter
#------------------------------------------------------------------------------ 
# -*- coding: utf-8 -*-
import sys
from Qt import QtCore, QtGui
import maya.cmds as cmds
from createActionSetGUI import Ui_Form
from McdActionFunctions import *

#import createActionSetGUI

#class CRW_SET(QtGui.QWidget):
class CRW_SET(QtGui.QDialog):
    def __init__(self, parent=None, Miarmy=None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.btnImport.clicked.connect(self.importFile)
        self.ui.btnTime.clicked.connect(self.timeline)
        self.ui.btnOrigin.clicked.connect(self.originAxis)
        self.ui.btnCreateAction.clicked.connect(self.createAction)

    def importFile(self):
        cmds.select(clear=True)
        cmds.select('Crw_Hips',r=True,hi=True)
        cmds.selectKey(keyframe=True)
        cmds.delete(all=True, c=True)
        cmds.select(clear=True)
        cmds.select('Crw_Hips',r=True)
        filename = cmds.fileDialog2(fileMode=1, caption="Import Anim")
        cmds.file(filename[0], i=True)

    def timeline(self):
        cmds.select('Crw_Hips',r=True)
        keylist=cmds.keyframe(query=True, lastSelected=True, timeChange=True)
        setKey=keylist[0]
        cmds.playbackOptions(minTime=1, maxTime=setKey)
        cmds.currentTime(1)
        
    def originAxis(self):
        cmds.toggleAxis(o=True)

    def createAction(self):
        #cmds.button(command=lambda x: McdCreateActionCmd())
        McdCreateActionCmd()
        

def crd_set_show():
    
    crd_set_window = CRW_SET(QtGui.QApplication.activeWindow())
    crd_set_window.show()


