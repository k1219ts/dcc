__author__ = 'gyeongheon.jeong'

import maya.OpenMayaUI as mui
import os

from Qt import QtCore, QtGui, QtWdigets
import dxUI

import maya.cmds as cmds
import maya.mel as mm

usrProfile = mm.eval('getenv("USERPROFILE")')
mayaVersion = "2014-x64"
UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CF_Instancer_NP.ui")

class CF_Instancer_NP(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(CF_Instancer_NP, self).__init__(parent)
        dxUI.setup_ui(UIFILE, self)
        #self.setupUi( self )

        self.connectSignals()

    def connectSignals(self):
        print "Connect Signals"

def runUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = CF_Instancer_NP()
    _win.show()