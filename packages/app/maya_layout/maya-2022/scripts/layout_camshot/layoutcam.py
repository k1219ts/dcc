# -*- coding: utf-8 -*-
import os, sys
from maya import cmds, mel
import getpass
from PySide2 import QtWidgets, QtGui, QtCore
import laycamui
reload(laycamui)
import layoutshot
reload(layoutshot)
CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

def main():
    mainVar = Cameraset(getMayaWindow())

if __name__ == "__main__":
    main()

class Cameraset(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window) # maya ui
        self.ui = laycamui.Ui_Form()
        self.ui.setupUi(self)
        self.cameraCheck()

    def mayascene(self):
        mayascene = cmds.file(q=True, sn=True)
        cammb = '/'.join(mayascene.split('/')[0:4])
        self.previzcam = '%s/camera/layoutcam.mb' % cammb
        if not os.path.exists(self.previzcam):
            self.previzcam = os.path.join(CURRENTPATH, 'layoutcam.mb')

    def cameraCheck(self):####camera check
        self.mayascene()
        user = getpass.getuser()
        self.ui.username.setText(user)
        cameralist = layoutshot.cameraList()

        if not cameralist:
            self.show()
            self.ui.createbtn.clicked.connect(self.createCamera)
        else:
            self.shotCreate()

    def createCamera(self):#######camera create
        self.camtotal = 0
        self.camtotal = int(self.ui.cameranum.value())
        cmds.group(em=True, n='cam')

        for i in range(0, self.camtotal):# camera create
            mel.eval('file -import -type "mayaBinary"  -ignoreVersion -rpr "PrevisCam" '
                     '-options "v=0;" "%s";'%self.previzcam)

        cameralist = layoutshot.cameraList()
        for i in range(0, self.camtotal):
            chname = str(10 * (i+1)).zfill(4)
            cam = unicode('C' + chname)
            cmds.rename(cameralist[i], cam)
            cmds.parent(cam, 'cam')
        self.shotCreate()

    def shotCreate(self):
    ## shot create
        cameralist = layoutshot.cameraList()
        if cmds.ls(type = 'shot'):# shot exits check
            self.shotEdit()
        else:
            stime = 1000
            etime = 1000
            for i in cameralist:
                shots = i + '_00'
                stime = etime + 1
                etime = stime + 99
                cmds.shot(shots, cc=i, sn=i, st=stime, et=etime, sst=stime, set=etime, track=1)
            self.shotEdit()

    def shotEdit(self):
    ## camera sequence window
        self.close()
        layoutshot.main()
