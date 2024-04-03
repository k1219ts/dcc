# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds

from Qt import QtCore, QtGui, QtWidgets, load_ui
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/spdAgent.ui")

class Window(QtGui.QMainWindow):

    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.connectSignal()
        trAtt = ["translateX", "translateY", "translateZ"]
        roAtt = ["rotateX", "rotateY", "rotateZ"]

    def connectSignal(self):
        self.ui.getBtn.clicked.connect(self.spdAgent)

    def spdAgent(self):
        cpName = cmds.modelPanel(cmds.getPanel(wf=True), q=True, camera=True)
        self.ui.lEdt.setText(cpName)

    def mainControl(self):
        selG = cmds.ls(sl=1)
        smAl = []
        ranGe = []
        for i in selG:
            em = 0
            for j in trAtt:
                ek.append(cmds.getAttr(str(i) + "." + j))
                em += (cmds.getAttr(str(i) + "." + j) - cmds.getAttr(curCam + "." + j)) ** 2
            smAl.append(em ** 0.5)
            ranGe.append(min(ek))
            ranGe.append(max(ek))
        for q in trAtt:
            ek=[]
            for w in selG:
                ek.append(cmds.getAttr(str(w) + "." + q))
        disRange = max(smAl) - min(smAl)
        xyzRange = []
        for e in range(len(ranGe) / 2):
            xyzRange.append(ranGe[2 * e + 1] - ranGe[2 * e])


def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()

if __name__ == '__main__':
    main()
