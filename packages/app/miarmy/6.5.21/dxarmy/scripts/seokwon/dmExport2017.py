# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
from Qt import QtCore, QtGui, QtWidgets, load_ui
import dxUI

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/dmExport.ui")


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        self.connectSignal()

    def connectSignal(self):
        self.dmBtn.clicked.connect(self.selD)
        self.expBtn.clicked.connect(self.export)

    def selD(self):
        sel = str(cmds.ls(sl=True)[0])
        self.szX = cmds.getAttr(sel + ".boundingBoxSizeX")
        self.szY = cmds.getAttr(sel + ".boundingBoxSizeY")
        self.szZ = cmds.getAttr(sel + ".boundingBoxSizeZ")
        self.lnEdit.setText(sel)

    def export(self):
        if self.allRdn.isChecked() == True:
            agList = cmds.ls("McdAgent*", type="transform")
        else:
            if len(cmds.ls(sl=True)) != 0:
                for b in cmds.ls(sl=True):
                    if str(b).count("McdAgent") != 1:
                        QtWidgets.QMessageBox.warning(self, "Warning", "Select agent please.")
                    else:
                        pass
                agList = cmds.ls(sl=True)
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", "Select agent please.")
        for i in agList:
            num = (i.index("McdAgent") + 8)
            agNum = i[num:]
            crBox = str(cmds.polyCube(n="agBox" + agNum)[0])
            cmds.setAttr(crBox + ".scaleX", self.szX)
            cmds.setAttr(crBox + ".scaleY", self.szY)
            cmds.setAttr(crBox + ".scaleZ", self.szZ)
            cmds.setAttr(crBox + ".rotatePivotY", -self.szY / 2)
            cmds.setAttr(crBox + ".translateY", self.szY / 2)
            cmds.parentConstraint(i, crBox, mo=False)
        cmds.select("agBox*")
        minT = int(cmds.playbackOptions(q=True, min=True))
        maxT = int(cmds.playbackOptions(q=True, max=True))
        cmds.bakeResults(cmds.ls(sl=True), simulation=True, t=(minT, maxT), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete("agBox*_parentConstraint1")
        cmds.select(cl=True)
        for e in cmds.ls("agBox*", type="transform"):
            cmds.select(str(e), add=True)
        selList = cmds.ls(sl=True)
        ops = str()
        for q in selList:
            ops += " -rt |" + str(q)
        diPath = str(cmds.fileDialog2(startingDirectory="/dexter/Cache_DATA/", fileMode=0, fileFilter="Alembic (*.abc)", caption="Export Alembic")[0])
        cmd = "-frameRange %d %d -uvWrite -worldSpace -dataFormat ogawa%s -file %s" % (minT, maxT, ops, diPath)
        cmds.AbcExport(verbose=True, j=cmd)
        cmds.delete("agBox*")

def main():
    global myWindow
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()