# encoding:utf-8
# !/usr/bin/env python

import os
import subprocess
import maya.cmds as cmds
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "simpleConst.ui")

class undoCheck(object):
    def __enter__(self):
        cmds.undoInfo(openChunk=True)
    def __exit__(self, *exc):
        cmds.undoInfo(closeChunk=True)

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
        self.atBtn.clicked.connect(self.connectParent)
        self.detBtn.clicked.connect(self.disconnectParent)
        self.moveBtn.clicked.connect(self.moveSel)
        self.allBtn.clicked.connect(self.allSel)
        self.eachBtn.clicked.connect(self.eachSel)
        self.fngBtn.clicked.connect(self.fngSel)
        self.fromBtn.clicked.connect(self.pickSel)
        self.toBtn.clicked.connect(self.assignSel)
        self.helpBtn.clicked.connect(self.openPDF)
        self.helpBtn.setIcon(QtGui.QIcon(QtGui.QPixmap("/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/icons/system-help.png")))
        self.helpBtn.setIconSize(QtCore.QSize(24, 24))

    def openPDF(self):
        exp = "/usr/bin/evince"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/simpleConst/selectionSet.pdf"
        subprocess.Popen([exp, fileName])

    def moveSel(self):
        try:
            cmds.manipMoveContext('Move', edit=1, mode=2)
            nsList = list()
            sel = cmds.ls(sl=True)
            for e in sel:
                ns = str(e).split(":")[0]
                if not ns in nsList:
                    nsList.append(ns)
            for k in nsList:
                if len(cmds.ls(k + ":root_CON")) != 0:
                    movecon = ["root_CON", "L_IK_hand_CON", "R_IK_hand_CON", "L_IK_foot_CON", "R_IK_foot_CON",
                               "L_IK_footVec_CON", "R_IK_footVec_CON", "L_IK_handVec_CON", "R_IK_handVec_CON"]
                elif len(cmds.ls(k + ":C_IK_root_CON")) != 0:
                    movecon = ["C_IK_root_CON", "L_IK_foreLeg_CON", "R_IK_foreLeg_CON", "L_IK_knee_CON",
                               "R_IK_knee_CON",
                               "L_IK_hindLeg_CON", "R_IK_hindLeg_CON", "L_IK_edbow_CON", "R_IK_edbow_CON"]
                else:
                    pass
            cmds.select(cl=True)
            for q in nsList:
                for i in movecon:
                    cmds.select(q + ":" + i, add=True)
        except:
            pass

    def allSel(self):
        try:
            sel = str(cmds.ls(sl=True)[0])
            nsChar = sel.split(":")[0]
            cmds.select(nsChar + ":*_CON")
        except:
            pass

    def eachSel(self):
        try:
            selName = str(cmds.ls(sl=True)[0]).rsplit(":")[-1]
            eachList = cmds.ls("*:" + selName, type="transform")
            cmds.select(cl=True)
            for i in eachList:
                cmds.select(str(i), add=True)
        except:
            pass

    def fngSel(self):
        try:
            sel = cmds.ls(sl=True)
            if len(sel) == 0:
                cmds.error("Select Con")
            elif len(sel) == 1:
                ob = str(sel[0]).split("_")
                ns = "_".join(ob[:-2]) + "_" + ob[-2][:-1] + "*_" + ob[-1]
                cmds.select(cl=True)
                for i in cmds.ls(ns):
                    if str(i).split("_")[-2][-1].isdigit():
                        cmds.select(str(i), add=True)
            else:
                cmds.select(cl=True)
                for k in sel:
                    ob = str(k).split("_")
                    ns = "_".join(ob[:-2]) + "_" + ob[-2][:-1] + "*_" + ob[-1]
                    for i in cmds.ls(ns):
                        if str(i).split("_")[-2][-1].isdigit():
                            cmds.select(str(i), add=True)
        except:
            pass

    def pickSel(self):
        try:
            with undoCheck():
                attr = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"]
                self.sel = str(cmds.ls(sl=True)[0]).split(":")
                fstFrm = int(cmds.currentTime(q=True))
                selAtt = list()
                self.selLoc = cmds.spaceLocator()[0]
                selStraint = cmds.parentConstraint(self.sel[0] + ":" + self.sel[1], self.selLoc, w=1, mo=False, st="none", sr="none")[0]
                cmds.delete(selStraint)
                cmds.select(self.sel[0] + ":" + self.sel[1])
                for i in attr:
                    selAtt.append(cmds.getAttr(self.sel[0] + ":" + self.sel[1] + "." + i, t=fstFrm))
                if len(cmds.ls(self.sel[0] + ":root_CON")) != 0:
                    self.movecon = ["root_CON", "L_IK_hand_CON", "R_IK_hand_CON", "L_IK_foot_CON", "R_IK_foot_CON",
                                    "L_IK_footVec_CON", "R_IK_footVec_CON", "L_IK_handVec_CON", "R_IK_handVec_CON"]
                    self.movecon.remove(self.sel[1])
                    self.locL = dict()
                    constL = list()
                    for j in self.movecon:
                        self.locL[j] = cmds.spaceLocator()[0]
                        constL.append(
                            cmds.parentConstraint(self.sel[0] + ":" + j, self.locL[j], w=1, mo=False, st="none", sr="none")[0])
                    cmds.delete(constL)
                    cmds.select(cl=True)
                    for k in self.locL.values():
                        cmds.select(k, add=True)
                    cmds.select(self.selLoc, add=True)
                    cmds.parent()
                    cmds.select(cl=True)
                elif len(cmds.ls(self.sel[0] + ":C_IK_root_CON")) != 0:
                    self.movecon = ["C_IK_root_CON", "L_IK_foreLeg_CON", "R_IK_foreLeg_CON", "L_IK_knee_CON", "R_IK_knee_CON",
                                    "L_IK_hindLeg_CON", "R_IK_hindLeg_CON", "L_IK_edbow_CON", "R_IK_edbow_CON"]
                    self.movecon.remove(self.sel[1])
                    self.locL = dict()
                    constL = list()
                    for j in self.movecon:
                        self.locL[j] = cmds.spaceLocator()[0]
                        constL.append(
                            cmds.parentConstraint(self.sel[0] + ":" + j, self.locL[j], w=1, mo=False, st="none", sr="none")[0])
                    cmds.delete(constL)
                    cmds.select(cl=True)
                    for k in self.locL.values():
                        cmds.select(k, add=True)
                    cmds.select(self.selLoc, add=True)
                    cmds.parent()
                    cmds.select(self.sel[0] + ":" + self.sel[1])
        except:
            pass

    def assignSel(self):
        try:
            with undoCheck():
                attr = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"]
                scFrm = int(cmds.currentTime(q=True))
                cmds.selectKey(self.sel[0] + ":" + self.sel[1], k=True)
                cmds.filterCurve(f="euler")
                tm = cmds.parentConstraint(self.sel[0] + ":" + self.sel[1], self.selLoc, w=1, mo=False, st="none", sr="none")[0]
                cmds.delete(tm)
                for z in self.movecon:
                    if cmds.listAttr(self.sel[0] + ":" + z, k=True).count("rotateX") == 1:
                        tms = cmds.parentConstraint(self.locL[z], self.sel[0] + ":" + z, w=1, mo=False, st="none", sr="none")[0]
                        for b in attr:
                            cmds.setKeyframe(self.sel[0] + ":" + z, at=b, t=(scFrm, scFrm))
                    else:
                        tms = cmds.pointConstraint(self.locL[z], self.sel[0] + ":" + z, w=1, mo=False, sk="none")[0]
                        cmds.setKeyframe(self.sel[0] + ":" + z, at="translateX", t=(scFrm, scFrm))
                        cmds.setKeyframe(self.sel[0] + ":" + z, at="translateY", t=(scFrm, scFrm))
                        cmds.setKeyframe(self.sel[0] + ":" + z, at="translateZ", t=(scFrm, scFrm))
                    cmds.delete(tms)
                cmds.select(self.sel[0] + ":*_CON")
                cmds.selectKey(self.sel[0] + ":*_CON", k=True)
                cmds.filterCurve(f="euler")
                cmds.delete(self.selLoc)
                cmds.select(self.sel[0] + ":" + self.sel[1])
        except:
            pass

    def selectFingers(self):
        sel = cmds.ls(sl=True)
        if len(sel) == 0:
            cmds.error("Select Con")
        elif len(sel) == 1:
            ob = str(sel[0]).split("_")
            ns = "_".join(ob[:-2]) + "_" + ob[-2][:-1] + "*_" + ob[-1]
            cmds.select(cl=True)
            for i in cmds.ls(ns):
                if str(i).split("_")[-2][-1].isdigit():
                    cmds.select(str(i), add=True)
        else:
            cmds.select(cl=True)
            for k in sel:
                ob = str(k).split("_")
                ns = "_".join(ob[:-2]) + "_" + ob[-2][:-1] + "*_" + ob[-1]
                for i in cmds.ls(ns):
                    if str(i).split("_")[-2][-1].isdigit():
                        cmds.select(str(i), add=True)

    def connectParent(self):
        with undoCheck():
            self.selc = cmds.ls(sl=True)
            if str(self.selc[0]).count(".vtx["):
                self.en = str(cmds.emitter(self.selc[0], dx=1, nuv=0, spd=1, sp=0, srn=0, sro=0, mnd=0, r=0, dz=0, cye='none', dy=0, nsp=1, cyi=1, type='omni', mxd=0, tsp=0)[-1])
                cmds.setAttr(self.en + ".visibility", 0)
                self.tempCon = str(cmds.parentConstraint(self.en, self.selc[1], mo=True, w=True, n="linked_const_#")[0])
            else:
                self.tempCon = str(cmds.parentConstraint(self.selc[0], self.selc[1], mo=True, w=True, n="linked_const_#")[0])
            self.curT = cmds.currentTime(q=True)
            try:
                cmds.setKeyframe(self.selc[1] + ".blendParent1", t=self.curT, v=1)
                cmds.setKeyframe(self.selc[1] + ".blendParent1", t=self.curT - 1, v=0)
            except:
                pass

    def disconnectParent(self):
        with undoCheck():
            curTs = cmds.currentTime(q=True)
            if self.curT == curTs:
                try:
                    cmds.delete(self.tempCon)
                    cmds.delete(self.en)
                except:
                    pass
                return
            cmds.bakeResults(self.selc[1], sm=False, t=(self.curT, curTs), mr=True, pok=True)
            try:
                cmds.delete(self.tempCon)
                cmds.delete(self.en)
            except:
                pass
            cmds.selectKey(self.selc[1], k=True)
            cmds.filterCurve(f="euler")

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()
    return myWindow

if __name__ == '__main__':
    main()