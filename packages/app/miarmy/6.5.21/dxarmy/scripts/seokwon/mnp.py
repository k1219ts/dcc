# encoding:utf-8
# !/usr/bin/env python

import os
import random
import maya.cmds as cmds
from PySide2 import QtCore, QtGui, QtWidgets, QtCompat

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/mnp.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()

    def connectSignal(self):
        self.sacBtn.clicked.connect(self.sac)
        self.secBtn.clicked.connect(self.sec)
        self.plBtn.clicked.connect(self.pl)
        self.alBtn.clicked.connect(self.al)
        self.cpBtn.clicked.connect(self.scp)
        self.getBtn.clicked.connect(self.get)
        self.rstBtn.clicked.connect(self.rst)

    def sac(self):
        try:
            cmds.select("*:*_CON")
        except:
            pass

    def sec(self):
        selName = str(cmds.ls(sl=True)[0]).rsplit(":")[-1]
        eachList = cmds.ls("*:" + selName, type="transform")
        cmds.select(cl=True)
        for i in eachList:
            cmds.select(str(i), add=True)


    # 네임스페이스가 다른 각각의 여러 캐릭터가 있을 때 direction_Con >> place_CON 애니메이션 옮기는 기능
    def pl(self):
        locRange = [13, 16, 17, 18, 20, 22, 25, 26, 28, 29, 31]
        wdCon = {"move_CON": ["move_CON", "direction_CON", "place_CON"],
                 "direction_CON": ["direction_CON", "place_CON"],
                 "place_CON": ["place_CON"]}
        sel = cmds.ls(sl=True)
        self.locList = dict()
        tempList = list()
        for i in sel:
            loc = str(cmds.spaceLocator()[0])
            locSh = str(cmds.ls(loc, ap=True, dag=True, type="shape")[0])
            keyName = str(i).rsplit(":")[0][-3:]
            self.locList[keyName] = loc
            cmds.setAttr(locSh + ".overrideEnabled", 1)
            cmds.setAttr(locSh + ".overrideColor", random.choice(locRange))
            tempCon = cmds.parentConstraint(str(i), loc, w=1, mo=False, st="none", sr="none")[0]
            tempList.append(tempCon)
            cmds.setAttr(loc + ".scaleX", 2)
            cmds.setAttr(loc + ".scaleY", 2)
            cmds.setAttr(loc + ".scaleZ", 2)
        self.minT = cmds.playbackOptions(q=True, min=True)
        self.maxT = cmds.playbackOptions(q=True, max=True)
        cmds.bakeResults(self.locList.values(), simulation=True, t=(int(self.minT), int(self.maxT)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete(tempList)
        cmds.currentTime(self.minT)
        tx = self.chk.checkState()
        if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            pass
        else:
            for k in sel:
                nsChar = str(k).rsplit(":")[0]
                if str(k).rsplit(":")[-1] in wdCon.keys():
                    for q in wdCon[str(k).rsplit(":")[-1]]:
                        if cmds.keyframe(nsChar + ":" + str(q), at=["translate", "rotate"], q=True):
                            cmds.selectKey(nsChar + ":" + str(q), at=["translate", "rotate"])
                            cmds.cutKey(an="keys", cl=True)
                        else:
                            pass
                else:
                    pass

    def al(self):
        sel = cmds.ls(sl=True)
        tempList = list()
        for j in sel:
            keyName = str(j).rsplit(":")[0][-3:]
            tempCon = cmds.parentConstraint(self.locList[keyName], str(j), w=1, mo=False, st="none", sr="none")[0]
            tempList.append(tempCon)
        cmds.bakeResults(sel, simulation=True, t=(int(self.minT), int(self.maxT)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        cmds.delete(tempList)
        cmds.delete(self.locList.values())
        del self.locList

    # gcd2 - ScorpionV03의 머리,목 애니메이션을 ScorpionV04로 옮기는 기능 (머리 위치 맞추고  V03 , V04 순으로 헤드 컨트롤러를 선택하고 실행한다.)
    def scp(self):
        sel = cmds.ls(sl=True)
        pr = str(sel[0].split(":")[0])
        pst = str(sel[-1].split(":")[0])

        # Timeline Set
        cmds.select(pr + ":*_CON")
        minT = int(min(cmds.keyframe(cmds.ls(sl=True), q=True)))
        maxT = int(max(cmds.keyframe(cmds.ls(sl=True), q=True)))
        cmds.playbackOptions(min=minT)
        cmds.playbackOptions(max=maxT)

        prHead = pr + ":C_FK_head_CON"
        prNeck = pr + ":C_IK_neck_CON"
        pstHead = pst + ":C_IK_head_CON"
        pstNeck = pst + ":C_IK_upNeck_CON"

        conL = [prHead, prNeck, pstHead, pstNeck]
        conD = dict()
        for i in conL:
            loc = str(cmds.spaceLocator()[0])
            conD[i] = loc
            temp = str(cmds.pointConstraint(i, loc, w=True, mo=False)[0])
            cmds.delete(temp)
        rCon = list()
        for k in [prHead, prNeck]:
            temps = str(cmds.orientConstraint(k, conD[k], w=True, mo=True)[0])
            rCon.append(temps)
        for j in [pstHead, pstNeck]:
            tempz = str(cmds.orientConstraint(conD[j], j, w=True, mo=True)[0])
            rCon.append(tempz)
        temp = str(cmds.pointConstraint(prHead, conD[prHead], w=True, mo=True)[0])
        rCon.append(temp)
        temp = str(cmds.pointConstraint(conD[pstHead], pstHead, w=True, mo=True)[0])
        rCon.append(temp)
        temp = str(cmds.pointConstraint(conD[prHead], conD[pstHead], w=True, mo=True)[0])
        rCon.append(temp)
        temp = str(cmds.orientConstraint(conD[prNeck], conD[pstNeck], w=True, mo=True)[0])
        rCon.append(temp)
        temp = str(cmds.orientConstraint(conD[prHead], conD[pstHead], w=True, mo=True)[0])
        rCon.append(temp)

        bkSet = [pstHead, pstNeck]
        cmds.bakeResults(bkSet, sm=True, t=(minT, maxT), sb=1.0, dic=True, pok=True, mr=True)
        cmds.delete(rCon)
        for n in conD.keys():
            cmds.delete(conD[n])

    def get(self):
        self.atts = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"]
        sel = str(cmds.ls(sl=True)[0])
        self.trt = list()
        for i in self.atts:
            self.trt.append(cmds.getAttr(sel + "." + i))
            cmds.setAttr(sel + "." + i, 0)

    def rst(self):
        tar = str(cmds.ls(sl=True)[0])
        for i,att in enumerate(self.atts):
            cmds.setAttr(tar + "." + att, self.trt[i])

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()