# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import McdPlacementFunctions as mpf
import math
from Qt import QtCore, QtGui, QtWidgets, load_ui

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/McdPlacement.ui")

def hconv(text):
    return unicode(text, 'utf-8')

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.connectSignal()

    def connectSignal(self):

        self.ui.pidBtn.clicked.connect(self.scanId)  # 버튼 등록 및 실행
        self.ui.allBtn.clicked.connect(self.allInv)
        self.ui.selBtn.clicked.connect(self.selInv)
        self.ui.aimBtn.clicked.connect(self.aimCircle)
        self.ui.mkdmBtn.clicked.connect(self.makeDummy)
        self.ui.targetBtn.clicked.connect(self.targetAim)
        self.ui.selPlaceBtn.clicked.connect(self.selPlaceEdit)

    def scanId(self):
        if cmds.ls(sl=1):
            outpName = str(cmds.ls(sl=True, dag=True, type="shape")[0])
            idName = str(cmds.getAttr(outpName + ".placeId"))
            self.ui.pidLabel.setText(outpName + " ------> ID : " + idName)
        else:
            cmds.warning("Select a placement")

    def makeDummy(self):
        sel = cmds.ls(sl=True)
        boundBoxes = list()
        for i in sel:
            alm = str(cmds.ls(i, dag=True, type="dxComponent")[0])
            x = cmds.getAttr(alm + "." + "boundingBoxSize")[0][0]
            y = cmds.getAttr(alm + "." + "boundingBoxSize")[0][1]
            z = cmds.getAttr(alm + "." + "boundingBoxSize")[0][2]
            em = str(cmds.polyCube()[0])
            boundBoxes.append(em)
            cmds.setAttr(em + ".scaleX", x)
            cmds.setAttr(em + ".scaleY", y)
            cmds.setAttr(em + ".scaleZ", z)
            cmds.xform(str(i), cpc=True)
            temCon = cmds.parentConstraint(str(i), em, mo=0, w=1)
            cmds.delete(temCon)
        cmds.createNode('transform', n="Dummies")
        for j in boundBoxes:
            cmds.parent(j, "Dummies")
        for k in sel:
            cmds.setAttr(str(k) + ".visibility", 0)

    def allInv(self):

        bkr = []
        for k in cmds.ls("McdPlace" + "*", type="McdPlace"):
            bkr.append(cmds.getAttr(k + ".numOfAgent"))

        pkr = []
        for l in cmds.ls("McdPlace" + "*", type="transform"):
            pkr.append(l)
        cmds.select(clear=True)
        num = 0
        plr = []
        for i in bkr:
            for j in range(i):
                num = num + 1
                cmds.select("McdAgent" + str(num), add=True)
            mpf.inversePlacementAgent()
            plr.append(str(cmds.ls("McdPlace" + "*", type='transform')[-1]))
            cmds.select(clear=True)

        mpf.dePlacementAgent()

        for w in range(len(pkr)):
            cmds.delete(pkr[w])
            cmds.rename(plr[w], pkr[w])

        mpf.placementAgent()

    def selInv(self):
        pcList = list()
        for i in cmds.ls("McdPlace*", type="transform"):
            if not str(i) in pcList:
                pcList.append(str(i))
            else:
                cmds.error("There is a duplicated Placement ID.")
        selPcList = list()
        for j in cmds.ls(sl=True, type="transform"):
            num = cmds.getAttr(str(j) + ".placeId")
            for k in pcList:
                numId = cmds.getAttr(str(k) + ".placeId")
                if num == numId and not str(k) in selPcList:
                    selPcList.append(str(k))
        placeDict = dict()
        for n in selPcList:
            placeDict[n] = list()
            plid = cmds.getAttr(n + ".placeId")
            for e in cmds.ls("McdAgent*", type="transform"):
                agid = cmds.getAttr(str(e) + ".placeId")
                if plid == agid:
                    placeDict[n].append(str(e))
                else:
                    pass
        placeNameList = dict()
        for w in placeDict.keys():
            cmds.select(cl=True)
            for q in placeDict[w]:
                cmds.select(str(q), add=True)
            mpf.inversePlacementAgent()
            for c in cmds.ls("McdPlace*", type="transform"):
                if not str(c) in placeDict.keys():
                    placeNameList[w] = str(c)
                else:
                    pass
        mpf.dePlacementAgent()
        for h in range(len(placeDict)):
            cmds.delete(placeNameList.keys()[h])
            cmds.rename(placeNameList.values()[h], placeNameList.keys()[h])
        mpf.placementAgent()

    def checkPlace(self):
        checkList = list()
        for i in cmds.ls("McdPlace*", type="transform"):
            idz = cmds.getAttr(str(i) + ".placeId")
            if idz in checkList:
                cmds.error("There is a overlapped placement ID.")
            else:
                checkList.append(idz)

    def selPlaceEdit(self):
        agL = cmds.ls(sl=True, ap=True, dag=True, type="shape")
        self.checkPlace()
        placeDict = dict()
        for place in cmds.ls("McdPlace*", type="transform"):
            ids = cmds.getAttr(str(place) + ".placeId")
            placeDict[str(ids)] = str(place)
        plList = list()
        for w in agL:
            id = cmds.getAttr(str(w) + ".placeId")
            shp = placeDict[str(id)] + "Shape"
            if shp in plList:
                pass
            else:
                plList.append(shp)
        agList = dict()
        for i in agL:
            agplId = cmds.getAttr(str(i) + ".placeAgentId")
            tpl = cmds.getAttr(str(i) + ".placeId")
            if (placeDict[str(tpl)] + "Shape") in agList.keys():
                pass
            else:
                agList[placeDict[str(tpl)] + "Shape"] = dict()
            plx = cmds.xform(placeDict[str(tpl)], q=True, t=True)[0]
            ply = cmds.xform(placeDict[str(tpl)], q=True, t=True)[1]
            plz = cmds.xform(placeDict[str(tpl)], q=True, t=True)[2]
            orx = cmds.getAttr(placeDict[str(tpl)] + "Shape.placement[" + str(agplId) + "].agentPlace[1]")
            ory = cmds.getAttr(placeDict[str(tpl)] + "Shape.placement[" + str(agplId) + "].agentPlace[2]")
            orz = cmds.getAttr(placeDict[str(tpl)] + "Shape.placement[" + str(agplId) + "].agentPlace[3]")
            tLoc = str(cmds.spaceLocator()[0])
            trAg = str(cmds.listRelatives(str(i), p=True)[0])
            tCon = str(cmds.parentConstraint(trAg, tLoc, mo=False, w=1)[0])
            cmds.delete(tCon)
            lox = cmds.xform(tLoc, q=True, t=True)[0]
            loy = cmds.xform(tLoc, q=True, t=True)[1]
            loz = cmds.xform(tLoc, q=True, t=True)[2]
            lortx = cmds.xform(tLoc, q=True, ro=True)[0]
            lorty = cmds.xform(tLoc, q=True, ro=True)[1]
            lortz = cmds.xform(tLoc, q=True, ro=True)[2]
            dx = lox - (orx + plx)
            dy = loy - (ory + ply)
            dz = loz - (orz + plz)
            if dx != 0 or dy != 0 or dz != 0:
                agList[placeDict[str(tpl)] + "Shape"][str(agplId)] = [dx, dy, dz, lortx, lorty, lortz]
            else:
                pass
            cmds.delete(tLoc)

        for m in plList:
            for c in range(cmds.getAttr(m + ".numOfAgent")):
                if str(c) in agList[m].keys():
                    cmds.setAttr(m + ".userOffset[" + str(c * 8) + "]", c)
                    ttx = cmds.getAttr(m + ".userOffset[" + str(c * 8 + 1) + "]")
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 1) + "]", ttx + agList[m][str(c)][0])
                    tty = cmds.getAttr(m + ".userOffset[" + str(c * 8 + 2) + "]")
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 2) + "]", tty + agList[m][str(c)][1])
                    ttz = cmds.getAttr(m + ".userOffset[" + str(c * 8 + 3) + "]")
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 3) + "]", ttz + agList[m][str(c)][2])
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 4) + "]", agList[m][str(c)][3])
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 5) + "]", agList[m][str(c)][4])
                    cmds.setAttr(m + ".userOffset[" + str(c * 8 + 6) + "]", agList[m][str(c)][5])
                else:
                    cmds.setAttr(m + ".userOffset[" + str(c * 8) + "]", c)

    def euler(self, deg):
        if abs(deg) > 180:
            return -(180 - deg % 180)
        else:
            return deg

    def aimCircle(self):
        selp = str(cmds.ls(sl=True, ap=True, dag=True, type="shape")[0])
        if cmds.getAttr(selp + ".placeType") == 0:
            pass
        else:
            cmds.error("PlaceType must be Vortex")
        nAgent = cmds.getAttr(selp + ".numOfAgent")
        i = 1
        if nAgent <= 1:
            pass
        else:
            while (True):
                if (3 * (i ** 2) - 3 * i) < nAgent - 1 < (3 * (i ** 2) + 3 * i + 1):
                    break
                else:
                    pass
                i += 1

        for k in range(nAgent):
            if k == 0:
                cmds.setAttr(selp + ".userOffset[" + str(k * 8) + "]", k)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 1) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 2) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 3) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 4) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 5) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 6) + "]", 0)
            else:
                for j in range(i):
                    num = j + 1
                    if ((3 * (num ** 2) - 3 * num) < k < (3 * (num ** 2) + 3 * num + 1)):
                        deg = float(360.00 / (6 * num))  # deg = 해당 num의 단위각.
                        cNum = k - (3 * (num ** 2) - 3 * num)  # cNum = 각 서클에서 몇 번째 에이전트인지
                        angle = self.euler((-2 * num + (cNum - 1)) * deg)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8) + "]", k)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 1) + "]", 0)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 2) + "]", 0)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 3) + "]", 0)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 4) + "]", 0)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 5) + "]", angle)
                        cmds.setAttr(selp + ".userOffset[" + str(k * 8 + 6) + "]", 0)
                    else:
                        pass

    def targetAim(self):
        placeList = cmds.ls(sl=True)[:-1]
        target = cmds.ls(sl=True)[-1]
        tartx = cmds.getAttr(target + ".translate")[0][0]
        tartz = cmds.getAttr(target + ".translate")[0][2]
        for e in placeList:
            pltx = cmds.getAttr(e + ".translate")[0][0]
            pltz = cmds.getAttr(e + ".translate")[0][2]
            selp = str(cmds.ls(e, ap=True, dag=True, type="shape")[0])
            for i in range(cmds.getAttr(selp + ".numOfAgent")):
                agtx = cmds.getAttr(selp + ".placement[" + str(i) + "].agentPlace[1]")
                agtz = cmds.getAttr(selp + ".placement[" + str(i) + "].agentPlace[3]")
                cmds.setAttr(selp + ".userOffset[" + str(i * 8) + "]", i)
                cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 1) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 2) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 3) + "]", 0)
                cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 4) + "]", 0)
                distance = abs(((pltx + agtx - tartx) ** 2 + (pltz + agtz - tartz) ** 2) ** 0.5)
                if (pltx + agtx) < tartx:
                    if (pltz + agtz) < tartz:
                        ang = math.acos(abs(pltz + agtz - tartz) / distance)
                        cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 5) + "]", (ang * (180 / math.pi)))
                    else:
                        ang = math.acos(abs(pltx + agtx - tartx) / distance)
                        cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 5) + "]", (ang * (180 / math.pi)) + 90)
                else:
                    if (pltz + agtz) < tartz:
                        ang = math.acos(abs(pltz + agtz - tartz) / distance)
                        cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 5) + "]", -(ang * (180 / math.pi)))
                    else:
                        ang = math.acos(abs(pltx + agtx - tartx) / distance)
                        cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 5) + "]", -(ang * (180 / math.pi)) - 90)
                cmds.setAttr(selp + ".userOffset[" + str(i * 8 + 6) + "]", 0)

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.ui.show()

if __name__ == '__main__':
    main()





