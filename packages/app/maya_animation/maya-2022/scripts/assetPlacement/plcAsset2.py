# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import maya.mel as mel
import random
import sgUI
from PySide2 import QtCore, QtGui, QtWidgets
from plcAssetUI import Ui_plcAsset

chList = list()
chName = list()
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/plcAsset.ui")

class DragDropTest(QtWidgets.QListWidget):
    def __init__(self, parent):
        QtWidgets.QListWidget.__init__(self, parent)

        self.setAcceptDrops(True)
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
            lst = list()
            for i in event.mimeData().urls():
                lst.append(i.toLocalFile().split(os.sep)[-1])
                chList.append(str(i.toLocalFile()))
                chName.append(str(i.toLocalFile().split(os.sep)[-1]))
            self.addItems(lst)
        else:
            event.ignore()

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.ui = Ui_plcAsset()
        self.ui.setupUi(self)
        self.connectSignal()

    def connectSignal(self):
        self.ui.listWidget.setAcceptDrops(True)
        self.ui.listWidget.setDragEnabled(True)
        self.ui.numOri.setValue(0)
        self.ui.numOri.setMaximum(100)
        self.ui.slOri.setMaximum(100)
        self.ui.slOri.valueChanged.connect(self.randOr)
        self.ui.selGeoBtn.clicked.connect(self.selGeo)
        self.ui.scnum.setValue(1)
        self.ui.scsl.valueChanged.connect(self.locScale)
        self.ui.scnum.setMaximum(50)
        self.ui.scsl.setMaximum(50)
        self.ui.crBtn.clicked.connect(self.mkLoc)
        self.ui.getrBtn.clicked.connect(self.getRange)
        self.ui.geoConBtn.clicked.connect(self.geoConst)
        self.ui.scattBtn.clicked.connect(self.scatter)
        self.ui.aimBtn.clicked.connect(self.aimSet)
        self.ui.impBtn.clicked.connect(self.impCh)
        self.ui.delBtn.clicked.connect(self.delList)
        self.ui.loadBtn.clicked.connect(self.loadCache)
        self.ui.dmBtn.clicked.connect(self.mkDummy)
        self.ui.repBtn.clicked.connect(self.reImpRepl)
        self.ui.scaleMin.setText("1.00")
        self.ui.scaleMax.setText("1.00")
        self.ui.offMin.setText("0")
        self.ui.offMax.setText("0")
        self.ui.sclMin.setText("1.00")
        self.ui.sclMax.setText("1.00")
        self.rtLst = dict()

    def getLocNum(self):
        LocNum = len(cmds.ls("Placements", type="transform", dag=True, ap=True)) - 1
        return LocNum

    def mkLoc(self):
        self.locList = list()
        agNum = int(self.ui.numCh.text())
        for i in range(agNum):
            cvName = cmds.curve(d=1, p=[(-0.3, 0, -0.5), (0, 0, 1.5), (0.3, 0, -0.5), (-0.3, 0, -0.5)], k=[0, 1, 2, 3])
            if len(cmds.ls("Placements")) == 0:
                cmds.createNode("transform", n="Placements")
                cmds.parent(cvName, "Placements")
            else:
                cmds.parent(cvName, "Placements")
            cmds.rename(cvName, "Loc" + str(self.getLocNum()))
        for ia in cmds.ls("Loc*", type="transform"):
            if str(ia).count("Constraint") == 1:
                pass
            else:
                self.locList.append(str(ia))
        for j in self.locList:
            self.rtLst[str(j)] = [float(random.randrange(-300, 300, step=1) / 100.00), cmds.getAttr(str(j) + ".ry")]
        cmds.select(cl=True)
        sel = self.ui.targetGeo.text()
        if len(cmds.ls("Ground_Terrain_LAY")) == 0:
            cmds.select(sel)
            cmds.createDisplayLayer(name="Ground_Terrain_LAY", number=1, nr=True)
            mel.eval("layerEditorLayerButtonTypeChange Ground_Terrain_LAY;")
            mel.eval("layerEditorLayerButtonTypeChange Ground_Terrain_LAY;")
        else:
            pass
        cmds.select(cl=True)
        for nm in self.locList:
            cmds.select(str(nm), add=True)

    def mkDummy(self):
        if self.ui.chkfc.isChecked() == True:
            if len(str(self.ui.listWidget.selectedItems()[0].text())) != 0:
                selCh = chList[chName.index(str(self.ui.listWidget.selectedItems()[0].text()))]
                if self.ui.rdDir.isChecked() == True:
                    for con in os.listdir(selCh):
                        if str(con).count(":") == 1 and str(con).count(".abc") == 1:
                            selName = str(con).rsplit(":")[0]
                else:
                    selName = str(self.ui.listWidget.selectedItems()[0].text())
                self.ui.dummy.setText(selName)
                if self.ui.rdDir.isChecked() == True:
                    ciClass = sgUI.ComponentImport(Files=selCh, World=1)
                    ciClass.m_mode = 0
                    ciClass.m_display = 3
                    ciClass.m_fitTime = 0
                    ciClass.doIt()
                    cmds.namespace(ren=(selName, "DM"))
                    sel = "DM:world_CON"
                else:
                    cmds.file(selCh, i=True, type='Alembic', ra=True, options='v=0', mergeNamespacesOnClash=True, namespace="DM")
                    grm = cmds.ls("DM:*")
                    gr = cmds.createNode("transform", n="DM_GRP")
                    cmds.parent(grm, gr)
                    cmds.select(gr)
                    sel = self.ui.dummy.text()
                self.dupList = list()
                self.dmCons = list()
                locList = list()
                for ia in cmds.ls("Loc*", type="transform"):
                    if str(ia).count("Constraint") == 1:
                        pass
                    else:
                        locList.append(str(ia))
                for k in locList:
                    cmds.select(cl=True)
                    if self.ui.rdDir.isChecked() == True:
                        cmds.select(sel)
                    else:
                        cmds.select("DM_GRP")
                    cmds.currentTime(1)
                    cmds.Duplicate(sel)
                    selc = str(cmds.ls(sl=True)[0])
                    cmds.currentTime(cmds.playbackOptions(q=True, min=True))
                    self.dupList.append(selc)
                    self.dmCons.append(cmds.parentConstraint(str(k), selc, w=1, mo=False)[0])
                cmds.createNode("transform", n="Cache_Dummies_GRP")
                for a in self.dupList:
                    cmds.parent(a, "Cache_Dummies_GRP")
                cmds.select("Cache_Dummies_GRP")
                cmds.createDisplayLayer(name="Cache_dummies_LAY", number=1, nr=True)
                mel.eval("layerEditorLayerButtonTypeChange Cache_dummies_LAY;")
                if self.ui.rdDir.isChecked() == True:
                    cmds.delete("DM:world_CON")
                else:
                    cmds.delete(gr)
                cmds.namespace(rm="DM")
            else:
                cmds.warning("Select Cache, please")

        else:
            seld = str(cmds.ls(sl=True)[0])
            self.ui.dummy.setText(seld)
            sel = self.ui.dummy.text()
            self.dupList = list()
            self.dmCons = list()
            locList = list()
            for ia in cmds.ls("Loc*", type="transform"):
                if str(ia).count("Constraint") == 1:
                    pass
                else:
                    locList.append(str(ia))
            for k in locList:
                cmds.select(cl=True)
                cmds.select(sel)
                cmds.currentTime(1)
                cmds.Duplicate(sel)
                selc = str(cmds.ls(sl=True)[0])
                cmds.currentTime(cmds.playbackOptions(q=True, min=True))
                self.dupList.append(selc)
                self.dmCons.append(cmds.parentConstraint(str(k), selc, w=1, mo=False)[0])
            cmds.createNode("transform", n="Cache_Dummies_GRP")
            for a in self.dupList:
                cmds.parent(a, "Cache_Dummies_GRP")
            cmds.select("Cache_Dummies_GRP")
            cmds.createDisplayLayer(name="Cache_dummies_LAY", number=1, nr=True)
            mel.eval("layerEditorLayerButtonTypeChange Cache_dummies_LAY;")

    def getRange(self):
        if len(cmds.ls("getSctRange")) == 0:
            cmds.polyPlane(w=1, h=1, sx=1, sy=1, ax=(0, 1, 0), cuv=2, ch=1, n="getSctRange")
            cmds.setAttr("getSctRange.scaleX", 50)
            cmds.setAttr("getSctRange.scaleZ", 50)
        else:
            pass
        sel = str(cmds.ls("getSctRange", ap=True, dag=True, type="shape")[0])
        mel.eval("createAndAssignShader blinn %s;" % sel)
        shd = str(cmds.listConnections(str(cmds.listConnections(sel)[0]), type="blinn")[0])
        cmds.setAttr(shd + ".transparency", 0.75, 0.75, 0.75)
        cmds.setAttr(shd + ".color", 0, 0, 0)

    def scatter(self):
        cmds.select(cl=True)
        sel = "getSctRange"
        getY = cmds.getAttr(sel + ".ty")
        selX = cmds.getAttr(sel + ".boundingBoxSize")[0][0]
        selZ = cmds.getAttr(sel + ".boundingBoxSize")[0][2]
        minX = int((cmds.getAttr(sel + ".tx") - abs(selX / 2)) * 100)
        maxX = int((cmds.getAttr(sel + ".tx") + abs(selX / 2)) * 100)
        minZ = int((cmds.getAttr(sel + ".tz") - abs(selZ / 2)) * 100)
        maxZ = int((cmds.getAttr(sel + ".tz") + abs(selZ / 2)) * 100)
        for i in self.locList:
            cmds.setAttr(i + ".tx", float(random.randrange(minX, maxX, step=1) / 100))
            cmds.setAttr(i + ".tz", float(random.randrange(minZ, maxZ, step=1) / 100))
            cmds.setAttr(i + ".ty", getY)
        cmds.select(cl=True)
        cmds.select("Placements")
        for j in self.locList:
            self.rtLst[str(j)] = [float(random.randrange(-300, 300, step=1) / 100.00), cmds.getAttr(str(j) + ".ry")]

    def selGeo(self):
        sel = str(cmds.ls(sl=True)[0])
        self.ui.targetGeo.setText(sel)
        if len(cmds.ls("Ground_Terrain_LAY")) == 0:
            cmds.select(sel)
            cmds.createDisplayLayer(name="Ground_Terrain_LAY", number=1, nr=True)
            mel.eval("layerEditorLayerButtonTypeChange Ground_Terrain_LAY;")
            mel.eval("layerEditorLayerButtonTypeChange Ground_Terrain_LAY;")
        else:
            pass
        cmds.select(cl=True)

    def geoConst(self):
        tarGeo = self.ui.targetGeo.text()
        self.geoList = list()
        if len(cmds.ls("Loc*", sl=True)) == 0:
            for j in self.locList:
                if cmds.ls(str(j) + "_geometry*", ap=True, dag=True, type="constraint"):
                    pass
                else:
                    geoConst = cmds.geometryConstraint(tarGeo, j, weight=1)[0]
                    self.geoList.append(geoConst)
        else:
            seList = cmds.ls("Loc*", sl=True)
            for j in seList:
                if cmds.ls(str(j) + "_geometry*", ap=True, dag=True, type="constraint"):
                    pass
                else:
                    geoConst = cmds.geometryConstraint(tarGeo, j, weight=1)[0]
                    self.geoList.append(geoConst)

    def randOr(self):
        value = int(self.ui.numOri.value())
        if len(cmds.ls("Loc*", sl=True)) == 0:
            for k in self.locList:
                multValue = self.rtLst[str(k)][0]
                cmds.setAttr(str(k) + ".ry", self.rtLst[k][1] + multValue * value)
        else:
            seList = cmds.ls("Loc*", sl=True)
            for k in seList:
                multValue = self.rtLst[str(k)][0]
                cmds.setAttr(str(k) + ".ry", self.rtLst[k][1] + multValue * value)

    def aimSet(self):
        selAim = str(cmds.ls(sl=True)[0])
        self.ui.aimTarget.setText(selAim)
        aimTarget = self.ui.aimTarget.text()
        if len(cmds.ls("Loc*", sl=True)) == 0:
            for ac in self.locList:
                if cmds.getAttr(str(ac) + ".tz") < 0:
                    temAcon = cmds.aimConstraint(aimTarget, str(ac), offset=(0, 0, 0), weight=1, aimVector=(0, 0, 1),
                                                 upVector=(0, 1, 0), worldUpType="vector", worldUpVector=(0, 1, 0), skip=["x", "z"])
                    cmds.delete(temAcon)
                else:
                    temAcon = cmds.aimConstraint(aimTarget, str(ac), offset=(0, 180, 0), weight=1, aimVector=(0, 0, 1),
                                                 upVector=(0, 1, 0), worldUpType="vector", worldUpVector=(0, -1, 0), skip=["x", "z"])
                    cmds.delete(temAcon)
        else:
            seList = cmds.ls("Loc*", sl=True)
            for ac in seList:
                if cmds.getAttr(str(ac) + ".tz") < 0:
                    temAcon = cmds.aimConstraint(aimTarget, str(ac), offset=(0, 0, 0), weight=1, aimVector=(0, 0, 1),
                                                 upVector=(0, 1, 0), worldUpType="vector", worldUpVector=(0, 1, 0), skip=["x", "z"])
                    cmds.delete(temAcon)
                else:
                    temAcon = cmds.aimConstraint(aimTarget, str(ac), offset=(0, 180, 0), weight=1, aimVector=(0, 0, 1),
                                                 upVector=(0, 1, 0), worldUpType="vector", worldUpVector=(0, -1, 0), skip=["x", "z"])
                    cmds.delete(temAcon)
        for j in self.locList:
            self.rtLst[str(j)][1] = cmds.getAttr(str(j) + ".ry")

    def impCh(self):
        '''
        디렉토리로 부를 경우:
            chList = 경로  리스트
            chName = 마지막 디렉토리  리스트
        파일로 부를 경우:
            chList = 경로 + 파일명  리스트
            chName = 파일명  리스트
            리스트 위젯에 chName 띄우기
        :return:
        '''
        getCacheList = []
        if self.ui.rdDir.isChecked() == True:
            getDir = str(cmds.fileDialog2(startingDirectory="/show", fileMode=3, fileFilter="Maya Alembic (*.abc)", caption="Import Cache")[0])
            chList.append(getDir)
            chName.append(getDir.split(os.sep)[-1])
            getCacheList.append(getDir.split(os.sep)[-1])
            self.ui.listWidget.addItems(getCacheList)
        elif self.ui.rdFile.isChecked() == True:
            getChFile = cmds.fileDialog2(startingDirectory="/show", fileMode=4, fileFilter="Maya Alembic (*.abc)", caption="Import Cache")
            for i in getChFile:
                chList.append(str(i))
                chName.append(os.path.basename(str(i)).rsplit(".")[0])
                getCacheList.append(os.path.basename(str(i)).rsplit(".")[0])
            self.ui.listWidget.addItems(getCacheList)

    def delList(self):
        selItem = self.ui.listWidget.selectedItems()
        if not selItem:
            return
        for j in selItem:
            del chList[chName.index(str(j.text()))]
            chName.remove(str(j.text()))
            self.ui.listWidget.takeItem(self.ui.listWidget.row(j))

    def checkNs(self, nameSpace):
        for i in range(500000):
            i += 1
            if len(cmds.ls(nameSpace + str(i) + ":*")) != 0 or cmds.namespace(ex=nameSpace + str(i)) == True:
                pass
            else:
                output = str(i)
                break
        return output

    def locScale(self):
        scAtt = ["scaleX", "scaleY", "scaleZ"]
        locSc = int(self.ui.scnum.value())
        for i in self.locList:
            for j in scAtt:
                cmds.setAttr(str(i) + "." + j, locSc)

    def loadCache(self):
        scMin = int(float("%0.2f" % float(self.ui.scaleMin.text())) * 100)
        scMax = int(float("%0.2f" % float(self.ui.scaleMax.text())) * 100)
        offMin = int(self.ui.offMin.text())
        offMax = int(self.ui.offMax.text())
        sclMin = int(float("%0.2f" % float(self.ui.sclMin.text())) * 100)
        sclMax = int(float("%0.2f" % float(self.ui.sclMax.text())) * 100)
        if self.ui.rdRan.isChecked() == True:  # Auto
            locList = list()
            for ia in cmds.ls("Loc*", type="transform"):
                if str(ia).count("Constraint") == 1:
                    pass
                else:
                    locList.append(str(ia))
            cnt = self.ui.listWidget.count()
            for k in locList:
                rand = random.randrange(1, cnt + 1) - 1
                selCh = chList[chName.index(str(self.ui.listWidget.item(rand).text()))]
                if self.ui.rdDir.isChecked() == True:  # Directory NameSpace
                    for con in os.listdir(selCh):
                        if str(con).count(":") == 1 and str(con).count(".abc") == 1:
                            selName = str(con).rsplit(":")[0]
                else:  # File NameSpace
                    selName = str(self.ui.listWidget.item(rand).text())
                ns = self.checkNs(selName)
                nodeN = selName + ns
                if self.ui.rdDir.isChecked() == True:  # Directory Import
                    ciClass = sgUI.ComponentImport(Files=selCh, World=1)
                    ciClass.m_mode = 0
                    ciClass.m_display = 3
                    ciClass.m_fitTime = 0
                    ciClass.doIt()
                    cmds.namespace(ren=(selName, nodeN))
                else:  # File Import
                    cmds.file(selCh, i=True, type='Alembic', ra=True, options='v=0', mergeNamespacesOnClash=True, namespace=nodeN)
                if len(cmds.ls(nodeN + ":*", type="AlembicNode")) != 0:  # Alembic Node exist.
                    selOb = str(cmds.ls(nodeN + ":*", type="AlembicNode")[0])
                    if self.ui.rdDir.isChecked() == True:  # Directory Attr Create
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.item(rand).text()), type="string")
                        cmds.connectAttr(trNode + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(trNode + ".speed", selOb + ".speed", f=True)
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:  # File Attr Create
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.item(rand).text()), type="string")
                        cmds.connectAttr(gr + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(gr + ".speed", selOb + ".speed", f=True)
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr
                    if scMin != scMax:
                        randResult = random.randrange(scMin, scMax, step=1) / 100.00
                        cmds.setAttr(selT + ".speed", randResult)
                    else:
                        cmds.setAttr(selT + ".speed", scMax / 100)
                    if offMin != offMax:
                        randOff = random.randrange(offMin, offMax, step=1)
                        cmds.setAttr(selT + ".offset", 1001 + randOff)
                    else:
                        cmds.setAttr(selT + ".offset", 1001 + offMax)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                else:  # Alembic Node not exist.
                    if self.ui.rdDir.isChecked() == True:  # Directory Attr Create
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.item(rand).text()), type="string")
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.item(rand).text()), type="string")
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr
                    cmds.setAttr(selT + ".speed", 1)
                    cmds.setAttr(selT + ".offset", 1001)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                try:
                    selConn = str(cmds.listConnections(str(k), type="parentConstraint")[0])
                    consCube = str(cmds.listConnections(selConn + ".constraintParentInverseMatrix")[0])
                    cmds.delete(consCube)
                except:
                    pass
                cmds.delete(str(k))
            if len(cmds.ls("Cache_Dummies_GRP", dag=True, ap=True)) == 1:
                cmds.delete("Cache_Dummies_GRP")
                cmds.delete("Cache_dummies_LAY")
            else:
                pass
            if len(cmds.ls("Placements", dag=True, ap=True)) == 1:
                cmds.delete("Placements")
                try:
                    cmds.delete("getSctRange")
                    cmds.delete("Ground_Terrain_LAY")
                except:
                    pass
            else:
                pass
        elif self.ui.rdSel.isChecked() == True:  # Manual
            sel = cmds.ls(sl=True)
            selCh = chList[chName.index(str(self.ui.listWidget.selectedItems()[0].text()))]
            if self.ui.rdDir.isChecked() == True:
                for con in os.listdir(selCh):
                    if str(con).count(":") == 1 and str(con).count(".abc") == 1:
                        selName = str(con).rsplit(":")[0]
            else:
                selName = str(self.ui.listWidget.selectedItems()[0].text())
            for k in sel:
                ns = self.checkNs(selName)
                nodeN = selName + ns
                if self.ui.rdDir.isChecked() == True:
                    ciClass = sgUI.ComponentImport(Files=selCh, World=1)
                    ciClass.m_mode = 0
                    ciClass.m_display = 3
                    ciClass.m_fitTime = 0
                    ciClass.doIt()
                    cmds.namespace(ren=(selName, nodeN))
                else:
                    cmds.file(selCh, i=True, type='Alembic', ra=True, options='v=0', mergeNamespacesOnClash=True, namespace=nodeN)
                if len(cmds.ls(nodeN + ":*", type="AlembicNode")) != 0:  # Alembic Node exist.
                    selOb = str(cmds.ls(nodeN + ":*", type="AlembicNode")[0])
                    if self.ui.rdDir.isChecked() == True:
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.connectAttr(trNode + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(trNode + ".speed", selOb + ".speed", f=True)
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.connectAttr(gr + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(gr + ".speed", selOb + ".speed", f=True)
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr

                    if scMin != scMax:
                        randResult = random.randrange(scMin, scMax, step=1) / 100.00
                        cmds.setAttr(selT + ".speed", randResult)
                    else:
                        cmds.setAttr(selT + ".speed", scMax / 100)
                    if offMin != offMax:
                        randOff = random.randrange(offMin, offMax, step=1)
                        cmds.setAttr(selT + ".offset", 1001 + randOff)
                    else:
                        cmds.setAttr(selT + ".offset", 1001 + offMax)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                else:  # Alembic Node not exist.
                    if self.ui.rdDir.isChecked() == True:
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr
                    cmds.setAttr(selT + ".speed", 1)
                    cmds.setAttr(selT + ".offset", 1001)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                try:
                    selConn = str(cmds.listConnections(str(k), type="parentConstraint")[0])
                    consCube = str(cmds.listConnections(selConn + ".constraintParentInverseMatrix")[0])
                    cmds.delete(consCube)
                except:
                    pass
                cmds.delete(str(k))
            if len(cmds.ls("Cache_Dummies_GRP", dag=True, ap=True)) == 1:
                cmds.delete("Cache_Dummies_GRP")
                cmds.delete("Cache_dummies_LAY")
            else:
                pass
            if len(cmds.ls("Placements", dag=True, ap=True)) == 1:
                cmds.delete("Placements")
                try:
                    cmds.delete("getSctRange")
                    cmds.delete("Ground_Terrain_LAY")
                except:
                    pass
            else:
                pass
        elif self.ui.rdCus.isChecked() == True:
            sel = cmds.ls(sl=True)
            selCh = chList[chName.index(str(self.ui.listWidget.selectedItems()[0].text()))]
            if self.ui.rdDir.isChecked() == True:
                for con in os.listdir(selCh):
                    if str(con).count(":") == 1 and str(con).count(".abc") == 1:
                        selName = str(con).rsplit(":")[0]
            else:
                selName = str(self.ui.listWidget.selectedItems()[0].text())
            for k in sel:
                ns = self.checkNs(selName)
                nodeN = selName + ns
                if self.ui.rdDir.isChecked() == True:
                    ciClass = sgUI.ComponentImport(Files=selCh, World=1)
                    ciClass.m_mode = 0
                    ciClass.m_display = 3
                    ciClass.m_fitTime = 0
                    ciClass.doIt()
                    cmds.namespace(ren=(selName, nodeN))
                else:
                    cmds.file(selCh, i=True, type='Alembic', ra=True, options='v=0', mergeNamespacesOnClash=True, namespace=nodeN)
                if len(cmds.ls(nodeN + ":*", type="AlembicNode")) != 0:  # Alembic Node exist.
                    selOb = str(cmds.ls(nodeN + ":*", type="AlembicNode")[0])
                    if self.ui.rdDir.isChecked() == True:
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.connectAttr(trNode + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(trNode + ".speed", selOb + ".speed", f=True)
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.connectAttr(gr + ".offset", selOb + ".offset", f=True)
                        cmds.connectAttr(gr + ".speed", selOb + ".speed", f=True)
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr
                    if scMin != scMax:
                        randResult = random.randrange(scMin, scMax, step=1) / 100.00
                        cmds.setAttr(selT + ".speed", randResult)
                    else:
                        cmds.setAttr(selT + ".speed", scMax / 100)
                    if offMin != offMax:
                        randOff = random.randrange(offMin, offMax, step=1)
                        cmds.setAttr(selT + ".offset", 1001 + randOff)
                    else:
                        cmds.setAttr(selT + ".offset", 1001 + offMax)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                else:
                    if self.ui.rdDir.isChecked() == True:
                        trNode = str(cmds.ls(nodeN + ":world_CON")[0])  # 최상위 Translation 노드
                        cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(trNode + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        temp = cmds.parentConstraint(str(k), trNode, mo=0, w=True)
                        cmds.delete(temp)
                        selT = trNode
                    else:
                        grm = cmds.ls(nodeN + ":*", type="transform")
                        gr = cmds.createNode("transform", n=nodeN + "_GRP")
                        cmds.addAttr(gr, ln='offset', nn='offset', at='long', dv=0)
                        cmds.addAttr(gr, ln='speed', nn='speed', at='float', dv=0)
                        cmds.addAttr(gr, ln='cacheName', nn='cacheName', dt='string')
                        cmds.setAttr(gr + ".cacheName", str(self.ui.listWidget.selectedItems()[0].text()), type="string")
                        cmds.parent(grm, gr)
                        temp = cmds.parentConstraint(str(k), gr, mo=0, w=True)
                        cmds.delete(temp)
                        selT = gr
                    cmds.setAttr(selT + ".speed", 1)
                    cmds.setAttr(selT + ".offset", 1001)
                    if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                        if sclMin != sclMax:
                            randScl = random.randrange(sclMin, sclMax, step=1) / 100.00
                            cmds.setAttr(selT + ".initScale", randScl)
                        else:
                            cmds.setAttr(selT + ".initScale", sclMax / 100)
                    else:
                        pass
                try:
                    selConn = str(cmds.listConnections(str(k), type="parentConstraint")[0])
                    consCube = str(cmds.listConnections(selConn + ".constraintParentInverseMatrix")[0])
                    cmds.delete(consCube)
                except:
                    pass
                cmds.setAttr(str(k) + ".visibility", 0)
            if len(cmds.ls("Cache_Dummies_GRP", dag=True, ap=True)) == 1:
                cmds.delete("Cache_Dummies_GRP")
                cmds.delete("Cache_dummies_LAY")
            else:
                pass
            if len(cmds.ls("Placements", dag=True, ap=True)) == 1:
                cmds.delete("Placements")
                try:
                    cmds.delete("getSctRange")
                    cmds.delete("Ground_Terrain_LAY")
                except:
                    pass
            else:
                pass

    def getAssetRoot(self):
        sel = cmds.ls(sl=True)
        rootList = list()
        for i in sel:
            path = cmds.listRelatives(str(i), f=1, p=1)
            if path == None:
                rootList.append(str(i))
            else:
                rootList.append(str(path[0].split('|')[1]))
        return rootList

    def getAttDict(self):
        selRigList = self.getAssetRoot()
        selAtt = dict()
        for j in selRigList:
            attList = list()
            attList.append(float("%0.2f" % cmds.getAttr(j + ".cacheSpeed")))
            attList.append(cmds.getAttr(j + ".cacheOffset") - 1001.0)
            attList.append(os.sep.join(str(cmds.getAttr(j + ".cacheName")).split(os.sep)[0:-1]))
            attList.append(float("%0.2f" % cmds.getAttr(j.split(":")[0] + ":place_CON.initScale")))
            selAtt[j] = attList
        return selAtt

    def replace(self):
        getAtts = self.getAttDict()
        for i in getAtts.keys():
            for con in os.listdir(getAtts[str(i)][2]):
                if str(con).count(":") == 1 and str(con).count(".abc") == 1:
                    selName = str(con).rsplit(":")[0]
            ns = self.checkNs(selName)
            nodeN = selName + ns
            ciClass = sgUI.ComponentImport(Files=getAtts[str(i)][2], World=1)
            ciClass.m_mode = 0
            ciClass.m_display = 3
            ciClass.m_fitTime = 0
            ciClass.doIt()
            cmds.namespace(ren=(selName, nodeN))

            if len(cmds.ls(nodeN + ":*", type="AlembicNode")) != 0:
                selOb = str(cmds.ls(nodeN + ":*", type="AlembicNode")[0])
                trNode = str(cmds.ls(nodeN + ":world_CON")[0])
                cmds.addAttr(trNode, ln='offset', nn='offset', at='long', dv=0)
                cmds.addAttr(trNode, ln='speed', nn='speed', at='float', dv=0)
                cmds.addAttr(trNode, ln='cacheName', nn='cacheName', dt='string')
                cmds.setAttr(trNode + ".cacheName", getAtts[str(i)][2], type="string")
                cmds.connectAttr(trNode + ".offset", selOb + ".offset", f=True)
                cmds.connectAttr(trNode + ".speed", selOb + ".speed", f=True)
                temp = cmds.parentConstraint(str(i).split(":")[0] + ":place_CON", trNode, mo=0, w=True)
                cmds.delete(temp)
                selT = trNode
                cmds.setAttr(selT + ".speed", getAtts[str(i)][0])
                cmds.setAttr(selT + ".offset", 1001 + getAtts[str(i)][1])
                if cmds.attributeQuery("initScale", n=selT, ex=True) == True:
                    cmds.setAttr(selT + ".initScale", getAtts[str(i)][3])
                else:
                    pass
            rn = str(cmds.referenceQuery(str(i), rfn=True))
            cmds.file(rr=True, rfn=rn)

    def reImpRepl(self):
        if len(cmds.ls(sl=True)) != 0:
            self.replace()
        else:
            cmds.warning("Select object(s) to replace.")

def main():
    global myWindow
    myWindow = Window()
    myWindow.show()


if __name__ == '__main__':
    main()