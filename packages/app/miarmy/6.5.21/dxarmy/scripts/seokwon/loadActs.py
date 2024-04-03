# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import McdAgentManager
import McdLoadActions
from PySide2 import QtCore, QtGui, QtWidgets, QtCompat

chList = list()
chName = list()
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "ui", "loadActs.ui")
print uiFile

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

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
                lst.append(i.toLocalFile().split(os.sep)[-1].split(".ma")[0])
                chList.append(str(i.toLocalFile()))
                chName.append(str(i.toLocalFile().split(os.sep)[-1]).split(".ma")[0])
            self.addItems(lst)
        else:
            event.ignore()
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self._del_item()

    def _del_item(self):
        selItem = self.selectedItems()
        if not selItem:
            return
        for j in selItem:
            self.takeItem(self.row(j))
            del chList[chName.index(str(j.text()))]
            del chName[chName.index(str(j.text()))]

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()
        self.agDict = {}
        self.updateAgentList()

    def connectSignal(self):
        self.allAct = DragDropTest(self.wd)
        self.allAct.resize(265,170)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.allAct.setFont(font)
        self.allAct.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.impBtn.clicked.connect(self.impAct)
        self.delBtn.clicked.connect(self.delAct)
        self.dropBtn.clicked.connect(self.dropAct)
        self.refBtn.clicked.connect(self.updateAgentList)
        self.taCombo.currentIndexChanged.connect(self.tarAgChange)
        self.delAgBtn.clicked.connect(self.delAg)
        self.loadBtn.clicked.connect(self.loadAgent)

    def impAct(self):
        getActLis = []
        getAct = cmds.fileDialog2(startingDirectory="/dexter/Cache_DATA/CRD", fileMode=4, fileFilter="Maya Ascii (*.ma)", caption="Import Action")
        for i in getAct:
            chList.append(str(i))
            chName.append(os.path.basename(str(i)).split(".")[0])
            getActLis.append(os.path.basename(str(i)).split(".")[0])
        self.allAct.addItems(getActLis)
        self.reData(chList, chName)

    def reData(self, path, name):
        self.pathInfo = {}
        for v in range(len(name)):
            self.pathInfo[name[v]] = path[v]

    def delAct(self):
        selItem = self.allAct.selectedItems()
        if not selItem:
            return
        for j in selItem:
            self.allAct.takeItem(self.allAct.row(j))
            del chList[chName.index(str(j.text()))]
            del chName[chName.index(str(j.text()))]
        self.reData(chList, chName)

    def dropAct(self):
        getList = self.allAct.selectedItems()
        temList = []
        for w in getList:
            self.agDict[self.taCombo.currentText()].append(str(w.text()))
            temList.append(w.text())
        self.agAct.addItems(temList)

    def updateAgentList(self):
        self.agAct.clear()
        try:
            agList = McdAgentManager.McdGetAllAgentTypeNIDWithColor()[0]
        except:
            pass
        if agList:
            self.taCombo.clear()
            self.taCombo.addItems(agList)
            self.agDict = {}
            for e in agList:
                self.agDict[e] = []

    def tarAgChange(self):
        if not len(self.agDict) == 0:
            if not len(str(self.taCombo.currentText())) == 0:
                curAg = str(self.taCombo.currentText())
                self.agAct.clear()
                if not len(self.agDict[curAg]) == 0:
                    self.agAct.addItems(self.agDict[curAg])
            else:
                return
        else:
            return

    def delAg(self):
        selItem = self.agAct.selectedItems()
        agItems = []
        for g in xrange(self.agAct.count()):
            agItems.append(str(self.agAct.item(g).text()))
        if not selItem:
            return
        for t in selItem:
            del self.agDict[self.taCombo.currentText()][agItems.index(str(t.text()))]
            self.agAct.takeItem(self.agAct.row(t))

    def loadAgent(self):
        getList = []
        loadList = []
        self.reData(chList, chName)
        for g in xrange(self.agAct.count()):
            getList.append(str(self.agAct.item(g).text()))
        for s in getList:
            loadList.append(self.pathInfo[s])
        cmds.setAttr("McdGlobal1.activeAgentName", str(self.taCombo.currentText()), type="string")
        for z in loadList:
            McdLoadActions.McdLoadActions(z)

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