# HVSpos0030
import maya.cmds as cmds
from PySide2 import QtWidgets, QtCore, QtUiTools

import dxRigUI
import aniCommon

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        # _win.deleteLater()
    _win = HVSpos0030Temp()
    _win.show()
    _win.resize(300, 100)

class HVSpos0030Temp(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(HVSpos0030Temp, self).__init__(parent)
        main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        self.copyMeshBtn = QtWidgets.QPushButton("Copy Mesh")
        self.removeMeshBtn = QtWidgets.QPushButton("Remove Mesh")

        main_layout.addWidget(self.copyMeshBtn)
        main_layout.addWidget(self.removeMeshBtn)

        self.copyMeshBtn.clicked.connect(self.copyMesh)
        self.removeMeshBtn.clicked.connect(self.removeMesh)

    def getSelection(self):
        sel = cmds.ls(sl=True)
        if not sel:
            raise Exception("Select Object First")
        return sel


    def copyMesh(self):
        sel = self.getSelection()
        dupedObj = cmds.duplicate(sel, returnRootsOnly=True)
        self.addRenderMesh(dupedObj)


    def removeMesh(self):
        selection = self.getSelection()
        self.removeRenderMesh(selection)


    def getRootMapData(self, objects):
        rootMap = {}
        for i in objects:
            if not rootMap.has_key(aniCommon.getRootNode(i, type="dxNode")):
                rootMap[aniCommon.getRootNode(i, type="dxNode")] = list()

            rootMap[aniCommon.getRootNode(i, type="dxNode")].append(i)

        return rootMap


    def addRenderMesh(self, objects):
        newObjectNames = list()

        for obj in objects:
            # namespace = aniCommon.getNameSpace(obj)
            if cmds.objExists(obj):
                newName = obj
            else:
                newName = obj + "_#"
            newObjectNames.append(cmds.rename(obj, newName))

        rootMap = self.getRootMapData(newObjectNames)
        print rootMap
        for root in rootMap:
            cmds.select(rootMap[root], r=True)
            dxRigUI.addSelectedMeshes(root + ".renderMeshes")
        cmds.select(rootMap.values()[0], r=True)


    def removeRenderMesh(self, selection):
        rootMap = self.getRootMapData(selection)

        for root in rootMap:
            currentValues = cmds.getAttr(root + ".renderMeshes")

            for i in selection:
                if i in currentValues:
                    currentValues.remove(i)
            cmds.setAttr(root + ".renderMeshes", *([len(currentValues)] + currentValues), type='stringArray')
        cmds.delete(selection)