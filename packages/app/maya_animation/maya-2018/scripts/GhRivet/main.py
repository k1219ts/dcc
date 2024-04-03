# -*- coding: utf-8 -*-
"""
author : gyeongheon.jeong

import GhRivet.window
reload(GhRivet.window)
GhRivet.window.showUI()

"""

import os
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools
import maya.cmds as cmds
import modules;reload(modules)

CURRENTPATH = os.path.abspath( __file__ )
UIROOT = os.path.dirname(CURRENTPATH)
UIFILE = os.path.join(UIROOT, "ui/GhRivet.ui")

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = Rivet()
    _win.show()

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

class Rivet(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Rivet, self).__init__(parent)

        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)

        setup_ui(ui, self)
        self.setWindowTitle('GhRivet')

        self.byCenter_radioBtn.setChecked(True)
        self.connectSignals()

    def connectSignals(self):
        self.doRivet_Btn.clicked.connect(self.doIt)
        self.cancel_Btn.clicked.connect(self.close)
        self.addTarget_Btn.clicked.connect(self.addTargetMesh)
        self.addObjects_Btn.clicked.connect(self.addObjectList)

    def getMayaSelection(self):
        """
        
        :return: A list of maya selections 
        """
        sel = cmds.ls(sl=True)
        return sel

    def addTargetMesh(self):
        targetMesh = self.getMayaSelection()
        countSelction = len(targetMesh)
        if countSelction >= 2:
            QtWidgets.QMessageBox.warning(self, "Warning!", "To Many Object Selected!")
            return

        targetMeshShape = cmds.listRelatives(targetMesh[0], s=True)[0]
        if cmds.objectType(targetMeshShape) != "mesh":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Mesh")
            return

        self.targetMesh_lineEdit.clear()
        self.targetMesh_lineEdit.setText(str(targetMesh[0]))

    def addObjectList(self):
        objectList = self.getMayaSelection()
        if not objectList:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Objects")
            return
        self.objectList_listWidget.clear()
        self.objectList_listWidget.addItems(objectList)

    def doIt(self):
        targetMesh = str(self.targetMesh_lineEdit.text())
        items = []
        for index in xrange(self.objectList_listWidget.count()):
            items.append(self.objectList_listWidget.item(index))
        objectList = [str(i.text()) for i in items]
        modules.rivet(objects=objectList, targetMesh=targetMesh)

