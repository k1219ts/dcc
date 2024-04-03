# -*- coding:utf-8 -*-
#!/usr/bin/env python

__author__ = 'gyeongheon.jeong'

import maya.cmds as cmds
import maya.OpenMayaUI as mui
import os
import getpass
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools
from shiboken2 import wrapInstance
import GH_QtPanel.GH_QtPanelModules as QtPanelModules
reload(QtPanelModules)
import HUD.HUDmodules as hud


currentpath = os.path.abspath( __file__ )
UIROOT = os.path.dirname(currentpath)
ROOT = "/netapp/backstage/pub/apps/maya2/global/DDPM"
mayaVersion = "2017"
UIFILE = os.path.join(UIROOT, "GH_QtPanel.ui")
RESOLUTION_DIC = { "HD1080" : [1920, 1080], "HD720" : [1280, 720]}

def hconv(text):
    return unicode(text, 'utf-8')

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    wrapInstance(long(ptr), QtWidgets.QWidget)

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

def getCurrentCamera():
    pan = cmds.getPanel( wf=True )
    if pan.find('modelPanel') > -1:
        return cmds.modelPanel( pan, q=True, camera=True )

class QtPanelUI(QtWidgets.QWidget):

    def __init__(self, parent = getMayaWindow()):
        super(QtPanelUI, self).__init__(parent)
        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)
        self.setWindowTitle('QtPanel')

        # First use SIP to unwrap the layout into a pointer
        # Then get the full path to the UI in maya as a string
        #layout = mui.MQtUtil.fullName(long(shiboken.getCppPointer(self.verticalLayout)[0]))
        cmds.setParent("QtPanel_Form")

        paneLayoutName = cmds.paneLayout()
        # Find a pointer to the paneLayout that we just created
        ptr = mui.MQtUtil.findControl(paneLayoutName)

        # Wrap the pointer into a python QObject
        self.paneLayout = wrapInstance(long(ptr), QtWidgets.QWidget)
        cam = getCurrentCamera()
        if cam:
            self.modelPanelName = cmds.modelPanel(label='QtPanel', camera=cam)
        else:
            self.modelPanelName = cmds.modelPanel(label="QtPanel")
        ptr = mui.MQtUtil.findControl(self.modelPanelName)

        # Wrap the pointer into a python QObject
        self.modelPanel = wrapInstance(long(ptr), QtWidgets.QWidget)
        self.verticalLayout.addWidget(self.paneLayout)


        self.initGUI()
        self.connectSignals()

    def initGUI(self):
        mayaCameras = cmds.listCameras()
        self.camList_comboBox.addItems(mayaCameras)
        self.ArtistNamelineEdit.setText( getpass.getuser() )
        self.fileOpenButton.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons/fileOpen.png'))))
        self.playblastButton.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons/timePlay.png'))))
        self.HUDcheckBox.setChecked(True)
        self.offScreencheckBox.setChecked(True)

        self.resolutionEdit()

    def connectSignals(self):
        # self.resWidthlineEdit.textChanged.connect(self.resizeEvent)
        # self.resHeightlineEdit.textChanged.connect(self.resizeEvent)
        # self.MaintainRatiocheckBox.stateChanged.connect(self.resizeEvent)

        self.ResolutioncomboBox.currentIndexChanged.connect(self.resolutionEdit)
        self.resizeWinButton.clicked.connect(reSizeWin)
        self.camList_comboBox.currentIndexChanged.connect(self.changePanelView)
        self.showTypeAllcheckBox.stateChanged.connect(lambda: self.showTypes(Type="allObjects"))
        self.showTypePolymeshcheckBox.stateChanged.connect(lambda: self.showTypes(Type="polymeshes"))
        self.showTypeImpcheckBox.stateChanged.connect(lambda: self.showTypes(Type="imagePlane"))
        self.showTypeGpuCachecheckBox.stateChanged.connect(lambda: self.showTypes(Type="gpucache"))
        self.playblastButton.clicked.connect(self.playIt)
        self.makeMovButton.clicked.connect(self.makeIt)
        self.fileOpenButton.clicked.connect(self.explorerButton)

    def resolutionEdit(self):
        resText = self.ResolutioncomboBox.currentText()

        if resText == "custom":
            self.resWidthlineEdit.setEnabled(True)
            self.resHeightlineEdit.setEnabled(True)

        elif resText == "HD1080":
            self.resWidthlineEdit.setEnabled(False)
            self.resHeightlineEdit.setEnabled(False)
            self.resWidthlineEdit.setText(str(RESOLUTION_DIC["HD1080"][0]))
            self.resHeightlineEdit.setText(str(RESOLUTION_DIC["HD1080"][1]))
        elif resText == "HD720":
            self.resWidthlineEdit.setEnabled(False)
            self.resHeightlineEdit.setEnabled(False)
            self.resWidthlineEdit.setText(str(RESOLUTION_DIC["HD720"][0]))
            self.resHeightlineEdit.setText(str(RESOLUTION_DIC["HD720"][1]))

    def showEvent(self, event):
        #super(QtPanelUI, self).showEvent(event)

        # maya can lag in how it repaints UI. Force it to repaint
        # when we show the window.
        self.paneLayout.repaint()

    def resizeEvent(self, event):
        #super(QtPanelUI, self).resizeEvent(event)
        width = int(self.resWidthlineEdit.text()) + 222
        height = int(self.resHeightlineEdit.text()) + 62
        print width, height

        new_size = QtCore.QSize(width, height)

        if self.MaintainRatiocheckBox.isChecked():
            try:
                new_size.scale(event.size(), QtCore.Qt.KeepAspectRatio)
                #self.setMinimumSize(new_size)
                self.resize(new_size)
            except:
                pass
        else:
            try:
                new_size.scale(event.size(), QtCore.Qt.IgnoreAspectRatio)
                #self.setMinimumSize(new_size)
                self.resize(new_size)
            except:
                pass

    # @QtCore.pyqtSlot(str)
    def showTypes(self, Type=str()):
        state = bool()

        allState = self.showTypeAllcheckBox.isChecked()
        polyState = self.showTypePolymeshcheckBox.isChecked()
        impState = self.showTypeImpcheckBox.isChecked()
        gpuState = self.showTypeGpuCachecheckBox.isChecked()

        if Type == "allObjects":
            if allState:
                if not polyState: self.showTypePolymeshcheckBox.setCheckState(QtCore.Qt.Checked)
                if not impState: self.showTypeImpcheckBox.setCheckState(QtCore.Qt.Checked)
                if not gpuState: self.showTypeGpuCachecheckBox.setCheckState(QtCore.Qt.Checked)
                state = True

            elif allState == 0:
                if polyState: self.showTypePolymeshcheckBox.setCheckState(QtCore.Qt.Unchecked)
                if impState: self.showTypeImpcheckBox.setCheckState(QtCore.Qt.Unchecked)
                if gpuState: self.showTypeGpuCachecheckBox.setCheckState(QtCore.Qt.Unchecked)
                state = False

        elif Type == "polymeshes":
            if polyState:
                state = True
            elif not polyState:
                state = False

        elif Type == "imagePlane":
            if impState:
                state = True
            elif not impState:
                state = False

        elif Type == "gpucache":
            if gpuState:
                state = True
            elif not gpuState:
                state = False

        QtPanelModules.showTypes(self.modelPanelName, Type, state)


    def changePanelView(self):
        selectedCam = str(self.camList_comboBox.currentText())
        cmds.modelPanel(self.modelPanelName, e=True, camera=selectedCam)


    def explorerButton(self):
        self.updateHUDdata(makeDir = False)
        os.system('/usr/bin/nautilus %s &' % self.rootDirectory)

    def updateHUDdata(self, makeDir = False):
        self.artistName = str(self.ArtistNamelineEdit.text())
        self.sceneName = cmds.file(q=1, namespace=1)
        self.status = str(self.HUDstatus_comboBox.currentText())
        self.progress = str(self.HUDprogress_comboBox.currentText())

        sceneFile = cmds.file(q=True, l=True)[0]

        self.rootDirectory = "%s/preview" % os.sep.join( sceneFile.split(os.sep)[:-2] )

        if not os.path.isdir(self.rootDirectory) and makeDir:
            try:
                os.mkdir(self.rootDirectory)
            except:
                pass

        if len(self.sceneName):
            movieFileName = self.sceneName + '.mov'
        else:
            movieFileName = 'untitled.mov'

        self.movfileName = os.path.join(self.rootDirectory, movieFileName)

    def playIt(self):

        if self.offScreencheckBox.isChecked():
            offScreenValue = 1
        else:
            offScreenValue = 0

        w_v = int(self.resWidthlineEdit.text())
        h_v = int(self.resHeightlineEdit.text())

        if self.HUDcheckBox.isChecked():
            self.updateHUDdata()
            hud.mg_CreateHUD(self.artistName, self.sceneName, self.status, self.progress)

        cmds.setFocus( self.modelPanelName )
        QtPanelModules.playBlast(offScreenValue, w_v, h_v)

        hud.mg_removeHUD()

    def makeIt(self):
        self.updateHUDdata(True)
        msg = QtWidgets.QMessageBox.question(self, hconv("실행안내"), hconv("mov 파일을 만드시겠습니까?"), QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if msg != QtWidgets.QMessageBox.Ok: return

        if self.offScreencheckBox.isChecked():
            offScreenValue = 1
        else:
            offScreenValue = 0

        w_v = int(self.resWidthlineEdit.text())
        h_v = int(self.resHeightlineEdit.text())

        if self.HUDcheckBox.isChecked():
            hud.mg_CreateHUD(self.artistName, self.sceneName, self.status, self.progress)

        cmds.setFocus( self.modelPanelName )

        QtPanelModules.runFFmpeg(self.movfileName, ROOT, w_v, h_v, offScreenValue)

        hud.mg_removeHUD()

        QtWidgets.QMessageBox.information(self, hconv("알림"), hconv("성공적으로 완료되었습니다."))

    def deleteControl(self, control):
        if cmds.workspaceControl(control, q=True, exists=True):
            cmds.workspaceControl(control, e=True, close=True)
            cmds.deleteUI(control, control=True)

    def run(self):
        self.setObjectName('QtPanelUI')
        workSpaceControlName = self.objectName() + 'WorkspaceControl'
        print workSpaceControlName
        self.deleteControl(workSpaceControlName)
        self.show()

def reSizeWin():
    width = int( win.resWidthlineEdit.text() ) + 222
    height = int( win.resHeightlineEdit.text() ) + 62

    #win.setMinimumSize( width, height)
    #win.setFixedSize( width, height)
    win.resize( width, height)
    win.repaint()
    print width


def showUI():
    global win
    try:
        win.close()
        win.deleteLater()
    except:
        pass
    win = QtPanelUI()
    win.show()
    #win.run()
    return win
