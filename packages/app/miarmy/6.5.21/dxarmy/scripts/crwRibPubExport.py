# encoding:utf-8

import maya.cmds as cmds
import os, site, webbrowser
import dxUI
from Qt import QtCore, QtGui, QtWidgets, load_ui
from McdGeneral import *
from McdSimpleCmd import *
from McdRender import *

import McdMeshDriveSetup
reload(McdMeshDriveSetup)
import dxArmy
import dxConfig
site.addsitedir(dxConfig.getConf('TRACTOR_API'))
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "./ui/ribPub.ui")

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        self.connectSignal()
        self.mayafile_tmp = ''

    def connectSignal(self):
        self.pubBtn.clicked.connect(self.publish)
        self.scFileBtn.clicked.connect(self.mayafileImport)
        self.pubFileBtn.clicked.connect(self.mayafileExport)
        self.brws.clicked.connect(self.trct)

    def mayafileImport(self):
        multipleFilter = "Maya Files (*.mb *.ma);;Maya ASCII (*.ma);;Maya Binary (*.mb)"
        inputFile = str(cmds.fileDialog2(fileMode=1, fileFilter=multipleFilter, caption="Import Maya Scene File")[0])
        self.scFile.setText(inputFile)
        defaultPath = os.sep.join(inputFile.split(os.sep)[:-2]) + "/data/"
        self.pubFile.setText(defaultPath)

    def mayafileExport(self):
        if len(self.pubFile.text()) == 0:
            outPath = "/"
        else:
            outPath = self.pubFile.text()
        animDir = str(cmds.fileDialog2(fileMode=3, fileFilter="outputPath", dir=outPath, caption="Output Path")[0])
        self.pubFile.setText(animDir)

    def publish(self):
        chunks = int(self.chunk.value())
        minTime = int(self.minTime.text())
        maxTime = int(self.maxTime.text())
        mayaFile = self.scFile.text()
        pubFile = self.pubFile.text()
        if cmds.ls("MDGGrp_*", type="transform"):
            McdMeshDriveSetup.McdMeshDrive2Clear()
        options = {'m_chunk':chunks, 'm_mayafile': mayaFile, 'm_outdir':pubFile, 'm_start':minTime, 'm_end':maxTime}
        if len(self.pubFile.text()) == 0 or len(self.scFile.text()) == 0:
            cmds.confirmDialog(title='Error', message='Path Error')
        else:
            self.createPath()
            dxArmy.ExportRibSpool(options)

    def trct(self):
        webbrowser.register('firefox', None)
        webbrowser.Mozilla('firefox').open('http://10.0.0.25/tv')

    def createPath(self):
        filePath = self.pubFile.text()
        if not os.path.exists(os.path.dirname(filePath)):
            os.makedirs(os.path.dirname(filePath))

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
