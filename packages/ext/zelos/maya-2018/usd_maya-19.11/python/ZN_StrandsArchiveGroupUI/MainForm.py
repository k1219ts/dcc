################################################
#
# author        : Dexter RND daeseok.chae
# create        : 2017.06.23
# filename      : MainForm.py
# last update   : 2017.06.23
#
################################################

MY_PYPATH_MODUEL = "/netapp/backstage/pub/apps/maya2/versions/2017/global/linux/lib/site-packages"
import site
site.addsitedir(MY_PYPATH_MODUEL)

from Qt import QtWidgets
from Qt import QtCore

import Qt
import maya.cmds as cmds
import maya.mel as mel
import os

from MainUI import Ui_Form

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)


        # set UI file
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.abcPushBtn.clicked.connect(self.abcButtonClick)
        self.ui.zennPushBtn.clicked.connect(self.zennButtonClick)
        self.ui.loadPushBtn.clicked.connect(self.loadButtonClick)

    def getOpenDirectory(self, titleCaption, startDirPath):
        fileName = ""
        if "PyQt" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getExistingDirectory(self, titleCaption, startDirPath)
        else:
            dialog = cmds.fileDialog2(fileMode = 3, caption = titleCaption, okCaption = "Select", startingDirectory = startDirPath)
            if dialog != None:
                fileName = str(dialog[0])

        return fileName
    def getOpenFile(self, titleCaption, startDirPath, exrCaption):
        fileName = ""
        if "PyQt" in Qt.__binding__:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self, titleCaption, startDirPath, exrCaption)
        else:
            fileName = str(cmds.fileDialog2(fileMode = 1, caption = titleCaption, okCaption = "Select", startingDirectory = startDirPath)[0])
        return fileName

    def abcButtonClick(self):
        abcpath = self.getOpenFile(titleCaption="select abc file", startDirPath = mel.eval('rman getvar RMSPROJ'), exrCaption = ".abc")
        self.ui.abcPathLineEdit.setText(abcpath)

    def zennButtonClick(self):
        zennpath = self.getOpenDirectory(titleCaption="select zenn strand directory", startDirPath = mel.eval('rman getvar RMSPROJ'))
        self.ui.zennPathLineEdit.setText(zennpath )

    def loadButtonClick(self):
        isRmanLoaded = cmds.pluginInfo('RenderMan_for_Maya', q=True, l=True)
        if not isRmanLoaded:
            cmds.loadPlugin('RenderMan_for_Maya')

        abcpath         = self.ui.abcPathLineEdit.text()
        zenncachepath   = self.ui.zennPathLineEdit.text()

        rmanOutputProceduralName = 'rmanOutputZN_StrandsArchiveRigidBindingProcedrual'


        #root = cmds.createNode('transform', name=os.path.basename(zenncachepath))
        root = cmds.createNode('transform', name='strandsGroup_%s' % os.path.basename(zenncachepath))


        for zenncachename in os.listdir(zenncachepath):
            node = cmds.createNode('ZN_StrandsArchive')
            cmds.connectAttr('time1.outTime', '%s.inTime' % node)
            cmds.addAttr(node, dt='string', ln='rman__torattr___preShapeScript')
            cmds.setAttr('%s.rman__torattr___preShapeScript' % node, rmanOutputProceduralName, type='string')
            cmds.setAttr('%s.inAbcCachePath' % node, abcpath, type='string')
            cmds.setAttr('%s.inZennCachePath' % node, zenncachepath, type='string')
            cmds.setAttr('%s.inZennCacheName' % node, zenncachename, type='string')

            node    = cmds.rename(node, '%sShape' % zenncachename)
            parent  = cmds.listRelatives(node, p=True, f=True)[0]
            parent  = cmds.rename(parent, '%s' % zenncachename)
            cmds.parent(parent, root)