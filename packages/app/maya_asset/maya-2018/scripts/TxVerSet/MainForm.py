#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2019.01.15'
__comment__ = 'Texture Version Setup'
__windowName__ = "Tx Version Setup"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds

from .TxVerSetUI import Ui_Form

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore

import os

def getMayaWindow():
    '''
    get Maya Window Process
    :return: Maya window Process
    '''
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QMainWindow)
    except:
        return None

class TextureVersionSetup(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        # Scene Setup
        # project auto setup
        scenePath = cmds.file(q=True, sn=True)
        if scenePath == "":
            scenePath = cmds.workspace(q=True, rd=True)
        self.showDir, self.showName = GetProjectPath(maya=scenePath)

        self.ui.showDirEdit.setText(self.showDir)
        splitPath = scenePath.split("/")

        if "show" in splitPath:
            showIndex = splitPath.index("show")
            assetName = splitPath[showIndex +  4]
            self.ui.assetNameEdit.setText(assetName)

        self.ui.elementCheckBox.stateChanged.connect(self.elementStateChanged)
        self.ui.execBtn.clicked.connect(self.execClicked)
        self.ui.assetNameEdit.editingFinished.connect(self.assetNameEditFinished)

        self.assetNameEditFinished()
        self.elementStateChanged(False)

    def elementStateChanged(self, state):
        self.ui.elementLabel.setVisible(state)
        self.ui.elementEdit.setVisible(state)

    def execClicked(self):
        # TODO

        pathJoin = self.ui.showDirEdit.text()
        if not os.path.exists(pathJoin):
            cmds.confirmDialog(m="not exists showDir", title='Warning', icon='warning', b=['ok'])
            return

        assetName = self.ui.assetNameEdit.text()
        pathJoin = os.path.join(pathJoin, "asset", assetName)

        if not os.path.exists(pathJoin):
            cmds.confirmDialog(m="not exists asset %s" % assetName, title='Warning', icon='warning', b=['ok'])
            return

        elementName = ""
        if self.ui.elementCheckBox.isChecked():
            elementName = self.ui.elementEdit.text()
            pathJoin = os.path.join(pathJoin, "element", elementName)
            if not os.path.exists(pathJoin):
                cmds.confirmDialog(m="not exists element %s" % elementName, title='Warning', icon='warning', b=['ok'])
                return

        modelVersion = self.ui.modelVersionEdit.text()
        modelPath = os.path.join(pathJoin, "model", modelVersion)
        if not os.path.exists(modelPath):
            cmds.confirmDialog(m="not exists modelVersion %s" % modelVersion, title='Warning', icon='warning', b=['ok'])
            return

        textureVersion = self.ui.textureVersionEdit.text()
        texturePath = os.path.join(pathJoin, "texture", "tex", textureVersion)
        if not os.path.exists(texturePath):
            cmds.confirmDialog(m="not exists textureVersion %s" % texturePath, title='Warning', icon='warning', b=['ok'])
            return

        # dxsUsd.MakeTexAttr(txPath=texturePath, modelVersion = modelVersion).doIt()
        AttrExport(txPath=txPath)
        cmds.confirmDialog(m="success model(%s) <=> texture(%s)" % (modelVersion, textureVersion), title='Success Texture Version Setup', icon='information', b=['ok'])

    def assetNameEditFinished(self):
        assetName = self.ui.assetNameEdit.text()
        if not assetName:
            return

        modelPath = os.path.join(self.ui.showDirEdit.text(), "asset", assetName, "model")
        modelVersion = GetVersion(modelPath, False)
        self.ui.modelVersionEdit.setText(modelVersion)

        texturePath = os.path.join(self.ui.showDirEdit.text(), "asset", assetName, "texture", "tex")
        textureVersion = GetVersion(texturePath, False)
        self.ui.textureVersionEdit.setText(textureVersion)


def GetProjectPath(show=None, maya=None):
    '''
    Output directory compute by pathRule.json
    Args:
        show (str): showName
        maya (str): maya filePath
    Returns:
        showDir (str):
        showName(str):
    '''
    showDir = None; showName = None;
    if show:
        if "/show/" in show:
            showDir = show
        else:
            showDir = '/show/{NAME}'.format(NAME=show)
        showDir = GetOutShowDir(showDir)
        showName= show.replace('_pub', '')

    if maya:
        maya = maya.replace('/netapp/dexter/show', '/show')
        maya = maya.replace('/space/dexter/show', '/show')
        if maya.find('/show/') > -1:
            splitPath = maya.split('/')
            showIndex = splitPath.index('show')
            showName  = splitPath[showIndex+1]
            showDir   = string.join(splitPath[:showIndex+2], '/')
            showDir = GetOutShowDir(showDir)
            showName= showName.replace('_pub', '')
        elif maya.find('/assetlib/') > -1:
            splitPath = maya.split('/')
            showDir = '/assetlib/3D'
            showName= '3D'
    return str(showDir), str(showName)



def GetVersion(dirPath, overWrite=True):
    if not os.path.exists(dirPath):
        return 'v001'

    last = GetLastVersion(dirPath)
    if overWrite:
        return 'v%03d' % (int(last[1:]) + 1)
    else:
        return last

def AttrExport(txPath):
    import DXUSD.Tweakers as twk
    arg = twk.ATexture()
    arg.texAttrDir = txPath
    if arg.Treat():
        TT = twk.Texture(arg)
        TT.DoIt()

def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = TextureVersionSetup()
    # app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    window.setObjectName(__windowName__)
    window.show()
