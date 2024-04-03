# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys
import os
import site
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
WORKSPACEFOLDER = os.path.join( CURRENTPATH, "workspace" )
from PySide2 import QtWidgets, QtCore

import maya.cmds as cmds
from spanner2_ui_addWorkCode import AddWorkCode_Ui_Form
from shutil import copyfile
import json
import getpass

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# ADD Directory
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
def addPrevAsset(titleShort='', assetType=''):
# new asset ui open : text input
    try:
        text = newAssetUI()
        addPath = '/show/%s/prev/asset/%s/%s' % (titleShort, assetType, text)
        print addPath
        if not text == None:
            if not os.path.exists(addPath):
                os.makedirs(addPath)
                print 'created successfully'
            else:
                existErrorUI()
    except: pass

def addAssetDir(titleShort=''):
# add asset type dir
    path = ['cam', 'char', 'effect', 'env', 'matt', 'prop', 'vehicle']
    addPath = '/show/%s/prev/asset/' % (titleShort)
    print addPath
    if not os.path.exists(addPath):
        os.makedirs(addPath)

    if not os.listdir(addPath):
        for i in path:
            makeDir = addPath + i
            if not os.path.exists(makeDir):
                os.makedirs(makeDir)
            else:
                break

def addAssetWorkDir(titleShort='', assetType='', assetName=''):
# new asset workcode dir
    path = ['ani', 'model', 'rig', 'texture']
    if assetType == "env":
        path = ['model', 'texture']
    elif assetType == "prop":
        path = ['model', 'rig', 'texture']

    addPath = '/show/%s/prev/asset/%s/%s/' % (titleShort, assetType, assetName)
    if not os.listdir(addPath):
        for i in path:
            makeDir = addPath + i
            print "makedir=", makeDir
            if not os.path.exists(makeDir):
                os.makedirs(makeDir)
                createWorkSpace(i, makeDir)
            else:
                break

def addPrevShot(titleShort='', shotType='', shotName=''):
# new shot ui open : text input
    try:
        text = newShotUI()
        addPath = '/show/%s/prev/shot/%s/%s/%s' % (titleShort, shotType, shotName, text)
        if not text == None:
            if not os.path.exists(addPath):
                os.makedirs(addPath)
                print 'created successfully'
            else:
                existErrorUI()
    except: pass

def addShotDir(titleShort='', shotType=''):
# new shot type dir
    path = 'edit'
    addPath = '/show/%s/prev/shot/%s/' % (titleShort, shotType)
    makeDir =  addPath + path
    if not os.path.exists(makeDir):
        os.makedirs(makeDir)

def addShotNameDir(titleShort='', shotType='', shotName=''):
# new shot workcode dir
    path = ['prev', 'tech', 'output', 'reference', 'post']
    addPath = '/show/%s/prev/shot/%s/%s/' % (titleShort, shotType, shotName)
    for i in path:
        makeDir = addPath + i
        if not os.path.exists(makeDir) and shotName.find('_') > 0:
            os.makedirs(makeDir)

def newAssetUI():
    gd = GetDirNameUI()
    gd.label.setText('new asset name or type\n (no spaces or special characters):')
    gd.setWindowTitle('new asset')
    gd.show()
    gd.exec_()
    return gd.text

def newShotUI():
    gd = GetDirNameUI()
    gd.label.setText('new shot name or type\n (no spaces or special characters):')
    gd.setWindowTitle('new shot')
    gd.show()
    gd.exec_()
    return gd.text

def newWorkCodeUI(filePath):
    dialog = NewWorkCodeDialog(QtWidgets.QDialog)
    dialog.show()
    result = dialog.exec_()
    if result == 1:
        addDir = os.path.join(filePath, dialog.item)
        print addDir
        if not os.path.exists(addDir):
            wokrCode = dialog.item
            createWorkSpace(wokrCode, filePath)
        else:
            result = existErrorUI()
            if result == 1:
                wokrCode = dialog.item
                createWorkSpace(wokrCode, filePath)
            else: pass

def createWorkSpace(workCode, filePath):
# workcode ani, rig, model... add json file
    workcodeFile = os.path.join( CURRENTPATH, 'resource/shotWorkcode.json')
    myFile = open(workcodeFile, "r")
    DirDic = json.load(myFile)
    myFile.close()

    # get folder list
    devList = DirDic[workCode]['dev']
    pubList = DirDic[workCode]['pub']
    filePath = '/'.join(filePath.split('/')[0:7])
    devBase = filePath + '/' + workCode + '/dev/'
    pubBase = filePath + '/' + workCode + '/pub/'
    # create folders
    for devFolder in devList:
        if workCode == 'texture':
            pass
        else:
            devPath = devBase + devFolder
            if not os.path.isdir(devPath):
                os.makedirs(devPath)

    for pubFolder in pubList:
        pubPath =  pubBase + pubFolder
        if not os.path.isdir(pubPath):
            os.makedirs(pubPath)

    # copy workspace.mel
    if not workCode == 'texture':
        workspaceFile = os.path.join(str(devBase), "workspace.mel")
        source_workspace = os.path.join(WORKSPACEFOLDER, workCode + ".mel")
        copyfile(source_workspace, workspaceFile)
        print 'created successfully'

def createFirstScene(filePath):
# shot, asset dev first scene create
    filePath = '/'.join(filePath.split('/')[0:8])
    devPath = filePath + '/dev/scenes'
    pubPath = filePath + '/pub/scenes'
    pathTempList = []
    if os.path.isdir(devPath):#####current dir mb exists
        for pathTemp in os.listdir(devPath):
            print 'pathTemp', pathTemp
            if pathTemp.find('.mb') > 0:
                pathTempList.append(pathTemp)
    if os.path.isdir(pubPath):  #####current dir mb exists
        for pathTemp in os.listdir(pubPath):
            if pathTemp.find('.mb') > 0:
                pathTempList.append(pathTemp)
    if len(pathTempList) == 0:
        taskName = filePath.split('/')[6]
        workCode = filePath.split('/')[7]

        if '/prev/asset' in filePath:###if asset or shot
            newMaya = '/%s_%s_v01_w01.mb' % (taskName, workCode)
        else:
            newMaya = '/%s_v01_w01.mb' % (taskName)
        if not os.path.exists(devPath):
            os.makedirs(devPath)
        openMessageBox(devPath, newMaya)# save or open

def openMessageBox(devPath=None, newMaya=None):
# current scene or new scene first scene window
    messageBox = QtWidgets.QMessageBox()
    messageBox.setText(
        "You are creating first scene.\n Would you like to start with a new scene, or currently open scene?")
    messageBox.setWindowModality(QtCore.Qt.WindowModal)
    messageBox.setIcon(QtWidgets.QMessageBox.Question)
    newSceneButton = messageBox.addButton('New Scene', QtWidgets.QMessageBox.AcceptRole)
    currentSceneButton = messageBox.addButton('Current Scene', QtWidgets.QMessageBox.AcceptRole)
    messageBox.setDefaultButton(currentSceneButton)
    messageBox.addButton(QtWidgets.QMessageBox.Cancel)
    messageBox.exec_()

    if messageBox.clickedButton() == newSceneButton:
        cmds.file(new=True, force=True, iv=True)
        cmds.file(rename=devPath + newMaya)
        cmds.file(save=True, type='mayaBinary')
    elif messageBox.clickedButton() == currentSceneButton:
        cmds.file(rename=devPath + newMaya)
        cmds.file(save=True, type='mayaBinary')
    elif messageBox.clickedButton() == messageBox.button(QtWidgets.QMessageBox.Cancel):
        return False

# show error if dir already exists.
def existErrorUI():
    dialog = DirectoryErrorDialog()
    dialog.show()
    result = dialog.exec_()
    return result

class GetDirNameUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        self.label = QtWidgets.QLabel("Type Directory Name:")
        self.commentBox = QtWidgets.QLineEdit()
        self.yes_btn = QtWidgets.QPushButton("OK")
        self.close_btn = QtWidgets.QPushButton("Cancel")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.yes_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.commentBox)
        layout.addLayout(layout2,3,0)
        self.setLayout(layout)
        self.setWindowTitle("New Directory")

        # connection
        self.close_btn.clicked.connect(self.reject)
        self.yes_btn.clicked.connect(self.result)

    def result(self):
        self.text = self.commentBox.text()
        self.accept()

# make workCode Dialog
class NewWorkCodeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = AddWorkCode_Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('Add WorkCode')

        items = ['ani', 'model', 'rig', 'texture', 'fx', 'mocap']
        items.sort()
        self.ui.Workcode_listWidget.addItems(items)
        self.ui.buttonBox.accepted.connect(self.result)
        self.ui.buttonBox.rejected.connect(self.reject)
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)

    def result(self):
        self.item = self.ui.Workcode_listWidget.currentItem().text()
        self.accept()

class DirectoryErrorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("Directory Already Exists.\nDo you want to continue?")
        self.yes_btn = QtWidgets.QPushButton("Yes")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.yes_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2,5,0)
        self.setLayout(layout)
        self.setWindowTitle("Error")
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        #connection
        self.yes_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
