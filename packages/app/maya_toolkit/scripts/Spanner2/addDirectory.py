# -*- coding: utf-8 -*-
import sys
import os
import site
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
WORKSPACEFOLDER = os.path.join( CURRENTPATH, "workspace" )
# import Qt
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
import maya.cmds as cmds
from spanner2_ui_addWorkCode import AddWorkCode_Ui_Form
from shutil import copyfile
import json
from dxstats import inc_tool_by_user
import getpass

def addAsset(titleShort='', assetType='', assetName=''):
    """
    create asset folder
    :param titleShort: short name of the title
    :param assetType: current asset type selected
    """
    try:
        text = newAssetUI()
        addPath = '/show/%s/asset/%s/%s' % (titleShort, assetType, text)
        if not text == None:
            if not os.path.exists(addPath):
                os.makedirs(addPath)
                print 'created successfully'
            else:
                existErrorUI()
    except: pass

def addShot(titleShort='', shotType='', shotName=''):
    """
    create shot folder
    :param titleShort: short name of the title
    :param assetType: current shot type selected
    """
    try:
        text = newShotUI()
        addPath = '/show/%s/shot/%s/%s' % (titleShort, shotType, text)
        if not text == None:
            if not os.path.exists(addPath):
                os.makedirs(addPath)
                print 'created successfully'
            else:
                existErrorUI()
    except: pass

# show ui to get asset type name
def newAssetTypeUI():
    gd = GetDirNameUI()
    gd.label.setText('new asset type name\n (no spaces or special characters):')
    gd.setWindowTitle('new asset type')
    gd.show()
    gd.exec_()
    return gd.text

# show ui to get sset name
def newAssetUI():
    gd = GetDirNameUI()
    gd.label.setText('new asset name\n (no spaces or special characters):')
    gd.setWindowTitle('new asset')
    gd.show()
    gd.exec_()
    return gd.text

def newShotTypeUI():
    gd = GetDirNameUI()
    gd.label.setText('shot type name\n (no spaces or special characters):')
    gd.setWindowTitle('new shot type')
    gd.show()
    gd.exec_()
    return gd.text

def newShotUI():
    gd = GetDirNameUI()
    gd.label.setText('shot type name\n (no spaces or special characters):')
    gd.setWindowTitle('new shot')
    gd.show()
    gd.exec_()
    return gd.text

# show ui to get workcode name
def newWorkCodeUI(filePath):
    dialog = NewWorkCodeDialog(QtWidgets.QDialog)
    dialog.show()
    result = dialog.exec_()
    if result == 1:
        addDir = os.path.join(filePath, dialog.item)
        print addDir
        if not os.path.exists(addDir):
            createWorkSpace(dialog.item, filePath)
        else:
            result = existErrorUI()
            if result == 1:
                createWorkSpace(dialog.item, filePath)
            else: pass

def createWorkSpace(workCode, filePath):
    """
    create workspace under task name
    str :param workCode: current workcode name
    str :param filePath: current file path on the file path line edit widget.
    """
    inc_tool_by_user.run('action.Spanner2.makeWorkSpace', getpass.getuser())
    workcodeFile = os.path.join( CURRENTPATH, 'resource/shotWorkcode.json' )
    myFile = open(workcodeFile, "r")
    DirDic = json.load(myFile)
    myFile.close()

    # get folder list
    devList = DirDic[workCode]['dev']
    pubList = DirDic[workCode]['pub']
    if '/prev' in filePath:
        filePath = '/'.join(filePath.split('/')[0:7])
    else:
        filePath = '/'.join(filePath.split('/')[0:6])

    devBase = filePath + '/' + workCode + '/dev/'
    # create folders
    for devFolder in devList:
        if workCode == 'texture':
            # dev
            devPath = devBase + devFolder
            if not os.path.isdir(devPath):
                os.makedirs(devPath)
            # ldv
            devPath = filePath + '/' + workCode + '/ldv/' + devFolder
            if not os.path.isdir(devPath):
                os.makedirs(devPath)
        else:
            devPath = devBase + devFolder
            if not os.path.isdir(devPath):
                os.makedirs(devPath)

    for pubFolder in pubList:
        pubPath = filePath + '/' + workCode + '/pub/' + pubFolder
        if not os.path.isdir(pubPath):
            os.makedirs(pubPath)

    # copy workspace.mel
    workspaceFile = os.path.join(str(devBase), "workspace.mel")
    source_workspace = os.path.join(WORKSPACEFOLDER, workCode + ".mel")
    copyfile(source_workspace, workspaceFile)
    print 'created successfully'

def createFirstScene(filePath):
    """
    create scene if there is no file on the dev path.
    str :param filePath: current file path on the file path line edit widget.
    """
    inc_tool_by_user.run('action.Spanner2.createFirstScene', getpass.getuser())
    filePath = '/'.join(filePath.split('/')[0:7])
    devPath = filePath + '/dev/scenes'
    pubPath = filePath + '/pub/scenes'
    pathTempList = []
    if os.path.isdir(devPath and pubPath):
        for pathTemp in os.listdir(devPath):
            if pathTemp.split('.')[-1] == 'mb':
                pathTempList.append(pathTemp)
        for pathTemp in os.listdir(pubPath):
            if pathTemp.split('.')[-1] == 'mb':
                pathTempList.append(pathTemp)

    if len(pathTempList) == 0:
        taskName = filePath.split('/')[5]
        workCode = filePath.split('/')[6]
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
        if not os.path.exists(devPath):
            os.makedirs(devPath)

        if messageBox.clickedButton() == newSceneButton:
            cmds.file(new=True, force=True)
            cmds.file(rename= devPath + '/%s_%s_v01_w01.mb'%(taskName,workCode))
            cmds.file(save=True, type='mayaBinary')

        if messageBox.clickedButton() == currentSceneButton:
            cmds.file(rename= devPath + '/%s_%s_v01_w01.mb'%(taskName,workCode))
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

        items = ['ani', 'cloth', 'finalize', 'fx', 'hair', 'hairSim', 'layout', 'lighting', 'matchmove', 'mocap',
                 'model', 'rig', 'texture']
        items.sort()
        self.ui.Workcode_listWidget.addItems(items)
        self.ui.buttonBox.accepted.connect(self.result)
        self.ui.buttonBox.rejected.connect(self.reject)
        font = QtGui.QFont()
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
        font = QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        #connection
        self.yes_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)

class DirectoryWarning(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("Directory not Exists.\nDo you want to create folder?")
        self.yes_btn = QtWidgets.QPushButton("Yes")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.yes_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2,5,0)
        self.setLayout(layout)
        self.setWindowTitle("Warning")
        font = QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        #connection
        self.yes_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
