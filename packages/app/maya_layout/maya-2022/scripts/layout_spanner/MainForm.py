# -*- coding: utf-8 -*-
import sys, os, subprocess, site, glob, requests, operator
from datetime import datetime
import dxConfig
import getpass
from mayaActions import SaveDevForm
from mayaActions import SavePubForm
from mayaActions import MayaActions

from PySide2 import QtWidgets, QtCore, QtGui

from pymongo import MongoClient
import dxConfig
import historyAction
import addDirectory

DB_IP = dxConfig.getConf('DB_IP')
SITE = dxConfig.getHouse()
WORKING = ['Waiting', 'In-Progress', 'Ready', 'Retake', 'Review', 'OK']
NONWORKING = ['Approved', 'Omit', 'Hold']
CURRENTPATH = os.path.dirname(os.path.abspath(__file__))
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

# import ui
import spanner_ui
reload(spanner_ui)
from spanner_ui import Ui_Form


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.startSet()

    def startSet(self):
        self.SnapShotClass = historyAction.SnapShots()
        self.CommentClass = historyAction.CommentDB()
        self.setWindowTitle('Layout Team Spanner')
        self.fileName = None
        self.filePath = None
        self.getUserName()
        self.projectDic = {}
        self.memberDic = {}
        self.taskDic = {}

        self.showComboBox()
        self.showTeamComboBox()
        self.setImage()
        try:
            self.getUserInfo()
        except Exception as e:
            print e.message
        self.connections()
        self.ui.team_comboBox.setMinimumWidth(180)
        self.setTaskUser()

    def setButtonImage(self, button, image):
        button.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/%s.png" % image))))

    def setPixmapImage(self, label, imagePath, width, height):
        image = QtGui.QPixmap(imagePath)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        label.setPixmap(image)

    # set default image for ui
    def setImage(self):
        # set pixmap
        self.setPixmapImage(self.ui.layout_label, os.path.join(CURRENTPATH, 'resource/laytitle.png'), 1030, 30)
        self.setPixmapImage(self.ui.snapshot_label, os.path.join(CURRENTPATH, 'resource/noPreview.jpg'), 355, 200)
        # set button image
        self.setButtonImage(self.ui.snapShot_pushButton, 'camera')
        self.setButtonImage(self.ui.openPath_btn, 'open')
        self.setButtonImage(self.ui.update_btn, 'reset')
        self.setButtonImage(self.ui.saveNote_btn, 'note')


        # styleSheet
        self.ui.show_comboBox.setView(QtWidgets.QListView())
        self.ui.show_comboBox.setStyleSheet('''
        QComboBox QAbstractItemView::item { min-height: 25px; min-width: 120px;}
        QComboBox QAbstractItemView { font-size: 10pt;}
        QComboBox QAbstractItemView::item:selected{ background:rgb(208, 139, 0);}''')
        self.ui.team_comboBox.setView(QtWidgets.QListView())
        self.ui.team_comboBox.setStyleSheet('''
        QComboBox QAbstractItemView::item { min-height: 25px; min-width: 120px;}
        QComboBox QAbstractItemView { font-size: 10pt;}
        QComboBox QAbstractItemView::item:selected{ background:rgb(208, 139, 0);}''')
        self.ui.name_comboBox.setView(QtWidgets.QListView())
        self.ui.name_comboBox.setStyleSheet('''
        QComboBox QAbstractItemView::item { min-height: 25px; min-width: 120px;}
        QComboBox QAbstractItemView { font-size: 10pt;}
        QComboBox QAbstractItemView::item:selected{ background:rgb(208, 139, 0);}''')

    # connect ui and modules
    def connections(self):
        # saveDevPub window
        self.ui.saveDev_btn.clicked.connect(self.openSaveDevWindow)
        self.ui.savePub_btn.clicked.connect(self.openSavePubWindow)
        self.ui.snapShot_pushButton.clicked.connect(self.takeSnapShot)
        self.ui.saveNote_btn.clicked.connect(self.saveComment)

        # open folder
        self.ui.openPath_btn.clicked.connect(self.openPath)
        self.ui.assetOpenDevPreview_btn.clicked.connect(self.openPath)
        self.ui.assetOpenPubPreview_btn.clicked.connect(self.openPath)
        self.ui.shotOpenDevPreview_btn.clicked.connect(self.openPath)
        self.ui.shotOpenPubPreview_btn.clicked.connect(self.openPath)

        # current scene path
        self.ui.update_btn.clicked.connect(self.updateSceneName)
        self.ui.show_comboBox.currentIndexChanged.connect(self.currentshowPath)

        # Previs asset Tab
        self.ui.prev_assetType_listWidget.itemClicked.connect(self.showPrevAsset)
        self.ui.prev_asset_listWidget.itemClicked.connect(self.showPrevWorkCode)
        self.ui.prev_workCode_listWidget.itemClicked.connect(self.showPrevAssetDevPub)
        self.ui.prev_assetDev_listWidget.itemPressed.connect(self.readHistory)
        self.ui.prev_assetPub_listWidget.itemPressed.connect(self.readHistory)
        self.ui.prev_addAssetType_btn.clicked.connect(self.addAsset)
        self.ui.prev_addAssetName_btn.clicked.connect(self.addAsset)
        self.ui.prev_addWorkCode_btn.clicked.connect(self.addAsset)
        self.ui.prev_workCode_listWidget.doubleClicked.connect(self.createFirstScene)
        self.ui.prev_assetDev_listWidget.doubleClicked.connect(self.openMaya)
        self.ui.prev_assetPub_listWidget.doubleClicked.connect(self.openMaya)

        # Previs shot Tab
        self.ui.prev_shotType_listWidget.itemClicked.connect(self.showPrevShot)
        self.ui.prev_seq_listWidget.itemClicked.connect(self.showPrevShotWorkCode)
        self.ui.prev_shot_listWidget.itemClicked.connect(self.showPrevShotDevPub)
        self.ui.prev_shotDev_listWidget.itemPressed.connect(self.readHistory)
        self.ui.prev_shotPub_listWidget.itemPressed.connect(self.readHistory)
        self.ui.prev_addShotType_btn.clicked.connect(self.addShot)
        self.ui.prev_addSeq_btn.clicked.connect(self.addShot)
        self.ui.prev_addShot_btn.clicked.connect(self.addShot)
        self.ui.prev_shot_listWidget.doubleClicked.connect(self.createFirstScene)
        self.ui.prev_shotDev_listWidget.itemDoubleClicked.connect(self.openMaya)
        self.ui.prev_shotPub_listWidget.itemDoubleClicked.connect(self.openMaya)

        # shot and asset find linetext
        self.ui.prev_findAsset_lineEdit.textChanged.connect(self.findAsset)
        self.ui.prev_findShot_lineEdit.textChanged.connect(self.findShot)

        # My Task Context Menu
        self.ui.myAsset_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.myShotName_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.nonMyAsset_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.nonMyShot_listWidget.itemPressed.connect(self.getTaskType)

        # My Task shot, asset Tab move
        self.ui.team_comboBox.currentIndexChanged.connect(self.showTeamMemberComboBox)
        self.ui.name_comboBox.currentIndexChanged.connect(self.currentName)
        self.ui.myAsset_listWidget.itemDoubleClicked.connect(self.assetQuick)
        self.ui.nonMyAsset_listWidget.itemDoubleClicked.connect(self.assetQuick)
        self.ui.myShotName_listWidget.itemDoubleClicked.connect(self.shotQuick)
        self.ui.nonMyShot_listWidget.itemDoubleClicked.connect(self.shotQuick)

        # right click action
        self.ui.prev_assetDev_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.prev_assetDev_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.prev_assetPub_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.prev_assetPub_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.prev_shotDev_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.prev_shotDev_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.prev_shotPub_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.prev_shotPub_listWidget.customContextMenuRequested.connect(self.contextMenu)

    def createFirstScene(self):
    # create first scene if there is none in the scene folder.
        addDirectory.createFirstScene(self.filePath)
        if self.ui.task_tab.currentIndex() == 0:
            self.showPrevAssetDevPub()
        if self.ui.task_tab.currentIndex() == 1:
            self.showPrevShotDevPub()

    def addShot(self):
    # create shot folder.
        if self.sender() == self.ui.prev_addShotType_btn:
            addDirectory.addPrevShot(self.titleShort)
            self.showPrevShotType()
        if self.sender() == self.ui.prev_addSeq_btn:
            addDirectory.addShotDir(self.titleShort, self.prev_shotType)
            addDirectory.addPrevShot(self.titleShort, self.prev_shotType)
            self.showPrevShot()
        if self.sender() == self.ui.prev_addShot_btn:
            addDirectory.addShotNameDir(self.titleShort, self.prev_shotType, self.prev_seq)
            addDirectory.addPrevShot(self.titleShort, self.prev_shotType, self.prev_seq)
            self.showPrevShotWorkCode()

    def addAsset(self):
    # create asset folder
        if self.sender() == self.ui.prev_addAssetType_btn:
            addDirectory.addAssetDir(self.titleShort)
            addDirectory.addPrevAsset(self.titleShort)
            self.showPrevAssetType()
        if self.sender() == self.ui.prev_addAssetName_btn:
            addDirectory.addPrevAsset(self.titleShort, self.prev_assetType)
            self.showPrevAsset()
        if self.sender() == self.ui.prev_addWorkCode_btn:
            addDirectory.addAssetWorkDir(self.titleShort, self.prev_assetType, self.prev_assetName)
            addDirectory.newWorkCodeUI(self.filePath)
            self.showPrevWorkCode()

    def openMaya(self):
    # open maya scene by the fileName. >> maya scene dbclicked save or open ?
        loadpath = cmds.file(q=True, sn=True)
        openfile = os.path.join(self.filePath, self.fileName)
        title = "Warning : Scene Save or Not Save"
        if loadpath:
            txt = "Save Changes to \"%s\" Scene?"%loadpath
        else:
            txt = "Save Changes to None Scene?"

        ow = OpenWaringDialog(title, txt)
        ow.exec_()

        if ow.result == 'Save':
            filePath = os.path.dirname(cmds.file(sn=1, q=1))
            fileName = cmds.fileDialog2(ds=2, startingDirectory=filePath,
                                        fileFilter="Maya Files (*.ma *.mb)")
            if fileName:
                cmds.file(rename=str(fileName[0]))
                cmds.file(save=True, type='mayaBinary')
        elif ow.result == 'Close':
            return

        cmds.file(openfile, open=True, force=True, iv=True)
        try:
            mel.eval('setProject "%s"' % (self.filePath.split('/scenes')[0]))
        except:
            print 'failed'

    def contextMenu(self, pos):
    # right click action on dev list widgets and pub list widgets.
    # param pos: get current position of the cursor.
        pos = pos + (QtCore.QPoint(20, 0))
        menu = QtWidgets.QMenu()
        ma = MayaActions(self.filePath, self.fileName)
        self.getRigMalfunction()

        if not self.fileName in self.rigMalFunctionFile:
            action2 = menu.addAction('Reference')
            action2.triggered.connect(ma.referenceAct)
            action3 = menu.addAction('Multi Reference')
            action3.triggered.connect(ma.multiReferenceAct)
            menu.addSeparator()
            action4 = menu.addAction('Import')
            action4.triggered.connect(ma.importAct)
            action5 = menu.addAction('Import (nameSpace:)')
            action5.triggered.connect(ma.importNSAct)
            action6 = menu.addAction('Multi Import (nameSpace:)')
            action6.triggered.connect(ma.multiImportAct)

        menu.exec_(self.focusWidget().mapToGlobal(pos))
#
    def openSaveDevWindow(self):
    # open dialog for save dev file
        dw = SaveDevForm(QtWidgets.QDialog, self.fileName, self.filePath)
        dw.show()
        dw.exec_()
        # reload state
        self.updateSceneName()

    def openSavePubWindow(self):
    # open dialog for save pub file/ publish.
        pw = SavePubForm(QtWidgets.QDialog, self.fileName, self.filePath)
        pw.show()
        pw.exec_()
        # reload state
        self.updateSceneName()

    def getUserName(self):
    # get name of current user.
        self.userName = getpass.getuser()
        self.ui.name_label.setText(self.userName)

    def getUserInfo(self):
    # get user info when app starts
        self.ui.show_comboBox.setCurrentIndex(0)
        self.currentshowPath()

        # view team member
        self.showTeamMemberComboBox()
        # view current task
        self.ui.name_comboBox.setCurrentIndex(0)
        name = unicode(self.ui.name_comboBox.currentText())
        for i in self.memberDic:
            if name == i:
                name_eng = self.memberDic[i]['code']
        self.currentTaskAssigned(name_eng)

    def openPath(self):
    # open nautilus current file path.
        if self.sender().objectName() == 'openPath_btn':
            previewPath = self.ui.filePath_lineEdit.text()
        elif self.sender().objectName() == 'assetOpenDevPreview_btn' or self.sender().objectName() == 'shotOpenDevPreview_btn':
            previewPath = '/'.join(self.filePath.split('/')[0:8]) + '/dev/preview'
        else:
            previewPath = '/'.join(self.filePath.split('/')[0:8]) + '/pub/preview'

        if os.path.isdir(previewPath):
            subprocess.Popen(['xdg-open', str(previewPath)])

    def getProjectDict(self):
    # show project list
        projectDic = {}
        params = {}
        params['api_key'] = API_KEY
        params['category'] = 'Active'
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        exceptList = ['test', 'testshot']
        for i in infos:
            if i['name'] in exceptList:
                pass
            else:
                showname = '/show/%s'%i['name']
                if os.path.exists(showname):###current exists project
                    projectDic[i['title']] = i
        return projectDic

    def showComboBox(self):
    # get show name from Tactic and append them to the show comboBox.
        self.projectDic = self.getProjectDict()
        projects = self.projectDic.keys()
        projects.sort()
        self.ui.show_comboBox.addItems(projects)

    def currentshowPath(self):
    # show asset type and shot type from the show.
        title = unicode(self.ui.show_comboBox.currentText())
        self.titleShort = self.projectDic[title]['name']
        print 'title ',self.titleShort
        oldshot = 'shotPrv'
        newshot = 'shot'
        preshot = '/show/%s/prev' % str(self.titleShort)

        if oldshot in os.listdir(preshot):####directory exists
            self.prevpath = oldshot
        else:
            self.prevpath = newshot

        self.prevShotTypePath = '/show/%s/prev/%s' % (str(self.titleShort), self.prevpath)
        self.prevAssetTypePath = '/show/%s/prev/asset' % str(self.titleShort)
        # show list
        self.showPrevAssetType()
        self.showPrevShotType()
        # show path
        self.ui.filePath_lineEdit.setText(preshot)
        # get name
        name = unicode(self.ui.name_comboBox.currentText())
        self.showCode = self.projectDic[title]['code']
        self.currentName(name)

    def updateSceneName(self):
     # get path of currently opened maya scene and go to the file directly in the app.
        temp = cmds.file(q=True, sn=True)
        self.filePath = os.path.dirname(temp)
        self.fileName = os.path.basename(temp)
        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.setText(self.fileName)

        pathTemp = self.filePath.split('/')
        show = pathTemp[2]
        for i in self.projectDic:
            if show == self.projectDic[i]['name']:
                show = i
        assetShot = pathTemp[4]

        # find show index
        index = self.ui.show_comboBox.findText(show, QtCore.Qt.MatchContains)
        self.ui.show_comboBox.setCurrentIndex(index)

        if assetShot == 'asset':# asset
            self.ui.task_tab.setCurrentIndex(0)
            self.assetType = pathTemp[5]
            self.asset = pathTemp[6]
            self.workCode = pathTemp[7]
            devPub = pathTemp[8]
            # find asset type
            item = self.ui.prev_assetType_listWidget.findItems(self.prev_assetType, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_assetType_listWidget.setCurrentItem(item)
            # find asset
            self.showPrevAsset()
            item = self.ui.prev_asset_listWidget.findItems(self.prev_assetName, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_asset_listWidget.setCurrentItem(item)
            # find asset work code
            self.showPrevWorkCode()
            item = self.ui.prev_workCode_listWidget.findItems(self.prev_workCode, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_workCode_listWidget.setCurrentItem(item)
            #find from dev/pub list
            self.showPrevAssetDevPub()
            if devPub == 'dev':
                item = self.ui.prev_assetDev_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.prev_assetDev_listWidget.setCurrentItem(item)
                self.ui.prev_assetDev_listWidget.setFocus()
                self.readHistory(item)
            else:
                item = self.ui.prev_assetPub_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.prev_assetPub_listWidget.setCurrentItem(item)
                self.ui.prev_assetPub_listWidget.setFocus()
                self.readHistory(item)

        if assetShot == self.prevpath:# shot
            self.ui.task_tab.setCurrentIndex(1)
            self.prev_shotType = pathTemp[5]
            self.prev_seq = pathTemp[6]
            self.prev_shot = pathTemp[7]
            devPub = pathTemp[8]
            # find shot type
            item = self.ui.prev_shotType_listWidget.findItems(self.prev_shotType, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_shotType_listWidget.setCurrentItem(item)
            # find shot
            self.showPrevShot()
            item = self.ui.prev_seq_listWidget.findItems(self.prev_seq, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_seq_listWidget.setCurrentItem(item)
            # find shot work code
            self.showPrevShotWorkCode()
            item = self.ui.prev_shot_listWidget.findItems(self.prev_shot, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_shot_listWidget.setCurrentItem(item)
            # find from dev/pub list
            self.showPrevShotDevPub()
            if devPub == 'dev':
                item = self.ui.prev_shotDev_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.prev_shotDev_listWidget.setCurrentItem(item)
                self.ui.prev_shotDev_listWidget.setFocus()
                self.readHistory(item)
            else:
                item = self.ui.prev_shotPub_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.prev_shotPub_listWidget.setCurrentItem(item)
                self.ui.prev_shotPub_listWidget.setFocus()
                self.readHistory(item)

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # ASSET Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def clearItem(self):
    # treewidget clear
        self.ui.fileName_lineEdit.clear()
        self.ui.history_treeWidget.clear()
        self.ui.comment_textEdit.clear()
        self.setPixmapImage(self.ui.snapshot_label, os.path.join(CURRENTPATH, 'resource/noPreview.jpg'), 355, 200)

    def assetClear(self):
    # asset dev pub listwidget clear
        self.ui.prev_assetDev_listWidget.clear()
        self.ui.prev_assetPub_listWidget.clear()

    def showPrevAssetType(self):
    # asset type list : char, effect, env, prop, ref, texture..
        self.ui.prev_addAssetName_btn.setEnabled(False)
        self.ui.prev_addWorkCode_btn.setEnabled(False)
        self.ui.prev_assetType_listWidget.clear()
        self.ui.prev_asset_listWidget.clear()
        self.ui.prev_workCode_listWidget.clear()
        self.ui.prev_findAsset_lineEdit.clear()
        self.ui.filePath_lineEdit.setText(self.prevAssetTypePath)
        self.assetClear()
        self.clearItem()
        # show dir
        assetDir = QtCore.QDir(self.prevAssetTypePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_assetType_listWidget)
            item.setDirName(os.path.join(self.prevAssetTypePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_assetType_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevAsset(self):
    # asset name list : prvCaract...
        self.ui.prev_addAssetName_btn.setEnabled(True)
        self.ui.prev_addWorkCode_btn.setEnabled(False)
        self.ui.prev_asset_listWidget.clear()
        self.ui.prev_workCode_listWidget.clear()
        self.ui.prev_findAsset_lineEdit.clear()
        self.assetClear()
        self.clearItem()
        self.prev_assetType = unicode(self.ui.prev_assetType_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/asset/%s' % (self.titleShort, self.prev_assetType)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_asset_listWidget)
            item.setDirName(os.path.join(self.filePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_asset_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevWorkCode(self):
    # asset workcode list : ani, model, rig, texture...
        self.ui.prev_addWorkCode_btn.setEnabled(True)
        self.ui.prev_workCode_listWidget.clear()
        self.assetClear()
        self.clearItem()
        self.prev_assetName = unicode(self.ui.prev_asset_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/asset/%s/%s' % (self.titleShort, self.prev_assetType, self.prev_assetName)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_workCode_listWidget)
            item.setDirName(os.path.join(self.filePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_workCode_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevAssetDevPub(self):
    # asset dev, pub list : /dev/scenes/ ###.mb
        self.assetClear()
        self.ui.fileName_lineEdit.clear()

        self.prev_workCode = unicode(self.ui.prev_workCode_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/asset/%s/%s/%s' \
                        % (self.titleShort, self.prev_assetType, self.prev_assetName, self.prev_workCode)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        if self.prevpath == 'shotPrv' and self.filePath.find('dev') > 0:
            devpath = self.filePath
        else:
            devpath = os.path.join(self.filePath, 'dev/scenes')
        if os.path.exists(devpath):
            for file in os.listdir(devpath):
                if os.path.splitext(file)[-1] == '.mb':
                    item = TaskItem(self.ui.prev_assetDev_listWidget)
                    item.setDirName(os.path.join(devpath, file))
                    item.setText(file)
            self.ui.prev_assetDev_listWidget.sortItems(QtCore.Qt.DescendingOrder)

        # show dir
        if self.prevpath == 'shotPrv' and self.filePath.find('pub') > 0:
            pubpath = self.filePath
        else:
            pubpath = os.path.join(self.filePath, 'pub/scenes')
        if os.path.exists(pubpath):
            for file in os.listdir(pubpath):
                if os.path.splitext(file)[-1] == '.mb':
                    item = TaskItem(self.ui.prev_assetPub_listWidget)
                    item.setDirName(os.path.join(pubpath, file))
                    item.setText(file)
            self.ui.prev_assetPub_listWidget.sortItems(QtCore.Qt.DescendingOrder)

        self.readDB()

    def findAsset(self):
    # find asset list by name
        if self.ui.task_tab.currentIndex() == 0:
            item = self.ui.prev_findAsset_lineEdit.text()
            try:
                item = self.ui.prev_asset_listWidget.findItems(item, QtCore.Qt.MatchContains)[0]
                self.ui.prev_asset_listWidget.setCurrentItem(item)
                self.showPrevWorkCode()
            except:
                pass

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # Shot Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def shotClear(self):
    # shot dev pub listwidget clear
        self.ui.prev_shotDev_listWidget.clear()
        self.ui.prev_shotPub_listWidget.clear()

    def showPrevShotType(self):
    # shot type list : sequence
        self.ui.prev_addSeq_btn.setEnabled(False)
        self.ui.prev_addShot_btn.setEnabled(False)
        self.ui.prev_shotType_listWidget.clear()
        self.ui.prev_seq_listWidget.clear()
        self.ui.prev_shot_listWidget.clear()
        self.shotClear()
        self.clearItem()
        self.ui.prev_findShot_lineEdit.clear()
        self.filePath = '/show/%s/prev/%s' % (self.titleShort, self.prevpath)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_shotType_listWidget)
            item.setDirName(os.path.join(self.prevShotTypePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_shotType_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevShot(self):
    # shot name list : shot
        self.ui.prev_addSeq_btn.setEnabled(True)
        self.ui.prev_addShot_btn.setEnabled(False)
        self.ui.prev_seq_listWidget.clear()
        self.ui.prev_shot_listWidget.clear()
        self.shotClear()
        self.clearItem()
        self.ui.prev_findShot_lineEdit.clear()
        self.prev_shotType = unicode(self.ui.prev_shotType_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/%s/%s' % (self.titleShort, self.prevpath, self.prev_shotType)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_seq_listWidget)
            item.setDirName(os.path.join(self.filePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_seq_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevShotWorkCode(self):
    # shot workcode list : output, post, prev, reference....
        self.oldtype = 0
        self.ui.prev_addShot_btn.setEnabled(True)
        self.ui.prev_shot_listWidget.clear()
        self.shotClear()
        self.clearItem()
        self.prev_seq = unicode(self.ui.prev_seq_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/%s/%s/%s' % (self.titleShort, self.prevpath, self.prev_shotType, self.prev_seq)
        self.ui.filePath_lineEdit.setText(self.filePath)
        # show dir
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.prev_shot_listWidget)
            item.setDirName(os.path.join(self.filePath, str(info.baseName())))
            item.setText(info.baseName())
        self.ui.prev_shot_listWidget.sortItems(QtCore.Qt.AscendingOrder)

    def showPrevShotDevPub(self):
    # shot dev, pub list : /dev/scenes/ ###.mb
        self.shotClear()
        self.ui.fileName_lineEdit.clear()
        self.prev_shot = unicode(self.ui.prev_shot_listWidget.currentItem().text())
        self.filePath = '/show/%s/prev/%s/%s/%s/%s' \
                        % (self.titleShort, self.prevpath, self.prev_shotType, self.prev_seq, self.prev_shot)
        self.ui.filePath_lineEdit.setText(self.filePath)

        # show dir (old show path add)
        if self.prevpath == 'shotPrv' and self.filePath.find('dev') > 0:
            devpath = self.filePath + '/scenes'
        else:
            devpath = os.path.join(self.filePath, 'dev/scenes')

        if os.path.exists(devpath):
            for file in os.listdir(devpath):
                if os.path.splitext(file)[-1] == '.mb':
                    item = TaskItem(self.ui.prev_shotDev_listWidget)
                    item.setDirName(os.path.join(devpath, file))
                    item.setText(file)
            self.ui.prev_shotDev_listWidget.sortItems(QtCore.Qt.DescendingOrder)

        # show dir
        if self.prevpath == 'shotPrv' and self.filePath.find('pub') > 0:
            pubpath = self.filePath + '/scenes'
        else:
            pubpath = os.path.join(self.filePath, 'pub/scenes')
        if os.path.exists(pubpath):
            for file in os.listdir(pubpath):
                if os.path.splitext(file)[-1] == '.mb':
                    item = TaskItem(self.ui.prev_shotPub_listWidget)
                    item.setDirName(os.path.join(pubpath, file))
                    item.setText(file)
            self.ui.prev_shotPub_listWidget.sortItems(QtCore.Qt.DescendingOrder)
        self.readDB()

    def findShot(self):
    # find shot list by name
        if self.ui.task_tab.currentIndex() == 1:
            item = self.ui.prev_findShot_lineEdit.text()
            try:
                item = self.ui.prev_shot_listWidget.findItems(item, QtCore.Qt.MatchContains)[0]
                self.ui.prev_shot_listWidget.setCurrentItem(item)
                self.showPrevShotDevPub()
            except:
                pass

    def getRigMalfunction(self):
    # reference rig list
        client = MongoClient(DB_IP)
        db = client['PUBLISH']
        coll = db['RIG_MALFUNCTION']
        self.rigMalFunctionFile = coll.find({'show': self.titleShort}).distinct('filename')
        self.rigMalFunctionPath = coll.find({'show': self.titleShort}).distinct('path')

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # Read history and read db and snapshot
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def readHistory(self, item):
    # read information from DB and xml. ex) comments, save date, artist info
    # :param item: selected item from dev, pub list widgets. (.mb filename)
        self.fileName = item.text()
        if self.focusWidget().objectName() == 'prev_assetDev_listWidget':
            addpath = 'prev/asset/%s/%s/%s/dev/scenes' % (self.prev_assetType, self.prev_assetName,
                                                          self.prev_workCode)

        if self.focusWidget().objectName() == 'prev_assetPub_listWidget':
            addpath = 'prev/asset/%s/%s/%s/pub/scenes' % (self.prev_assetType, self.prev_assetName,
                                                          self.prev_workCode)

        if self.focusWidget().objectName() == 'prev_shotDev_listWidget':
            addpath = 'prev/%s/%s/%s/%s/dev/scenes' % (self.prevpath, self.prev_shotType,
                                                       self.prev_seq, self.prev_shot)

        if self.focusWidget().objectName() == 'prev_shotPub_listWidget':
            addpath = 'prev/%s/%s/%s/%s/pub/scenes' % (self.prevpath, self.prev_shotType,
                                                       self.prev_seq, self.prev_shot)

        self.filePath = '/show/%s/%s' % (self.titleShort, addpath)
        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.setText(self.fileName)
        self.showSnapshot()
        self.readComment()

    def readDB(self):
    # read history from MongoDB and set infos on the history widget.
        self.ui.history_treeWidget.clear()
        self.dbDic = self.CommentClass.readDB(self.filePath)
        # sort
        sortDic = {}
        if not self.dbDic == None:
            for i in self.dbDic:
                if 'time' in self.dbDic[i].keys():
                    self.dbDic[i]['time'] = self.dbDic[i]['time'][0:16]
                    sortDic[i] = self.dbDic[i]['time']

            sortedList = sorted(sortDic.items(), key=operator.itemgetter(1))
            sortedKeys = [x[0] for x in sortedList]
            for i in reversed(sortedKeys):
                item = HistoryItem(self.ui.history_treeWidget)
                version = '_'.join(i.split('_')[2:])

                if self.dbDic[i]['artist'] != None:
                    item.setText(0, '  ' + self.dbDic[i]['artist'])

                item.setText(1, '  ' + self.dbDic[i]['event'])
                item.setText(2, '  ' + version)
                if self.dbDic[i]['time'] != None:
                    item.setText(3, '  ' + self.dbDic[i]['time'][0:16])

            self.ui.history_treeWidget.sortByColumn(3, QtCore.Qt.DescendingOrder)
            # for Maya2017
            try:
                self.ui.history_treeWidget.header().setResizeMode(QtWidgets.QHeaderView.ResizeToContents)
            except:
                self.ui.history_treeWidget.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

    def takeSnapShot(self):
    # take snapshot and show the image file on the snapshot label.
        fileName = self.SnapShotClass.takeSnapShot(self.filePath, self.fileName)
        self.showSnapshot()

    def showSnapshot(self):
    # show snapshot for asset task.
        imageFile = self.SnapShotClass.showSnapShot(self.filePath, self.fileName)
        self.setPixmapImage(self.ui.snapshot_label, imageFile, 350, 198)

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # My Task Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def setTaskUser(self):
    # My task Tab login user task first setting
        try:
            self.ui.task_tab.setCurrentIndex(2)
            index = self.ui.team_comboBox.findText(self.userdep, QtCore.Qt.MatchContains)
            self.ui.team_comboBox.setCurrentIndex(index)

            index = self.ui.name_comboBox.findText(self.userkor, QtCore.Qt.MatchContains)
            self.ui.name_comboBox.setCurrentIndex(index)
        except:
            pass

    def showTeamComboBox(self):
    # get team name from Tactic and add them to the comboBox
        params = {}
        params['api_key'] = API_KEY
        params['department'] = 'LAY'#ANI||LAY

        infos = requests.get("http://%s/dexter/search/user.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()

        self.ui.team_comboBox.clear()
        teamList = []
        for i in infos:
            self.memberDic[i['name_kr']] = i
            imsi = i['code']
            if self.userName == imsi:
                self.userkor = i['name_kr']
                self.userdep = i['department']

        for i in self.memberDic:
            if self.memberDic[i]['department'] in teamList:
                pass
            else:
                teamList.append(self.memberDic[i]['department'])

        teamList.sort()
        self.ui.team_comboBox.addItems(teamList)
        self.showTeamMemberComboBox()

    def showTeamMemberComboBox(self):
    # show team member's name and add names to the team member comboBox.
    # unicode:param team: selected name from the team name combobox
        team = self.ui.team_comboBox.currentText()
        memberList = []
        for i in self.memberDic:
            if self.memberDic[i]['department'] == team:
                memberList.append(i)

        memberList.sort()
        self.ui.name_comboBox.clear()
        self.ui.name_comboBox.addItems(memberList)

    def currentName(self, item):
    # get current text on the name_comboBox :param item: item from the member name comboBox
        if self.ui.name_comboBox.currentText():
            name = unicode(self.ui.name_comboBox.currentText())
            for i in self.memberDic:
                if name == i:
                    name_eng = self.memberDic[i]['code']
            self.currentTaskAssigned(name_eng)
#
    def currentTaskAssigned(self, name):
    # query from Tactic, a team member's tasks assigned. :param name: member name in english
        params = {}
        params['api_key'] = API_KEY
        params['login'] = name
        infos = requests.get("http://%s/dexter/search/task.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        self.ui.myAsset_listWidget.clear()
        self.ui.myShotName_listWidget.clear()
        self.ui.nonMyAsset_listWidget.clear()
        self.ui.nonMyShot_listWidget.clear()

        for i in infos:
            name = i['extra_name']
            self.taskDic[name] = i
            if i['search_type'] == u'%s/asset?project=%s' % (self.showCode, self.showCode):
                if i['status'] in WORKING:
                    item = TaskItem(self.ui.myAsset_listWidget)
                    item.setText(name)
                    item.setItemColor(i['status'])
                if i['status'] in NONWORKING:
                    item = TaskItem(self.ui.nonMyAsset_listWidget)
                    item.setText(name)
                    item.setItemColor(i['status'])

            if i['search_type'] == u'%s/shot?project=%s' % (self.showCode, self.showCode):
                if i['status'] in WORKING:
                    item = TaskItem(self.ui.myShotName_listWidget)
                    item.setText(name)
                    item.setItemColor(i['status'])
                if i['status'] in NONWORKING:
                    item = TaskItem(self.ui.nonMyShot_listWidget)
                    item.setText(name)
                    item.setItemColor(i['status'])

        self.ui.myAsset_listWidget.setSortingEnabled(1)
        self.ui.myShotName_listWidget.setSortingEnabled(1)
        self.ui.nonMyAsset_listWidget.setSortingEnabled(1)
        self.ui.nonMyShot_listWidget.setSortingEnabled(1)

    def getTaskType(self, item):
    # assigned task double clicked > listdir exist check >> assetQuick or shotQuick call
        try:
            self.workName = item.text()
        except:
            pass
        widget = self.sender().objectName()
        if widget in ['myAsset_listWidget', 'nonMyAsset_listWidget']:
            path = glob.glob('/show/%s/prev/asset'%self.titleShort)
            for i in path:
                if os.path.isdir(i):
                    for j in os.listdir(i):
                        path2 = '%s/%s'%(i, j)
                        if self.workName in os.listdir(path2):
                            self.workType = path2.split('/')[-1]
                            break
                        else:
                            self.workType = ""
        else:
            path = glob.glob('/show/%s/prev/%s/*'%(self.titleShort, self.prevpath))
            for i in path:
                if os.path.isdir(i):
                    if self.workName in os.listdir(i):
                        self.workType = i.split('/')[-1]
                        break
                    else:
                        self.workType = self.workName

    def assetQuick(self):
    # when double clicked, go directly to the asset task from my task menu
        self.ui.task_tab.setCurrentIndex(0)
        try:
            searchedItem = self.ui.prev_assetType_listWidget.findItems(self.workType, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_assetType_listWidget.setCurrentItem(searchedItem)
            self.showPrevAsset()
            searchAssetItem = self.ui.prev_asset_listWidget.findItems(self.workName, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_asset_listWidget.setCurrentItem(searchAssetItem)
            self.showPrevWorkCode()
        except:
            self.ui.fileName_lineEdit.setText('project folder not exist')
            self.ui.prev_workCode_listWidget.clear()
            self.ui.prev_assetDev_listWidget.clear()
            self.ui.prev_assetPub_listWidget.clear()
        self.ui.history_treeWidget.clear()

    def shotQuick(self):
    # when double clicked, go directly to the shot task from my task menu
        self.ui.task_tab.setCurrentIndex(1)
        try:
            searchedItem = self.ui.prev_shotType_listWidget.findItems(self.workType, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_shotType_listWidget.setCurrentItem(searchedItem)
            self.showPrevShot()
            searchShotItem = self.ui.prev_seq_listWidget.findItems(self.workName, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_seq_listWidget.setCurrentItem(searchShotItem)
            self.showPrevShotWorkCode()
            self.prev_shot = 'prev'
            searchWorkItem = self.ui.prev_shot_listWidget.findItems(self.prev_shot, QtCore.Qt.MatchExactly)[0]
            self.ui.prev_shot_listWidget.setCurrentItem(searchWorkItem)
            self.showPrevShotDevPub()
        except:
            self.ui.fileName_lineEdit.setText('project folder not exist')
            self.ui.prev_shot_listWidget.clear()
            self.ui.prev_shotDev_listWidget.clear()
            self.ui.prev_shotPub_listWidget.clear()
            self.ui.history_treeWidget.clear()

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # Comment Read and Save
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def readComment(self):
    # read comment from DB and add on the comment text widget.
        self.ui.comment_textEdit.clear()
        commentDB = self.CommentClass.readDBComment(self.filePath, self.fileName)
        if commentDB:
            # sort dictionary
            sortedKeys = (sorted(commentDB.keys(),
                                 key=lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')))
            for keys in sortedKeys:
                self.ui.comment_textEdit.append('\n' + '=' * 48 + '\n')
                self.ui.comment_textEdit.append('Date: ' + keys)
                self.ui.comment_textEdit.append('Artist: ' + commentDB[keys][1] + '\n')
                self.ui.comment_textEdit.append('Comment:\n' + commentDB[keys][0] + '\n')

        self.ui.comment_textEdit.moveCursor(QtGui.QTextCursor.End)

    def saveComment(self):
    # open comment box and save comment to MongoDB.
        filename = self.ui.fileName_lineEdit.text()

        if filename:
            gc = historyAction.GetCommentUI()
            gc.show()
            gc.fileBox.setText(self.fileName)
            gc.exec_()
            try:
                comment = gc.comment
                self.CommentClass.saveDBComment(self.filePath, self.fileName, comment)
                self.readComment()
            except:
                print 'No Comment to Save'
        else:
            cmds.confirmDialog(title= 'Warning Not Comment Save', message= 'Scene Select to Please !!',
                               messageAlign= 'center', icon= 'warning',
                               button = ['OK'], backgroundColor = [0.9, 0.6, 0.6])

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# My Task ListWidgetItem
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class TaskItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QListWidgetItem.__init__(self, parent)
        self.dirName = ''

    def setDirName(self, name):
        self.dirName = name

    def getDirName(self):
        return self.dirName

    def setItemColor(self, color):
        brush = QtGui.QBrush()
        if color == 'Ready':
            brush.setColor(QtGui.QColor(253, 255, 107, 255))
            self.setForeground(brush)
        if color == 'In-Progress':
            brush.setColor(QtGui.QColor(255, 177, 20, 255))
            self.setForeground(brush)
        if color == 'Review':
            brush.setColor(QtGui.QColor(131, 229, 151, 255))
            self.setForeground(brush)
        if color == 'OK':
            brush.setColor(QtGui.QColor(73, 252, 255, 255))
            self.setForeground(brush)
        if color == 'Approved':
            brush.setColor(QtGui.QColor(88, 144, 204, 255))
            self.setForeground(brush)
        if color == 'Retake':
            brush.setColor(QtGui.QColor(255, 123, 125, 255))
            self.setForeground(brush)
        if color in ['Omit', 'Hold']:
            brush.setColor(QtGui.QColor(130, 130, 130, 255))
            self.setForeground(brush)
        if color == 'BGred':
            brush.setColor(QtGui.QColor(255, 123, 125, 255))
            self.setForeground(brush)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# History TreeWidgetItem
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class HistoryItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Open maya save or open window
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class OpenWaringDialog(QtWidgets.QMessageBox):
    def __init__(self, title=None, txt=None):
        QtWidgets.QMessageBox.__init__(self)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.setFont(font)
        self.setWindowTitle(title)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setText(txt)
        self.setStandardButtons(QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Open |QtWidgets.QMessageBox.Close)
        self.buttonClicked.connect(self.msgbtn)

    def msgbtn(self, i):
        self.result = i.text()
