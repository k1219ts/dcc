# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import site
import json
from bson import ObjectId
import time

from datetime import datetime
from time import gmtime,strftime

# import Qt
from PySide2 import QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
SITE = dxConfig.getHouse()

import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)

import requests
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

if not SITE == 'CHN':
    import pika
    HOST = '10.0.0.13'
    # import dextok

from dxstats import inc_tool_by_user
import getpass
import operator

import mayaActions
from mayaActions import SaveDevForm
from mayaActions import SavePubForm
from mayaActions import MayaActions
import historyAction
import addDirectory
from InventorySpool import InventoryDialog
from rig_malfunctioned_checker.MainForm import MainForm as RigMalChecker

CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
WORKING = ['Waiting', 'In-Progress', 'Ready', 'Retake', 'Review', 'OK']
NONWORKING = ['Approved', 'Omit', 'Hold']

try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

# import ui
from spanner2_ui import Ui_Form
class MainForm(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        inc_tool_by_user.run('Spanner2', getpass.getuser())
        inc_tool_by_user.run('action.Spanner2.open', getpass.getuser())
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # if not SITE == 'CHN':
        #     self.talk_tab = dextok.Talk_tab()
        #     self.ui.task_tab.insertTab( 3, self.talk_tab, 'talk')

        # user data
        self.startSet()
        if not SITE == 'CHN':
            # MQ
            self.mqThread()
            self.showPrevMessage()
            # self.talk_tab.pushButton.clicked.connect(self.sendChatMessege)
            # self.talk_tab.tactic_popup_checkBox.clicked.connect(self.saveUserInfo)

    def startSet(self):
        # print "# Debug : startSet()"
        self.setWindowTitle('Spanner2')
        self.fileName = None
        self.filePath = None
        # print "111111111111111"
        self.getUserName()
        self.projectDic = {}
        self.memberDic = {}
        self.taskDic = {}
        self.myShowList = []
        if SITE == 'CHN':
            self.showComboBoxByDirectory()
        else:
            self.showComboBox()
        # print "222222222222222"
        self.showTeamComboBox()
        # print "333333333333333"
        self.setImage()
        # print "444444444444444"
        try:
            self.getUserInfo()
        except Exception as e:
            print e.message
        # print "555555555555555"
        self.connections()
        # print "666666666666666"
        self.ui.mb_radioButton.setChecked(1)
        self.ui.team_comboBox.setMinimumWidth(180)

        # user image
        if not SITE == 'CHN':
            # print "777777777777777"
            self.user_image = ""
            params = {}
            params = dict()
            params['api_key'] = API_KEY
            params['code'] = getpass.getuser()
            infos = requests.get("http://%s/dexter/search/user.php" %'10.0.0.51',
                                 params=params).json()

            if infos:
                # print "8888888888888888"
                # print '/tactic/assets', infos['relative_dir'], infos['file_name']
                try:
                    self.user_image = os.path.join( '/tactic/assets', infos['relative_dir'], infos['file_name'] )
                except:
                    self.user_image = ""
        # print "999999999999999999"

    def setButtonImage(self, button, image):
        button.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/%s.png" %image))))

    def setPixmapImage(self, label, imagePath, width, height ):
        image = QtGui.QPixmap(imagePath)
        image = image.scaled(width, height, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        label.setPixmap(image)

    # set default image for ui
    def setImage(self):
        # set pixmap
        self.setPixmapImage(self.ui.snapshot_label, os.path.join(CURRENTPATH, 'resource/noPreview.jpg'), 355, 200)
        # set button image
        self.setButtonImage(self.ui.snapShot_pushButton, 'camera')
        self.setButtonImage(self.ui.openPath_btn, 'Icon-Folder01-Yellow')
        self.setButtonImage(self.ui.update_btn, 'A29-CurvedArrow-Green')
        self.setButtonImage(self.ui.assetOpenDevPreview_btn, 'playButton')
        self.setButtonImage(self.ui.assetOpenPubPreview_btn, 'playButton')
        self.setButtonImage(self.ui.shotOpenDevPreview_btn, 'playButton')
        self.setButtonImage(self.ui.shotOpenPubPreview_btn, 'playButton')
        self.setButtonImage(self.ui.editNote_btn, 'playButton')
        self.setButtonImage(self.ui.addToInventory_pushButton, 'inventory')
        self.setButtonImage(self.ui.rig_malfunction_checker_pushButton, 'out_dxRig')
        self.ui.addToInventory_pushButton.setText("")
        # styleSheet
        self.ui.history_treeWidget.setStyleSheet('''
        QTreeView::item:selected{background: grey;}''')
        self.ui.nonMyAsset_listWidget.setStyleSheet('''
        QListWidget::item:selected{background: rgb(62, 126, 189);}''')
        self.ui.nonMyShot_listWidget.setStyleSheet('''
        QListWidget::item:selected{background: rgb(62, 126, 189);}''')
        self.ui.myAsset_listWidget.setStyleSheet('''
        QListWidget::item:selected{background: rgb(95, 143, 61);}''')
        self.ui.myShotName_listWidget.setStyleSheet('''
        QListWidget::item:selected{background: rgb(95, 143, 61);}''')
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
        # quick show
        self.ui.show_comboBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.comboBox.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)        
        self.ui.show_comboBox.customContextMenuRequested.connect(self.addMyShowMenu)
        self.ui.comboBox.customContextMenuRequested.connect(self.delMyShowMenu)        
    
        # inventory & rig ban list
        self.ui.addToInventory_pushButton.clicked.connect(self.openInventory)
        self.ui.rig_malfunction_checker_pushButton.clicked.connect(self.openRigMalChecker)
        # saveUserInfo
        self.ui.show_comboBox.currentIndexChanged.connect(self.saveUserInfo)
        self.ui.team_comboBox.currentIndexChanged.connect(self.saveUserInfo)
        self.ui.name_comboBox.currentIndexChanged.connect(self.saveUserInfo)
        # saveDevPub window
        self.ui.saveDev_btn.clicked.connect(self.openSaveDevWindow)
        self.ui.savePub_btn.clicked.connect(self.openSavePubWindow)
        self.ui.snapShot_pushButton.clicked.connect(self.takeSnapShot)
        self.ui.saveNote_btn.clicked.connect(self.saveComment)
        self.ui.editNote_btn.clicked.connect(self.editComment)
        # mb or ma
        self.ui.mb_radioButton.toggled.connect(self.setfileFormat)
        self.ui.ma_radioButton.toggled.connect(self.setfileFormat)
        # open folder
        self.ui.openPath_btn.clicked.connect(self.openPath)
        self.ui.assetOpenDevPreview_btn.clicked.connect(self.openDevPreview)
        self.ui.assetOpenPubPreview_btn.clicked.connect(self.openPubPreview)
        self.ui.shotOpenDevPreview_btn.clicked.connect(self.openDevPreview)
        self.ui.shotOpenPubPreview_btn.clicked.connect(self.openPubPreview)
        self.ui.shotOpenPubDataPreview_btn.clicked.connect(self.openPubDataPreview)
        # current scene path
        self.ui.update_btn.clicked.connect(self.updateSceneName)
        # search lineEdit
        self.ui.findAsset_lineEdit.textChanged.connect(self.findAsset)
        self.ui.findShot_lineEdit.textChanged.connect(self.findShot)

        # Asset Tab
        self.ui.comboBox.activated.connect(self.changeShowPath)
        self.ui.show_comboBox.currentIndexChanged.connect(self.currentshowPath)
        self.ui.assetType_listWidget.itemClicked.connect(self.showAsset)
        self.ui.asset_listWidget.itemClicked.connect(self.showAssetWorkCode)
        self.ui.workCode_listWidget.itemClicked.connect(self.showAssetDevPub)

        # Shot Tab
        self.ui.shotType_listWidget.itemClicked.connect(self.showShot)
        self.ui.shot_listWidget.itemClicked.connect(self.showShotWorkCode)
        self.ui.shotWorkCode_listWidget.itemClicked.connect(self.showShotDevPub)

        # My Task
        self.ui.team_comboBox.currentIndexChanged.connect(self.showTeamMemberComboBox)
        self.ui.name_comboBox.currentIndexChanged.connect(self.currentName)
        self.ui.myAsset_listWidget.itemDoubleClicked.connect(self.assetQuick)
        self.ui.nonMyAsset_listWidget.itemDoubleClicked.connect(self.assetQuick)
        self.ui.myShotName_listWidget.itemDoubleClicked.connect(self.shotQuick)
        self.ui.nonMyShot_listWidget.itemDoubleClicked.connect(self.shotQuick)

        # My Task Context Menu
        self.ui.myAsset_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.myAsset_listWidget.customContextMenuRequested.connect(self.myTaskMenu)
        self.ui.myAsset_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.myShotName_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.myShotName_listWidget.customContextMenuRequested.connect(self.myTaskMenu)
        self.ui.myShotName_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.nonMyAsset_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.nonMyAsset_listWidget.customContextMenuRequested.connect(self.myTaskMenu)
        self.ui.nonMyAsset_listWidget.itemPressed.connect(self.getTaskType)
        self.ui.nonMyShot_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.nonMyShot_listWidget.customContextMenuRequested.connect(self.myTaskMenu)
        self.ui.nonMyShot_listWidget.itemPressed.connect(self.getTaskType)

        # Show History
        self.ui.assetPub_listWidget.itemPressed.connect(self.readHistory)
        self.ui.assetDev_listWidget.itemPressed.connect(self.readHistory)
        self.ui.shotDev_listWidget.itemPressed.connect(self.readHistory)
        self.ui.shotPub_listWidget.itemPressed.connect(self.readHistory)

        # right click Action
        self.ui.assetDev_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.assetDev_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.assetPub_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.assetPub_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.shotDev_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.shotDev_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.shotPub_listWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.shotPub_listWidget.customContextMenuRequested.connect(self.contextMenu)
        self.ui.assetDev_listWidget.itemDoubleClicked.connect(self.openMaya)
        self.ui.assetPub_listWidget.itemDoubleClicked.connect(self.openMaya)
        self.ui.shotDev_listWidget.itemDoubleClicked.connect(self.openMaya)
        self.ui.shotPub_listWidget.itemDoubleClicked.connect(self.openMaya)

        # create folders
        self.ui.addAssetType_btn.clicked.connect(self.addAsset)
        self.ui.addAssetName_btn.clicked.connect(self.addAsset)
        self.ui.addWorkCode_btn.clicked.connect(self.addWorkCode)
        self.ui.addShotType_btn.clicked.connect(self.addShot)
        self.ui.addShotName_btn.clicked.connect(self.addShot)
        self.ui.addShotWorkCode_btn.clicked.connect(self.addWorkCode)
        self.ui.workCode_listWidget.doubleClicked.connect(self.createFirstScene)
        self.ui.shotWorkCode_listWidget.doubleClicked.connect(self.createFirstScene)

    ###############
    ##   DEXTOK  ##
    def showPrevMessage(self):
        self.tasklist = self.checkTask()
        tasklist = self.tasklist
        db = client['stats']
        coll = db['rabbitmq']
        result = coll.find( {'shot':{'$in':tasklist}} ).limit(10).sort('time', pymongo.DESCENDING)
        tacticlist = []

        # tactic message
        if result.count() > 0:
            for i in result:
                if i['status'] in ['Approved','Retake','Ready','Omit','Hold']:
                    if i['process'] in ['model','creature','animation']:
                        tacticlist.append(i)

            tacticlist = list(reversed(tacticlist))
            for k in tacticlist:
                k.pop("_id")
                data = json.dumps(k)
                self.messageFilter(data)

        # db message
        db = client['PUBLISH']
        coll = db['spanner2_talk']
        result = coll.find({'key':'spanner2.talk'}).limit(50).sort('time', pymongo.DESCENDING)
        chatlist = []
        if result.count() > 0:
            for i in result:
                chatlist.append(i)

            # chatlist = list(reversed(chatlist))
            # for c in chatlist:
            #     self.addTalkItem(c['sender'], c['message'], c['image'], 'chat', c['time'])

    # def sendChatMessege(self):
    #     text = self.talk_tab.chat_plainTextEdit.toPlainText()
    #     if not text:
    #         return

    #     msg = {}
    #     msg['sender'] = getpass.getuser()
    #     msg['name'] = getpass.getuser()
    #     msg['text'] = text
    #     msg['image'] = self.user_image

    #     # dextok.sendChatMessege_db( msg, 'spanner2.talk' )
    #     # dextok.sendChatMessege( msg, 'spanner2.talk', 'tactic')
    #     self.talk_tab.chat_plainTextEdit.clear()

    def mqThread(self):
        # connect to mqserver
        binding_keys = []
        binding_keys.append("*.*.*.*.%s" %getpass.getuser().replace('.','_'))
        binding_keys.append("spanner2.talk")

        # self.chat_process = dextok.PikaClass(self, binding_keys, 'tactic')
        # self.chat_process.start()
        # self.chat_process.emitMessage.connect(self.messageFilter)
        # self.chat_process.emitMessage.connect(self.tacticPopup)


    def messageFilter(self, text):
        data = json.loads(text)
        # user chatting message
        if not data['sender'] == 'tactic':
            image = data['image']
            # self.addTalkItem(data['name'], data['text'], image, 'chat')

        # tactic message
        else:
            # FIND TASK
            tasklist = self.tasklist
            if not data['shot'] in tasklist:
                return
            acceptlist = ['Approved','Retake','Ready','Omit','Hold']
            if not data['status'] in acceptlist:
                return
            acceptprocess = ['model','creature','animation']
            if not data['process'] in acceptprocess:
                return

            # FIND USER IMAGE, TEAM
            name = data['assigned'] #user name
            team = data['process'].upper() # work process
            image = data['process'].upper()

            # FIND SHOW NAME
            data = json.loads(text)
            show = data['body']['project_code']
            for i in self.projectDic.keys():
                if self.projectDic[i]['code'] == show:
                    show = self.projectDic[i]['name']
                    break

            # SET MESSEAGE
            colorDict = { "Approved":"ffffff", "Retake":"ff8080", "Ready":"ffff99", "Omit":"ff8080", "Hold":"ff8080" }
            name = u"%s <font size=\"4\" color=\"white\"><b>%s</b></font>" %( name,team )
            string = u"<b><font size=\"3\" color=\"white\"> %s &nbsp;%s</font></b>"  %( str(show).upper(), str(data['shot']).upper() )
            string += u"<br><br>changed to<font color=\"#%s\"><b>  &nbsp; %s</b></font> " %( colorDict[data['status']], data['status'].upper() )
            # self.addTalkItem(name, string, image, 'tactic', data['time'])

            self.tacticPopupString = u'from ' + name + u'<b>  &nbsp; <br><br></b>'+ string

    # def tacticPopup(self):
    #     # popup
    #     if self.talk_tab.tactic_popup_checkBox.isChecked():
    #         return
    #         dialog = mayaActions.WaringDialog()
    #         dialog.label.setText(self.tacticPopupString)
    #         dialog.setWindowTitle('TACTIC MESSAGE')
    #         dialog.exec_()

    # def addTalkItem(self, name, body, image, talktype, talktime=""):
    #     chatItem = dextok.ChatItem()
    #     chatItem.setName(name)
    #     chatItem.setText(unicode(body))

    #     if talktype == "chat":
    #         chatItem.setLabel(image)
    #         if name != getpass.getuser():
    #             chatItem.setColor('rgb(255,255,255,200)')

    #     if talktype == "tactic":
    #         chatItem.setColor('rgb(59,118,177,200)')
    #         chatItem.setLabelText(image)

    #     if talktime:
    #         chatItem.setTime(talktime)

    #     chat = QtWidgets.QListWidgetItem(self.talk_tab.talk_listWidget)
    #     chat.setSizeHint(chatItem.sizeHint())
    #     self.talk_tab.talk_listWidget.addItem(chat)
    #     self.talk_tab.talk_listWidget.setItemWidget(chat, chatItem )
    #     self.talk_tab.talk_listWidget.setSelectionMode(Qt.QtWidgets.QAbstractItemView.NoSelection)
    #     self.talk_tab.talk_listWidget.setFocusPolicy(QtCore.Qt.NoFocus)
    #     self.talk_tab.talk_listWidget.scrollToBottom()

    def checkTask(self):
        params = {}
        params = dict()
        params['api_key'] = API_KEY
        params['login'] = getpass.getuser()
        infos = requests.get("http://%s/dexter/search/task.php" %'10.0.0.51',
                             params=params).json()
        tasklist = []
        for i in infos:
            tasklist.append(i['extra_code'])
        return tasklist

    ###############
    ## START SET ##
    def addMyShowMenu(self, pos):
        pos = pos + (QtCore.QPoint(20, 0))
        menu = QtWidgets.QMenu()
        action1 = menu.addAction('add quick +')
        action1.triggered.connect(self.addMyShow)
        menu.exec_(self.ui.show_comboBox.mapToGlobal(pos))
        
    def delMyShowMenu(self, pos):
        pos = pos + (QtCore.QPoint(20, 0))
        menu = QtWidgets.QMenu()
        action1 = menu.addAction('del quick -')
        action1.triggered.connect(self.delMyShow)
        menu.exec_(self.ui.comboBox.mapToGlobal(pos))        
        
    def addMyShow(self):
        current = self.ui.show_comboBox.currentText()
        self.ui.comboBox.addItem(current)
        currentList = []
        for i in range(self.ui.comboBox.count()):
            currentList.append(self.ui.comboBox.itemText(i))
            
        self.myShowList = currentList
        self.saveUserInfo()
        self.getUserInfo()
        
    def delMyShow(self):
        for i in self.myShowList:
            if i == self.ui.comboBox.currentText():
                self.myShowList.remove(i)
        self.saveUserInfo()                
        self.getUserInfo()
        
    def changeShowPath(self):
        current =  self.ui.comboBox.currentText()
        index = self.ui.show_comboBox.findText(current, QtCore.Qt.MatchExactly)
        self.ui.show_comboBox.setCurrentIndex(index)
                
    def createFirstScene(self):
        """
        create first scene if there is none in the scene folder.
        """
        addDirectory.createFirstScene(self.filePath)
        if self.ui.task_tab.currentIndex() == 0:
            self.showAssetDevPub(self.ui.workCode_listWidget.currentItem())
        if self.ui.task_tab.currentIndex() == 1:
            self.showShotDevPub(self.ui.shotWorkCode_listWidget.currentItem())

    def saveComment(self):
        """
        open comment box and save comment to MongoDB.
        """
        gc = historyAction.GetCommentUI()
        gc.show()
        gc.fileBox.setText(self.fileName)
        gc.exec_()
        try:
            comment = gc.comment
            historyAction.saveDBComment(self.filePath, self.fileName, comment)
            self.readComment()
        except:
            print 'No Comment to Save'

    def editComment(self):
        """
        open comment edit box and edit comment according to the date.
        """
        historyAction.editDBComment(self.filePath, self.fileName)
        self.readComment()

    def setfileFormat(self):
        """
        get 'ma' or 'mb' if user checks the checkBox.
        """
        if self.ui.mb_radioButton.isChecked():
            self.fileFormat = 'mb'
        else:
            self.fileFormat = 'ma'

        # update state
        try:
            self.showAssetDev()
            self.showAssetPub()
        except: pass

    def addShot(self):
        """
        create shot folder.
        """
        if self.sender() == self.ui.addShotType_btn:
            addDirectory.addShot(self.titleShort)
            self.showShotType()
        if self.sender() == self.ui.addShotName_btn:
            addDirectory.addShot(self.titleShort, self.shotType)
            self.showShot()

    def addAsset(self):
        """
        create asset folder
        """
        if self.sender() == self.ui.addAssetType_btn:
            addDirectory.addAsset(self.titleShort)
            self.showAssetType()
        if self.sender() == self.ui.addAssetName_btn:
            addDirectory.addAsset(self.titleShort, self.assetType)
            self.showAsset()

    def addWorkCode(self):
        """
        create workcode folder for asset and shot
        """
        addDirectory.newWorkCodeUI(self.filePath)

        # update state
        if self.ui.task_tab.currentIndex() == 0:
            self.showAssetWorkCode()
        if self.ui.task_tab.currentIndex() == 1:
            self.showShotWorkCode()

    def openMaya(self):
        """
        open maya scene by the fileName.
        """
        inc_tool_by_user.run('action.Spanner2.openMaya', getpass.getuser())
        openfile = os.path.join(self.filePath, self.fileName)
        ow = OpenWaringDialog()
        ow.show()
        result = ow.exec_()
        if ow.result == 'save':
            filePath = os.path.dirname(cmds.file(sn=1,q=1))
            fileName = cmds.fileDialog2(ds=2, startingDirectory=filePath,
                                        fileFilter="Maya Files (*.ma *.mb)")
            if fileName:
                cmds.file(rename=str(fileName[0]))
                cmds.file(save=True, type='mayaBinary')

        elif ow.result == 'open':
            pass

        else:
            return

        cmds.file(openfile, open=True, force=True)
        try:
            mel.eval('setProject "%s"' % (self.filePath.split('/scenes')[0]))
        except:
            print 'failed'

    def contextMenu(self, pos ):
        """
        right click action on dev list widgets and pub list widgets.
        :param pos: get current position of the cursor.
        """
        pos = pos + (QtCore.QPoint(20,0))
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

        if self.ui.task_tab.currentIndex() == 0:
            workcode = self.ui.workCode_listWidget.currentItem().text()
            if workcode == 'rig':
                menu.addSeparator()
                action7 = menu.addAction('Rig Malfuctionize')
                action7.triggered.connect(self.rigMalFuction)
                action8 = menu.addAction('Rig Fuctionize')
                action8.triggered.connect(self.rigFunctionize)

        menu.exec_(self.focusWidget().mapToGlobal(pos))

    def rigFunctionize(self):
        wd = mayaActions.WaringDialog(text='Are you sure?')
        warn = wd.exec_()
        if not warn:
            return
        rigpath = os.path.join(self.filePath, self.fileName)
        client = MongoClient(DB_IP)
        db = client['PUBLISH']
        coll = db['RIG_MALFUNCTION']
        if self.ui.task_tab.currentIndex() == 0:
            coll.remove(
                {
                'name': self.ui.asset_listWidget.currentItem().text(),
                'show': self.titleShort,
                'type': self.ui.assetType_listWidget.currentItem().text(),
                'path': rigpath,
                'filename': os.path.basename(rigpath)
                }
            )
            self.showAssetDevPub(self.ui.workCode_listWidget.currentItem())

    def rigMalFuction(self):
        wd = mayaActions.WaringDialog(text='Are you sure?')
        warn = wd.exec_()
        if not warn:
            return
        rigpath = os.path.join( self.filePath, self.fileName )
        client = MongoClient(DB_IP)
        db = client['PUBLISH']
        coll = db['RIG_MALFUNCTION']
        if self.ui.task_tab.currentIndex() == 0:
            coll.insert({
                'name':self.ui.asset_listWidget.currentItem().text(),
                'show':self.titleShort,
                'type':self.ui.assetType_listWidget.currentItem().text(),
                'path':rigpath,
                'filename':os.path.basename(rigpath),
                'time':str(datetime.now()),
                'user':getpass.getuser()
            })
            self.showAssetDevPub(self.ui.workCode_listWidget.currentItem())

    def openSaveDevWindow(self):
        """
        open dialog for save dev file
        """
        dw = SaveDevForm(QtWidgets.QDialog, self.fileName, self.filePath)
        dw.show()
        dw.exec_()

        # reload state
        self.updateSceneName()

    def openSavePubWindow(self):
        """
        open dialog for save pub file/ publish.
        """
        pw = SavePubForm(QtWidgets.QDialog, self.fileName, self.filePath)
        pw.show()
        pw.exec_()

        # reload state
        self.updateSceneName()

    def getUserName(self):
        # print "# Debug : getUserName()"
        """
        get name of current user.
        """
        self.userName = os.getlogin()
        self.ui.name_label.setText(self.userName)

    def getUserInfo(self):
        # print "# Debug : getUserInfo()"
        """
        get user info when app starts
        """
        try:
            dataDict = historyAction.getUserInfo()
            if dataDict:
                self.ui.task_tab.setCurrentIndex(2)
                self.ui.show_comboBox.setCurrentIndex(dataDict['show'])
                self.ui.team_comboBox.setCurrentIndex(dataDict['teamName'])
                self.currentshowPath()

                # view team member
                self.showTeamMemberComboBox()

                # view current task
                self.ui.name_comboBox.setCurrentIndex(dataDict['userName'])
                name = unicode(self.ui.name_comboBox.currentText())
                for i in self.memberDic:
                    if name == i:
                        name_eng = self.memberDic[i]['code']
                self.currentTaskAssigned(name_eng)

                # tactic popup setting
                if 'tacticPopup' in dataDict:
                    self.talk_tab.tactic_popup_checkBox.setChecked(dataDict['tacticPopup'])
                else:
                    self.talk_tab.tactic_popup_checkBox.setChecked(True)

                if 'myShow' in dataDict:
                    self.myShowList = dataDict['myShow']
                    self.ui.comboBox.clear()
                    self.ui.comboBox.addItems(dataDict['myShow'])
                
            else:
                # preset
                self.ui.show_comboBox.setCurrentIndex(0)
                self.currentshowPath()
                
        except:
            pass

    def saveUserInfo(self):
        # print "# Debug : saveUserInfo()"
        """
        save user info to MongoDB.
        """
        data = {}
        data['show'] = self.ui.show_comboBox.currentIndex()
        data['teamName'] = self.ui.team_comboBox.currentIndex()
        data['userName'] = self.ui.name_comboBox.currentIndex()
        if not SITE == 'CHN': pass
            # data['tacticPopup'] = self.talk_tab.tactic_popup_checkBox.isChecked()
        data['myShow'] = self.myShowList
        historyAction.saveUserInfo(data)

    def openPath(self):
        """
        open nautilus current file path.
        """
        subprocess.Popen(['xdg-open', str(self.ui.filePath_lineEdit.text())])

    def openDevPreview(self):
        """
        open nautilus for dev preview.
        """
        previewPath = '/'.join(self.filePath.split('/')[0:7]) + '/dev/preview'
        subprocess.Popen(['xdg-open', str(previewPath)])

    def openPubPreview(self):
        """
        open nautilus for pub preview.
        """
        previewPath = '/'.join(self.filePath.split('/')[0:7]) + '/pub/preview'
        subprocess.Popen(['xdg-open', str(previewPath)])

    def openPubDataPreview(self):
        """
        open nautilus for pub data preview.
        """
        previewPath = '/'.join(self.filePath.split('/')[0:7]) + '/pub/data/preview'
        subprocess.Popen(['xdg-open', str(previewPath)])

    def findAsset(self):
        """
        find asset by name
        """
        if self.ui.task_tab.currentIndex() == 0:
            item = self.ui.findAsset_lineEdit.text()
            try:
                item = self.ui.asset_listWidget.findItems(item, QtCore.Qt.MatchContains)[0]
                self.ui.asset_listWidget.setCurrentItem(item)
                self.showAssetWorkCode()
            except: pass

    def getProjectDict(self):
        # print "# Debug : getProjectDict()"
        projectDic = {}
        params = {}
        params['api_key'] = API_KEY
        params['category'] = 'Active'
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        exceptList = ['test', 'testshot']
        for i in infos:
            if i['name'] in exceptList:
                if i['name'] == 'testshot':
                    i['name'] = 'test_shot'
                    projectDic[i['title']] = i
            else:
                projectDic[i['title']] = i

        return projectDic

    def showComboBox(self):
        # print "# Debug : showComboBox()"
        """
        get show name from Tactic and append them to the show comboBox.
        """
        self.projectDic = self.getProjectDict()
        self.ui.show_comboBox.addItems(sorted(self.projectDic.keys()))

    def showComboBoxByDirectory(self):
        # print "# Debug : showComboBoxByDirectory()"
        path = '/show/'
        for i in os.listdir(path):
            if not '_pub' in i:
                self.projectDic[i] = {}
                self.projectDic[i]['name'] = i
                self.projectDic[i]['title'] = i
                self.projectDic[i]['code'] = i

        projectDic = self.getProjectDict()
        for i in projectDic.keys():
            self.projectDic[projectDic[i]['name']] = {}
            self.projectDic[projectDic[i]['name']]['name']  = projectDic[i]['name']
            self.projectDic[projectDic[i]['name']]['title'] = projectDic[i]['title']
            self.projectDic[projectDic[i]['name']]['code']  = projectDic[i]['code']

        self.ui.show_comboBox.addItems(sorted(self.projectDic.keys()))

    def currentshowPath(self):
        # print "# Debug : currentshowPath()"
        """
        show asset type and shot type from the show.
        """
        title = unicode(self.ui.show_comboBox.currentText())
        # print title
        self.titleShort = self.projectDic[title]['name']
        # print self.titleShort
        self.assetTypePath = '/show/%s/asset' % str(self.titleShort)
        self.shotTypePath = '/show/%s/shot' % str(self.titleShort)

        # print self.assetTypePath, self.shotTypePath

        # show list
        self.showAssetType()
        self.showShotType()

        # get name
        name = unicode(self.ui.name_comboBox.currentText())
        self.showCode = self.projectDic[title]['code']
        self.currentName(name)

        # show path
        self.ui.filePath_lineEdit.setText( '/show/%s' % str(self.titleShort) )

    def updateSceneName(self):
        # print "# Debug : updateSceneName()"
        """
        get path of currently opened maya scene and go to the file directly in the app.
        """
        temp = cmds.file(q=True, sn=True)
        self.filePath = os.path.dirname(temp)
        self.fileName = os.path.basename(temp)
        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.setText(self.fileName)

        # print self.filePath
        pathTemp = self.filePath.split('/')
        show = pathTemp[2]
        for i in self.projectDic:
            if show == self.projectDic[i]['name']:
                show = i
        assetShot = pathTemp[3]

        # find show index
        index = self.ui.show_comboBox.findText(show, QtCore.Qt.MatchContains)
        self.ui.show_comboBox.setCurrentIndex(index)

        if assetShot == 'asset':
            self.ui.task_tab.setCurrentIndex(0)
            self.assetType = pathTemp[4]
            self.asset = pathTemp[5]
            self.workCode = pathTemp[6]
            devPub = pathTemp[7]
            # find asset type
            item = self.ui.assetType_listWidget.findItems(self.assetType, QtCore.Qt.MatchExactly)[0]
            self.ui.assetType_listWidget.setCurrentItem(item)
            # find asset
            self.showAsset()
            item = self.ui.asset_listWidget.findItems(self.asset, QtCore.Qt.MatchExactly)[0]
            self.ui.asset_listWidget.setCurrentItem(item)
            # find asset work code
            self.showAssetWorkCode()
            item = self.ui.workCode_listWidget.findItems(self.workCode, QtCore.Qt.MatchExactly)[0]
            self.ui.workCode_listWidget.setCurrentItem(item)
            # find from dev/pub list
            self.showAssetDevPub(item)
            if devPub == 'dev':
                item = self.ui.assetDev_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.assetDev_listWidget.setCurrentItem(item)
                self.ui.assetDev_listWidget.setFocus()
                self.readHistory(item)
            else:
                item = self.ui.assetPub_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.assetPub_listWidget.setCurrentItem(item)
                self.ui.assetPub_listWidget.setFocus()
                self.readHistory(item)

        if assetShot == 'shot':
            self.ui.task_tab.setCurrentIndex(1)
            self.shotType = pathTemp[4]
            self.shot = pathTemp[5]
            self.shotWorkCode = pathTemp[6]
            devPub = pathTemp[7]
            # find shot type
            item = self.ui.shotType_listWidget.findItems(self.shotType, QtCore.Qt.MatchExactly)[0]
            self.ui.shotType_listWidget.setCurrentItem(item)
            # find shot
            self.showShot()
            item = self.ui.shot_listWidget.findItems(self.shot, QtCore.Qt.MatchExactly)[0]
            self.ui.shot_listWidget.setCurrentItem(item)
            # find shot work code
            self.showShotWorkCode()
            item = self.ui.shotWorkCode_listWidget.findItems(self.shotWorkCode, QtCore.Qt.MatchExactly)[0]
            self.ui.shotWorkCode_listWidget.setCurrentItem(item)
            # find from dev/pub list
            self.showShotDevPub(item)
            if devPub == 'dev':
                item = self.ui.shotDev_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.shotDev_listWidget.setCurrentItem(item)
                self.ui.shotDev_listWidget.setFocus()
                self.readHistory(item)
            else:
                item = self.ui.shotPub_listWidget.findItems(self.fileName, QtCore.Qt.MatchExactly)[0]
                self.ui.shotPub_listWidget.setCurrentItem(item)
                self.ui.shotPub_listWidget.setFocus()
                self.readHistory(item)

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # ASSET Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def getRigMalfunction(self):
        client = MongoClient(DB_IP)
        db = client['PUBLISH']
        coll = db['RIG_MALFUNCTION']
        self.rigMalFunctionFile = coll.find({'show': self.titleShort}).distinct('filename')
        self.rigMalFunctionPath = coll.find({'show': self.titleShort}).distinct('path')

    def showAssetType(self):
        # print "# Debug : showAssetType()"
        """
        show asset type path on the asset type list widget.
        """
        self.ui.addAssetName_btn.setEnabled(False)
        self.ui.addWorkCode_btn.setEnabled(False)
        self.ui.assetType_listWidget.clear()
        self.ui.asset_listWidget.clear()
        self.ui.workCode_listWidget.clear()
        self.ui.assetDev_listWidget.clear()
        self.ui.assetPub_listWidget.clear()

        # get dir
        self.assetBaseDir = QtCore.QDir(self.assetTypePath)
        for info in self.assetBaseDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.assetType_listWidget)
            item.setDirName(os.path.join(self.assetTypePath, str(info.baseName())
                                         )
                            )
            item.setText(info.baseName())

    def showAsset(self):
        """
        get list of asset name folders and show them on the asset list widget.
        """
        self.ui.addAssetName_btn.setEnabled(True)
        self.ui.addWorkCode_btn.setEnabled(False)
        self.ui.asset_listWidget.clear()
        self.ui.workCode_listWidget.clear()
        self.ui.fileName_lineEdit.clear()
        self.ui.assetDev_listWidget.clear()
        self.ui.assetPub_listWidget.clear()
        self.ui.findAsset_lineEdit.clear()
        self.assetType = unicode(self.ui.assetType_listWidget.currentItem().text())
        self.filePath = '/show/%s/asset/%s' % (self.titleShort, self.assetType)
        # show dir
        self.ui.filePath_lineEdit.setText(self.filePath)
        assetDir = QtCore.QDir(self.filePath)
        for info in assetDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.asset_listWidget)
            item.setDirName(os.path.join(self.filePath, str(info.baseName())
                                         )
                            )
            item.setText(info.baseName())

    def showAssetWorkCode(self):
        """
        get list of asset work code dir and show asset work code on asset workcode list widget.
        """
        self.ui.addWorkCode_btn.setEnabled(True)
        self.ui.workCode_listWidget.clear()
        self.asset = self.ui.asset_listWidget.currentItem().text()
        self.filePath = '/show/%s/asset/%s/%s' % (self.titleShort, self.assetType, self.asset)
        # show path
        self.ui.filePath_lineEdit.setText(self.filePath)
        workDir = QtCore.QDir(self.filePath)
        for info in workDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.workCode_listWidget)
            item.setDirName(self.filePath)
            item.setText(info.baseName())
        self.ui.fileName_lineEdit.clear()
        self.ui.fileName_lineEdit.setStyleSheet("color:(0,0,0)")
        try:
            self.showSnapshot()
        except:
            pass




    def showAssetDev(self):
        """
        show list of asset dev files on asset dev list widget.
        """
        self.ui.assetDev_listWidget.clear()
        self.workCode = self.ui.workCode_listWidget.currentItem().text()
        self.filePath = '/show/%s/asset/%s/%s/%s/dev/scenes' % (
            self.titleShort, self.assetType, self.asset, self.workCode)


        self.getRigMalfunction()
        if os.path.isdir(self.filePath):
            for pathTemp in os.listdir(self.filePath):
                if pathTemp.split('.')[-1] == self.fileFormat:
                    item = TaskItem(self.ui.assetDev_listWidget)
                    item.setDirName( os.path.join(self.assetTypePath, os.path.basename(pathTemp) ) )
                    item.setText( os.path.basename(pathTemp) )
                    if os.path.basename(pathTemp) in self.rigMalFunctionFile:
                        item.setItemColor('BGred')

            self.ui.assetDev_listWidget.sortItems(QtCore.Qt.DescendingOrder)

    def showAssetPub(self):
        """
        show list of asset pub files on asset pub list widget.
        show mb, abc, json file format.
        """
        if self.fileFormat == 'mb':
            format = ['mb','abc','json']
        else:
            format = 'ma'
        self.ui.assetPub_listWidget.clear()
        self.workCode = self.ui.workCode_listWidget.currentItem().text()
        self.filePath = '/show/%s/asset/%s/%s/%s/pub/scenes' % (
        self.titleShort, self.assetType, self.asset, self.workCode)
        # rig check
        self.getRigMalfunction()
        if os.path.isdir(self.filePath):
            for pathTemp in os.listdir(self.filePath):
                if pathTemp.split('.')[-1] in format:
                    item = TaskItem(self.ui.assetPub_listWidget)
                    item.setDirName(os.path.join(self.assetTypePath, os.path.basename(pathTemp)))
                    item.setText(os.path.basename(pathTemp))
                    if os.path.basename(pathTemp) in self.rigMalFunctionFile:
                        item.setItemColor('Omit')

            self.ui.assetPub_listWidget.sortItems(QtCore.Qt.DescendingOrder)

        # SHOW ASB FILES
        if self.workCode == "model":
            asbPath = self.filePath.replace("/scenes", "/envlayout")
            if os.path.isdir( asbPath ):
                for pathTemp in os.listdir(asbPath):
                    for asbfile in os.listdir(os.path.join(asbPath,pathTemp)):
                        if asbfile.split('.')[-1] == "asb":
                            item = TaskItem(self.ui.assetPub_listWidget)
                            item.setDirName(os.path.join(pathTemp, os.path.basename(asbfile)))
                            item.setText(os.path.join(pathTemp, os.path.basename(asbfile)))


    def showAssetDevPub(self, item):
        """
        call both showAssetDev() and showAssetPub()
        :param item: item from workcode list widget.
        """
        self.workCode = item.text()
        self.filePath = '/show/%s/asset/%s/%s/%s' % (self.titleShort, self.assetType, self.asset, self.workCode)
        # show dev pub
        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.clear()
        self.showAssetDev()
        self.showAssetPub()
        # Read DB
        self.filePath = self.ui.filePath_lineEdit.text()
        self.readDB()
        self.asset = self.ui.asset_listWidget.currentItem().text()

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # Shot Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def findShot(self):
        """
        search shot name
        """
        if self.ui.task_tab.currentIndex() == 1:
            item = self.ui.findShot_lineEdit.text()
            try:
                item = self.ui.shot_listWidget.findItems(item, QtCore.Qt.MatchContains)[0]
                self.ui.shot_listWidget.setCurrentItem(item)
                self.showShotWorkCode()
            except:
                pass

    def showShotType(self):
        # print "# Debug : showShotType()"
        """
        show list of shot type folders on shot type list widget.
        """
        self.ui.addShotName_btn.setEnabled(False)
        self.ui.addShotWorkCode_btn.setEnabled(False)
        self.ui.shotType_listWidget.clear()
        # print self.shotTypePath
        self.shotBaseDir = QtCore.QDir(self.shotTypePath)
        # print self.shotBaseDir
        for info in self.shotBaseDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.shotType_listWidget)
            # print self.assetTypePath, str(info.baseName())
            item.setDirName(os.path.join(self.assetTypePath, str(info.baseName())))
            # print info.baseName()
            item.setText(info.baseName())

    def showShot(self):
        """
        show list of shot name on the shot name list widget.
        """
        self.ui.addShotName_btn.setEnabled(True)
        self.ui.addShotWorkCode_btn.setEnabled(False)
        self.ui.shot_listWidget.clear()
        self.ui.fileName_lineEdit.clear()
        self.ui.shotWorkCode_listWidget.clear()
        self.ui.shotDev_listWidget.clear()
        self.ui.shotPub_listWidget.clear()
        self.ui.findShot_lineEdit.clear()
        self.shotType = unicode(self.ui.shotType_listWidget.currentItem().text())
        self.filePath = '/show/%s/shot/%s' % (self.titleShort, self.shotType)
        self.ui.filePath_lineEdit.setText(self.filePath)

        # get list and sort
        itemList = []
        for info in os.listdir(self.filePath):
            if info and not info.startswith('.'):
                itemList.append(info)
        itemList.sort()
        self.ui.shot_listWidget.addItems(itemList)

    def showShotWorkCode(self):
        """
        show list of shot workcode on the shot workcode list widget.
        """
        self.ui.addShotWorkCode_btn.setEnabled(True)
        self.ui.shotWorkCode_listWidget.clear()
        self.ui.fileName_lineEdit.clear()
        self.ui.fileName_lineEdit.setStyleSheet("color:(0,0,0)")
        self.shot = self.ui.shot_listWidget.currentItem().text()
        self.filePath = '/show/%s/shot/%s/%s' % (self.titleShort, self.shotType, self.shot)
        # show path
        self.ui.filePath_lineEdit.setText(self.filePath)
        workDir = QtCore.QDir(self.filePath)
        for info in workDir.entryInfoList(filters=QtCore.QDir.NoDotAndDotDot | QtCore.QDir.Dirs):
            item = TaskItem(self.ui.shotWorkCode_listWidget)
            item.setDirName(self.filePath)
            item.setText(info.baseName())
        try:
            self.showSnapshot()
        except:
            pass

    def showShotDevPub(self, item):
        """
        call both showShotDev() and showShotPub()
        :param item: item from shot workcode list widget.
        """
        self.shotWorkCode = item.text()
        self.filePath = '/show/%s/shot/%s/%s/%s' % (self.titleShort, self.shotType, self.shot, self.shotWorkCode)
        # show path
        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.clear()
        self.showShotDev()
        self.showShotPub()
        self.filePath = self.ui.filePath_lineEdit.text()
        self.readDB()

    def showShotDev(self):
        """
        show list of shot dev files on shot dev list widget.
        """
        self.ui.shotDev_listWidget.clear()
        self.shotWorkCode = self.ui.shotWorkCode_listWidget.currentItem().text()
        self.filePath = '/show/%s/shot/%s/%s/%s/dev/scenes' % (
        self.titleShort, self.shotType, self.shot, self.shotWorkCode)
        pathTempList = []
        if os.path.isdir(self.filePath):
            for pathTemp in os.listdir(self.filePath):
                if pathTemp.split('.')[-1] == 'mb':
                    pathTempList.append(pathTemp)
            pathTempList.sort(reverse=1)
            self.ui.shotDev_listWidget.addItems(pathTempList)

    # show shot pub files
    def showShotPub(self):
        """
        show list of shot pub files on shot pub list widget.
        """
        self.ui.shotPub_listWidget.clear()
        self.shotWorkCode = self.ui.shotWorkCode_listWidget.currentItem().text()
        self.filePath = '/show/%s/shot/%s/%s/%s/pub/scenes' % (
        self.titleShort, self.shotType, self.shot, self.shotWorkCode)
        pathTempList = []
        if os.path.isdir(self.filePath):
            for pathTemp in os.listdir(self.filePath):
                if self.shotWorkCode == 'rig':
                    if pathTemp.split('.')[-1] in ['mb', 'abc', 'json']:
                        pathTempList.append(pathTemp)
                else:
                    if pathTemp.split('.')[-1] == 'mb':
                        pathTempList.append(pathTemp)
                        
            pathTempList.sort(reverse=1)
            self.ui.shotPub_listWidget.addItems(pathTempList)

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # Read Info (Asset)
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def readHistory(self, item):
        """
        read information from DB and xml. ex) comments, save date, artist info
        :param item: selected item from dev, pub list widgets. (.mb filename)
        """
        self.fileName = item.text()
        if self.focusWidget().objectName() == 'assetPub_listWidget':
            self.filePath = '/show/%s/asset/%s/%s/%s/pub/scenes' % (
            self.titleShort, self.assetType, self.asset, self.workCode)

            currentfile = self.ui.assetPub_listWidget.currentItem().getDirName()
            if os.path.splitext(currentfile)[-1] == ".asb":
                self.filePath = '/show/%s/asset/%s/%s/%s/pub/envlayout' % (
                    self.titleShort, self.assetType, self.asset, self.workCode)

            self.ui.assetDev_listWidget.setCurrentItem(None)

        if self.focusWidget().objectName() == 'assetDev_listWidget':
            self.filePath = '/show/%s/asset/%s/%s/%s/dev/scenes' % (
            self.titleShort, self.assetType, self.asset, self.workCode)

            self.ui.assetPub_listWidget.setCurrentItem(None)

        if self.focusWidget().objectName() == 'shotDev_listWidget':
            self.filePath = '/show/%s/shot/%s/%s/%s/dev/scenes' % (
            self.titleShort, self.shotType, self.shot, self.shotWorkCode)
            self.ui.shotPub_listWidget.setCurrentItem(None)

        if self.focusWidget().objectName() == 'shotPub_listWidget':
            self.filePath = '/show/%s/shot/%s/%s/%s/pub/scenes' % (
            self.titleShort, self.shotType, self.shot, self.shotWorkCode)
            self.ui.shotDev_listWidget.setCurrentItem(None)

        self.ui.filePath_lineEdit.setText(self.filePath)
        self.ui.fileName_lineEdit.setText(self.fileName)
        self.showSnapshot()
        self.readComment()

    def readDB(self):
        """
        read history from MongoDB and set infos on the history widget.
        """
        self.ui.history_treeWidget.clear()
        self.dbDic = historyAction.readDB(self.filePath)
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
                # asset
                if  self.ui.task_tab.currentIndex() == 0:
                    item = HistoryItem(self.ui.history_treeWidget)
                    version = '_'.join(i.split('_')[2:])

                # shot
                elif  self.ui.task_tab.currentIndex() == 1:
                    item = HistoryItem(self.ui.history_treeWidget)
                    version = '_'.join(i.split('_')[3:])

                else:
                    item = HistoryItem(self.ui.history_treeWidget)
                    version = '_'.join(i.split('_')[2:])

                if self.dbDic[i]['artist'] != None :
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

    def readComment(self):
        """
        read comment from DB and add on the comment text widget.
        """
        if os.path.splitext(self.fileName)[-1] == ".asb":
            return
        self.ui.comment_textEdit.clear()
        comment = historyAction.readComment(self.filePath, self.fileName)
        self.ui.comment_textEdit.setText(comment)
        commentDB = historyAction.readDBComment(self.filePath, self.fileName)
        if commentDB:
            # sort dictionary
            sortedKeys = (sorted(commentDB.keys(),
                                key=lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')))
            for keys in sortedKeys:
                self.ui.comment_textEdit.append('\n' + '=' * 48 + '\n')
                self.ui.comment_textEdit.append('Date: '+keys)
                self.ui.comment_textEdit.append('Artist: ' + commentDB[keys][1] + '\n')
                self.ui.comment_textEdit.append('Comment:\n'+commentDB[keys][0] + '\n')

        self.ui.comment_textEdit.moveCursor(QtGui.QTextCursor.End)

    def takeSnapShot(self):
        """
        take snapshot and show the image file on the snapshot label.
        """
        fileName = historyAction.takeSnapShot(self.filePath, self.fileName)
        self.showSnapshot()

    def showSnapshot(self):
        """
        show snapshot for asset task.        """

        try:
            showcode = self.projectDic[self.ui.show_comboBox.currentText()]['code']
        except:
            pass
        # # print 'filepath=', self.filePath, self.fileName, showcode
        imageFile = historyAction.showSnapShot(self.filePath, self.fileName, showcode=showcode )
        # # print 'image file=',imageFile, showcode
        self.setPixmapImage(self.ui.snapshot_label, imageFile, 350, 198)

    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    # My Task Tab
    # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
    def showTeamComboBox(self):
        """
        get team name from Tactic and add add them to the comboBox
        """
        params = {}
        params['api_key'] = API_KEY
        if dxConfig.getHouse() == 'KOR':
            params['department'] = 'LNR|RIG|ANI|AST|MCP|CRD|MMV'
        if dxConfig.getHouse() == 'CHN':
            params['department'] = 'CLR|CAN|CAS'
        infos = requests.get("http://%s/dexter/search/user.php" %( dxConfig.getConf('TACTIC_IP') ),
                             params=params).json()

        self.ui.team_comboBox.clear()
        teamList = []
        for i in infos:
            self.memberDic[i['name_kr']] = i

        for i in self.memberDic:
            if self.memberDic[i]['department'] in teamList:
                pass
            else:
                teamList.append(self.memberDic[i]['department'])

        teamList.sort()
        # if dxConfig.getHouse() == 'KOR':
        #     teamList.remove('ANI')
        #     teamList.remove('AST')
        self.ui.team_comboBox.addItems(teamList)
        self.showTeamMemberComboBox()

    def showTeamMemberComboBox(self):
        """
        show team member's name and add names to the team member comboBox.
        unicode:param team: selected name from the team name combobox
        """
        team = self.ui.team_comboBox.currentText()
        memberList = []
        for i in self.memberDic:
            if self.memberDic[i]['department'] == team:
                memberList.append(i)

        memberList.sort()
        self.ui.name_comboBox.clear()
        self.ui.name_comboBox.addItems(memberList)

    def currentName(self, item):
        """
        get current text on the name_comboBox
        :param item: item from the member name comboBox
        """
        if self.ui.name_comboBox.currentText():
            name = unicode(self.ui.name_comboBox.currentText())
            for i in self.memberDic:
                if name == i:
                    name_eng = self.memberDic[i]['code']

            self.currentTaskAssigned(name_eng)

    def currentTaskAssigned(self, name):
        """
        query from Tactic, a team member's tasks assigned.
        :param name: member name in english
        """
        params = {}
        params['api_key'] = API_KEY
        params['login'] = name
        infos = requests.get("http://%s/dexter/search/task.php" %( dxConfig.getConf('TACTIC_IP') ),
                             params=params).json()
        self.ui.myAsset_listWidget.clear()
        self.ui.myShotName_listWidget.clear()
        self.ui.nonMyAsset_listWidget.clear()
        self.ui.nonMyShot_listWidget.clear()
        assetList = []
        shotList = []
        for i in infos:
            name = i['extra_name']
            self.taskDic[name] = i
            if i['search_type'] == u'%s/asset?project=%s' % (self.showCode, self.showCode):
                if i['status'] in WORKING:
                    if not name in assetList:
                        assetList.append(name)
                        item = TaskItem(self.ui.myAsset_listWidget)
                        item.setText(name)
                        item.setItemColor(i['status'])
                if i['status'] in NONWORKING:
                    item = TaskItem(self.ui.nonMyAsset_listWidget)
                    item.setText(name)
                    item.setItemColor(i['status'])

            if i['search_type'] == u'%s/shot?project=%s' % (self.showCode, self.showCode):
                if i['status'] in WORKING:
                    if not name in shotList:
                        shotList.append(name)
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

    def myTaskMenu(self, pos):
        """
        right click action on my task tab.
        :param pos: position of the mouse cursor.
        """
        pos = pos + (QtCore.QPoint(20, 0))
        widget = self.focusWidget().objectName()
        if widget in ['myAsset_listWidget','nonMyAsset_listWidget']:
            path = '/show/%s/asset/%s/%s' % (self.titleShort, self.workType, self.workName)
        else:
            path = '/show/%s/shot/%s/%s' % (self.titleShort, self.workType, self.workName)

        self.path = path
        if not os.path.isdir(path):
            dialog = addDirectory.DirectoryWarning(QtWidgets.QDialog)
            dialog.show()
            result = dialog.exec_()
            if result == 1:
                os.makedirs(path)
                print 'folder created successfully.'
                return
            else:
                return

        else:
            menu = QtWidgets.QMenu()
            action1 = menu.addAction('Create folder')
            action1.triggered.connect(self.createFolderMenu)
            menu2 = menu.addMenu('Go To')
            menu.addSeparator()
            menu2.triggered.connect(self.menuQuick)

        dirList = []
        for tempPath in os.listdir(path):
            if os.path.isdir(os.path.join(path, tempPath)):
                dirList.append(tempPath)

        dirList.sort()
        for i in dirList:
            menu2.addAction(str(i))

        if self.focusWidget().objectName() == 'myAsset_listWidget':
            menu.exec_(self.ui.myAsset_listWidget.mapToGlobal(pos))
        if self.focusWidget().objectName() == 'myShotName_listWidget':
            menu.exec_(self.ui.myShotName_listWidget.mapToGlobal(pos))
        if self.focusWidget().objectName() == 'nonMyAsset_listWidget':
            menu.exec_(self.ui.nonMyAsset_listWidget.mapToGlobal(pos))
        if self.focusWidget().objectName() == 'nonMyShot_listWidget':
            menu.exec_(self.ui.nonMyShot_listWidget.mapToGlobal(pos))

    def createFolderMenu(self):
        dialog = addDirectory.NewWorkCodeDialog(QtWidgets.QDialog)
        dialog.show()
        result = dialog.exec_()
        if result == 1:
            self.workCode = dialog.item
            addDirectory.createWorkSpace(self.workCode,self.path)

    def getTaskType(self, item):
        self.workName = item.text()
        self.workType = self.taskDic[self.workName]['category_code']

    def menuQuick(self, item):
        """
        when right clicked, go directly to the task workcode from my task menu
        :param item: item selected from the context menu
        """
        inc_tool_by_user.run('action.Spanner2.menuQuick', getpass.getuser())
        self.workCode = item.text()
        if self.focusWidget().objectName() in ['myAsset_listWidget','nonMyAsset_listWidget']:
            self.assetQuick()
            searchItem = self.ui.workCode_listWidget.findItems(self.workCode, QtCore.Qt.MatchExactly)[0]
            self.ui.workCode_listWidget.setCurrentItem(searchItem)
            self.showAssetDevPub(searchItem)

        if self.focusWidget().objectName() in ['myShotName_listWidget','nonMyShot_listWidget']:
            self.shotQuick()
            searchItem = self.ui.shotWorkCode_listWidget.findItems(self.workCode, QtCore.Qt.MatchExactly)[0]
            self.ui.shotWorkCode_listWidget.setCurrentItem(searchItem)
            self.showShotDevPub(searchItem)

    def assetQuick(self):
        """
        when double clicked, go directly to the asset task from my task menu
        """
        inc_tool_by_user.run('action.Spanner2.assetQuick', getpass.getuser())
        self.workType = self.taskDic[self.workName]['category_code']
        self.ui.task_tab.setCurrentIndex(0)
        try:
            searchedItem = self.ui.assetType_listWidget.findItems(self.workType, QtCore.Qt.MatchExactly)[0]
            self.ui.assetType_listWidget.setCurrentItem(searchedItem)
            self.showAsset()
            searchAssetItem = self.ui.asset_listWidget.findItems(self.workName, QtCore.Qt.MatchExactly)[0]
            self.ui.asset_listWidget.setCurrentItem(searchAssetItem)
            self.showAssetWorkCode()
        except:
            self.ui.fileName_lineEdit.setText('project folder not exist')
            self.ui.fileName_lineEdit.setStyleSheet("color: rgb(255,0,0)")
            self.ui.workCode_listWidget.clear()

        self.ui.assetDev_listWidget.clear()
        self.ui.assetPub_listWidget.clear()

    def shotQuick(self):
        """
        when double clicked, go directly to the shot task from my task menu
        """
        inc_tool_by_user.run('action.Spanner2.shotQuick', getpass.getuser())
        self.workType = self.taskDic[self.workName]['category_code']
        self.ui.task_tab.setCurrentIndex(1)
        try:
            searchedItem = self.ui.shotType_listWidget.findItems(self.workType, QtCore.Qt.MatchExactly)[0]
            self.ui.shotType_listWidget.setCurrentItem(searchedItem)
            self.showShot()
            searchShotItem = self.ui.shot_listWidget.findItems(self.workName, QtCore.Qt.MatchExactly)[0]
            self.ui.shot_listWidget.setCurrentItem(searchShotItem)
            self.showShotWorkCode()
        except:
            self.ui.fileName_lineEdit.setText('project folder not exist')
            self.ui.fileName_lineEdit.setStyleSheet("color: rgb(255,0,0)")

        self.ui.shotDev_listWidget.clear()
        self.ui.shotPub_listWidget.clear()

    def openInventory(self):
        op = InventoryDialog(self)
        op.show()
        op.exec_()

    def openRigMalChecker(self):
        rmc = RigMalChecker()
        rmc.show()
        rmc.exec_()

    def closeEvent(self, event):
        self.saveUserInfo()

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

class HistoryItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

class OpenWaringDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("You have currently opened scene.\nDo you want to save current scene and proceed?\n")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.open_btn = QtWidgets.QPushButton("Don't Save")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.save_btn)
        layout2.addWidget(self.open_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2, 3, 0)
        self.setLayout(layout)
        self.setWindowTitle("Warning")
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        # connection
        self.close_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.save)
        self.open_btn.clicked.connect(self.open)

    def save(self):
        self.result = 'save'
        self.close()

    def open(self):
        self.result = 'open'
        self.close()

