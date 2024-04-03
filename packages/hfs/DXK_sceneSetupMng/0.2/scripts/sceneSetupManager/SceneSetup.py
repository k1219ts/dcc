# -*- coding: utf-8 -*-
import os
import sys
import datetime
import getpass
from pprint import pprint
import time
import json

# QT
import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore

# MAYA & HOUDINI
mainWindowName = "Maya"
try:
    import mayaAction as Action
    mainWindowName = "Maya"
except:
    import houdiniAction as Action
    mainWindowName = "Houdini"
print mainWindowName

# config setting
import pymongo
from pymongo import MongoClient
import dxConfig
SITE = dxConfig.getHouse()
DB_IP = dxConfig.getConf('DB_IP')
TACTIC_IP = dxConfig.getConf('TACTIC_IP')
DB_NAME = 'PIPE_PUB'
client = MongoClient(DB_IP)
db = client[DB_NAME]

# get task from tactic
import requests
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
from tactic_client_lib import TacticServerStub

# tool stats
from dxstats import inc_tool_by_user

# ui
from ui.ui_SceneSetup import Ui_Form
from items import *
from viewers import *

CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        inc_tool_by_user.run('sceneSetupManager', getpass.getuser())
        inc_tool_by_user.run('action.sceneSetupManager.open', getpass.getuser())
        self.startFrame = 0
        self.endFrame = 0
        self.alembicOpt = 'GPU'
        self.worldOpt = 'baked'
        self.zennOpt = 'static'
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle('Scene Setup Manager')

        # Maya & Houdini ui
        with open( CURRENTPATH + '/ui/%sStyleSheet' % mainWindowName, mode = 'r') as f:
            styleSheet = ""
            for str in f.readlines():
                styleSheet += str
            self.setStyleSheet(styleSheet)

        self.getUserSettings()
        self.setStyle()
        self.connections()
        self.getShowDB()
        self.getSequence()

        # from workspace MAYA
        if mainWindowName == 'Maya':
            show, seq, shot = Action.getWorkSpace()
            if show:
                index = self.ui.show_comboBox.findText(show, QtCore.Qt.MatchExactly)
                self.ui.show_comboBox.setCurrentIndex(index)
            if seq:
                index = self.ui.seq_comboBox.findText(seq, QtCore.Qt.MatchExactly)
                self.ui.seq_comboBox.setCurrentIndex(index)
            if shot:
                index = self.ui.shot_comboBox.findText(shot, QtCore.Qt.MatchExactly)
                self.ui.shot_comboBox.setCurrentIndex(index)
        elif mainWindowName  == "Houdini":
            show, seq, shot = Action.getWorkSpace()
            if show:
                index = self.ui.show_comboBox.findText(show, QtCore.Qt.MatchExactly)
                self.ui.show_comboBox.setCurrentIndex(index)
            if seq:
                index = self.ui.seq_comboBox.findText(seq, QtCore.Qt.MatchExactly)
                self.ui.seq_comboBox.setCurrentIndex(index)
            if shot:
                index = self.ui.shot_comboBox.findText(shot, QtCore.Qt.MatchExactly)
                self.ui.shot_comboBox.setCurrentIndex(index)

    def connections(self):
        ### SHOT
        self.ui.show_comboBox.currentIndexChanged.connect(self.getSequence)
        self.ui.seq_comboBox.currentIndexChanged.connect(self.getShot)
        self.ui.shot_comboBox.currentIndexChanged.connect(self.getShotData)
        self.ui.shot_comboBox.currentIndexChanged.connect(self.saveUserInfo)
        self.ui.quick_listWidget.itemClicked.connect(self.goToShot)
        self.ui.findShot_lineEdit.textChanged.connect(self.goToShot)
        self.ui.lineEdit.textChanged.connect(self.goToShot)
        self.ui.dataTypeList_treeWidget.itemClicked.connect(self.getDatabase)

        ### ASSET
        self.ui.show_comboBox.currentIndexChanged.connect(self.getAssetType)
        self.ui.assetType_comboBox.activated.connect(self.getAssetName)
        self.ui.assetName_comboBox.currentIndexChanged.connect(self.getAssetData)
        self.ui.asset_dataTypeList_treeWidget.itemClicked.connect(self.getDatabase)

        self.ui.cancel_pushButton.clicked.connect(self.close)
        self.ui.cancel_pushButton.clicked.connect(self.saveUserInfo)
        self.ui.ReadMoreDB_pushButton.clicked.connect(self.readMoreDataBase)
        self.ui.db_treeWidget.itemClicked.connect(self.updateVer)
        self.ui.db_treeWidget.itemDoubleClicked.connect(self.showDatabase)
        self.ui.addQuick_pushButton.clicked.connect(self.addToQuick)
        self.ui.delQuick_pushButton.clicked.connect(self.delQuick)
        self.ui.spanner2_pushButton.clicked.connect(showSpanner2)
        self.ui.renderSpool_pushButton.clicked.connect(showRenderSpool)

        if mainWindowName == "Houdini":
            self.ui.dataTypeList_treeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.ui.dataTypeList_treeWidget.customContextMenuRequested.connect(self.rmbClicked)

        try:
            self.ui.import_pushButton.clicked.connect(self.doImport)
            self.ui.import_pushButton.clicked.connect(self.saveUserInfo)
        except:
            pass

    # -------------------------------------------------------------------------------
    #   SETTINGS
    # -------------------------------------------------------------------------------
    def setStyle(self):
        # self.resize(1300, 800)
        self.setMinimumSize(1500, 800)
        self.ui.splitter.handle(1).setMinimumHeight(10)

        ### STYLESHEET
        TreeWidget_styles = """
        QTreeWidget { background: rgb(80, 80, 80); color: white; }
        QTreeWidget::item { padding: 0 10 0 10 px; margin: 0px; border: 0 px}
        QTreeWidget::item:selected{background: rgb(67, 124, 185);}
        QTreeWidget:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        QTreeWidget:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        """
        TreeWidget_styles2 = """
        QTreeWidget { background: rgb(80, 80, 80); color: white; }
        QTreeWidget::item { padding: 5 10 5 10 px; margin: 0px; border: 0 px}
        QTreeWidget::item:selected{background: rgb(67, 124, 185);}
        QTreeWidget:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        QTreeWidget:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        """
        comboBox_Style = '''
        QComboBox { padding : 0 5 0 5 }
        QComboBox QAbstractItemView::item {
                                            background: rgb(60, 60, 60);
                                            padding: 0 5 0 5 px; margin: 0px; border: 0 px;
                                            min-height: 25px; min-width: 120px; max-height: 250px;
                                            }
        QComboBox QAbstractItemView::item:selected { background: rgb(110, 100, 160); }
        QComboBox QAbstractItemView { font-size: 10pt;}
        '''

        lineEdit_Style = '''
        QLineEdit:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        QLineEdit:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        '''

        listWidget_style = '''
        QListWidget { background: rgb(80, 80, 80); color: white; font: 10px; }
        QListWidget::item { padding: 7 10 5 10 px; margin: 0px; border: 0 px}
        QListWidget::item:hover { background: rgb(170, 170, 255, 150); }
        QListWidget::item:selected{background: rgb(170, 170, 255, 150);}
        QListWidget:focus { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        QListWidget:hover { border: 2px solid rgb(170, 170, 255, 150); border-radius: 1px; }
        '''

        tabWidget_style = '''
        QTabBar {background: rgb(70, 70, 70); color: white; }
        '''
        self.ui.groupBox_2.setStyleSheet(''' QFrame { background-color: #494949; }
                                           QRadioButton { background-color: #494949; } ''')
        self.ui.DBdatatype_label.setStyleSheet('''QLabel { color: #c59ef1; }''')
        self.ui.findShot_lineEdit.setStyleSheet(lineEdit_Style)
        self.ui.lineEdit.setStyleSheet(lineEdit_Style)
        self.ui.dataTypeList_treeWidget.setStyleSheet(TreeWidget_styles)
        self.ui.asset_dataTypeList_treeWidget.setStyleSheet(TreeWidget_styles)
        self.ui.db_treeWidget.setStyleSheet(TreeWidget_styles2)
        self.ui.quick_listWidget.setStyleSheet(listWidget_style)
        self.ui.tabWidget.setStyleSheet(tabWidget_style)
        # self.ui.tabWidget.setDocumentMode(1)

        # SHOT
        self.ui.show_comboBox.setView(QtWidgets.QListView())
        self.ui.seq_comboBox.setView(QtWidgets.QListView())
        self.ui.shot_comboBox.setView(QtWidgets.QListView())
        self.ui.show_comboBox.setStyleSheet(comboBox_Style)
        self.ui.seq_comboBox.setStyleSheet(comboBox_Style)
        self.ui.shot_comboBox.setStyleSheet(comboBox_Style)
        # ASSET
        self.ui.assetType_comboBox.setView(QtWidgets.QListView())
        self.ui.assetName_comboBox.setView(QtWidgets.QListView())
        self.ui.assetType_comboBox.setStyleSheet(comboBox_Style)
        self.ui.assetName_comboBox.setStyleSheet(comboBox_Style)

        ### IMAGE & ICON
        self.ui.delQuick_pushButton.setIcon(
            QtGui.QIcon(QtGui.QPixmap( os.path.join( CURRENTPATH, "resource/trashCan.png" ) )))
        self.ui.spanner2_pushButton.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/spanner_icon.png"))))
        self.ui.renderSpool_pushButton.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/pxrman.png"))))
        image = QtGui.QPixmap(os.path.join( CURRENTPATH, "resource/sceneSetup.png" ) )
        image = image.scaled(50, 50, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        self.ui.logo_label.setPixmap(image)
        label = QtWidgets.QLabel(self.ui.splitter.handle(1))
        image = QtGui.QPixmap(os.path.join(CURRENTPATH, "resource/drop_arrow.png"))
        image = image.scaled(15, 15, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        label.setPixmap(image)

        # SHOT
        self.ui.dataTypeList_treeWidget.header().resizeSection(0, 40)
        self.ui.dataTypeList_treeWidget.setHeaderLabel('')
        self.ui.dataTypeList_treeWidget.header().resizeSection(1, 40)
        self.ui.dataTypeList_treeWidget.header().resizeSection(2, 40)
        self.ui.dataTypeList_treeWidget.header().resizeSection(3, 200)
        self.ui.dataTypeList_treeWidget.header().resizeSection(4, 80)
        self.ui.dataTypeList_treeWidget.header().resizeSection(5, 120)
        self.ui.dataTypeList_treeWidget.header().resizeSection(6, 80)
        self.ui.dataTypeList_treeWidget.header().resizeSection(7, 120)
        # ASSET
        self.ui.asset_dataTypeList_treeWidget.header().resizeSection(0, 40)
        self.ui.asset_dataTypeList_treeWidget.setHeaderLabel('')
        self.ui.asset_dataTypeList_treeWidget.header().resizeSection(1, 80)
        self.ui.asset_dataTypeList_treeWidget.header().resizeSection(2, 80)

        ### DBVIEWER
        self.ui.db_treeWidget.header().resizeSection(0, 80)
        self.ui.db_treeWidget.header().resizeSection(1, 50)
        self.ui.db_treeWidget.header().resizeSection(2, 100)


    def getUserSettings(self):
        db = client['PUBLISH']
        coll = db['sceneSetup_user']
        result = coll.find({'name': getpass.getuser()})
        if result.count() > 0:
            if 'tabIndex' in result[0]:
                self.ui.tabWidget.setCurrentIndex(result[0]['tabIndex'])
            self.quickList = result[0]['quickList']
            self.ui.quick_listWidget.clear()
            self.ui.quick_listWidget.addItems(self.quickList.keys())
            if result[0]['alembic_setting'] == 'GPU':
                self.ui.GPU_radioButton.setChecked(True)
            if result[0]['alembic_setting'] == 'mesh':
                self.ui.Mesh_radioButton.setChecked(True)

            if result[0]['world_setting'] == 'none':
                self.ui.None_radioButton.setChecked(True)
            if result[0]['world_setting'] == 'baked':
                self.ui.baked_radioButton.setChecked(True)
            if result[0]['world_setting'] == 'seperate':
                self.ui.seperate_radioButton.setChecked(True)

            if result[0]['zenn_setting'] == 'simulation':
                self.ui.simulation_radioButton.setChecked(True)
            if result[0]['zenn_setting'] == 'static':
                self.ui.static_radioButton.setChecked(True)

        else:
            self.quickList = {}
            self.ui.GPU_radioButton.setChecked(True)
            self.ui.baked_radioButton.setChecked(True)
            self.ui.static_radioButton.setChecked(True)

    def saveUserInfo(self):
        userShow = self.ui.show_comboBox.currentIndex()
        userSeq = self.ui.show_comboBox.currentIndex()
        userShot = self.ui.shot_comboBox.currentIndex()
        db = client['PUBLISH']
        coll = db['sceneSetup_user']
        userData = ({
            'tabIndex' : self.ui.tabWidget.currentIndex(),
            'name'     : getpass.getuser(),
            'show'     : userShow,
            'seq'      : userSeq,
            'shot'     : userShot,
            'quickList': self.quickList,
            'alembic_setting': self.alembicOpt,
            'world_setting': self.worldOpt,
            'zenn_setting': self.zennOpt
        })
        coll.update({ 'name': getpass.getuser() }, { '$set': userData }, upsert=True)

    def addToQuick(self):
        show = str ( self.ui.show_comboBox.currentText() )
        if self.ui.tabWidget.currentIndex() == 0:
            seq = str( self.ui.seq_comboBox.currentText() )
            shot = str( self.ui.shot_comboBox.currentText() )
            if not shot in self.quickList.keys():
                self.quickList[shot] = [ show, seq ]

        for i in self.quickList.keys():
            if not self.ui.quick_listWidget.findItems(i, QtCore.Qt.MatchExactly):
                self.ui.quick_listWidget.addItem(i)
        self.ui.quick_listWidget.sortItems(QtCore.Qt.AscendingOrder)
        self.saveUserInfo()

    def delQuick(self):
        sels = self.ui.quick_listWidget.selectedItems()
        for i in sels:
            if i.text() in self.quickList:
                self.quickList.pop( i.text() )
                row = self.ui.quick_listWidget.row( i )
                self.ui.quick_listWidget.takeItem( row )
        self.saveUserInfo()

    def goToShot(self, item=None):
                
        if self.focusWidget().objectName() == 'quick_listWidget':
            shot = item.text()
            show = self.quickList[shot][0]
            seq = self.quickList[shot][1]
            index = self.ui.show_comboBox.findText(show, QtCore.Qt.MatchExactly)
            self.ui.show_comboBox.setCurrentIndex(index)
            index = self.ui.seq_comboBox.findText(seq, QtCore.Qt.MatchExactly)
            self.ui.seq_comboBox.setCurrentIndex(index)
            index = self.ui.shot_comboBox.findText( shot, QtCore.Qt.MatchExactly )
            self.ui.shot_comboBox.setCurrentIndex( index )

        if self.focusWidget().objectName() == 'findShot_lineEdit':
            try:
                text = self.ui.findShot_lineEdit.text()
                index = self.ui.shot_comboBox.findText(text, QtCore.Qt.MatchContains)
                self.ui.shot_comboBox.setCurrentIndex( index )
            except: pass

        if self.focusWidget().objectName() == 'lineEdit':
            try:
                text = self.ui.lineEdit.text()
                index = self.ui.assetName_comboBox.findText(text, QtCore.Qt.MatchContains)
                self.ui.assetName_comboBox.setCurrentIndex( index )
            except: pass

    def getShowDB(self):
        params = dict()
        params['api_key'] = API_KEY
        params['category'] = 'Active'
        infos = requests.get("http://%s/dexter/search/project.php" %TACTIC_IP,
                             params=params).json()
        self.titleDic = {}
        for i in infos:
            if os.path.exists( os.path.join( '/show', i['name'] ) ):
                self.titleDic[i['name']] = i['code']
        self.titleDic['testshot'] = ['testshot']
        self.ui.show_comboBox.addItems(self.titleDic.keys())

    # -------------------------------------------------------------------------------
    #   DATABASE - SHOT
    # -------------------------------------------------------------------------------
    def getSequence(self):
        if SITE == 'CHN':
            self.title = self.ui.show_comboBox.currentText()
            coll = db[self.title]
            result = coll.find().distinct('sequence')
            self.ui.seq_comboBox.clear()
            self.ui.seq_comboBox.addItems( sorted(result) )

        else:
            self.title = self.ui.show_comboBox.currentText()
            showPath = os.path.join('/show/%s/shot' % self.title)
            self.ui.seq_comboBox.clear()
            if os.path.exists(showPath):
                for i in sorted(os.listdir(showPath)):
                    if os.path.exists(os.path.join(showPath, i)) and not i.startswith('.'):
                        self.ui.seq_comboBox.addItem(i)
            else:
                return

    def getShot(self):
        if SITE == 'CHN':
            self.title = self.ui.show_comboBox.currentText()
            coll = db[self.title]
            seq = self.ui.seq_comboBox.currentText()
            result = coll.find({'sequence':seq}).distinct('shot')
            self.ui.shot_comboBox.clear()
            self.ui.shot_comboBox.addItems( sorted(result) )

        else:
            self.seq = self.ui.seq_comboBox.currentText()
            shotPath = os.path.join('/show/%s/shot/%s' % (self.title, self.seq))
            self.ui.shot_comboBox.clear()
            shotList = []
            if os.path.isdir(shotPath):
                for i in os.listdir(shotPath):
                    if os.path.exists(os.path.join(shotPath, i)) and not i.startswith('.'):
                        shotList.append(i)
                self.ui.shot_comboBox.addItems( sorted(shotList) )

    def getShotData(self):
        coll = db[self.title]
        shot = self.ui.shot_comboBox.currentText()
        result = coll.find({ 'shot': shot }).distinct('data_type')
        
        # tree widget clear
        while self.ui.dataTypeList_treeWidget.topLevelItemCount() > 0:
            self.ui.dataTypeList_treeWidget.takeTopLevelItem(0)
            treeitem = self.ui.dataTypeList_treeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()
                
        if 'camera' in result:
            camDic = self.getEnvVer( 'camera')
        if 'assembly' in result:
            assemDic = self.getEnvVer('assembly')

        # get asset list
        if 'geoCache' in result and 'zenn' not in result:
            self.getAssetVer()
        if 'geoCache' in result and 'zenn' in result:
            self.getAssetVer(zenn='zenn')

        sr = coll.find({'shot':shot, 'data_type':'scene_graph_reference'})
        if sr.count() > 0:
            self.sceneSetupData = sr[0]

    def getEnvVer(self, datatype):
        coll = db[self.title]
        shot = self.ui.shot_comboBox.currentText()
        latestDic = coll.find({'shot': shot, 'data_type': datatype, 'enabled':True },
                           sort=[('version', pymongo.DESCENDING)]).limit(1)
        if latestDic.count():
            typeItem = TreeWidget_CheckableItem(self.ui.dataTypeList_treeWidget, latestDic[0])
            typeItem.importVer.setText(str(latestDic[0]['version']))

            if latestDic[0]['task_publish'].has_key('startFrame'):
                self.startFrame = latestDic[0]['task_publish']['startFrame']
                self.endFrame = latestDic[0]['task_publish']['endFrame']

            return latestDic[0]

    def getAssetVer(self, zenn=None):
        coll = db[self.title]
        shot = self.ui.shot_comboBox.currentText()
        assetDic = {}
        result = coll.find({"data_type": "geoCache", 'enabled': True, 'shot': shot
                            },sort=[('version', pymongo.DESCENDING)]).distinct('files.assets')
        for i in result:
            assetDic.update(i)
            
        assets = []
        for i in assetDic.keys():
            if not i.split(':')[-1] in assets:
                assets.append(i.split(':')[-1])

        assetTopItem = TreeWidget_CheckableItem(self.ui.dataTypeList_treeWidget, assetItem=True)
        assetTopItem.setText(4, 'assets')

        assetlist = []
        for i in assets:
            childItem = TreeWidget_CheckableItem(self.ui.dataTypeList_treeWidget)
            childItem.setText(3, i.split(':')[-1])
            assetlist.append(childItem)

        geover = 0
        for asset in assetDic.keys():
            found = self.ui.dataTypeList_treeWidget.findItems(asset.split(':')[-1], QtCore.Qt.MatchExactly, 3)[0]
            assetChild = TreeWidgetChild_CheckableItem(found, asset, assetDic)

            # get GEO versions
            geoList = []
            result = coll.find({"data_type": "geoCache", 'enabled': True,
                                'shot': shot,
                                'files.assets.%s' % asset: {"$exists": True}
                                }).distinct('version')
            if result:
                geoList = [str(i).zfill(2) for i in result]
                assetChild.geoVer.addItems(sorted(geoList, reverse=1))

            # get ZENN versions
            if zenn:
                zennList = []
                result = coll.find({"data_type": "zenn", 'enabled': True,
                                    'shot': shot,
                                    'files.assets.%s' % asset: {"$exists": True}
                                    }).distinct('version')
                if result:
                    zennList = [str(i).zfill(2) for i in result]
                    assetChild.zennVer.addItems(sorted(zennList, reverse=1))

            self.setCheckAssets(found)
            
        self.ui.dataTypeList_treeWidget.expandAll()
        self.ui.dataTypeList_treeWidget.sortItems(3, QtCore.Qt.AscendingOrder)

    def setCheckAssets(self, assetParent):
        # CHECK LAST VERSION
        for i in range(assetParent.childCount()):
            childItem = assetParent.child(i)
            childItem.geoCheck.setChecked(True)
            childItem.zennCheck.setChecked(True)

    def updateVer(self, item):
        datatype = item.text(0)
        dataver = item.text(1)
        index = self.ui.tabWidget.currentIndex()
        if index == 0:
            if datatype not in 'geoCache, zenn':
                found = self.ui.dataTypeList_treeWidget.findItems(datatype, QtCore.Qt.MatchContains, 4)[0]
                found.importVer.setText(dataver)
                found.setDict(item.getDict())

        if index == 1:
            found = self.ui.asset_dataTypeList_treeWidget.findItems(datatype, QtCore.Qt.MatchContains, 1)[0]
            found.importVer.setText(dataver)
            found.setDict(item.getDict())

    def getDatabase(self, item):
        coll = db[self.title]
        shot = self.ui.shot_comboBox.currentText()
        assetName = self.ui.assetName_comboBox.currentText()
        if self.focusWidget().objectName() == 'dataTypeList_treeWidget':
            seltype = item.text(self.ui.dataTypeList_treeWidget.currentColumn())
            if item.text(4) == 'geoCache':
                assetKey = self.ui.dataTypeList_treeWidget.currentItem().text(3)
            self.ui.DBdatatype_label.setText(seltype.upper())
            result = coll.find({ 'shot': shot,'data_type':seltype, 'enabled': True },
                                   sort=[('version', pymongo.DESCENDING)]).limit(10)

        if self.focusWidget().objectName() == 'asset_dataTypeList_treeWidget':
            seltype = item.text(self.ui.asset_dataTypeList_treeWidget.currentColumn())
            self.ui.DBdatatype_label.setText(seltype.upper())
            if not seltype == 'rig':
                result = coll.find({ 'asset_name': assetName,'data_type':seltype, 'enabled': True },
                                   sort=[('version', pymongo.DESCENDING)]).limit(10)
            else:
                result = coll.find({'asset': assetName, 'data_type': seltype, 'enabled': True},
                                   sort=[('version', pymongo.DESCENDING)]).limit(10)

        self.ui.db_treeWidget.clear()
        try:
            for i in result:
                dbItem = DataBaseViewerItem(self.ui.db_treeWidget)
                dbItem.setDict(i)
                dbItem.setTexts()
                if 'files' in i:
                    if self.focusWidget().objectName() == 'dataTypeList_treeWidget':
                        if 'assets' in i['files']:
                            if assetKey in i['files']['assets']:
                                dbItem.setText( 4, str(os.path.basename( i['files']['assets'][assetKey]['path'][0] )) )
                                
                        if 'camera_path' in i['files']:
                            dbItem.setText( 4, str(os.path.basename( i['files']['camera_path'][0] )) )
                            
                    if self.focusWidget().objectName() == 'asset_dataTypeList_treeWidget':
                        if 'abc' in i['files']:
                            dbItem.setText(4, str( os.path.basename( i['files']['abc'][0] ) ))
                        if 'maya_dev_file' in i['files']:
                            dbItem.setText( 4, str(os.path.basename( i['files']['maya_dev_file'][0] ) ) )
                    
                    if i['data_type'] == 'assembly':
                        if 'path' in i['files']:
                            for k in i['files']['path']:
                                if os.path.splitext(k)[-1] == '.asb':
                                    dbItem.setText(4, str( os.path.basename( k ) ))
                        if 'assembly' in i['files']:
                            dbItem.setText(4, str( os.path.basename( i['files']['assembly'][0] ) ))


                b = QtGui.QBrush(QtGui.QColor(255, 100, 100, 255))
                dbItem.dataPath = i

        except Exception as e:
            print e.message
            print 'NO DB RECORDS'

    def readMoreDataBase(self):
        # tool stats
        inc_tool_by_user.run('action.sceneSetupManager.readMoreDataBase', getpass.getuser())
        coll = db[self.title]
        shot = self.ui.shot_comboBox.currentText()
        dataType = self.ui.db_treeWidget.itemAt(0, 0).text(0)
        endNum = self.ui.db_treeWidget.topLevelItemCount()
        lastItem = self.ui.db_treeWidget.topLevelItem( int(endNum-1) )
        lastVer = lastItem.text(1)
        if self.ui.tabWidget.currentIndex() == 0:
            if self.ui.dataTypeList_treeWidget.currentItem().text(4) == 'geoCache':
                assetKey = self.ui.dataTypeList_treeWidget.currentItem().text(3)
            result = coll.find({'shot': shot, 'data_type': dataType, 'enabled': True,
                                'version': {'$gt': int(lastVer) - 10, '$lt': int(lastVer) }},
                               sort=[('version', pymongo.DESCENDING)]).limit(10)

        if self.ui.tabWidget.currentIndex() == 1:
            assetName = self.ui.assetName_comboBox.currentText()
            if not dataType == 'rig':
                result = coll.find({ 'asset_name': assetName,'data_type':dataType, 'enabled': True,
                                     'version': {'$gt': int(lastVer) - 10, '$lt': int(lastVer)}},
                                   sort=[('version', pymongo.DESCENDING)]).limit(10)
            else:
                result = coll.find({'asset': assetName, 'data_type': dataType, 'enabled': True,
                                    'version': {'$gt': int(lastVer) - 10, '$lt': int(lastVer)}},
                                   sort=[('version', pymongo.DESCENDING)]).limit(10)

        if result.count() > 0:
            for i in result:
                dbItem = DataBaseViewerItem(self.ui.db_treeWidget)
                dbItem.setDict(i)
                dbItem.setTexts()
                if 'files' in i:
                    if self.ui.tabWidget.currentIndex() == 0:
                        if 'assets' in i['files']:
                            if assetKey in i['files']['assets']:
                                dbItem.setText(4, str(os.path.basename( i['files']['assets'][assetKey]['path'][0] ) ))
                                
                        if 'camera_path' in i['files']:
                            dbItem.setText( 4, str(os.path.basename( i['files']['camera_path'][0] )) )
                            
                    if self.ui.tabWidget.currentIndex() == 1:
                        if 'abc' in i['files']:
                            dbItem.setText(4, str(os.path.basename( i['files']['abc'][0] ) ))
                        if 'maya_dev_file' in i['files']:
                            dbItem.setText( 4, str(os.path.basename( i['files']['maya_dev_file'][0] )) )
                            
                    if i['data_type'] == 'assembly':
                        if 'path' in i['files']:
                            for k in i['files']['path']:
                                if os.path.splitext(k)[-1] == '.asb':
                                    dbItem.setText(4, str( os.path.basename( k ) ))
                        if 'assembly' in i['files']:
                            dbItem.setText(4, str( os.path.basename( i['files']['assembly'][0] ) ))

                dbItem.dataPath = i

    def showDatabase(self, item):
        # tool stats
        inc_tool_by_user.run('action.sceneSetupManager.showDataBase', getpass.getuser())
        self.dviewer = DataBaseViewer( None, item.dataDict)
        self.dviewer.show()

    # -------------------------------------------------------------------------------
    #   DATABASE - ASSET
    # -------------------------------------------------------------------------------
    def getAssetType(self):
        self.title = self.ui.show_comboBox.currentText()
        code = self.titleDic[self.ui.show_comboBox.currentText()]
        server = TacticServerStub(login='taehyung.lee', password='dlxogud',
                                  server=dxConfig.getConf('TACTIC_IP'),
                                  project=code)

        shot_exp = "@SOBJECT(%s/asset[])" % code
        infos = server.eval(shot_exp)
        self.assetDic = {}
        for i in infos:
            if i['asset_category_code'] in self.assetDic.keys():
                if not i['name'] in self.assetDic[i['asset_category_code']]:
                    self.assetDic[i['asset_category_code']].append(i['name'])
            else:
                self.assetDic[i['asset_category_code']] = [i['asset_category_code']]

        firstkey = sorted( self.assetDic.keys() )[0]
        self.ui.assetType_comboBox.clear()
        self.ui.assetName_comboBox.clear()
        self.ui.assetType_comboBox.addItems( sorted( self.assetDic.keys() ) )
        self.ui.assetName_comboBox.addItems( sorted( self.assetDic[ firstkey ])  )

    def getAssetName(self):
        assetType = self.ui.assetType_comboBox.currentText()
        assetNames = self.assetDic[assetType]
        self.ui.assetName_comboBox.clear()
        self.ui.assetName_comboBox.addItems( sorted(assetNames) )

    def getAssetData(self):
        coll = db[self.title]
        assetName = self.ui.assetName_comboBox.currentText()
        self.ui.asset_dataTypeList_treeWidget.clear()

        # ASSET
        dataTypes = coll.find({'asset_name': assetName }).distinct('data_type')
        if dataTypes:
            for i in sorted( dataTypes ):
                result = coll.find({ 'asset_name': assetName, 'data_type':i }).sort('version',pymongo.DESCENDING).limit(1)
                if result.count() > 0:
                    typeItem = Asset_TreeWidget_CheckableItem( self.ui.asset_dataTypeList_treeWidget, i )
                    typeItem.setDict(result[0])
                    typeItem.importVer.setText( str(result[0]['version']) )

        # RIG
        result = coll.find({'asset': assetName, 'data_type':'rig'}).sort('version',pymongo.DESCENDING).limit(1)
        if result.count() > 0:
            typeItem = Asset_TreeWidget_CheckableItem( self.ui.asset_dataTypeList_treeWidget, 'rig' )
            typeItem.setDict(result[0])
            typeItem.importVer.setText(str(result[0]['version']))

    def getAssetCheckedKey(self):
        self.camData = dict()
        self.geoData = dict()
        self.zennData = dict()
        self.assemData = dict()
        self.assetData = dict()
        for i in range(self.ui.asset_dataTypeList_treeWidget.topLevelItemCount()):
            coll = db[self.ui.show_comboBox.currentText()]
            assetName = self.ui.assetName_comboBox.currentText()
            item = self.ui.asset_dataTypeList_treeWidget.topLevelItem(i)
            self.assetData['name'] = assetName
            if item.text(1) == 'model':
                if item.importCheck.isChecked():
                    self.assetData['model_path'] = item.dataDic['files']['abc']
                    self.assetData['model_backup'] = item.dataDic['files']['backup']
                    if 'scene' in item.dataDic['files']:
                        self.assetData['model_scene'] = item.dataDic['files']['scene']

            if item.text(1) == 'assembly':
                if item.importCheck.isChecked():
                    self.assetData['assembly_path'] = item.dataDic['files']['assembly']
                    self.assetData['assembly_json'] = item.dataDic['files']['json_file']
                    if 'file' in item.dataDic['files']:
                        self.assetData['assembly_file'] = item.dataDic['files']['file']

            if item.text(1) == 'rig':
                if item.importCheck.isChecked():
                    self.assetData['rig_path'] = item.dataDic['files']['maya_path']

            if item.text(1) == 'texture' and SITE == 'CHN':
                self.assetData['texture_proxy'] = item.dataDic['files']['texture_proxy_dir_path']


    # -------------------------------------------------------------------------------
    #   IMPORT - SHOT
    # -------------------------------------------------------------------------------
    def doImport(self):
        # tool stats
        inc_tool_by_user.run('action.sceneSetupManager.doImport', getpass.getuser())
        # IMPORT SHOT
        imd = ImportDialog()
        if imd.result == 'import':
            if self.ui.tabWidget.currentIndex() == 0:
                self.doImportInfo()
                self.getCheckedKey()
                if SITE == 'CHN':
                    result = self.fileCheck()
                Action.importCache(
                    self.startFrame, self.endFrame,
                    camData=self.camData, assemData=self.assemData, geoData=self.geoData, zennData=self.zennData,
                    worldOpt= self.worldOpt, alembicOpt=self.alembicOpt, shot = self.ui.shot_comboBox.currentText())

            # IMPORT ASSET
            if self.ui.tabWidget.currentIndex() == 1:
                self.getAssetCheckedKey()
                print self.assetData
                if SITE == 'CHN':
                    result = self.fileCheck()
                Action.importAssetCache(self.assetData)

        if imd.result == 'update':
            # tool stats
            inc_tool_by_user.run('action.sceneSetupManager.doUpdate', getpass.getuser())
            if self.ui.tabWidget.currentIndex() == 0:
                self.doImportInfo()
                self.getCheckedKey()
                if SITE == 'CHN':
                    result = self.fileCheck()
                Action.updateCache(
                    self.startFrame, self.endFrame,
                    camData=self.camData, assemData=self.assemData, geoData=self.geoData, zennData=self.zennData,
                    worldOpt=self.worldOpt, alembicOpt=self.alembicOpt, shot = self.ui.shot_comboBox.currentText())
            # if self.ui.tabWidget.currentIndex() == 1:
            #     self.getAssetCheckedKey()
            #     if SITE == 'CHN':
            #         result = self.fileCheck()
            #     Action.updateAssetCache(self.assetData)


    def getCheckedKey(self):
        self.camData = dict()
        self.geoData = dict()
        self.zennData = dict()
        self.assemData = dict()
        self.assetData = dict()
        for i in range(self.ui.dataTypeList_treeWidget.topLevelItemCount()):
            item = self.ui.dataTypeList_treeWidget.topLevelItem(i)

            if item.text(4) == 'camera':
                if item.importCheck.isChecked():
                    print item.dataDic['files']
                    self.camData = {
                    'maya_dev_file' : item.dataDic['files']['maya_dev_file'],
                    'camera_path' : item.dataDic['files']['camera_path']
                    }
                    if item.dataDic['files'].has_key('panzoom_json_path'):
                        self.camData['panzoom_json_path'] = item.dataDic['files']['panzoom_json_path']

                    if item.dataDic['files'].has_key("imageplane_json_path"):
                        self.camData['imageplane_json_path'] = item.dataDic['files']['imageplane_json_path']

                    if item.dataDic['task'] == 'matchmove':
                        if item.dataDic['files'].has_key('camera_geo_path'):
                            self.camData['camera_geo_path'] = item.dataDic['files']['camera_geo_path']

                        if item.dataDic['files'].has_key('camera_loc_path'):
                            self.camData['camera_loc_path'] = item.dataDic['files']['camera_loc_path']
                        
                        if item.dataDic['files'].has_key('camera_asset_loc_path'):
                            self.camData['camera_asset_loc_path'] = item.dataDic['files']['camera_asset_loc_path']
                        

            if item.text(4) == 'assembly':
                if item.importCheck.isChecked():
                    self.assemData = {
                        'path' :  item.dataDic['files']['assembly'],
                        'json':  item.dataDic['files']['json_file']
                        }
                    if 'file' in  item.dataDic['files']:
                        self.assemData.update({'assembly_file': item.dataDic['files']['file']})

            else:
                ### geoCache ###
                for j in range(item.childCount()):
                    childItem = item.child(j)
                    if childItem.geoCheck.isChecked() and childItem.text(4) == 'geoCache':
                        if childItem.geoVer.count() > 0:
                            assetName = childItem.text(3)
                            print assetName
                            geoVer = int( childItem.geoVer.currentText() )
                            coll = db[self.title]
                            shot = self.ui.shot_comboBox.currentText()
                            result = coll.find({ 'shot': shot, 'data_type':'geoCache', 'version': geoVer, 'enabled': True })
                            print '### geo:', result[0]['version']
                            if result.count() > 0:
                                childItem.geoPath = result[0]['files']['assets'][assetName]
                                self.geoData[assetName] = childItem.geoPath
                                self.geoData['maya_dev_file'] = result[0]['files']['maya_dev_file']

                ### zenn ###
                for j in range(item.childCount()):
                    childItem = item.child(j)
                    if childItem.zennCheck.isChecked() and childItem.text(6) == 'zenn':
                        assetName = childItem.text(3)
                        assetBasename = assetName.split(':')[-1]
                        print '###assetname : ', assetBasename
                        coll = db[self.title]
                        shot = self.ui.shot_comboBox.currentText()

                        ### STATIC ###
                        print '### zennOpt', self.zennOpt
                        if self.zennOpt == 'static':
                            result = coll.find({'task': 'asset',
                                                'data_type': 'zenn',
                                                'asset_name': assetBasename
                                                }).sort('version', pymongo.DESCENDING).limit(1)
                            if result.count() > 0:
                                self.zennData[assetName] = result[0]['files']
                                self.zennData[assetName]['hairTask'] = False
                            else:
                                print '### NO ZENN STATIC CACHE FROM ASSET'

                        ### SIMULATION ###
                        if self.zennOpt == 'simulation':
                            if childItem.zennVer.count() > 0:
                                zennVer = int(childItem.zennVer.currentText())
                                coll = db[self.title]
                                result = coll.find({
                                    'shot': shot,
                                    'data_type': 'zenn',
                                    'enabled': True,
                                    'version': zennVer}).limit(1)
                                if result.count() > 0:
                                    self.zennData[assetName] = result[0]['files']['assets'][assetName]
                                    self.zennData[assetName]['hairTask'] = True

                        ### AUTO ###
                        if self.zennOpt == 'auto':
                            coll = db[self.title]
                            ### zenn SIMULATION task ###
                            sim_result = coll.find({
                                                    'files.assets.%s' %assetName: {"$exists":True},
                                                    'shot': shot,
                                                    'data_type': 'zenn',
                                                    'enabled': True,
                                                    'task':{'$nin':['ani','asset']}
                                                    }, sort=[('version', pymongo.DESCENDING)]).limit(1)
                            if sim_result.count() > 0:
                                print '### sim'
                                self.zennData[assetName] = sim_result[0]['files']['assets'][assetName]
                                self.zennData[assetName]['hairTask'] = True

                            else:
                                ### zenn ANIMATION task ###
                                shot_result = coll.find({
                                                          'files.assets.%s' %assetName: {"$exists": True},
                                                          'data_type': 'zenn',
                                                          'task': 'ani',
                                                          'enabled': True
                                                          }, sort=[('version', pymongo.DESCENDING)]).limit(1)
                                if shot_result.count() > 0:
                                    print '### shot'
                                    self.zennData[assetName] = shot_result[0]['files']['assets'][assetName]
                                    self.zennData[assetName]['hairTask'] = True

                                else:
                                    ### zenn ASSET task ###
                                    asset_result = coll.find({
                                                              'asset_name': assetBasename,
                                                              'data_type': 'zenn',
                                                              'task': 'asset',
                                                              'enabled': True
                                                              }, sort=[('version', pymongo.DESCENDING)]).limit(1)
                                    if asset_result.count() > 0:
                                        print '### asset'
                                        self.zennData[assetName] = asset_result[0]['files']
                                        self.zennData[assetName]['hairTask'] = False

                                    else:
                                        print '### NO ZENN STATIC CACHE FROM ASSET'

        pprint ( self.camData )
        pprint ( self.assemData )
        pprint ( self.geoData )
        pprint ( self.zennData )

    def doImportInfo(self):
        if self.ui.GPU_radioButton.isChecked():
            self.alembicOpt = 'GPU'
        if self.ui.Mesh_radioButton.isChecked():
            self.alembicOpt = 'mesh'

        if self.ui.None_radioButton.isChecked():
            self.worldOpt = 'none'
        if self.ui.baked_radioButton.isChecked():
            self.worldOpt = 'baked'
        if self.ui.seperate_radioButton.isChecked():
            self.worldOpt = 'seperate'

        if self.ui.simulation_radioButton.isChecked():
            self.zennOpt = 'simulation'
        if self.ui.static_radioButton.isChecked():
            self.zennOpt = 'static'
        if self.ui.auto_radioButton.isChecked():
            self.zennOpt = 'auto'

    # -------------------------------------------------------------------------------
    #   CHINA DOWNLOAD
    # -------------------------------------------------------------------------------
    def fileCheck(self):
        fileDict = {}
        downChecker = False
        if self.camData:
            fileDict['cam'] = list()
            if self.camData.has_key('camera_path'):
                if not os.path.exists(self.camData['camera_path'][0]):
                    fileDict['cam'].append( self.camData['camera_path'][0] )
                    downChecker = True

            if self.camData.has_key('camera_geo_path'):
                for path in self.camData['camera_geo_path']:
                    if not os.path.exists(path):
                        fileDict['cam_geo'].append(path)
                        downChecker = True

            if self.camData.has_key('camera_loc_path'):
                for path in self.camData['camera_loc_path']:
                    if not os.path.exists(path):
                        fileDict['cam_loc'].append(path)
                        downChecker = True

        if self.assemData:
            fileDict['assembly'] = list()
            for assemKey in self.assemData.keys():
                for path in self.assemData[assemKey]:
                    if not os.path.exists(path):
                        fileDict['assembly'].append( path )
                        downChecker = True
            fileDict['assembly'] = list(set(fileDict['assembly']))

        if self.geoData:
            fileDict['geoCache'] = list()
            for asset in self.geoData:
                print asset
                if asset == 'maya_dev_file':
                    continue
                if not os.path.exists(self.geoData[asset]['path'][0]):
                    fileDict['geoCache'].append( self.geoData[asset]['path'][0] )
                    downChecker = True
            fileDict['geoCache'].append(self.geoData['maya_dev_file'][0])

            for key in self.geoData[asset].keys():
                if len(self.geoData[asset][key]) > 0 and not os.path.exists(self.geoData[asset][key][0]):
                    fileDict['geoCache'].append( self.geoData[asset][key][0] )
                    downChecker = True
                if not asset == 'maya_dev_file':
                    for key in self.geoData[asset].keys():
                        if len(self.geoData[asset][key]) > 0 and not os.path.exists(self.geoData[asset][key][0]):
                            fileDict['geoCache'].append( self.geoData[asset][key][0] )
                            downChecker = True
            if not os.path.exists(self.geoData['maya_dev_file'][0]):
                fileDict['geoCache'].append( self.geoData['maya_dev_file'][0] )
                downChecker = True

        if self.zennData:
            fileDict['zenn'] = list()
            for asset in self.zennData:
                if not os.path.exists(self.zennData[asset]['zenn_path'][0]):
                    fileDict['zenn'].append( self.zennData[asset]['zenn_path'][0] )
                    downChecker = True

        if self.assetData:
            fileDict['asset'] = list()
            for type in self.assetData:
                if not type in ['name','texture_proxy'] and not os.path.exists(self.assetData[type][0]):
                    fileDict['asset'] += self.assetData[type]
                    downChecker = True
                if type == 'texture_proxy' and not os.path.exists(self.assetData[type][0]):
                    fileDict['asset'] += [ os.path.join( self.assetData['texture_proxy'][0], '*' ) ]
                    downChecker = True

        if downChecker:
            ed = ExistDialog(fileDict=fileDict)
            result = ed.exec_()
            return result

    # HOUDINI RMB CLICKED
    def rmbClicked(self):
        show = self.ui.show_comboBox.currentText()
        seq = self.ui.seq_comboBox.currentText()
        shot = self.ui.shot_comboBox.currentText()

        menu = QtWidgets.QMenu(self)
        menu.addAction(QtGui.QIcon(), u"import %s Light" % shot, lambda : Action.importLight(show, seq, shot))
        menu.popup(QtGui.QCursor.pos())


def showSpanner2():
    import Spanner2.main as main;
    main.main()

def showRenderSpool():
    import TractorSpool.spoolWindow as tsp
    tsp.show_ui()
