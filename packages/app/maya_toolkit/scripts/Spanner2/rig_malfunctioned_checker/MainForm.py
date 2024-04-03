# -*- coding: utf-8 -*-

import os, sys
import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

# Mongo DB
import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)

import requests

from ui_Main import Ui_Form

class MainForm(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('Rig Banned')
        self.resize(600,500)
        self.ui.treeWidget.header().resizeSection(0, 150)
        self.ui.treeWidget.header().resizeSection(1, 250)
        self.ui.comboBox.setMaxCount(10)
        self.addShowList()
        self.ui.comboBox.activated.connect(self.addRigList)
        self.ui.comboBox.activated.connect(self.addShotList)

    def addShowList(self):
        params = {}
        params['api_key'] = API_KEY
        params['category'] = 'Active'
        infos = requests.get("http://%s/dexter/search/project.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        self.ui.comboBox.addItems([i['name'] for i in infos])

    def addRigList(self):
        show = self.ui.comboBox.currentText()
        self.ui.listWidget.clear()
        db = client['PUBLISH']
        coll = db['RIG_MALFUNCTION']
        self.malFunctionList = coll.find({'show':show}).distinct('path')
        for i in sorted(self.malFunctionList):
            self.ui.listWidget.addItem(os.path.basename(i))

    def addShotList(self):
        show = self.ui.comboBox.currentText()
        self.ui.treeWidget.clear()
        db = client['WORK']
        coll = db[show]
        result = coll.find({
            'etc.rig_referenced':{'$exists':True}
        })

        usingMalList = []
        for i in result:
            for k in i['etc']['rig_referenced']:
                if k in self.malFunctionList and not i['name'] in usingMalList:
                    usingMalList.append(i['name'])

        for i in usingMalList:
            result = coll.find({
                'name':i, 'platform':'maya'}).sort('time', pymongo.DESCENDING).limit(1)
            if result.count() > 0:
                if result[0]['etc'].has_key('rig_referenced'):
                    for k in result[0]['etc']['rig_referenced']:
                        if k in self.malFunctionList:
                            item = UsingMalfunctionedRigShot(self.ui.treeWidget)
                            item.setDict(result[0])
                            item.setTexts()
                            break

        self.ui.treeWidget.sortByColumn(2,QtCore.Qt.DescendingOrder)

class UsingMalfunctionedRigShot(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None ):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        dataDict = {}

    def setDict(self, dict):
        self.dataDict = dict

    def getDict(self):
        return self.dataDict

    def setTexts(self):
        self.setText(0, self.dataDict['user'])
        self.setText(1, os.path.basename(self.dataDict['filepath']))
        self.setText(2, self.dataDict['time'][0:16])
        for i in self.dataDict['etc']['rig_referenced']:
            item = QtWidgets.QTreeWidgetItem(self)
            item.setText(1, os.path.basename(i))
            self.addChild(item)

