# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys
import os
import pymongo
from pymongo import MongoClient
import dxConfig

CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
DB_NAME = 'PUBLISH'
COLL = 'spanner2_task'
DB_IP = dxConfig.getConf('DB_IP')

import HUD.HUDmodules
# import Qt
# from Qt import QtWidgets
# from Qt import QtCore
from PySide2 import QtWidgets, QtCore

from time import gmtime, strftime

import getpass
try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# SnapShot
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class SnapShots():
    def __init__(self):
        self.artist = getpass.getuser()

    def showSnapShot(self, filePath=None, fileName=None):
    # show snapshot
        dafaultImage = os.path.join( CURRENTPATH, 'resource/noPreview.jpg' )
        imageFile = filePath + '/.%s.thumb.jpg' % (fileName)
        snapFile = '/'.join(filePath.split('/')[0:9]) + '/preview/snapshot/' + fileName.replace('mb', 'jpg')
        if os.path.isfile(snapFile):
            return snapFile
        elif os.path.isfile(imageFile):
            return imageFile
        else:
            return dafaultImage

    def takeSnapShot(self, filePath=None, fileName=None):
    # take snapshot
        workCode = filePath.split('/')[7]
        # HUD
        HUD.HUDmodules.mg_CreateHUD(self.artist, fileName, workCode,'')

        # take snapshot
        imgPath = '/'.join(filePath.split('/')[0:9])+'/preview/snapshot/'
        imgName = imgPath + fileName.replace('mb', 'jpg')

        if not os.path.exists(imgPath):
            os.makedirs(imgPath)

        currFrame = cmds.currentTime(query=True)
        format = cmds.getAttr("defaultRenderGlobals.imageFormat")
        cmds.setAttr("defaultRenderGlobals.imageFormat", 8)
        cmds.playblast(frame=currFrame, format="image", completeFilename=str(imgName), showOrnaments=True,
                       viewer=False, widthHeight=[1280, 720], percent=80)
        cmds.setAttr("defaultRenderGlobals.imageFormat", format)
        HUD.HUDmodules.mg_removeHUD()
        return fileName

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Comment
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class CommentDB():
    def __init__(self):
        self.artist = getpass.getuser()
        self.client = MongoClient(DB_IP)
        self.db = self.client[DB_NAME]
        self.coll = self.db[COLL]

    def readDBComment(self, filePath=None, fileName=None):
    # mongo db comment read
        path = '/'.join(filePath.split('/')[0:7])
        file = fileName.split('.')[0]
        workCode = filePath.split('/')[7]

        temp = self.coll.find({'path': path})
        try:
            timeList = (temp[0][workCode][file].keys())
            commentList = {}
            for dbkey in timeList:
                commentList[dbkey] = [temp[0][workCode][file][dbkey]['comment']
                    , temp[0][workCode][file][dbkey]['artist']]

            return commentList
        except:
            pass

    def saveDBComment(self, filePath=None, fileName=None, comment=None):
    # mongo db comment save
        artist = self.artist
        path = '/'.join(filePath.split('/')[0:7])
        show = filePath.split('/')[2]
        name = filePath.split('/')[6]
        workCode = filePath.split('/')[7]
        file = fileName.split('.')[0]
        dbTime = strftime('%Y-%m-%dT%H:%M:%S')

        isDBExists = self.coll.find({'path': path, 'name': name}).limit(1)
        if isDBExists.count():
            data = {'artist': artist, 'file': file, 'time': dbTime,
                    'comment': comment}
            self.coll.update({'name': name, 'path': path},
                        {'$set': {'%s.%s.%s' % (workCode, file, dbTime): data}}
                        , upsert=True)
            print 'saved to DB Add successfully'
        else:
            post = {}
            post['name'] = name
            post['path'] = path
            post['show'] = show
            post[workCode] = {}
            self.coll.insert(post)
            # add
            data = {'artist': artist, 'file': file, 'time': dbTime,
                    'comment': comment}
            self.coll.update({'name': name, 'path': path},
                        {'$set': {'%s.%s.%s' % (workCode, file, dbTime): data}}
                        , upsert=True)
            print 'saved to DB Create successfully'

    def readDB(self, filePath=None):
    # mongo db comment read
        show = unicode(filePath.split('/')[2])

        if '/prev' in filePath:
            self.client = MongoClient(DB_IP)
            db = self.client['WORK']
            coll = db[show]
            fileDic = {}
            if os.path.isdir(filePath):
                devfolder = os.path.join(filePath, 'dev/scenes')
                pubfolder = os.path.join(filePath, 'pub/scenes')
                if os.path.isdir(devfolder):
                    for i in os.listdir(devfolder):
                        path = os.path.join(filePath,'dev/scenes',i)
                        if os.path.isfile(path) and not i.startswith('.'):
                            result = coll.find({'filepath':path}).sort('count',pymongo.DESCENDING).limit(1)
                            if result.count():
                                name = os.path.splitext(i)[0]
                                fileDic[name] = {}
                                fileDic[name]['time'] = result[0]['time']
                                fileDic[name]['artist'] = result[0]['user']
                                fileDic[name]['file'] = path
                                fileDic[name]['event'] = 'devel'

                pubDic = {}
                if os.path.isdir(pubfolder):
                    for i in os.listdir(pubfolder):
                        path = os.path.join(filePath,'pub/scenes',i)
                        if os.path.isfile(path) and not i.startswith('.'):
                            result = coll.find({'filepath':path}).sort('count',pymongo.DESCENDING).limit(1)
                            if result.count():
                                name = os.path.splitext(i)[0]
                                pubDic[name] = {}
                                pubDic[name]['time'] = result[0]['time']
                                pubDic[name]['artist'] = result[0]['user']
                                pubDic[name]['file'] = path
                                pubDic[name]['event'] = 'publish'

                fileDic.update(pubDic)
            return fileDic

# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Comment QDialog
# ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
class GetCommentUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("File name editing:")
        self.fileBox = QtWidgets.QLineEdit()
        label2 = QtWidgets.QLabel("Type comment:")
        self.commentBox = QtWidgets.QTextEdit()
        self.yes_btn = QtWidgets.QPushButton("Save")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.yes_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addWidget(self.fileBox)
        layout.addWidget(label2)
        layout.addWidget(self.commentBox)
        layout.addLayout(layout2,5,0)
        self.setLayout(layout)
        self.setWindowTitle("Add Memo")
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)
        label2.setFont(font)
        self.fileBox.setFont(font)
        self.commentBox.setFont(font)

        # connection
        self.close_btn.clicked.connect(self.reject)
        self.yes_btn.clicked.connect(self.result)

    def result(self):
        self.comment = self.commentBox.toPlainText()
        self.accept()
