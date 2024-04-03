# -*- coding: utf-8 -*-
import sys
import os
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
import site
import pymongo
from pymongo import MongoClient
DB_NAME = 'PUBLISH'
COLL = 'spanner2_task'
COLL_USER = 'spanner2_user'
import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
client = MongoClient(DB_IP)
db = client[DB_NAME]
coll = db[COLL]
from xml.etree.ElementTree import parse
import HUD.HUDmodules
# import Qt
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui
from time import gmtime,strftime
import time
from shutil import copyfile
from dxstats import inc_tool_by_user
import getpass
try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass


def showSnapShot(filePath, fileName, showcode=''):
    """
    show snapshot on the snapshot label.
    str :param filePath: current file path on the filePath_lineEdit widget.
    :param fileName: current file name from the fileName_lineEdit widget.
    """
    assetShot = filePath.split('/')[3]
    taskName = filePath.split('/')[5]
    dafaultImage = os.path.join( CURRENTPATH, 'resource/noPreview.jpg' )
    try:
        imageFile = filePath + '/.%s.thumb.jpg' % (fileName)
        if os.path.isfile(imageFile):
            return imageFile
        else:
            return dafaultImage
        # if assetShot == 'asset':
        #     imageFile = filePath + '/.%s.thumb.jpg' % (fileName)
        #     if os.path.isfile(imageFile):
        #         return imageFile
        #     else:
        #         return dafaultImage
        #
        # elif assetShot == 'shot':
        #     tacticpath = '/tactic/assets/%s/shot/%s/icon' %(showcode, taskName)
        #     if os.path.exists(tacticpath):
        #         return os.path.join(tacticpath, os.listdir(tacticpath)[0])
        #     else:
        #         return dafaultImage

    except:
        print 'click file name for a snapshot'
        return dafaultImage


def takeSnapShot(filePath, fileName):
    """
    show snapshot on the snapshot label.
    str :param filePath: current file path on the filePath_lineEdit widget.
    :param fileName: current file name from the fileName_lineEdit widget.
    """
    inc_tool_by_user.run('action.Spanner2.takeSnapshot', getpass.getuser())
    assetShot = filePath.split('/')[3]
    workCode = filePath.split('/')[6]
    artist = os.getlogin()

    # HUD
    HUD.HUDmodules.mg_CreateHUD(artist, fileName, workCode,'')

    # take snapshot
    path = filePath + '/.' + fileName + '.thumb.jpg'
    currFrame = cmds.currentTime(query=True)
    format = cmds.getAttr("defaultRenderGlobals.imageFormat")
    cmds.setAttr("defaultRenderGlobals.imageFormat", 8)
    cmds.playblast(frame=currFrame, format="image", completeFilename=str(path), showOrnaments=True,
                   viewer=False, widthHeight=[1280, 720], percent=80)
    cmds.setAttr("defaultRenderGlobals.imageFormat", format)
    HUD.HUDmodules.mg_removeHUD()
    
    # copy to preview
    copyPath = '/'.join(filePath.split('/')[0:8])+'/preview/snapshot/'+fileName.replace('mb','jpg')
    if not os.path.exists(os.path.dirname(copyPath)):
        os.makedirs(os.path.dirname(copyPath))
    try:
        copyfile(path, copyPath)
    except:
        pass
    return fileName

# previous spanner version

def readComment(filePath, fileName):
    """
    read comment of current selected scene file
    :param filePath: current file path from the filePath_lineEdit widget.
    :param fileName: current file name from the fileName_lineEdit widget.
    """
    assetShot = filePath.split('/')[3]
    taskName = filePath.split('/')[5]
    workCode = filePath.split('/')[6]
    filebase = '/'.join(filePath.split('/')[0:7])
    xmlFile = filebase + '/dev/scenes/%s_%s_ComponentNote.xml' % (taskName, workCode)
    if os.path.isfile(xmlFile):
        xmlIns = parse(xmlFile)
        root = xmlIns.getroot()
        noteDic = {}
        for note in root:
            versionStr = ''
            comment = ''
            for noteElement in note:
                # print noteElement.tag, noteElement.text
                if noteElement.tag == 'version':
                    versionStr = 'v' + str(noteElement.text).zfill(2)
                if noteElement.tag == 'wipversion':
                    versionStr += '_w' + str(noteElement.text).zfill(2)
                if noteElement.tag == 'comment':
                    comment = noteElement.text
                if noteElement.tag == 'subject':
                    subject = noteElement.text

            if (subject == None or subject == '\n'):
                noteDic[versionStr] = comment
            else:
                if '\n' in subject:
                    subject = subject - '\n'
                noteDic[versionStr + '_' + subject] = comment

        if assetShot == 'asset':
            if len(fileName.split('_')) > 3:
                fileVersion = '_'.join(os.path.splitext(fileName)[0].split('_')[2:])
                if fileVersion in noteDic.keys():
                    return noteDic[fileVersion]

            if len(fileName.split('_')) == 3:
                fileVersion = os.path.splitext(fileName)[0].split('_')[2:]
                if fileVersion in noteDic.keys():
                    return noteDic[fileVersion]

        if assetShot == 'shot':
            if len(fileName.split('_')) > 4:
                fileVersion = '_'.join(os.path.splitext(fileName)[0].split('_')[3:])
                if fileVersion in noteDic.keys():
                    return noteDic[fileVersion]

            if len(fileName.split('_')) == 4:
                fileVersion = os.path.splitext(fileName)[0].split('_')[3:]
                if fileVersion in noteDic.keys():
                    return noteDic[fileVersion]


def readDB(filePath):
    """
    read data from mongoDB ex) save date, artist info
    :param filePath: current file path from the filePath_lineEdit widget.
    """
    assetShot = unicode(filePath.split('/')[3])
    show = unicode(filePath.split('/')[2])

    # asset DB information
    if assetShot == 'asset':
        path = '/'.join(filePath.split('/')[0:6])
        assetName = unicode(filePath.split('/')[5])
        workCode = unicode(filePath.split('/')[6])
        DB_NAME = 'ASSET'
        COLL = show
        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[COLL]
        temp = coll.find({'name': assetName, 'show': show, 'path': path})
        fileDic = {}
        try:
            if workCode in temp[0].keys() and 'dev' in temp[0][workCode].keys():
                for i in temp[0][workCode]['dev']:
                    fileDic[i] = {}
                    fileDic[i]['time'] = temp[0][workCode]['dev'][i]['time']
                    fileDic[i]['artist'] = temp[0][workCode]['dev'][i]['artist']
                    fileDic[i]['file'] = temp[0][workCode]['dev'][i]['file']
                    fileDic[i]['event'] = 'devel'

            if workCode in temp[0].keys() and 'pub' in temp[0][workCode].keys():
                pubDic = {}
                for i in temp[0][workCode]['pub']:
                    pubDic[i] = {}
                    try:
                        pubDic[i]['time'] = temp[0][workCode]['pub'][i]['time']
                    except: pass

                    try:
                        pubDic[i]['artist'] = temp[0][workCode]['pub'][i]['artist']
                    except: pass

                    try:
                        pubDic[i]['file'] = temp[0][workCode]['pub'][i]['file']
                    except: pass

                    pubDic[i]['event'] = 'publish'

                fileDic.update(pubDic)

            return fileDic

        except:
            print 'Error : No DB Information'

    if assetShot == 'shot':
        shotName = unicode(filePath.split('/')[5])
        workCode = unicode(filePath.split('/')[6])
        DB_NAME = 'SHOT'
        COLL = show
        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[COLL]
        temp = coll.find({'shot': shotName, 'show': show})
        fileDic = {}
        try:
            if workCode in temp[0]['work'].keys() and 'dev' in temp[0]['work'][workCode].keys():
                for i in temp[0]['work'][workCode]['dev']:
                    fileDic[i] = {}
                    fileDic[i]['time'] = temp[0]['work'][workCode]['dev'][i]['time']
                    fileDic[i]['artist'] = temp[0]['work'][workCode]['dev'][i]['artist']
                    fileDic[i]['file'] = temp[0]['work'][workCode]['dev'][i]['file']
                    fileDic[i]['event'] = 'devel'

            if workCode in temp[0]['work'].keys() and 'pub' in temp[0]['work'][workCode].keys():
                pubDic = {}
                for i in temp[0]['work'][workCode]['pub']:
                    pubDic[i] = {}
                    try:
                        pubDic[i]['time'] = temp[0]['work'][workCode]['pub'][i]['time']
                        pubDic[i]['artist'] = temp[0]['work'][workCode]['pub'][i]['artist']
                        pubDic[i]['file'] = temp[0]['work'][workCode]['pub'][i]['file']
                    except: pass

                    pubDic[i]['event'] = 'publish'

                fileDic.update(pubDic)

            return fileDic

        except:
            print 'Error : No DB Information'

def readDBComment(filePath,fileName):
    """
    read comment from MongoDB
    :param filePath: current file path from the filePath_lineEdit widget.
    :param fileName: current file name from the fileName_lineEdit widget.
    """
    path = '/'.join(filePath.split('/')[0:6])
    file = fileName.split('.')[0]
    workCode = filePath.split('/')[6]
    temp = coll.find({'path': path})
    try:
        timeList = (temp[0][workCode][file].keys())
        commentList = {}
        for dbkey in timeList:
            commentList[dbkey] = [temp[0][workCode][file][dbkey]['comment']
                                , temp[0][workCode][file][dbkey]['artist']]

        return commentList

    except: pass

def saveDBComment(filePath, fileName, comment):
    """
    save comment to MongoDB
    :param filePath: current file path on the filePath_lineEdit widget.
    :param fileName: current file name on the fileName_lineEdit widget.
    :param comment: comment input by the user.
    """
    inc_tool_by_user.run('action.Spanner2.saveMemo', getpass.getuser())
    path = '/'.join(filePath.split('/')[0:6])
    show = filePath.split('/')[2]
    name = filePath.split('/')[5]
    workCode = filePath.split('/')[6]
    artist = os.getlogin()
    file = fileName.split('.')[0]
    DBtime = strftime('%Y-%m-%dT%H:%M:%S')

    # set DB
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]

    # if exists
    if coll.find({'path':path, 'name': name}).count() == 1:
        DBtime = strftime('%Y-%m-%dT%H:%M:%S')
        data = {'artist': artist, 'file': file, 'time': DBtime,
                'comment': comment}
        coll.update({'name': name, 'path': path},
                    {'$set': {'%s.%s.%s' % (workCode, file, DBtime): data}}
                    , upsert=True)

        print 'saved to DB successfully'

    # if object not exists
    if coll.find({'path':path, 'name': name}).count() == 0:
        # insert
        post = {}
        post['name'] = name
        post['path'] = path
        post['show'] = show
        post[workCode] = {}
        coll.insert(post)

        # add
        DBtime = strftime('%Y-%m-%dT%H:%M:%S')
        data = {'artist': artist, 'file': file, 'time': DBtime,
                'comment': comment}
        coll.update({'name': name, 'path': path},
                    {'$set': {'%s.%s.%s' % (workCode, file, DBtime): data}}
                    , upsert=True)

        print 'saved to DB successfully'


def editDBComment(filePath, fileName):
    """
    edit comment already in MongoDB, find comment by time.
    :param filePath: current file path from the filePath_lineEdit widget.
    :param fileName: current file name from the fileName_lineEdit widget.
    """
    inc_tool_by_user.run('action.Spanner2.editMemo', getpass.getuser())
    workCode = filePath.split('/')[6]
    workType = filePath.split('/')[3]
    path = '/'.join(filePath.split('/')[0:6])
    # show dialog
    ec = EditCommentUI()
    ec.show()
    ec.fileBox.setText(fileName)
    result = ec.exec_()
    if result == 1:
        if workType == 'asset':
            name = fileName.split('_')[0]
        if workType == 'shot':
            name = '_'.join(fileName.split('_')[0:2])
        print name
        fileName = fileName.split('.')[0]
        comment = ec.comment

        try:
            if comment != '' and comment != None:
                time.strptime(str(ec.time), '%Y-%m-%dT%H:%M:%S')
                # show warning
                wd = WaringDialog()
                wd.show()
                result = wd.exec_()
                # edit
                if result == 1:
                    coll.update({'name': name, 'path':path},
                                {'$set': {"%s.%s.%s.comment" % (workCode, fileName, ec.time)
                                          : comment}})
                    print 'edited successfully'

                else: pass

            else: print 'No comment to save'

        except: print 'Invalid time'


def saveUserInfo(data):
    """
    save user information on MongoDB
    str :param data: combobox index, user name
    """
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL_USER]
    name = os.getlogin()
    coll.update({'name': name }, {'$set':data}, upsert = True)

def getUserInfo():
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL_USER]
    name = os.getlogin()
    try:
        temp = coll.find({'name':name})
        return temp[0]

    except: pass

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
        font = QtGui.QFont()
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


class EditCommentUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("File name editing:")
        self.fileBox = QtWidgets.QLineEdit()
        label2 = QtWidgets.QLabel("Insert 'time' of memo editing:")
        self.timeBox = QtWidgets.QLineEdit()
        self.timeBox.setText('yyyy-mm-ddTHH:MM:SS')
        label3 = QtWidgets.QLabel("Type comment:")
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
        layout.addWidget(self.timeBox)
        layout.addWidget(label3)
        layout.addWidget(self.commentBox)
        layout.addLayout(layout2, 6, 0)
        font = QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)
        label2.setFont(font)
        label3.setFont(font)
        self.fileBox.setFont(font)
        self.timeBox.setFont(font)
        self.commentBox.setFont(font)
        self.setLayout(layout)
        self.setWindowTitle("Edit Memo")

        # connection
        self.close_btn.clicked.connect(self.reject)
        self.yes_btn.clicked.connect(self.result)

    def result(self):
        self.comment = self.commentBox.toPlainText()
        self.time = self.timeBox.text()
        self.accept()


class WaringDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("Are you sure to save?\n")
        self.ok_btn = QtWidgets.QPushButton("Ok")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.ok_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2, 3, 0)
        self.setLayout(layout)
        self.setWindowTitle("Warning")
        font = QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        # connection
        self.ok_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)

# DEXTER CHAT
class ChatItem(QtWidgets.QWidget):
    def __init__(self, parent=None ):
        super(ChatItem, self).__init__(parent)
        self.textLayout = QtGui.QVBoxLayout()
        self.name = QtGui.QLabel()
        self.text = QtGui.QLabel()
        self.textLayout.addWidget(self.name)
        self.textLayout.addWidget(self.text)
        
        self.allLayout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel()
        self.allLayout.addWidget(self.label, 0)
        self.allLayout.addLayout(self.textLayout, 1)
        
        self.setLayout(self.allLayout)
        
        self.name.setStyleSheet("color: black")
        self.text.setStyleSheet("color: black")

    def setName(self, text):
        self.name.setText(text)
        
    def setText(self, text):
        self.text.setText(text)
        
    def setLabel(self, path):
        pixmap = QtGui.QPixmap(QtCore.QSize(50,50))
        picture = QtGui.QPixmap(path)
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(0,0,50,50,QtGui.QColor(255,255,255,255))
        circle = QtGui.QPainterPath()
        circle.addEllipse(0,0,45,45)
        painter.setClipPath(circle)
        painter.drawPixmap(0,0,50,50,picture)

        self.label.setPixmap(pixmap)   
        painter.end()   
        
    # def showGIF(self,rawFile):
#     gifFile = rawFile + '.gif'
#     movFile = rawFile + '.mov'
#     print gifFile
#     if os.path.isfile(gifFile) :
#         # set gif
#         gif = QtGui.QMovie(gifFile)
#         self.ui.snapshot_label.setMovie(gif)
#         size = QtCore.QSize(355, 200)
#         gif.setScaledSize(size)
#         gif.start()
#
#
# def makeGIF(self,rawFile,movFile,gifFile):
#
#     # creat gif
#     file1 = movFile
#     file2 = gifFile
#     command = '/opt/ffmpeg/bin/ffmpeg -i %s -vf scale=300:-1,format=rgb8,format=rgb24 -r 5 -an -vcodec ppm -b:a 3000k -y %s' \
#               % (file1, file2)
#     subCommand = command.split(' ')
#     job = subprocess.Popen(subCommand)
#
#     print job.poll()
#     self.showGIF(rawFile)
