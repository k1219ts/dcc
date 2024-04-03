# -*- coding: utf-8 -*-
import os, sys
import getpass
import requests
import json

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as omu

import sgUI
import dxCameraUI
import sgAnimation
import sgCommon

import DXUSD.Message as msg

from shiboken2 import wrapInstance
from PySide2 import QtCore, QtGui, QtWidgets

import dxConfig
import pymongo
from pymongo import MongoClient

if msg.DEV:
    reload(sgCommon)

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

import ui_DxCameraImporter


def get_maya_window():
    main_window_ptr = omu.MQtUtil.mainWindow()
    return wrapInstance(long(main_window_ptr), QtWidgets.QWidget)

def getShotPath(show,seq=None,shot=None):
    if show == 'testshot':  show = 'test_shot'
    shotPath = '/show/%s/_3d/shot' % show
    if seq:
        shotPath = os.path.join(shotPath, seq)
        if shot:
            shotPath = os.path.join(shotPath, shot)
    return shotPath

def getLatestPubVersion(show, seq, shot, data_type,plateType=None):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[show]
    if plateType:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type,
                                   'task_publish.plateType':plateType},
                                  sort=[('version', pymongo.DESCENDING)])
    else:
        recentDoc = coll.find_one({'show': show,
                                   'shot': shot,
                                   'data_type': data_type},
                                  sort=[('version', pymongo.DESCENDING)])

    if recentDoc:
        return recentDoc['version']
    else:
        return 0

def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return '%.1f%s%s' % (num, 'P', suffix)


class DxCameraImporter(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, get_maya_window())
        self.ui = ui_DxCameraImporter.Ui_Dialog()
        self.ui.setupUi(self)

        plugins = ['AbcImport', 'backstageMenu']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        # UNLOAD OBJEXPORT PLUGIN / THIS MAKE obj REFERENCE BUG
        cmds.unloadPlugin('objExport')

        self.ui.cameraTree.setRootIsDecorated(False)
        self.ui.cameraTree.setColumnCount(6)

        self.ui.cameraTree.headerItem().setText(0, 'Type')
        self.ui.cameraTree.headerItem().setText(1, 'Version')
        self.ui.cameraTree.headerItem().setText(2, 'Team')
        self.ui.cameraTree.headerItem().setText(3, 'Stereo')
        self.ui.cameraTree.headerItem().setText(4, 'Publisher')
        self.ui.cameraTree.headerItem().setText(5, 'Time')

        self.ui.cameraTree.header().resizeSection(0, 100)
        self.ui.cameraTree.header().resizeSection(1, 50)
        self.ui.cameraTree.header().resizeSection(2, 100)
        self.ui.cameraTree.header().resizeSection(3, 50)

        #self.ui.fileTree.setRootIsDecorated(False)
        self.ui.fileTree.setColumnCount(4)
        self.ui.fileTree.headerItem().setText(0, '')
        self.ui.fileTree.headerItem().setText(1, 'Key')
        self.ui.fileTree.headerItem().setText(2, 'File')
        self.ui.fileTree.headerItem().setText(3, 'Size')

        self.ui.fileTree.header().resizeSection(0, 60)
        self.ui.fileTree.header().resizeSection(1, 160)
        self.ui.fileTree.header().resizeSection(2, 300)

        self.ui.fileTree.setStyleSheet("""
        background-color: rgb(55,55,55)
        """)
        self.ui.cameraTree.setStyleSheet("""
        background-color: rgb(55,55,55)
        """)

        self.ui.shotCombo.setMaxVisibleItems(20)

        # PROJECT SETTING FROM TACTIC
        self.projectInfo = self.queryProjects('Active', 'in_progres')
        self.projectInfo.append({'name': u'cdh1', 'title': u'외계인new (cdh1)'})
        if msg.DEV:
            self.projectInfo.append({'name': u'pipe', 'title': u'신규파이프라인 (pipe)'})

        self.titleDic = {}

        # MAIN CAMERA ID
        self.mainCameraID = None

        self.prepareInfo()
        self.connectSetting()
        setting = QtCore.QSettings("DEXTER_DIGITAL", "dxCameraImporter.project")
        if setting.value('project'):
            prjIndex = self.ui.showCombo.findText(setting.value('project'))
            self.ui.showCombo.setCurrentIndex(prjIndex)
        else:
            self.setSeq(0)

    def queryProjects(self, active, status):
        param = {}
        param['api_key'] = API_KEY
        param['category'] = active
        param['status'] = status

        return requests.get("http://%s/dexter/search/project.php" % dxConfig.getConf('TACTIC_IP'),
                            params=param).json()

    def prepareInfo(self):
        for prj in sorted(self.projectInfo,
                          key=lambda k:k['title']):

            self.ui.showCombo.addItem(prj['title'])
            self.titleDic[prj['title']] = prj

        # FOR DEBUG SHOT TEST
        # self.ui.showCombo.insertItem(0, 'mmv')
        # self.titleDic['mmv'] = {'name':'mmv'}

    def connectSetting(self):
        self.ui.showCombo.currentIndexChanged.connect(self.setSeq)
        self.ui.seqCombo.currentIndexChanged.connect(self.setShot)
        self.ui.shotCombo.currentIndexChanged.connect(self.getCameraList)
        self.ui.cameraTree.itemClicked.connect(self.refreshCameraFile)
        self.ui.fileTree.itemClicked.connect(self.toggleCheck)

        self.ui.cancelButton.clicked.connect(self.accept)
        self.ui.importButton.clicked.connect(self.importCamera)

    def toggleCheck(self, item, col):
        item.importCheck.toggle()

    def setSeq(self, index):
        setting = QtCore.QSettings("DEXTER_DIGITAL", "dxCameraImporter.project")
        setting.setValue("project", self.ui.showCombo.currentText())

        self.ui.seqCombo.clear()
        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        shotDir = getShotPath(show=project)
        seqs = sorted([i for i in os.listdir(shotDir) if (not (i.startswith('.')))])
        seqs.remove('shot.usd')

        self.ui.seqCombo.clear()
        self.ui.seqCombo.addItems(seqs)

    def setShot(self, index):
        self.ui.shotCombo.clear()

        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        seq = unicode(self.ui.seqCombo.currentText())
        seqDir = getShotPath(show=project, seq=seq)

        shots = sorted([i for i in os.listdir(seqDir) if (not (i.startswith('.')))])

        self.ui.shotCombo.clear()
        self.ui.shotCombo.addItems(shots)

    def getCameraList(self):
        self.ui.cameraTree.clear()
        self.ui.fileTree.clear()

        prjData = self.titleDic[unicode(self.ui.showCombo.currentText())]
        project = prjData['name']
        if project == 'testshot':     project = 'test_shot'
        seq = unicode(self.ui.seqCombo.currentText())
        shot = self.ui.shotCombo.currentText()
        if not(shot):
            return

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[project]

        for i in coll.find({'show':project,
                            'sequence':seq,
                            'shot':shot,
                            'data_type':'camera'}):

            item = CameraTreeItem(self.ui.cameraTree)
            item.setRecord(i)
            if i['task_publish'].has_key('plateType'):
                if i['task_publish']['plateType'] == 'layout':
                    #subCameras = i['sub_camera_id']

                    for sub in i['sub_camera_id']:
                        # IF SUB IS DICT TYPE IMPORT ABC FILE
                        # ELSE DB REFERENCE THEN DEREFERENCE IT

                        if isinstance(sub, dict):
                            cpath = os.path.basename(sub['abc_path'])
                            item.subRecord[cpath] = {'files':{ 'camera_geo_path':[sub['abc_path']]}}

                        else:
                            subCameraRecord = db.dereference(sub)
                            item.subRecord[subCameraRecord['_id']] = subCameraRecord

                            if sub._DBRef__kwargs.has_key('dxc_path'):
                                item.subRecord[subCameraRecord['_id']]['dxc_path'] = sub._DBRef__kwargs['dxc_path']

                item.setText(0, str(i['task_publish']['plateType']))

            item.setText(1, str(i['version']))
            item.setText(2, str(i['task']))
            try:
                item.setText(3, str(i['task_publish']['stereo']))
            except:
                item.setText(3, str('???'))
            item.setText(4, str(i['artist']))
            item.setText(5, str(i['time']).split('.')[0])


    def makeFileTreeItem(self, record, rootName, checkAll=False):
        optionList = ['camera_path',
                      'camera_loc_path',
                      'camera_geo_path',
                      'camera_asset_geo_path',
                      'camera_asset_loc_path',
                      'camera_asset_key_path'
                      ]
        print record
        files = record['files']

        rootItem = FileTreeItem(self.ui.fileTree)
        rootItem.setRecord(record)
        rootItem.setText(1, rootName)

        for i in [i for i in optionList if files.has_key(i)]:
            for f in files[i]:
                fItem = FileTreeItem(rootItem)
                fItem.setFilePath(f)
                if i == 'camera_asset_key_path':
                    fItem.setForeground(1, QtGui.QBrush(QtCore.Qt.darkBlue))
                    fItem.setForeground(2, QtGui.QBrush(QtCore.Qt.darkBlue))

                fItem.setText(1, i)
                fItem.setText(2, os.path.basename(f))

                try:
                    fileSize = os.stat(f).st_size
                    fItem.setText(3, sizeof_fmt(fileSize))
                except:
                    fileSize = 0
                    fItem.setText(3, "Can't find file.")

                if i == 'camera_path':
                    fItem.importCheck.setChecked(True)
                else:
                    fItem.importCheck.setChecked(checkAll)

                if (fileSize > 100000000) or (fileSize == 0):
                    fItem.setForeground(3, QtGui.QBrush(QtCore.Qt.red))

        rootItem.setExpanded(True)
        return rootItem

    def refreshCameraFile(self, item, column):
        self.ui.fileTree.clear()
        record = item.getRecord()
        self.mainCameraID = record['_id']

        self.makeFileTreeItem(record=record,
                              rootName='Main',
                              checkAll=False)

        if record['task_publish'].has_key('plateType'):
            if record['task_publish']['plateType'] == 'layout':
                subRecord = item.getSubRecord()
                for sub in subRecord.values():
                    self.makeFileTreeItem(record=sub,
                                          rootName='Sub',
                                          checkAll=False)


    def importCamera(self):
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))

        item = self.ui.cameraTree.selectedItems()[0]
        keyDic = self.getCheckedKey()
        print 'keyDic', keyDic
        mainInfo = item.getRecord()
        mainImportInfo = keyDic[self.mainCameraID]

        print 'subRecord', item.getSubRecord()


        optionList = ['camera_loc_path',
                      'camera_geo_path',
                      'camera_asset_geo_path',
                      'camera_asset_loc_path',
                      ]

        # # IMPORT MAIN CAMERA FILES
        dxcam = cmds.createNode('dxCamera')

        if mainImportInfo.has_key('camera_path'):
            cmds.fileInfo('overscan', mainInfo['task_publish']['overscan'])
            if mainInfo['task_publish'].has_key('overscan_value'):
                cmds.fileInfo('overscan_value', mainInfo['task_publish']['overscan_value'])

            # IMPORT CAMERA UNDER dxCamera NODE
            for idx, maincam in enumerate(mainImportInfo['camera_path']):
                # dxcam = cmds.createNode('dxCamera')
                cmds.addAttr(dxcam, longName='objectId', niceName='objectId', dataType="string")
                cmds.setAttr('%s.objectId' % dxcam, self.mainCameraID, type='string')

                dxCameraUI.import_cameraFile('%s.fileName' % dxcam,
                                             maincam)

                # LOAD IMAGEPLANE IF EXISTS
                if mainInfo['files'].has_key('imageplane_path'):
                    imgFile = mainInfo['files']['imageplane_path'][idx]
                    imgJsonFile = mainInfo['files']['imageplane_json_path'][0]
                    dxCameraUI.alembicCamera_imagePlane(imgFile, imgJsonFile)
                elif mainInfo['files'].has_key('imageplane_json_path'):
                    imgJsonFile = mainInfo['files']['imageplane_json_path'][0]
                    dxCameraUI.imagePlaneFromJson(imgJsonFile)

                if mainInfo['files'].has_key('panzoom_json_path'):
                    self.alembicCamera_2DPanZoom(mainInfo['files']['panzoom_json_path'][0])

                # print 'optionList:', optionList
                # for k in optionList:
                #     if mainImportInfo.has_key(k):
                #         # print idx, 'mainInfo:', mainInfo['files'][k]
                #         # print k, idx, mainImportInfo
                #         for abcFile in mainImportInfo[k]:
                #         # if 'geom' in os.path.basename(abcFile):
                #         #     ciClass = sgUI.ComponentImport(Files=[abcFile])
                #         #     returnNodes = ciClass.doIt()
                #         #     for node in returnNodes:
                #         #         cmds.parent(node, dxcam)
                #         # else:
                #             mel.eval('AbcImport -d -m import -rpr "%s" "%s"' % (dxcam,abcFile))

        # DXC TRANSFORM
        if mainInfo['files'].has_key('dxc_path'):
            sgCommon.AbcXformApplyKey(filepath=mainInfo['files']['dxc_path'][0],
                                      nodename=dxcam)

        # IMPORT ABC FILES
        print 'optionList:', optionList
        for k in optionList:
            if mainImportInfo.has_key(k):
                for abcFile in mainImportInfo[k]:
                    # IF GEO IN FILENAME THEN ZGPUMESH
                    if 'Geo' in os.path.basename(abcFile):
                        ciClass = sgUI.ComponentImport(Files=[abcFile])
                        returnNodes = ciClass.doIt()
                        for node in returnNodes:
                            cmds.parent(node, dxcam)
                    else:
                        mel.eval('AbcImport -d -m import -rpr "%s" "%s"' % (dxcam, abcFile))

        # IMPORT dxRig NODE FOR MAIN CAMERA
        #if mainInfo['files'].has_key('camera_asset_key_path'):
        # IF USER CHECKED camera_asset_key_path <- RIG MATCH JSON FILE
        if mainImportInfo.has_key('camera_asset_key_path'):
            for rigJson in mainImportInfo['camera_asset_key_path']:
                dxRig = sgAnimation.read(rigJson)

                # OBJECTID FROM DATABASE IN DXRIG
                cmds.addAttr(dxRig, longName='objectId', niceName='objectId', dataType="string")
                cmds.setAttr('%s.objectId' % dxRig, self.mainCameraID, type='string')

                # IF dxCamera has TRANSFORM KEY
                # THEN MAKE CONSTRAINT
                if mainInfo['files'].has_key('dxc_path'):
                    ns = dxRig.split(':')[0]
                    placeCon = ns + ':place_CON'
                    cmds.parentConstraint(dxcam, placeCon, mo=1)


        # RENDER SETTING
        try:
            cmds.setAttr("defaultResolution.width",
                         int(mainInfo['task_publish']['render_width']))
            cmds.setAttr("defaultResolution.height",
                         int(mainInfo['task_publish']['render_height']))
            cmds.setAttr("defaultResolution.deviceAspectRatio",
                         float(mainInfo['task_publish']['render_width']) / float(mainInfo['task_publish']['render_height']))

            # FRAME/ANIMATION SETTING
            cmds.setAttr('defaultRenderGlobals.animation', True)
            cmds.setAttr('defaultRenderGlobals.outFormatControl', 0)

        except:
            # DEPRECATED KEY NAME SUPPORT
            cmds.setAttr("defaultResolution.width",
                         int(mainInfo['task_publish']['renderWidth']))
            cmds.setAttr("defaultResolution.height",
                         int(mainInfo['task_publish']['renderHeight']))
            cmds.setAttr("defaultResolution.deviceAspectRatio",
                         float(mainInfo['task_publish']['renderWidth']) / float(mainInfo['task_publish']['renderHeight']))

            # FRAME/ANIMATION SETTING
            cmds.setAttr('defaultRenderGlobals.animation', True)
            cmds.setAttr('defaultRenderGlobals.outFormatControl', 0)


        # FRAME SETTING
        try:
            cmds.setAttr("defaultRenderGlobals.startFrame",
                         mainInfo['task_publish']['startFrame'])
            cmds.setAttr("defaultRenderGlobals.endFrame",
                         mainInfo['task_publish']['endFrame'])

            cmds.playbackOptions(ast=mainInfo['task_publish']['startFrame'],
                                 aet=mainInfo['task_publish']['endFrame'],
                                 min=mainInfo['task_publish']['startFrame'],
                                 max=mainInfo['task_publish']['endFrame']
                                 )
            cmds.currentTime(mainInfo['task_publish']['startFrame'])
        except:
            pass

        # IMPORT SUB CAMERA FILES
        # print "self.subCameraIDs : ", self.subCameraIDs
        # for subCameraID in self.subCameraIDs:
        for subCameraID in item.getSubRecord().keys():
            print subCameraID
            subImportInfo = keyDic[subCameraID]

            dxcam = cmds.createNode('dxCamera')
            subInfo = item.subRecord[subCameraID]

            if subImportInfo.has_key('camera_path'):
                # IMPORT CAMERA UNDER dxCamera NODE
                dxCameraUI.import_cameraFile('%s.fileName' % dxcam,
                                             subImportInfo['camera_path'][0])
                cmds.addAttr(dxcam, longName='objectId', niceName='objectId', dataType="string")
                cmds.setAttr('%s.objectId' % dxcam, subCameraID, type='string')

                # DISABLE SUB CAMERA RENDERABLE OPTION
                for i in cmds.ls(dxcam, dag=True, type='camera'):
                    try:
                        camNode = cmds.listRelatives(i, p=True)[0]
                        cmds.setAttr("%s.renderable" % camNode, False)
                    except:
                        print "fail to delete %s from renderable camera" % camNode

                # LOAD IMAGEPLANE IF EXISTS
                if subInfo['files'].has_key('imageplane_path'):
                    imgFile = subInfo['files']['imageplane_path'][0]
                    imgJsonFile = subInfo['files']['imageplane_json_path'][0]
                    dxCameraUI.alembicCamera_imagePlane(imgFile, imgJsonFile)

            # IMPORT ABC FILES
            for k in optionList:
                if subImportInfo.has_key(k):
                    for abcFile in subImportInfo[k]:
                        mel.eval('AbcImport -d -m import -rpr "%s" "%s"' % (dxcam, abcFile))
                        print abcFile

            # APPLY MATRIX TO SUB dxCamera NODE
            if subInfo.has_key('dxc_path'):
                sgCommon.AbcXformApplyKey(filepath=subInfo['dxc_path'],
                                          nodename=dxcam)

            # IMPORT dxRig NODE FOR SUB CAMERA
            #if subInfo['files'].has_key('camera_asset_key_path'):
            if subImportInfo.has_key('camera_asset_key_path'):
                for rigJson in subImportInfo['camera_asset_key_path']:
                    dxRig = sgAnimation.read(rigJson)

                    # OBJECTID FROM DATABASE IN DXRIG
                    cmds.addAttr(dxRig, longName='objectId', niceName='objectId', dataType="string")
                    cmds.setAttr('%s.objectId' % dxRig, subCameraID, type='string')

                    if subInfo.has_key('dxc_path'):
                        ns = dxRig.split(':')[0]
                        placeCon = ns + ':place_CON'
                        cmds.parentConstraint(dxcam, placeCon, mo=1)
                        # sgCommon.AbcXformApplyKey(filepath=mainInfo['files']['dxc_path'][0],
                        #                           nodename=placeCon)

        QtWidgets.QApplication.restoreOverrideCursor()
        self.accept()
        #return

    def alembicCamera_2DPanZoom(self, pzFile):
        # modified version of dxCameraUI alembicCamera_2DPanZOom function
        f = open(pzFile, 'r')
        body = json.load(f)
        f.close()

        if not body.has_key('2DPanZoom'):
            return

        data = body['2DPanZoom']
        for node in data:
            curShapes = cmds.ls(node.split(':')[-1], r=True, type='camera')
            attrs = data[node].keys()
            for shape in curShapes:
                for a in attrs:
                    #
                    cmds.setAttr('%s.panZoomEnabled' % shape, 1)
                    cmds.setAttr('%s.renderPanZoom' % shape, 1)
                    #
                    keyData = data[node][a]
                    if type(keyData).__name__ == 'dict':
                        if keyData.has_key('frame'):
                            sgCommon.coreKeyLoad(shape, a, keyData)
                        else:
                            gv = keyData['value']
                            gt = keyData['type']
                            if gt == 'string':
                                cmds.setAttr('%s.%s' % (shape, a), gv, type='string')
                            else:
                                cmds.setAttr('%s.%s' % (shape, a), gv)
                    else:
                        cmds.setAttr('%s.%s' % (shape, a), keyData)



    def getCheckedKey(self):
        keyDic = {}

        for i in range(self.ui.fileTree.topLevelItemCount()):
            item = self.ui.fileTree.topLevelItem(i)
            record = item.getRecord()
            if record.has_key('_id'):
                itemId = record['_id']
            else:
                itemId = os.path.basename(record['files']['camera_geo_path'][0])

            keyDic[itemId] = {}

            for index in range(item.childCount()):
                childItem = item.child(index)

                if childItem.importCheck.isChecked():
                    if keyDic[itemId].has_key(childItem.text(1)):
                        keyDic[itemId][childItem.text(1)].append(childItem.getFilePath())
                    else:
                        keyDic[itemId][childItem.text(1)] = [childItem.getFilePath()]

        return keyDic


class CameraTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(CameraTreeItem, self).__init__(parent)
        self.dbRecord = None
        self.subRecord = {}

    def getRecord(self):
        return self.dbRecord

    def setRecord(self, record):
        self.dbRecord = record

    def getSubRecord(self):
        return self.subRecord


class FileTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(FileTreeItem, self).__init__(parent)
        self.dbRecord = None
        self.filePath = ''

        self.importCheck = QtWidgets.QCheckBox()
        styles = """
        QCheckBox:checked { color: rgb(0,255,0);}
        """
        self.importCheck.setStyleSheet(styles)
        self.treeWidget().setItemWidget(self, 0, self.importCheck)
        self.importCheck.stateChanged.connect(self.checkChanged)


    def getRecord(self):
        return self.dbRecord

    def setRecord(self, record):
        self.dbRecord = record

    def setFilePath(self, filepath):
        self.filePath = filepath

    def getFilePath(self):
        return self.filePath

    def checkChanged(self, state):
        # IF ITEM IS TOP LEVEL
        if not(self.parent()):
            checkState = self.importCheck.isChecked()
            for index in range(self.childCount()):
                childItem = self.child(index)
                childItem.importCheck.setChecked(checkState)

        # # IF ITEM IS CHILD
        # else:
        #     if not(self.importCheck.isChecked()):
        #         self.parent().importCheck.setChecked(state)

def showUI():
    mainWidget = DxCameraImporter()
    mainWidget.show()
