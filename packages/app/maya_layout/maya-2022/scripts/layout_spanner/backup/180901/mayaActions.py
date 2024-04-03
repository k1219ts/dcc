# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import site
from time import gmtime,strftime
import time
import datetime
import json
from bson import ObjectId
import getpass
import shutil
import re

try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')

import Qt
from Qt import QtGui
from Qt import QtWidgets
from Qt import QtCore

import pymongo
from pymongo import MongoClient
DB_PUBLISH = 'PUBLISH'
DB_PIPEPUB = 'PIPE_PUB'
COLL = 'spanner2_task'

from dxAssetExport import ExportMesh
from dxstats import inc_tool_by_user
from dxname import rulebook
from dxname import tag_parser
import rigPub

# import ui
import historyAction
from spanner2_ui_saveDevel import saveDev_Ui_Form
from spanner2_ui_savePub import savePub_Ui_Form
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )

class MayaActions(object):
    def __init__(self, filePath=None, fileName=None):
        # type: (object, object) -> object
        self.filePath = filePath
        self.fileName = fileName

    def layoutCamera(self):
        """
        import layout camera
        """
        inc_tool_by_user.run('action.Spanner2.layoutCamera', getpass.getuser())
        jsonFile = self.fileName.replace(".mb", ".shotdb")
        jsonPath = os.path.join(self.filePath, jsonFile)
        if os.path.isfile(jsonPath):
            f = open(jsonPath, "r")
            camJson = json.load(f)

        layoutCam = camJson["mayaCam"]
        cmds.file(layoutCam,
                        type="mayaBinary",
                        i=True,
                        reference=False,
                        groupLocator=False,
                        groupReference=False,
                        loadReferenceDepth="all",
                        sharedNodes="renderLayersByName",
                        preserveReferences=True,
                        mergeNamespacesOnClash=True,
                        renameAll=True,
                        renamingPrefix="layoutCamera",
                        options="v=0")

    def referenceCamera(self):
        """
        import matchmove mayabinary file by reference
        """
        inc_tool_by_user.run('action.Spanner2.ReferenceCamera', getpass.getuser())
        jsonFile = self.fileName.replace(".mb", ".shotdb")
        jsonPath = os.path.join(self.filePath, jsonFile)
        if os.path.isfile(jsonPath):
            f = open(jsonPath, "r")
            camJson = json.load(f)

        j_publisher = camJson["user"]
        j_startFrame = float(camJson["startFrame"])
        j_endFrame = float(camJson["endFrame"])
        j_renderWidth = float(camJson["renderWidth"])
        j_renderHeight = float(camJson["renderHeight"])
        j_overscan = camJson["overscan"]
        j_stereo = camJson["stereo"]
        j_mayaScene = camJson["mayaScene"]
        j_plate = camJson["plate"]
        j_showName = camJson["show"]
        j_sequence = camJson["seq"]
        j_shotName = camJson["shot"]

        cmds.setAttr("defaultResolution.width", j_renderWidth)
        cmds.setAttr("defaultResolution.height", j_renderHeight)
        cmds.setAttr("defaultResolution.deviceAspectRatio", j_renderWidth / j_renderHeight)
        cmds.setAttr("defaultRenderGlobals.animation", 1)
        cmds.setAttr("defaultRenderGlobals.extensionPadding", 4)
        cmds.setAttr("defaultRenderGlobals.startFrame", j_startFrame)
        cmds.setAttr("defaultRenderGlobals.endFrame", j_endFrame)
        cmds.playbackOptions(ast=j_startFrame, aet=j_endFrame, min=j_startFrame, max=j_endFrame)
        cmds.currentTime(j_startFrame)

        try:
            openfile = os.path.join(self.filePath, self.fileName)
            fileNamespace = self.fileName.split('_matchmove')[0] + '_matchmove'
            SDBNode = fileNamespace + "_SDBNode"
            cmds.file(openfile,
                    type="mayaBinary",
                    i=False,
                    reference=True,
                    groupLocator=False,
                    groupReference=True,
                    groupName=SDBNode,
                    loadReferenceDepth="all",
                    sharedNodes="renderLayersByName",
                    preserveReferences=True,
                    mergeNamespacesOnClash=True,
                    namespace=fileNamespace,
                    options="v=0")

            cmds.xform(SDBNode,
                     translation=(0.000000000000000, 0.000000000000000, 0.000000000000000),
                     scale=(1.000000000000000, 1.000000000000000, 1.000000000000000),
                     rotation=(-0.000000000000000, 0.000000000000000, 0.000000000000000),
                     zeroTransformPivots=True,
                     rotateOrder="zxy")

            # add attribute to SDBNode
            attrList = ["publisher", "stereo", "from", "showName","sequence", "shotName", "overscan", "plate", "startFrame", "endFrame", "renderWidth", "renderHeight", "date", "aniSubUser", "mmvPUB_id"]
            for attr in attrList:
                cmds.addAttr(SDBNode, longName=attr, niceName=attr, dataType="string")

            cmds.setAttr(SDBNode + ".publisher", j_publisher, type="string", lock=True)
            cmds.setAttr(SDBNode + ".stereo", j_stereo, type="string", lock=True)    
            cmds.setAttr(SDBNode + ".from", j_mayaScene, type="string", lock=True)       
            cmds.setAttr(SDBNode + ".showName", j_showName, type="string", lock=True)
            cmds.setAttr(SDBNode + ".sequence", j_sequence, type="string", lock=True)
            cmds.setAttr(SDBNode + ".shotName", j_shotName, type="string", lock=True)
            cmds.setAttr(SDBNode + ".overscan", j_overscan, type="string", lock=True)
            cmds.setAttr(SDBNode + ".plate", j_plate, type="string", lock=True)
            cmds.setAttr(SDBNode + ".startFrame", j_startFrame, type="string", lock=True)
            cmds.setAttr(SDBNode + ".endFrame", j_endFrame, type="string", lock=True)
            cmds.setAttr(SDBNode + ".renderWidth", j_renderWidth, type="string", lock=True)
            cmds.setAttr(SDBNode + ".renderHeight", j_renderHeight, type="string", lock=True)
            cmds.setAttr(SDBNode + ".date", cmds.date(), type="string", lock=True)
            cmds.setAttr(SDBNode + ".aniSubUser", getpass.getuser(), type="string", lock=True)

        except: "print error reading file"

    def showCamInfo(self):
        """
        show camera info from MongoDB
        """
        inc_tool_by_user.run('action.Spanner2.cameraInfo', getpass.getuser())
        DB_NAME = 'PIPE_PUB'
        COLL = 'real'
        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[COLL]
        mi = CamInfoDialog()

        result = coll.find({'_id':ObjectId("58d48f4e0ac71a219e187d1b")})
        mi.showLine.setText(str(result[0]['show']))
        mi.sequenceLine.setText(str(result[0]['sequence']))
        mi.shotNameLine.setText(str(result[0]['shot']))
        mi.dataTypeLine.setText(str(result[0]['data_type']))
        mi.artistLine.setText(str(result[0]['artist']))
        mi.enableLine.setText(str(result[0]['enabled']))
        mi.versionLine.setText(str(result[0]['version']))
        mi.timeLine.setText(str(result[0]['time']))
        mi.taskLine.setText(str(result[0]['task']))

        mi.show()
        result = mi.exec_()

    def getNamespace(self):
        if self.filePath.split('/')[3] == 'asset' or '/prev/asset' in self.filePath:
            namespace = self.fileName.split('_')[0]

        else:
            namespace = '_'.join(self.fileName.split('_')[0:2])
            if cmds.namespace( exists=namespace) == True:
                namespace += '_1'
        return namespace

    def referenceAct(self):
        """
        import scene by reference
        """
        # inc_tool_by_user.run('action.Spanner2.importRefShot', getpass.getuser())
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        node = mel.eval("file -r -gl -namespace \"%s\" -lrd \"all\" -options \"v=0\" \"%s\"" % (namespace, openfile))
        
        try:
            topNode = cmds.referenceQuery( node, nodes=True)[0]
            assetName = cmds.getAttr('%s.assetName'%topNode)
            rnNode = cmds.referenceQuery( node, referenceNode=True, topReference=True)
            cmds.lockNode(rnNode, lock=False)
            cmds.addAttr(rnNode, longName ='assetName',niceName='assetName', dataType="string" )
            cmds.addAttr(rnNode, longName ='assetName',niceName='assetName', dataType="string" )
            cmds.setAttr('%s.assetName'%rnNode, assetName, type = 'string')
            cmds.lockNode(rnNode, lock=True)
        except: print '// Error: ref Node set attr error'

    def multiReferenceAct(self):
        """
        import multiple asset as reference
        """
        # inc_tool_by_user.run('action.Spanner2.importRefMulti', getpass.getuser())
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        # get number
        mi = multiImportDialog()
        mi.show()
        result = mi.exec_()
        if result == 1:
            num = mi.item
            for i in range(num):
                node = mel.eval(
                    "file -r -gl -namespace \"%s\" -lrd \"all\" -options \"v=0\" \"%s\"" % (namespace, openfile))
                try:
                    topNode = cmds.referenceQuery(node, nodes=True)[0]
                    assetName = cmds.getAttr('%s.assetName' % topNode)
                    rnNode = cmds.referenceQuery(node, referenceNode=True, topReference=True)
                    cmds.lockNode(rnNode, lock=False)
                    cmds.addAttr(rnNode, longName='assetName', niceName='assetName', dataType="string")
                    cmds.setAttr('%s.assetName' % rnNode, assetName, type='string')
                    cmds.lockNode(rnNode, lock=True)
                except:
                    print '// Error: ref Node set attr error'

                # import asset
    def importAct(self):
        """
        just import maya scene / alembic
        """
        inc_tool_by_user.run('action.Spanner2.import', getpass.getuser())
        openfile = os.path.join(self.filePath, self.fileName)
        if os.path.splitext(str(self.fileName))[-1] == '.abc':
            if not cmds.pluginInfo('AbcImport', l=True, q=True):
                cmds.loadPlugin('AbcImport')
            print openfile
            cmd = 'AbcImport -mode import "%s"'  % openfile
            mel.eval(cmd)
        else:
            mel.eval(
                "file -import -type \"mayaBinary\" -rdn -rpr \"clash\" -options \"v=0;p=17\"  -pr -loadReferenceDepth \"all\" \"%s\"" % (
                    openfile))

    def importNSAct(self):
        """
        import asset by namespace
        """
        openfile = os.path.join(self.filePath, self.fileName)
        inc_tool_by_user.run('action.Spanner2.importAsset', getpass.getuser())
        namespace = self.getNamespace()
        mel.eval("file -import -namespace \"%s\" -ra true -options \"v=0\"  -pr -loadReferenceDepth \"all\" \"%s\"" % (
        namespace, openfile))

    def multiImportAct(self):
        """
        import multiple scene
        """
        inc_tool_by_user.run('action.Spanner2.importMulti', getpass.getuser())
        openfile = os.path.join(self.filePath, self.fileName)
        namespace = self.getNamespace()
        # get number
        mi = multiImportDialog()
        mi.show()
        result = mi.exec_()
        if result == 1:
            num = mi.item
            for i in range(num):
                cmds.file(openfile, i=True, ra=True, ns=(namespace), pr=True, lrd='all', op='v=0')

# import multiple asset by namespace
class multiImportDialog(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        ma = MayaActions()
        # ui
        label = QtWidgets.QLabel("Number of Copies:")
        self.numRerference = QtWidgets.QSpinBox()
        self.numRerference.setRange(1, 1000)
        self.ok_btn = QtWidgets.QPushButton("Ok")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.ok_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.numRerference)
        layout.addLayout(layout2)
        self.setLayout(layout)
        self.setWindowTitle("Multiple Import")

        #connection
        self.ok_btn.clicked.connect(self.outNum)
        self.close_btn.clicked.connect(self.reject)

    def outNum(self):
        self.item = self.numRerference.value()
        self.accept()

# save Dev Dialog
class SaveDevForm(QtWidgets.QDialog):
    def __init__(self, parent=None, fileName='', filePath=''):
        QtWidgets.QDialog.__init__(self)
        self.dsc = ''
        temp = cmds.file(q=True, sn=True)
        self.fileName = temp.split('/')[-1]
        self.filePath = '/'.join(temp.split('/')[0:-1])
        if self.filePath.find('/pub'):
            self.filePath = self.filePath.replace('/pub','/dev')
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = saveDev_Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('save Devel')
        self.connections()
        self.getDevInfo()
        self.setDescription()
        self.ui.snapshot_checkBox.setChecked(1)

    def connections(self):
        self.ui.dsc_lineEdit.textChanged.connect(self.setDescription)
        self.ui.dsc_lineEdit.editingFinished.connect(self.setDescription)
        self.ui.dsc_lineEdit.textEdited.connect(self.setDescription)
        self.ui.version_spinBox.valueChanged.connect(self.addPub)
        self.ui.wipVersion_spinBox.valueChanged.connect(self.addDev)
        self.ui.buttonBox.accepted.connect(self.saveDev)
        self.ui.buttonBox.rejected.connect(self.reject)

    def getDevInfo(self):
        """
        get currently opened maya scene info
        """
        self.ui.fileName_lineEdit.setText(self.fileName)
        temp = self.fileName.split('.mb')[0].split('_')
        src = self.filePath.split('/')

        if '/prev/shot' in self.filePath:
            self.assetShot = 'prevShot'
        else:
            self.assetShot = 'asset'

        if len(temp) == 5:
            self.dsc = temp[4]

        if self.assetShot == 'asset':
            self.assetName = temp[0]
            self.assetWorkCode = temp[1]
            self.version = int(temp[2].split('v')[1])
            if self.fileName.find('_w') > -1:
                self.wipVersion = int(temp[3].split('w')[1])
            else:
                dirList = []
                for i in os.listdir(self.filePath):
                    if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' and len(i.split('_')) > 3:
                        dirList.append(i)
                if dirList:
                    lastWipFile = sorted(dirList)[-1].split('.')[0]
                    self.wipVersion = int(lastWipFile.split('_')[3].split('w')[1])
                else:
                    self.wipVersion = 0

            # add version number
            self.ui.version_spinBox.setValue(self.version)

            # add wipversion number
            self.ui.wipVersion_spinBox.setValue(self.wipVersion + 1)
            self.addDev()

        if self.assetShot == 'prevShot':
            self.shotName = temp[0]
            self.shotWorkCode = temp[1]

            if self.fileName.find('_v') > -1:
                self.version = int(temp[2].split('v')[1])

            if self.fileName.find('_w') > -1:
                self.wipVersion = int(temp[3].split('w')[1])
            else:
                dirList = []
                for i in os.listdir(self.filePath):
                    if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' and len(i.split('_')) > 3:
                        dirList.append(i)
                if dirList:
                    lastWipFile = sorted(dirList)[-1].split('.')[0]
                    self.wipVersion = int(lastWipFile.split('_')[3].split('w')[1])
                else:
                    self.wipVersion = 0

            # add version number
            nextversion = self.version
            self.ui.version_spinBox.setValue(nextversion)

            # add wipversion number
            nextWipVer = self.wipVersion + 1
            self.ui.wipVersion_spinBox.setValue(nextWipVer)

        self.setDevInfo()

    def setDevInfo(self):
        """
        set work version up
        """
        if not self.dsc == '':
            self.ui.dsc_lineEdit.setText(self.dsc)
            if self.assetShot == 'asset':
                self.fileName = '%s_%s_v%s_w%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2), self.dsc)
            if self.assetShot == 'prevShot':
                self.fileName = '%s_%s_v%s_w%s_%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2), self.dsc)
        else:
            if self.assetShot == 'asset':
                self.fileName = '%s_%s_v%s_w%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.upVersion).zfill(2), str(self.wipVersion).zfill(2))
            if self.assetShot == 'prevShot':
                self.fileName = '%s_%s_v%s_w%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.upVersion).zfill(2),
                    str(self.wipVersion).zfill(2))

        self.ui.fileName_lineEdit.setText(self.fileName)

    def addDev(self):
        self.wipVersion = self.ui.wipVersion_spinBox.value()
        self.setDevInfo()

    def addPub(self):
        self.upVersion = self.ui.version_spinBox.value()
        self.setDevInfo()

    def setDescription(self):
        self.dsc = self.ui.dsc_lineEdit.text()
        self.setDevInfo()

    def saveDev(self):
        """
        save maya scene and record information to MongoDB
        """
        self.savePath = os.path.join(self.filePath, self.fileName)
        if os.path.isfile(self.savePath):
            wd = WaringDialog(QtWidgets.QDialog)
            wd.show()
            result = wd.exec_()
            if result == 1:
                pass
            else:
                return

        # save comment to DB
        self.comment = self.ui.saveDevComment_textEdit.toPlainText()
        if self.comment != '':
            historyAction.saveDBComment(self.filePath, self.fileName, self.comment)

        # save file
        cmds.file(rename=str(self.savePath))
        cmds.file(save=True, type='mayaBinary')

        # take snapshot
        try:
            if self.ui.snapshot_checkBox.isChecked() == True:
                historyAction.takeSnapShot(self.filePath, self.fileName)
        except: pass
        inc_tool_by_user.run('action.Spanner2.saveDev', getpass.getuser())
        self.accept()

# save Pub Dialog
class SavePubForm(QtWidgets.QDialog):
    def __init__(self, parent=None, fileName='', filePath=''):
        QtWidgets.QDialog.__init__(self)
        self.description = ''
        temp = cmds.file(q=True, sn=True)
        self.fileName = temp.split('/')[-1]
        self.filePath = '/'.join(temp.split('/')[0:-1])
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = savePub_Ui_Form()
        self.ui.setupUi(self)

        self.setWindowTitle('save Publish')
        self.getPubInfo()
        self.setDevInfo()
        self.setDescription()
        self.ui.model_groupBox.setEnabled(0)

        # connection
        self.ui.buttonBox.rejected.connect(self.reject)
        self.ui.buttonBox.accepted.connect(self.savePub)
        self.ui.dsc_lineEdit.textChanged.connect(self.setDescription)
        self.ui.dsc_lineEdit.editingFinished.connect(self.setDescription)
        self.ui.dsc_lineEdit.textEdited.connect(self.setDescription)

    def getPubInfo(self):
        temp = self.fileName.split('.mb')[0].split('_')
        src = self.filePath.split('/')
        if 'shot' in self.filePath:
            self.assetShot = 'prevShot'
        else:
            self.assetShot = 'asset'

        if len(temp) == 5:
            self.description = temp[4]
            
        # published file
        if self.assetShot == 'asset':
            self.assetName = temp[0]
            self.assetWorkCode = temp[1]
            self.version = int(temp[2].split('v')[1])
            if self.fileName.find('_w') > -1:
                self.wipVersion = int(temp[3].split('w')[1])

        if self.assetShot == 'prevShot':
            self.shotName = temp[0]
            self.shotWorkCode = temp[1]

            if self.fileName.find('_v') > -1:
                self.version = int(temp[2].split('v')[1])
            if self.fileName.find('_w') > -1:
                self.wipVersion = int(temp[3].split('w')[1])

        self.workCode = temp[1]
        # dev file
        self.setPubInfo()

    def setPubInfo(self):
        if self.assetShot == 'asset':
            if self.description:
                self.fileName = '%s_%s_v%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2), self.description)
            else:
                self.fileName = '%s_%s_v%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2))

        if self.assetShot == 'prevShot':
            if self.description:
                self.fileName = '%s_%s_v%s_%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2), self.description)
            else:
                self.fileName = '%s_%s_v%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2))

        self.ui.fileName_lineEdit.setText(self.fileName)

        # dev pub file path
        if self.filePath.find('/pub') > -1:
            self.devPath = self.filePath.replace('/pub','/dev')
        else:
            self.devPath = self.filePath
        self.pubPath = self.filePath.replace('/dev/','/pub/')

    def setDevInfo(self):
        lastWipFile = ''
        temp = cmds.file(q=True, sn=True)
        if temp.find('/pub'):
            temp = temp.replace('/pub','/dev')
            path = os.path.join(temp.split('/scenes')[0], 'scenes')
        print 'temp ', temp
        print 'path ', path

        # add wipversion number
        if self.assetShot == 'asset':
            dirList = []
            for i in os.listdir(path):
                if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' and i.find('_w') > -1:
                    dirList.append(i)
            if dirList:
                lastWipFile = sorted(dirList)[-1].split('.')[0]
                wipVersion = lastWipFile.split('_')[3].split('w')[-1]
            else:
                wipVersion = 0

        if self.assetShot == 'prevShot':
            dirList = []
            for i in os.listdir(path):
                if '_v' + str(self.version).zfill(2) in i and i.split('.')[-1] == 'mb' and i.find('_w') > -1:
                    dirList.append(i)
            if dirList:
                lastWipFile = sorted(dirList)[-1].split('.')[0]
                wipVersion = lastWipFile.split('_')[3].split('w')[-1]
            else:
                wipVersion = 0

        nextWipVer = int(wipVersion) + 1

        # description exists:
        if not self.description == '':
            if self.assetShot == 'asset':
                self.nextDev = '%s_%s_v%s_w%s_%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2), self.description)
            if self.assetShot == 'prevShot':
                self.nextDev = '%s_%s_v%s_w%s.mb' % (
                    self.shotName, self.shotWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2), self.description)
            self.ui.dsc_lineEdit.setText(self.description)

        # description not exists:
        else:
            if self.assetShot == 'asset':
                self.nextDev = '%s_%s_v%s_w%s.mb' % (
                    self.assetName, self.assetWorkCode, str(self.version).zfill(2),
                    str(nextWipVer).zfill(2))
            if self.assetShot == 'prevShot':
                self.nextDev = '%s_%s_v%s_w%s.mb' % (
                    self.shotName, self.shotWorkCode,
                    str(self.version).zfill(2), str(nextWipVer).zfill(2))

        self.ui.nextDev_lineEdit.setText(self.nextDev)

    def saveRigPub(self):
        # for rigging data publish ( *.json, *.abc )
        saveList = []
        saveList.append(self.savePubPath)
        saveList.append(self.savePubPath.replace('.mb', '.abc'))
        saveList.append(self.savePubPath.replace('.mb', '.json'))
        print saveList

        for path in saveList:
            if os.path.isfile(path):
                os.chmod(path, 0777)

        # from dexcmd module
        rigPub.rigPubSpanner(self.savePubPath)
        try:
            rigPubDB(self.fileName, self.savePubPath)
        except:
            print 'rig Publish DB insert Failed'

        print '# Result : Rig published successfully.'
        inc_tool_by_user.run('action.Spanner2.savePub_Rig', getpass.getuser())

    def savePub(self):
        # publish path
        self.saveNextDevPath = os.path.join(self.devPath, self.nextDev)
        self.savePubPath = os.path.join(self.pubPath, self.fileName)
        self.workCode = self.savePubPath.split('/')[6]
        self.taskType = self.savePubPath.split('/')[3]
        print 'saveNextDevPath=', self.saveNextDevPath
        print 'savePubpath = ', self.savePubPath
        if os.path.isfile(self.savePubPath):
            # show exist error!
            wd = WaringDialog(QtWidgets.QDialog)
            wd.show()
            result = wd.exec_()
            if result == 1:
                print 'save Pub'
            else:
                return

        # save Dev
        # check whether to save dev file
        devChecking = self.ui.saveDev_groupBox.isChecked()
        if devChecking == True:
            cmds.file(rename=str(self.saveNextDevPath))
            cmds.file(save=True, type='mayaBinary')
            inc_tool_by_user.run('action.Spanner2.saveDev', getpass.getuser())
        else:
            print 'save Current File'
            cmds.file(save=True, type='mayaBinary')

        # pub for rig
        if self.workCode == 'rig':
            self.saveRigPub()

        # pub for model
        elif self.workCode == 'model':
            self.workType = self.filePath.split('/')[4]
            self.workName = self.filePath.split('/')[5]
            self.mb = self.ui.mb_checkBox.isChecked()
            self.abc = self.ui.abc_checkBox.isChecked()
            self.tex = self.ui.tex_checkBox.isChecked()
            self.savePubPath = '/show/real/asset/%s/%s/model/pub/scenes/%s'\
                               % (self.workType, self.workName,self.fileName.split('.')[0])
            # for modeling data publish ( *.json, *.abc )
            # from dexcmd module
            ex = ExportMesh(self.savePubPath, cmds.ls(sl=True, l=True))
            ex.mesh_export(self.mb, self.abc, self.tex)
            print '# Result : Modeling published successfully.'
            inc_tool_by_user.run('action.Spanner2.savePub_Model', getpass.getuser())
        # pub for shot
        else:
            if os.path.isfile(self.savePubPath):
                os.chmod(self.savePubPath, 0777)

            if not os.path.exists(os.path.dirname(self.savePubPath)):
                os.makedirs(os.path.dirname(self.savePubPath))

            cmds.file(rename=str(self.savePubPath))
            cmds.file(save=True, type='mayaBinary')

            if os.path.isfile(self.savePubPath):
                os.chmod(self.savePubPath, 0555)
            inc_tool_by_user.run('action.Spanner2.savePub_%s'% self.workCode, getpass.getuser())

        # take snapshot
        try:
            historyAction.takeSnapShot(self.pubPath, self.fileName)
        except: pass
        # save to DB ( publish info, comment)
        self.addDBInfo()
        self.comment = self.ui.savePubComment_textEdit.toPlainText()
        if self.comment != '':
            historyAction.saveDBComment(self.pubPath, self.fileName, self.comment)

        print '# Result : Information saved to MongoDB successfully.'
        self.accept()

    def addDBInfo(self):
        show = self.pubPath.split('/')[2]
        name = self.pubPath.split('/')[5]
        path = '/'.join(self.pubPath.split('/')[0:6])
        artist = os.getlogin()
        workCode = self.pubPath.split('/')[6]
        file = self.fileName.split('.')[0]
        time = strftime('%Y-%m-%dT%H:%M:%S')

        client = MongoClient(DB_IP)
        db = client['ASSET']
        coll = db[show]

        time = strftime('%Y-%m-%dT%H:%M:%S')
        data = {'artist': artist, 'file': file, 'time': time}
        coll.update({'name': name, 'path': path},
                    {'$set': {'%s.pub.%s' % (workCode, file): data}}
                    , upsert=True)

    def setDescription(self):
        self.description = self.ui.dsc_lineEdit.text()
        self.setDevInfo()
        self.setPubInfo()

# insert rig pub DB
def rigPubDB(devFileName, pubPath):
    jsonPath = pubPath.replace('.mb', '.json')
    print jsonPath
    if os.path.isfile(jsonPath):
        rigLog = json.loads(open(jsonPath, 'r').read())
        print rigLog
    else:
        return

    # get db version
    src = pubPath.split('/')
    show = src[src.index('show') + 1]
    asset = src[src.index('asset')+2]
    client = MongoClient(DB_IP)
    db = client[DB_PIPEPUB]
    coll = db[show]
    result = coll.find({'show': show, 'asset': asset, 'data_type': 'rig' }
                           ).sort('version', pymongo.DESCENDING).limit(1)
    for i in result:
        dbVer = i['version']
    try:
        dbVer += 1
    except:
        dbVer = 1

    logDB = {}
    logDB['files'] = {}
    logDB['files']['maya_dev_file'] = [rigLog['_Header']['context']]
    logDB['files']['maya_path'] = [pubPath]
    logDB['files']['abc_path'] = [rigLog['RigPublish']['mesh']]
    logDB['files']['json_path'] = [jsonPath]
    data = {"version": dbVer,
            "task": "rig",
            "show": src[src.index('show') + 1],
            "asset": src[src.index('asset')+2],
            "data_type": "rig",
            "enabled": True,
            "task_publish": {
                'type': src[src.index('show') + 2],
                'version': rigLog['_Header']['version']
            },
            "time": datetime.datetime.now().isoformat(),
            "artist": getpass.getuser(),
            }
    logDB.update(data)
    result = tag_parser.run(pubPath)
    logDB['tags'] = result
    coll.insert(logDB)
    print 'rig DB inserted successfully (PIPELINE 2.0) '

    # warnig Dialog
class WaringDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, text=''):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("File exists. Are you sure to save?\n")
        if text:
            label = QtWidgets.QLabel(text)
        self.ok_btn = QtWidgets.QPushButton("Ok")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.ok_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2,3,0)
        self.setLayout(layout)
        self.setWindowTitle("Warning")

        #connection
        self.ok_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)

class CamInfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)

        label1 = QtWidgets.QLabel("Shot Name")
        label2 = QtWidgets.QLabel("Data Type")
        label3 = QtWidgets.QLabel("Artist")
        label4 = QtWidgets.QLabel("Enabled")
        label5 = QtWidgets.QLabel("Version")
        label6 = QtWidgets.QLabel("Time")
        label7 = QtWidgets.QLabel("Task")
        label8 = QtWidgets.QLabel("Show")
        label9 = QtWidgets.QLabel("Sequence")

        self.shotNameLine = QtWidgets.QLabel()
        self.dataTypeLine = QtWidgets.QLabel()
        self.artistLine = QtWidgets.QLabel()
        self.enableLine = QtWidgets.QLabel()
        self.versionLine = QtWidgets.QLabel()
        self.timeLine = QtWidgets.QLabel()
        self.taskLine = QtWidgets.QLabel()
        self.showLine = QtWidgets.QLabel()
        self.sequenceLine = QtWidgets.QLabel()

        layout = QtWidgets.QGridLayout()

        layout.addWidget(label2, 2, 0)
        layout.addWidget(label3, 3, 0)
        layout.addWidget(label4, 4, 0)
        layout.addWidget(label5, 5, 0)
        layout.addWidget(label6, 6, 0)
        layout.addWidget(label7, 7, 0)
        layout.addWidget(label8, 8, 0)
        layout.addWidget(label9, 9, 0)
        layout.addWidget(label1, 10, 0)


        layout.addWidget(self.dataTypeLine, 2, 1)
        layout.addWidget(self.artistLine, 3, 1)
        layout.addWidget(self.enableLine, 4, 1)
        layout.addWidget(self.versionLine, 5, 1)
        layout.addWidget(self.timeLine, 6, 1)   
        layout.addWidget(self.taskLine, 7, 1)
        layout.addWidget(self.showLine, 8, 1)
        layout.addWidget(self.sequenceLine, 9, 1)
        layout.addWidget(self.shotNameLine, 10, 1)

        self.setLayout(layout)
        self.setWindowTitle("Camera Info")
        self.setMinimumWidth(400)
        self.setMaximumWidth(400)

