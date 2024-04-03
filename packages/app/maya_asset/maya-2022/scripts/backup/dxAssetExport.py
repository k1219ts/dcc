#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#    2015.06.20 $2
#
#-------------------------------------------------------------------------------

import datetime
import getpass
import os
import string

import Qt.QtWidgets as QtWidgets
import Qt.QtCore as QtCore
import Qt.QtGui as QtGui

import maya.cmds as cmds
import maya.mel as mel
import pymongo

# 2017.04.12 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
from pymongo import MongoClient

import ModelPubJobScript
import dxCommon

from dxExportMesh import ExportMesh
from SelectDeformUI import Ui_Dialog as SelectDeformUI

import dxConfig

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"

# 2017.09.18 by daeseok.chae
import dxAsset

def getPubVersion(show, task, data_type, asset_name = "", shot = ""):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print show, task, data_type, asset_name
    record = {'show': show,
                               'task': task,
                               'data_type': data_type}
    if asset_name != "":
        record['asset_name'] = asset_name

    if shot != "":
        record['shot'] = shot

    recentDoc = coll.find_one(record,
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1

def getLastPub(show, task, data_type, asset_name="", shot=""):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print show, task, data_type, asset_name
    record = {'show': show,
              'task': task,
              'data_type': data_type}
    if asset_name != "":
        record['asset_name'] = asset_name

    if shot != "":
        record['shot'] = shot

    recentDoc = coll.find_one(record,
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc
    else:
        return None

#-------------------------------------------------------------------------------
#
#    Dialog UI
#   1. Select Publish Type
#   2.
#
#-------------------------------------------------------------------------------
from SelectPubTypeUI import Ui_Dialog

class SelectPubUI(QtWidgets.QDialog):
    def __init__(self, parent = dxCommon.getMayaWindow()):
        super(SelectPubUI, self).__init__(parent = parent)

        self.move(parent.frameGeometry().center() -  self.frameGeometry().center())

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # connect Signal
        self.ui.okButton.clicked.connect(self.okButtonClick)
        self.ui.cancelButton.clicked.connect(self.cancelButtonClick)

    def okButtonClick(self):
        widget = None
        if self.ui.meshRadioButton.isChecked() == True:
            widget = BasicMeshClass(self)
        elif self.ui.envRadioButton.isChecked() == True:
            widget = EnvDataClass(self)
        elif self.ui.zenvRadioButton.isChecked() == True:
            widget = ZEnvDataClass(self)
        elif self.ui.shotRadioButton.isChecked() == True:
            widget = ShotClass(self)
        elif self.ui.zennRadioButton.isChecked() == True:
            widget = ZennClass(self)
        elif self.ui.zennShotRadioButton.isChecked() == True:
            widget = ZennShotClass(self)

        widget.exec_()
        # self.close()

    def cancelButtonClick(self):
        self.close()

#-------------------------------------------------------------------------------
#
#    Dialog UI
#
#-------------------------------------------------------------------------------
# 2017.05.30 by daeseok.chae
class BaseUIClass(QtWidgets.QDialog):
    def __init__(self, parent = None, child = None):
        QtWidgets.QDialog.__init__(self, parent)
        self.child = child

        self.resize(700, 280)

        self.defaultGroupBox = QtWidgets.QGroupBox(self)
        self.defaultGroupBox.setGeometry(QtCore.QRect(10, 10, 680, 120))
        self.defaultGroupBox.setTitle("Default Setting")

        showLabel = QtWidgets.QLabel(self.defaultGroupBox)
        showLabel.setText("Show")
        showLabel.setGeometry(QtCore.QRect(20, 30, 40, 40))

        self.showComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.showComboBox.setGeometry(QtCore.QRect(60, 35, 100, 30))
        self.showComboBox.addItems(os.listdir('/show'))

        pubOptionGroupBox = QtWidgets.QGroupBox(self)
        pubOptionGroupBox.setGeometry(QtCore.QRect(10, 140, 300, 80))
        pubOptionGroupBox.setTitle("Publish Option")

        gridLayoutWidget = QtWidgets.QWidget(pubOptionGroupBox)
        gridLayoutWidget.setGeometry(QtCore.QRect(10, 30, 295, 50))
        gridLayout = QtWidgets.QGridLayout(gridLayoutWidget)
        self.alembicCheckBox = QtWidgets.QCheckBox(gridLayoutWidget)
        self.alembicCheckBox.setChecked(True)
        self.alembicCheckBox.setText("Alembic")
        gridLayout.addWidget(self.alembicCheckBox, 0, 0, 1, 1)
        self.texAlembicCheckBox = QtWidgets.QCheckBox(gridLayoutWidget)
        self.texAlembicCheckBox.setText("Alembic for Texture")
        self.texAlembicCheckBox.setChecked(True)
        gridLayout.addWidget(self.texAlembicCheckBox, 1, 0, 1, 1)
        self.mayaBinaryCheckBox = QtWidgets.QCheckBox(gridLayoutWidget)
        self.mayaBinaryCheckBox.setText("Maya Binary")
        gridLayout.addWidget(self.mayaBinaryCheckBox, 0, 1, 1, 1)
        self.isPreviewCheckBox = QtWidgets.QCheckBox(gridLayoutWidget)
        self.isPreviewCheckBox.setText("Preview")
        self.isPreviewCheckBox.stateChanged.connect(self.previewCheckBoxStateChanged)
        gridLayout.addWidget(self.isPreviewCheckBox, 1, 1, 1, 1)

        publishGroupBox = QtWidgets.QGroupBox(self)
        publishGroupBox.setGeometry(QtCore.QRect(320, 140, 370, 110))
        publishGroupBox.setTitle("Publish")

        self.exOutputPathEdit = QtWidgets.QLineEdit(publishGroupBox)
        self.exOutputPathEdit.setGeometry(QtCore.QRect(10, 30, 305, 35))
        # self.exOutputPathEdit.setReadOnly(True)

        self.customDirPath = QtWidgets.QPushButton(publishGroupBox)
        self.customDirPath.setGeometry(QtCore.QRect(320, 30, 35, 35))
        self.customDirPath.clicked.connect(self.selectDirPath)
        imagePath_folder = '/dexter/Cache_DATA/RND/jeongmin/CacheExport_jm/resources/folder.png'
        self.customDirPath.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))

        self.tractorCheckBox = QtWidgets.QCheckBox(publishGroupBox)
        self.tractorCheckBox.setGeometry(QtCore.QRect(10, 70, 80, 35))
        self.tractorCheckBox.setText("Tractor")

        self.exportButton = QtWidgets.QPushButton(publishGroupBox)
        self.exportButton.setGeometry(QtCore.QRect(90, 70, 265, 35))
        self.exportButton.setText("Export")
        self.exportButton.clicked.connect(self.export_process)

        dexterLabel = QtWidgets.QLabel(self)
        dexterLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        dexterLabel.setGeometry(QtCore.QRect(395, 245, 300, 35))
        dexterLabel.setText('@DexterDigital Asset workflow R&D')

        self.ppRulebook = rulebook.Coder()
        self.ppRulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_publish.yaml")

    def selectDirPath(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setMinimumSize(800, 400)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        result = dialog.exec_()
        if result == 1:
            print dialog.selectedFiles()
            dirPath = dialog.selectedFiles()[-1]
            basePath = 'test_model_v01'
            self.exOutputPathEdit.setText(os.path.join(dirPath, basePath))
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.red)
            self.exOutputPathEdit.setPalette(palette)
            self.isWriteDB = False

    def setRulebookTask(self, taskNode):
        self.taskNode = taskNode

    def setPreviewPublishPath(self, path):
        self.exOutputPathEdit.setText(path)

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        self.exOutputPathEdit.setPalette(palette)
        self.isWriteDB = True

    def setPreview(self, state):
        if state:
            self.ppRulebook.flag['PUBDEV'] = 'dev'
        else:
            self.ppRulebook.flag['PUBDEV'] = 'pub'

    def previewCheckBoxStateChanged(self, state):
        self.setPreview(self.isPreviewCheckBox.checkState() == QtCore.Qt.Checked)
        self.child.updatePreviewLabel()

    def backupData(self):
        try:
            data_type = 'model'

            if "zenn" in self.taskNode.name:
                data_type = 'zenn'

            if str(self.ppRulebook.flag['SHOT']) != 'SEQUENCE_NUMBER':
                data = getLastPub(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type=data_type,
                                                  shot=str(self.ppRulebook.flag['SHOT']))
            # record other
            else:
                data = getLastPub(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type=data_type,
                                                  asset_name=str(self.ppRulebook.asset.flag["ASSET"]))

                if data == None:
                    return

                for pubFile in data['files']:
                    if pubFile == 'backup':
                        continue

                    filePath = data['files'][pubFile][0]
                    dirName = os.path.dirname(filePath)
                    baseName = os.path.basename(filePath)

                    backupPath = os.path.join(dirName, 'backup', baseName)

                    if not os.path.isdir(os.path.join(dirName, 'backup')):
                        os.mkdir(os.path.join(dirName, 'backup'))

                    os.system('cp -rf {0} {1}'.format(filePath, backupPath))
        except:
            print "Warning!!!!!!!!!!!!!!! backup fail"

    def export_file( self ):
        if 'assetname' in self.exOutputPathEdit.text() or 'type' in self.exOutputPathEdit.text() or 'dataName' in self.exOutputPathEdit.text():
            cmds.confirmDialog(title='Warning',
                               message='Check Publish Path',
                               messageAlign='center',
                               icon='warning',
                               button='OK')
            return False

        self.backupData()

        zennOutFiles = None
        zennDeformNodes = None
        logs = ""
        if "zenn" in self.taskNode.name:
            result, cachePath, deformNodes = self.zennBakeCache()
            if result == False:
                return
            else:
                zennOutFiles = cachePath
                zennDeformNodes = deformNodes
                logs = 'ZENN Cache\t:'
                logs += '%s' % cachePath

        # assetAttr = dxAsset.AssetAttribute()
        exportPath = self.exOutputPathEdit.text()
        if exportPath.startswith('/netapp/dexter/show'):
            exportPath.replace('/netapp/dexter/show', '/show')

        # assetAttr.m_assetname = string.join(exportPath.split('/')[3:6], '/')
        # processes = []
        # processes.append('objectname')
        # processes.append('texture')
        # processes.append('uniquename')
        # processes.append('renderstats')
        # print "result :", assetAttr.add(processes)

        expClass = ExportMesh( self.exOutputPathEdit.text(), cmds.ls(sl=True, l=True) )
        if not logs == "":
            expClass.appendLogs(logs)
        expClass.mesh_export( self.mayaBinaryCheckBox.isChecked(), self.alembicCheckBox.isChecked(), self.texAlembicCheckBox.isChecked() )

        if str(self.ppRulebook.flag['PUBDEV']) == 'pub' and self.isWriteDB == True:
            self.recordDB(expClass.outputFilePath, zennOutFiles, zennDeformNodes)

    def zennBakeCache(self):
        if not cmds.pluginInfo('ZENNForMaya', q = True, l = True):
            cmds.confirmDialog(title='Warning',
                               message='ZENNForMaya is not loaded.',
                               messageAlign='center',
                               icon='warning',
                               button='OK')
            self.close()
            return False, None, None

        if not cmds.objExists("ZN_ExportSet"):
            cmds.confirmDialog(title='Error',
                               message="don't find ZN_ExportSet",
                               messageAlign='center',
                               icon='warning',
                               button='OK')
            self.close()
            return False, None, None

        nodeNames = cmds.sets('ZN_ExportSet', q=True)

        # dialog = SelectDeformUI()
        #
        # dialog.deformList.addItems(cmds.ls(type = ['ZN_Deform', 'ZN_FeatherInstance']))
        # dialog.okBtn.clicked.connect(dialog.close)
        # dialog.exec_()
        #
        # deformList = dialog.deformList.selectedItems()
        #
        # if len(deformList) == 0:
        #     cmds.confirmDialog(title='Warning',
        #                        message="don't select ZN_Deform Nodes",
        #                        messageAlign='center',
        #                        icon='warning',
        #                        button='OK')
        #     return False, None, None

        # ZN_StrandsViewers = cmds.ls(type='ZN_StrandsViewer')
        #
        # print ZN_StrandsViewers
        #
        nodeNames = list(set(nodeNames))

        startFrame = endFrame = float(1.0)

        # /show/gcd1/asset/char/normalSpider/hair/pub/scenes/normalSpider_hair_v01.mb

        fileName = os.path.basename(self.exOutputPathEdit.text())
        cachePath = self.exOutputPathEdit.text().split('/scenes')[0] + '/data/zenn/' + fileName

        ZN_StrandsViewerModes = list()
        ZN_StrandsViewers = cmds.ls(type = 'ZN_StrandsViewer')

        for strandsViewer in ZN_StrandsViewers:
            ZN_StrandsViewerModes.append(cmds.getAttr(strandsViewer + ".colorMode"))
            cmds.setAttr(strandsViewer + '.colorMode', 0)

        ZN_FeatherSetViewerModes = list()
        ZN_FeatherSetViewers = cmds.ls(type='ZN_FeatherSetViewer')

        for featherViewer in ZN_FeatherSetViewers:
            ZN_FeatherSetViewerModes.append(cmds.getAttr(featherViewer + ".colorMode"))
            cmds.setAttr(featherViewer + '.colorMode', 0)

        ZN_MeshViewerModes = list()
        ZN_MeshViewers = cmds.ls(type='ZN_MeshViewer')

        for meshViewer in ZN_MeshViewers:
            ZN_MeshViewerModes.append(cmds.getAttr(meshViewer + ".whichMesh"))
            cmds.setAttr(meshViewer + '.whichMesh', 0)

        for nodeName in nodeNames:
            type = cmds.nodeType(nodeName)

            if ((type == "ZN_FeatherInstance") or (type == "ZN_Instance")):
                cmds.setAttr(nodeName + '.cacheGenMode', 1)

        cmds.select(cl = True)

        nodeNamesStr = ""
        for nodeName in nodeNames:
            nodeNamesStr += nodeName + ' '

        print "nodeNamesStr :", nodeNamesStr
        print "startFrame :", startFrame
        print "endFrame :", endFrame
        print "cachePath :", cachePath

        step = float(1.0)

        if not os.path.exists( cachePath ):
            os.makedirs( cachePath )

        print 'ZN_CacheGenCmd -nodeNames "{0}" -startFrame {1} -endFrame {2} -step {3} -cachePath "{4}"'.format(nodeNamesStr,
                                                                                                               startFrame,
                                                                                                               endFrame,
                                                                                                               step,
                                                                                                               cachePath)

        mel.eval('ZN_CacheGenCmd -nodeNames "{0}" -startFrame {1} -endFrame {2} -step {3} -cachePath "{4}"'.format(nodeNamesStr,
                                                                                                               startFrame,
                                                                                                               endFrame,
                                                                                                               step,
                                                                                                               cachePath))
        # restore
        for index, strandsViewer in enumerate(ZN_StrandsViewers):
            cmds.setAttr(strandsViewer + '.colorMode', ZN_StrandsViewerModes[index])

        for index, featherViewer in enumerate(ZN_FeatherSetViewers):
            cmds.setAttr(featherViewer + '.colorMode', ZN_FeatherSetViewerModes[index])

        for index, meshViewer in enumerate(ZN_MeshViewers):
            cmds.setAttr(meshViewer + '.whichMesh', ZN_MeshViewerModes[index])

        for index, nodeName in enumerate(nodeNames):
            type = cmds.nodeType(nodeName)

            if ((type == "ZN_FeatherInstance") or (type == "ZN_Instance")):
                cmds.setAttr(nodeName + '.cacheGenMode', 0)

        cmds.select(cl = True)

        return True, cachePath, nodeNames


    def recordDB(self, outputFileInfo, zennOutFiles = None, deformNodes = None):
        print "pub file info :", outputFileInfo
        data_type = 'model'
        try:

            if "zenn" in self.taskNode.name:
                data_type = 'zenn'

            typeTaskCoder = self.taskNode

            files = outputFileInfo

            if not zennOutFiles == None:
                files['zenn_path'] = [zennOutFiles]

            # if self.mayaBinaryCheckBox.isChecked():
            #     files['model_scene_path'] = [typeTaskCoder.product['model_scene_path']]
            #     files['model_scene_json_path'] = [typeTaskCoder.product['model_scene_json_path']]
            #
            # if self.alembicCheckBox.isChecked():
            #     files['model_abc_path'] = [typeTaskCoder.product['model_abc_path']]
            #     files['model_json_path'] = [typeTaskCoder.product['model_json_path']]
            #
            # if self.texAlembicCheckBox.isChecked():
            #     files['model_tex_abc_path'] = [typeTaskCoder.product['model_tex_abc_path']]
            #     files['model_tex_json_path'] = [typeTaskCoder.product['model_tex_json_path']]

            ######### 내일 이 밑에서부터 수정 #########
            record = {}

            # record Basic
            record['show'] = str(self.ppRulebook.flag["PROJECT"])
            record['task'] = "asset"
            record['data_type'] = data_type
            record['files'] = files
            if not zennOutFiles == None:
                record['task_publish'] = {"nodes": deformNodes}
            else:
                record['task_publish'] = {}
            blockNode = cmds.ls(type = 'xBlock')
            if blockNode:
                rigVersion = cmds.getAttr("%s.rigVersion" % blockNode[0])
                record['task_publish']['rigVersion'] = rigVersion
            record['time'] = datetime.datetime.now().isoformat()
            record['enabled'] = True
            record['artist'] = getpass.getuser()
            record['maya_version'] = 'maya2_2017'
            record['tags'] = tag_parser.run(self.exOutputPathEdit.text())

            # record Shot
            if str(self.ppRulebook.flag['SHOT']) != 'SEQUENCE_NUMBER':
                if data_type == "zenn":
                    record['asset_name'] = str(self.ppRulebook.flag['SHOT'])
                else:
                    record['shot'] = str(self.ppRulebook.flag['SHOT'])
                record['version'] = getPubVersion(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type=data_type,
                                                  shot = str(self.ppRulebook.flag['SHOT']))
            # record other
            else:
                record['version'] = getPubVersion(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type=data_type,
                                                  asset_name=str(self.ppRulebook.asset.flag["ASSET"]))

                record['asset_type'] = str(self.ppRulebook.asset.flag["TYPE"])
                record['asset_name'] = str(self.ppRulebook.asset.flag["ASSET"])

            COLLNAME = str(self.ppRulebook.flag["PROJECT"])
            client = MongoClient(DBIP)
            database = client[DBNAME]
            dbColl = database[COLLNAME]

            dbColl.insert_one(record)
            print "success db write", record
        except Exception as e:
            COLLNAME = "puberror"
            dbName = "test"
            client = MongoClient(DBIP)
            database = client[dbName]
            dbColl = database[COLLNAME]

            record = {"user": getpass.getuser(),
                      "errorMsg": str(e),
                      "time": datetime.datetime.now().isoformat(),
                      "project": str(self.ppRulebook.flag["PROJECT"]),
                      "asset": str(self.ppRulebook.asset.flag["ASSET"]),
                      "type": data_type,
                      "outputpath": self.exOutputPathEdit.text(),
                      "maya_version": "maya2_2017"}

            dbColl.insert_one(record)

    def export_tractor( self ):
        currentfile = cmds.file( q=True, sn=True )
        opt = {
            'm_basename': os.path.basename(currentfile).split('.')[0],
            'm_mayafile': currentfile,
            'm_exportfile': self.exOutputPathEdit.text(),
            'm_maya': int( self.mayaBinaryCheckBox.isChecked() ),
            'm_abc': int( self.alembicCheckBox.isChecked() ),
            'm_tex': int( self.texAlembicCheckBox.isChecked() ),
            'm_expGroup': ''
                }
        if cmds.ls(sl=True):
            opt['m_expGroup'] = string.join( cmds.ls(sl=True, l=True), ',' )
        job = ModelPubJobScript.JobMain( opt )
        job.doIt()

    def export_process( self ):
        current_file = cmds.file( sn=True, q=True )
        if not current_file:
            cmds.confirmDialog( title = 'Warning',
                                message = 'Save File',
                                messageAlign = 'center',
                                icon = 'warning',
                                button = 'OK' )
            self.close()
            return

        if self.isWriteDB == False:
            result = cmds.confirmDialog(title='Warning!',
                               message="DB information is not saved\nDo you want to proceed?",
                               icon='warning',
                               button=['OK', 'CANCEL'])
            if result == "CANCEL":
                return

        # save current file
        cmds.file( save=True )
        if self.tractorCheckBox.isChecked():
            self.export_tractor()
            msg = cmds.confirmDialog( title = 'Export',
                                      message = 'Export to Tractor',
                                      messageAlign = 'center',
                                      icon = 'information',
                                      button = 'OK' )
        else:
            if self.export_file() == False:
                return
        self.close()


class BasicMeshClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('Basic Mesh Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        showLabel = QtWidgets.QLabel(self.defaultGroupBox)
        showLabel.setText("Type")
        showLabel.setGeometry(QtCore.QRect(165, 30, 40, 40))

        self.typeComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.typeComboBox.setGeometry(QtCore.QRect(205, 35, 100, 30))
        self.typeComboBox.addItems(['char', 'env', 'vehicle', 'prop'])
        self.typeComboBox.currentIndexChanged.connect(self.typeComboBoxIndexChanged)

        assetNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        assetNameLabel.setText("AssetName")
        assetNameLabel.setGeometry(QtCore.QRect(315, 30, 90, 40))

        self.assetNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.assetNameLineEdit.setGeometry(QtCore.QRect(395, 35, 100, 30))
        self.assetNameLineEdit.textChanged.connect(self.assetNameTextChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(505, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(560, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.model.basic.decode(outputPath, product_name='root')
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.asset.flag['TYPE'] = decodingRule['TYPE']
                self.ppRulebook.asset.flag['ASSET'] = decodingRule['ASSET']
                self.ppRulebook.flag['VER'] = decodingRule['VER']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.typeComboBox.setCurrentIndex(self.typeComboBox.findText(decodingRule['TYPE']))
                self.assetNameLineEdit.setText(decodingRule['ASSET'])
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])

                self.updatePreviewLabel()
            except Exception as e:
                self.ppRulebook.asset.flag['TYPE'] = 'char'
                print e.message
        else:
            self.ppRulebook.asset.flag['TYPE'] = 'char'

        self.setRulebookTask(self.ppRulebook.asset.model.basic)


    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.updatePreviewLabel()

    def typeComboBoxIndexChanged(self, index):
        self.ppRulebook.asset.flag['TYPE'] = self.typeComboBox.currentText()
        self.updatePreviewLabel()

    def assetNameTextChanged(self, text):
        self.ppRulebook.asset.flag['ASSET'] = self.assetNameLineEdit.text()
        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + self.versionLineEdit.text()
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        self.setPreviewPublishPath(self.ppRulebook.asset.model.basic.product['root'])

class ZennClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('Zenn Scene Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        showLabel = QtWidgets.QLabel(self.defaultGroupBox)
        showLabel.setText("Type")
        showLabel.setGeometry(QtCore.QRect(165, 30, 40, 40))

        self.typeComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.typeComboBox.setGeometry(QtCore.QRect(205, 35, 100, 30))
        self.typeComboBox.addItems(['char', 'env', 'vehicle', 'prop'])
        self.typeComboBox.currentIndexChanged.connect(self.typeComboBoxIndexChanged)

        assetNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        assetNameLabel.setText("AssetName")
        assetNameLabel.setGeometry(QtCore.QRect(315, 30, 90, 40))

        self.assetNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.assetNameLineEdit.setGeometry(QtCore.QRect(395, 35, 100, 30))
        self.assetNameLineEdit.textChanged.connect(self.assetNameTextChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(505, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(560, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        self.alembicCheckBox.setChecked(False)
        self.alembicCheckBox.setDisabled(True)
        self.texAlembicCheckBox.setChecked(False)
        self.texAlembicCheckBox.setDisabled(True)
        self.mayaBinaryCheckBox.setChecked(True)

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.model.zenn_data.decode(outputPath, product_name='root')
                print decodingRule
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.asset.flag['TYPE'] = decodingRule['TYPE']
                self.ppRulebook.asset.flag['ASSET'] = decodingRule['ASSET']
                self.ppRulebook.flag['VER'] = decodingRule['VER']
                self.ppRulebook.flag['PUBDEV'] = decodingRule['PUBDEV']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.typeComboBox.setCurrentIndex(self.typeComboBox.findText(decodingRule['TYPE']))
                self.assetNameLineEdit.setText(decodingRule['ASSET'])
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])
                self.previewCheckBoxStateChanged(0)

                self.updatePreviewLabel()
            except Exception as e:
                self.ppRulebook.asset.flag['TYPE'] = 'char'
                print e.message
        else:
            self.ppRulebook.asset.flag['TYPE'] = 'char'

        self.setRulebookTask(self.ppRulebook.asset.model.zenn_data)


    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.updatePreviewLabel()

    def typeComboBoxIndexChanged(self, index):
        self.ppRulebook.asset.flag['TYPE'] = self.typeComboBox.currentText()
        self.updatePreviewLabel()

    def assetNameTextChanged(self, text):
        self.ppRulebook.asset.flag['ASSET'] = self.assetNameLineEdit.text()
        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + self.versionLineEdit.text()
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        self.setPreviewPublishPath(self.ppRulebook.asset.model.zenn_data.product['root'])

class ZennShotClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('Zenn Scene by Shot Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        showLabel = QtWidgets.QLabel(self.defaultGroupBox)
        showLabel.setText("SEQ")
        showLabel.setGeometry(QtCore.QRect(165, 30, 40, 40))

        self.typeComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.typeComboBox.setGeometry(QtCore.QRect(205, 35, 100, 30))
        self.typeComboBox.currentIndexChanged.connect(self.typeComboBoxIndexChanged)

        assetNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        assetNameLabel.setText("SHOT NUM")
        assetNameLabel.setGeometry(QtCore.QRect(315, 30, 90, 40))

        self.assetNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.assetNameLineEdit.setGeometry(QtCore.QRect(395, 35, 100, 30))
        self.assetNameLineEdit.textChanged.connect(self.assetNameTextChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(505, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(560, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        self.alembicCheckBox.setChecked(False)
        self.alembicCheckBox.setDisabled(True)
        self.texAlembicCheckBox.setChecked(False)
        self.texAlembicCheckBox.setDisabled(True)
        self.mayaBinaryCheckBox.setChecked(True)

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.model.zenn_scene_data.decode(outputPath, product_name='root')
                print decodingRule
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.flag['SEQUENCE'] = decodingRule['SEQUENCE']
                self.ppRulebook.flag['SHOT'] = decodingRule['SHOT']
                self.ppRulebook.flag['VER'] = decodingRule['VER']
                self.ppRulebook.flag['PUBDEV'] = decodingRule['PUBDEV']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.typeComboBox.setCurrentIndex(self.typeComboBox.findText(decodingRule['SEQUENCE']))
                self.assetNameLineEdit.setText(decodingRule['SHOT'])
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])
                self.previewCheckBoxStateChanged(0)

                self.updatePreviewLabel()
            except Exception as e:
                print e.message
        else:
            pass

        self.setRulebookTask(self.ppRulebook.asset.model.zenn_scene_data)


    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.typeComboBox.addItems(os.listdir(os.path.join('/show', self.showComboBox.currentText(), 'shot')))
        self.updatePreviewLabel()

    def typeComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['SEQUENCE'] = self.typeComboBox.currentText()
        self.updatePreviewLabel()

    def assetNameTextChanged(self, text):
        self.ppRulebook.flag['SHOT'] = self.typeComboBox.currentText() + "_" + self.assetNameLineEdit.text()
        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + self.versionLineEdit.text()
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        self.setPreviewPublishPath(self.ppRulebook.asset.model.zenn_scene_data.product['root'])


class ShotClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('Shot Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        showLabel = QtWidgets.QLabel(self.defaultGroupBox)
        showLabel.setText("Seq")
        showLabel.setGeometry(QtCore.QRect(165, 30, 40, 40))

        self.seqComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.seqComboBox.setGeometry(QtCore.QRect(205, 35, 100, 30))
        self.seqComboBox.currentIndexChanged.connect(self.seqComboBoxIndexChanged)

        seqLabel = QtWidgets.QLabel(self.defaultGroupBox)
        seqLabel.setText("Shot N")
        seqLabel.setGeometry(QtCore.QRect(315, 30, 90, 40))

        self.shotComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.shotComboBox.setGeometry(QtCore.QRect(395, 35, 100, 30))
        self.shotComboBox.currentIndexChanged.connect(self.shotComboBoxIndexChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(505, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(560, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.shot.decode(outputPath, product_name='root')
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.flag['SHOT'] = decodingRule['SHOT']
                self.ppRulebook.flag['SEQUENCE'] = decodingRule['SEQUENCE']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.seqComboBox.setCurrentIndex(self.seqComboBox.findText(decodingRule['SEQUENCE']))
                self.shotComboBox.setCurrentIndex(self.shotComboBox.findText(decodingRule['SHOT']))
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])

                self.updatePreviewLabel()
            except Exception as e:
                print e.message

        self.setRulebookTask(self.ppRulebook.asset.shot)

    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.seqComboBox.clear()
        for directory in os.listdir('/show/{0}/shot'.format(self.showComboBox.currentText())):
            if not directory.startswith('.'):
                self.seqComboBox.addItem(directory)
        self.updatePreviewLabel()

    def seqComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['SEQUENCE'] = self.seqComboBox.currentText()
        self.shotComboBox.clear()
        for directory in sorted(os.listdir('/show/{0}/shot/{1}'.format(self.showComboBox.currentText(), self.seqComboBox.currentText()))):
            if not directory.startswith('.'):
                self.shotComboBox.addItem(directory)
        self.updatePreviewLabel()

    def shotComboBoxIndexChanged(self, index):
        if not self.shotComboBox.currentText():
            return
        self.ppRulebook.flag['SHOT'] = self.shotComboBox.currentText()

        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + self.versionLineEdit.text()
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        self.setPreviewPublishPath(self.ppRulebook.asset.shot.product['root'])


class EnvDataClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('Env Data Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        assetNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        assetNameLabel.setText("AssetName")
        assetNameLabel.setGeometry(QtCore.QRect(165, 30, 90, 40))

        self.assetNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.assetNameLineEdit.setGeometry(QtCore.QRect(240, 35, 100, 30))
        self.assetNameLineEdit.textChanged.connect(self.assetNameTextChanged)

        dataNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        dataNameLabel.setText("dataName")
        dataNameLabel.setGeometry(QtCore.QRect(345, 30, 80, 40))

        self.dataNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.dataNameLineEdit.setGeometry(QtCore.QRect(410, 35, 100, 30))
        self.dataNameLineEdit.textChanged.connect(self.dataNameTextChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(515, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(565, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        dataName_Label = QtWidgets.QLabel(self.defaultGroupBox)
        dataName_Label.setText("dataName_?")
        dataName_Label.setGeometry(QtCore.QRect(345, 70, 100, 40))

        self.dataName_ComboBox = QtWidgets.QComboBox(self.defaultGroupBox)
        self.dataName_ComboBox.setGeometry(QtCore.QRect(430, 75, 100, 30))
        self.dataName_ComboBox.addItem(' ')
        self.dataName_ComboBox.addItems(string.uppercase)
        self.dataName_ComboBox.currentIndexChanged.connect(self.dataName_IndexChanged)

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.model.env_data.decode(outputPath, product_name='root')
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.asset.flag['TYPE'] = decodingRule['TYPE']
                self.ppRulebook.asset.flag['ASSET'] = decodingRule['ASSET']
                self.ppRulebook.flag['VER'] = decodingRule['VER']
                self.ppRulebook.asset.model.env_data.flag['DATANAME'] = decodingRule['DATANAME']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.assetNameLineEdit.setText(decodingRule['ASSET'])
                self.dataNameLineEdit.setText(decodingRule['DATANAME'])
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])
                self.dataName_ComboBox.setCurrentIndex(self.dataName_ComboBox.findText(str(decodingRule['DATANAME_'])[-1]))

                self.updatePreviewLabel()
            except Exception as e:
                print e.message
        self.ppRulebook.asset.flag['TYPE'] = 'env'

        self.setRulebookTask(self.ppRulebook.asset.model.env_data)

    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.updatePreviewLabel()

    def assetNameTextChanged(self, text):
        self.ppRulebook.asset.flag['ASSET'] = text
        self.updatePreviewLabel()

    def dataNameTextChanged(self, text):
        self.ppRulebook.asset.model.env_data.flag['DATANAME'] = text
        self.dataName_IndexChanged(self.dataName_ComboBox.currentIndex())
        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + text
        self.updatePreviewLabel()

    def dataName_IndexChanged(self, index):
        self.ppRulebook.asset.model.env_data.flag['DATANAME_'] = self.dataNameLineEdit.text() + self.dataName_ComboBox.currentText()
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        if self.dataName_ComboBox.currentText() == " ":
            path = '/show/{PROJECT}/asset/{TYPE}/{ASSET}/model/{PUBDEV}/data/abc/{DATANAME}/{DATANAME}_model{LOW}_{VER}'.format(
                PROJECT=self.ppRulebook.flag['PROJECT'],
                TYPE=self.ppRulebook.asset.flag['TYPE'],
                ASSET=self.ppRulebook.asset.flag['ASSET'],
                PUBDEV=self.ppRulebook.flag['PUBDEV'],
                DATANAME=self.ppRulebook.asset.model.env_data.flag['DATANAME'],
                LOW=self.ppRulebook.asset.flag['LOW'],
                VER=self.ppRulebook.flag['VER'])
            self.setPreviewPublishPath(path)
        else:
            self.setPreviewPublishPath(self.ppRulebook.asset.model.env_data.product['root'])


class ZEnvDataClass(BaseUIClass):
    def __init__(self, parent = None):
        BaseUIClass.__init__(self, child = self, parent = parent)

        self.setWindowTitle('ZEnv Data Publish')

        self.showComboBox.currentIndexChanged.connect(self.showComboBoxIndexChanged)

        assetNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        assetNameLabel.setText("AssetName")
        assetNameLabel.setGeometry(QtCore.QRect(165, 30, 90, 40))

        self.assetNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.assetNameLineEdit.setGeometry(QtCore.QRect(240, 35, 100, 30))
        self.assetNameLineEdit.textChanged.connect(self.assetNameTextChanged)

        dataNameLabel = QtWidgets.QLabel(self.defaultGroupBox)
        dataNameLabel.setText("Source N")
        dataNameLabel.setGeometry(QtCore.QRect(345, 30, 80, 40))

        self.dataNameLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.dataNameLineEdit.setGeometry(QtCore.QRect(410, 35, 100, 30))
        self.dataNameLineEdit.textChanged.connect(self.dataNameTextChanged)

        versionLabel = QtWidgets.QLabel(self.defaultGroupBox)
        versionLabel.setText("version")
        versionLabel.setGeometry(QtCore.QRect(515, 30, 70, 40))

        self.versionLineEdit = QtWidgets.QLineEdit(self.defaultGroupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(565, 35, 100, 30))
        self.versionLineEdit.setPlaceholderText("ex) 01")
        self.versionLineEdit.textChanged.connect(self.versionTextChanged)
        self.versionLineEdit.setValidator(QtGui.QIntValidator(0, 100, self))

        outputPath = cmds.file(q=True, sn=True).split('.mb')[0]

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        if outputPath:
            try:
                decodingRule = self.ppRulebook.asset.model.zenv_data.decode(outputPath, product_name='root')
                self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
                self.ppRulebook.asset.flag['TYPE'] = decodingRule['TYPE']
                self.ppRulebook.asset.flag['ASSET'] = decodingRule['ASSET']
                self.ppRulebook.flag['VER'] = decodingRule['VER']
                self.ppRulebook.asset.model.zenv_data.flag['SOURCE'] = decodingRule['SOURCE']

                self.showComboBox.setCurrentIndex(self.showComboBox.findText(decodingRule['PROJECT']))
                self.assetNameLineEdit.setText(decodingRule['ASSET'])
                self.dataNameLineEdit.setText(decodingRule['SOURCE'])
                self.versionLineEdit.setText(decodingRule['VER'].split('v')[1])

                self.updatePreviewLabel()
            except Exception as e:
                print e.message
        self.ppRulebook.asset.flag['TYPE'] = 'env'

        self.setRulebookTask(self.ppRulebook.asset.model.zenv_data)

    def showComboBoxIndexChanged(self, index):
        self.ppRulebook.flag['PROJECT'] = self.showComboBox.currentText()
        self.updatePreviewLabel()

    def assetNameTextChanged(self, text):
        self.ppRulebook.asset.flag['ASSET'] = text
        self.updatePreviewLabel()

    def dataNameTextChanged(self, text):
        self.ppRulebook.asset.model.zenv_data.flag['SOURCE'] = text
        self.updatePreviewLabel()

    def versionTextChanged(self, text):
        self.ppRulebook.flag['VER'] = 'v' + text
        self.updatePreviewLabel()

    def updatePreviewLabel(self):
        self.setPreviewPublishPath(self.ppRulebook.asset.model.zenv_data.product['root'])

def show_ui():
#    if not cmds.ls(sl=True):
#        msg = cmds.confirmDialog( title = 'Warning',
#                                  message = 'You must select an object.',
#                                  messageAlign = 'center',
#                                  icon = 'warning',
#                                  button = 'OK' )
#        return
    ExportMeshWin = SelectPubUI()
    ExportMeshWin.show()