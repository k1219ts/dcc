# encoding=utf-8
# !/usr/bin/env python

# -------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	SceneGraph Nodes UI
#
#	2017.03.1	$4
# -------------------------------------------------------------------------------

import os, sys
import re
import time
import json

import maya.cmds as cmds
import maya.mel as mel

import sgComponent
import sgAssembly
import sgAlembic
import sgCommon
import sgZenn
# import sgZEnv
import dxCameraUI

# 2017.06.16 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
import pymongo
from pymongo import MongoClient
import datetime
import getpass
import requests

import dxConfig

from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"


def getPubVersion(show, task, data_type, asset_name = "", shot = ""):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print(show, task, data_type, asset_name)
    record = {'show': show,
              'task': task,
              'data_type': data_type}

    if asset_name != "":
        record['asset_name'] = asset_name

    if shot != "":
        record["shot"] = shot

    recentDoc = coll.find_one(record,
                              sort=[('version', pymongo.DESCENDING)])

    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1


_MODEMAP = {'meshmode': 0, 'gpumode': 1}
_DISPMAP = {'BoundingBox': 0, 'Render': 1, 'Mid': 2, 'Low': 3, 'Simulation': 4}
_WORLDMAP = {'None': 0, 'Baked': 1, 'Separate': 2}
_CON_MAP = {
    'dexter': {'nodes': ['place_CON', 'direction_CON', 'move_CON'],
               'attrs': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']},
    'toneplus': {'nodes': ['world_ctrl', 'global_ctrl', 'COG_ctrl'],
                 'attrs': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']}
}


# -------------------------------------------------------------------------------
class ComponentImport:
    def __init__(self, Files=None, World=True):
        # plugin setup
        plugins = ['AbcImport', 'backstageMenu', 'ZENNForMaya']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        self.m_files = Files
        self.m_mode = 1  # 0:mesh, 1:gpu
        self.m_display = 3  # 0:bbox, 1:render, 2:mid, 3:low, 4:sim
        self.m_world = World  # 0:none, 1: baked, 2: separate
        self.m_fitTime = True

    def doIt(self):
        startTime = time.time()

        files = self.getFiles()
        files.sort()

        createNodeList = list()
        for f in files:
            baseName = os.path.basename(f)
            splitVer = re.compile(r'_v\d+.abc').findall(baseName)
            if splitVer:
                nodeName = baseName.split(splitVer[0])[0]
            else:
                nodeName = baseName.split('.abc')[0]

            if not ":" in nodeName and not "/model/" in os.path.dirname(f):
                namespace = os.path.basename(os.path.dirname(f))
                nodeName = namespace + ":" + nodeName

            compoNode = cmds.createNode('dxComponent', n=nodeName)
            createNodeList.append(compoNode)
            cmds.setAttr('%s.abcFileName' % compoNode, f, type='string')
            cmds.setAttr('%s.mode' % compoNode, self.m_mode)
            cmds.setAttr('%s.display' % compoNode, self.m_display)

            if self.m_world:
                worldFile = None
                # old-style world file
                wf = f.replace('.abc', '.world')
                if os.path.exists(wf):
                    worldFile = wf
                # new-style alembic world file
                wf = f.replace('.abc', '.wrd')
                if os.path.exists(wf):
                    worldFile = wf
                if worldFile:
                    cmds.setAttr('%s.worldFileName' % compoNode, worldFile, type='string')

            cpClass = sgComponent.Archive(compoNode)
            if self.m_world == 1:
                cpClass.m_baked = True
            cpClass.doIt()

        return createNodeList

    def getFiles(self):
        src_files = list()

        if type(self.m_files).__name__ == 'list':
            for i in self.m_files:
                if i.find('.abc') > -1:
                    src_files.append(i)
        else:
            for i in os.listdir(self.m_files):
                if i.find('.abc') > -1:
                    src_files.append(os.path.join(self.m_files, i))

        if not src_files:
            return

        files = list()
        for i in src_files:
            new = i
            new = os.path.join( os.path.dirname(new), os.path.basename(new).replace('_mid', '') )
            new = os.path.join( os.path.dirname(new), os.path.basename(new).replace('_low', '') )
            new = os.path.join( os.path.dirname(new), os.path.basename(new).replace('_sim', '') )
            files.append(new)
        return list(set(files))


# -------------------------------------------------------------------------------
# backstageMenu Dialog
def importComponentFileDialog():
    fn = cmds.fileDialog2(fm=4,
                          ff='Alembic (*.abc)',
                          cap='Import Alembic Component (Select File)',
                          okc='import',
                          ocr='dxsgnImportComponent_UICreate',
                          oin='dxsgnImportComponent_UIInit',
                          ocm='dxsgnImportComponent_UICommit')
    if not fn:
        return

    mode = cmds.optionVar(q='dxCompoImportMode')
    disp = cmds.optionVar(q='dxCompoDisplayMode')
    wopt = cmds.optionVar(q='dxCompoWorldAnim')
    fit = cmds.optionVar(q='dxCompoFitTime')

    #	print(mode, disp, wopt, fit)

    ciClass = ComponentImport(Files=fn, World=_WORLDMAP[wopt])
    ciClass.m_mode = _MODEMAP[mode]
    ciClass.m_display = _DISPMAP[disp]
    ciClass.m_fitTime = fit
    ciClass.doIt()


def importComponentDirDialog():
    fn = cmds.fileDialog2(fm=3,
                          cap='Import Alembic Component (Select Directory)',
                          okc='import',
                          ocr='dxsgnImportComponent_UICreate',
                          oin='dxsgnImportComponent_UIInit',
                          ocm='dxsgnImportComponent_UICommit')
    if not fn:
        return

    mode = cmds.optionVar(q='dxCompoImportMode')
    disp = cmds.optionVar(q='dxCompoDisplayMode')
    wopt = cmds.optionVar(q='dxCompoWorldAnim')
    fit = cmds.optionVar(q='dxCompoFitTime')

    ciClass = ComponentImport(Files=fn[0], World=_WORLDMAP[wopt])
    ciClass.m_mode = _MODEMAP[mode]
    ciClass.m_display = _DISPMAP[disp]
    ciClass.m_fitTime = fit
    ciClass.doIt()


def importComponentJsonDialog():
    fn = cmds.fileDialog2(fm=1,
                          ff='JSON (*.json)',
                          cap='Import Alembic Component (Select JSON File)',
                          okc='import',
                          ocr='dxsgnImportComponent_UICreate',
                          oin='dxsgnImportComponent_UIInit',
                          ocm='dxsgnImportComponent_UICommit')
    if not fn:
        return

    mode = cmds.optionVar(q='dxCompoImportMode')
    disp = cmds.optionVar(q='dxCompoDisplayMode')
    wopt = cmds.optionVar(q='dxCompoWorldAnim')
    fit = cmds.optionVar(q='dxCompoFitTime')

    f = open(fn[0], 'r')
    data = json.load(f)
    f.close()

    if not data.has_key('AlembicCache'):
        return

    start = 0
    end = 0

    cache = data['AlembicCache']

    # time unit setup (fps)
    if cache.has_key('fps'):
        cmds.currentUnit(time=str(cache['fps']))

    abcfiles = list()
    camfile = None
    rendercams = list()
    layouts = list()
    start = cache['start']
    end = cache['end']

    if cache.has_key('mesh'):
        abcfiles += cache['mesh']
    if cache.has_key('maya_camera'):
        camfile = cache['maya_camera']
    if cache.has_key('abc_camera'):
        camfile = cache['abc_camera']
    if cache.has_key('render_camera'):
        rendercams += cache['render_camera']
    if cache.has_key('layout'):
        layouts += cache['layout']

    if abcfiles:
        ciClass = ComponentImport(Files=abcfiles, World=_WORLDMAP[wopt])
        ciClass.m_mode = _MODEMAP[mode]
        ciClass.m_display = _DISPMAP[disp]
        ciClass.doIt()

    if camfile:
        dxcam = cmds.createNode('dxCamera')
        dxCameraUI.import_cameraFile('%s.fileName' % dxcam, camfile)

    if rendercams:
        # all camera off
        for s in cmds.ls(type='camera'):
            cmds.setAttr('%s.renderable' % s, 0)
        for c in rendercams:
            cam = cmds.ls(c, r=True)
            if cam:
                cmds.setAttr('%s.renderable' % cam[0], 1)

    if layouts:
        for f in layouts:
            node = sgAssembly.importAssemblyFile(f)
            if node:
                if cmds.pluginInfo('backstageLight', l=True, q=True):
                    arcnode = cmds.createNode('dxAssemblyArchive', n='%s_Arc' % node)
                    cmds.addAttr(arcnode, ln='rman__torattr___postTransformScript', dt='string')
                    cmds.setAttr('%s.rman__torattr___postTransformScript' % arcnode, 'dxarc', type='string')
                    import lgtUI
                    lgtUI.pointsArchive_fileSetup('%s.fileName' % arcnode, f)
            if node == None:
                if not cmds.pluginInfo('ZMayaTools', l=True, q=True):
                    cmds.unloadPlugin( 'ZMayaTools' )
                rdbNode = cmds.createNode('ZAssemblyArchive')
                cmds.setAttr('%s.asbFilePath'%rdbNode, f, type='string')

    # fit time range
    if fit:
        if start != end and start > 0 and end > 0:
            cmds.playbackOptions(minTime=start)
            cmds.playbackOptions(maxTime=end)
            cmds.playbackOptions(animationStartTime=start)
            cmds.playbackOptions(animationEndTime=end)
            cmds.currentTime(float(start))

    cmds.select(cl=True)


# -------------------------------------------------------------------------------
#
#	dxAssembly
#
# -------------------------------------------------------------------------------
#	AETemplate
def assemblyImport(attr, fileName):
    if not fileName:
        return

    curNode = attr.split('.')[0]
    cmds.setAttr('%s.fileName' % curNode, fileName, type='string')


#    mode 	= cmds.getAttr( '%s.mode' % curNode )
#    display = cmds.getAttr( '%s.display' % curNode )
#
#    asbClass = sgAssembly.CacheImport( fileName )
#    asbClass.m_mode = mode
#    asbClass.m_display = display
#    rootNode = asbClass.doIt( curNode )
#    if rootNode:
#        return '%s.%s' % (rootNode, attr.split('.')[-1])

def assemblyReload(attr):
    curNode = cmds.ls(attr.split('.')[0], l=True)[0]
    mode = cmds.getAttr('%s.mode' % curNode)
    display = cmds.getAttr('%s.display' % curNode)
    for i in cmds.ls(curNode, dag=True, type='dxComponent'):
        cmds.setAttr('%s.mode' % i, mode)
        cmds.setAttr('%s.display' % i, display)
        cpClass = sgComponent.Archive(i)
        cpClass.doIt()
    cmds.select(curNode)

#	backstageMenu Dialog
def importAssemblyFileDialog():
    fn = cmds.fileDialog2(fm=4,
                          ff='Assembly (*.asb);; All(*.*)',
                          cap='Import Assembly (Select Files)',
                          okc='import')
    if not fn:
        return

    for f in fn:
        try:
            node = sgAssembly.importAssemblyFile(f)
            if node:
                if cmds.pluginInfo('backstageLight', l=True, q=True):
                    arcnode = cmds.createNode('dxAssemblyArchive', n='%s_Arc' % node)
                    cmds.addAttr(arcnode, ln='rman__torattr___postTransformScript', dt='string')
                    cmds.setAttr('%s.rman__torattr___postTransformScript' % arcnode, 'dxarc', type='string')
                    import lgtUI
                    lgtUI.pointsArchive_fileSetup('%s.fileName' % arcnode, f)
            if node == None:
                if not cmds.pluginInfo('ZMayaTools', l=True, q=True):
                    cmds.unloadPlugin( 'ZMayaTools' )
                rdbNode = cmds.createNode('ZAssemblyArchive')
                cmds.setAttr('%s.asbFilePath'%rdbNode, f, type='string')

        except Exception as e:
            print(e.message)

    cmds.select(cl=True)


class AssemblyPubUI(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)

        self.resize(530, 270)
        self.setWindowTitle("Export Assembly (Select Directory)")

        self.team = 'layout'
        try:
            # MEMBER QUERY
            query_members = {}
            query_members['api_key'] = API_KEY
            # query_members['code'] = 'minchul.kim' # lighting
            # query_members['code'] = 'yeojin.lee'  # asset
            # query_members['code'] = 'chihun.kim'    # layout
            # query_members['code'] = 'rmantd'  # layout
            # query_members['code'] = 'sanghun.kim'  # layout
            query_members['code'] = getpass.getuser()

            print(query_members['code'])


            info = requests.get("http://{0}/dexter/search/user.php".format(dxConfig.getConf("TACTIC_IP")),
                                params=query_members).json()

            self.team = (info['department'].split(' ')[0]).lower()

            if not (self.team == "asset" or self.team == 'lighting' or self.team == "layout"):
                self.team = 'asset'
        except:
            self.team = 'layout'

        groupBox = QtWidgets.QGroupBox(self)
        groupBox.setGeometry(QtCore.QRect(10, 10, 510, 105))
        groupBox.setTitle("publish Type")

        firstY = 25
        astShtWidget = QtWidgets.QWidget(groupBox)
        astShtWidget.setGeometry(QtCore.QRect(10, firstY, 310, 15))

        self.assetRadio = QtWidgets.QRadioButton(astShtWidget)
        self.assetRadio.setGeometry(QtCore.QRect(0, 0, 50, 15))
        self.assetRadio.setText("Asset")
        self.assetRadio.setChecked(True)
        self.assetRadio.clicked.connect(self.clickedRadioButton)

        self.shotRadio = QtWidgets.QRadioButton(astShtWidget)
        self.shotRadio.setGeometry(QtCore.QRect(60, 0, 50, 15))
        self.shotRadio.setText("Shot")
        self.shotRadio.clicked.connect(self.clickedRadioButton)

        self.pubdev = "pub"

        pubdevWidget = QtWidgets.QWidget(groupBox)
        pubdevWidget.setGeometry(QtCore.QRect(460, firstY, 110, 15))

        self.devCheck = QtWidgets.QCheckBox(pubdevWidget)
        self.devCheck.setGeometry(QtCore.QRect(0, 0, 50, 15))
        self.devCheck.setText("dev?")
        self.devCheck.clicked.connect(self.clickedPubDevCheckBox)

        secondaryY = 50

        projectLabel = QtWidgets.QLabel(groupBox)
        projectLabel.setGeometry(QtCore.QRect(10, secondaryY, 40, 15))
        projectLabel.setStyleSheet("font:bold 13px")
        projectLabel.setText("show")

        self.projectComboBox = QtWidgets.QComboBox(groupBox)
        self.projectComboBox.setGeometry(QtCore.QRect(50, secondaryY - 2.5, 100, 20))
        self.projectComboBox.addItems(os.listdir('/show'))
        self.projectComboBox.currentIndexChanged.connect(self.updatePreview)

        self.assetLabel = QtWidgets.QLabel(groupBox)
        self.assetLabel.setGeometry(QtCore.QRect(160, secondaryY, 40, 15))
        self.assetLabel.setStyleSheet("font:bold 13px")
        self.assetLabel.setText("asset")

        self.assetLineEdit = QtWidgets.QLineEdit(groupBox)
        self.assetLineEdit.setGeometry(QtCore.QRect(200, secondaryY - 2.5, 100, 20))
        self.assetLineEdit.textChanged.connect(self.updatePreview)

        self.dataLabel = QtWidgets.QLabel(groupBox)
        self.dataLabel.setGeometry(QtCore.QRect(305, secondaryY, 40, 15))
        self.dataLabel.setStyleSheet("font:bold 13px")
        self.dataLabel.setText("type")

        self.dataLineEdit = QtWidgets.QLineEdit(groupBox)
        self.dataLineEdit.setGeometry(QtCore.QRect(335, secondaryY - 2.5, 65, 20))
        self.dataLineEdit.textChanged.connect(self.updatePreview)

        interval = 100

        self.versionLabel = QtWidgets.QLabel(groupBox)
        self.versionLabel.setGeometry(QtCore.QRect(305 + interval, secondaryY, 40, 15))
        self.versionLabel.setStyleSheet("font:bold 13px")
        self.versionLabel.setText("ver")

        self.versionLineEdit = QtWidgets.QLineEdit(groupBox)
        self.versionLineEdit.setGeometry(QtCore.QRect(330 + interval, secondaryY - 2.5, 70, 20))
        self.versionLineEdit.setPlaceholderText("01")
        intValidator = QtGui.QIntValidator(1, 99, self)
        self.versionLineEdit.setValidator(intValidator)
        self.versionLineEdit.setMaxLength(2)
        self.versionLineEdit.textChanged.connect(self.updatePreview)

        self.previewLabel = QtWidgets.QLabel(groupBox)
        self.previewLabel.setGeometry(QtCore.QRect(10, secondaryY + 25, 100, 20))
        self.previewLabel.setText("output path")

        self.previewLineEdit = QtWidgets.QLineEdit(groupBox)
        self.previewLineEdit.setGeometry(QtCore.QRect(70, secondaryY + 25, 400, 20))
        # self.previewLineEdit.setReadOnly(True)

        self.customDirPath = QtWidgets.QPushButton(groupBox)
        self.customDirPath.setGeometry(QtCore.QRect(480, secondaryY + 25, 20, 20))
        self.customDirPath.clicked.connect(self.selectDirPath)
        imagePath_folder = '/dexter/Cache_DATA/RND/jeongmin/CacheExport_jm/resources/folder.png'
        self.customDirPath.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))

        ##########################################################################################

        frameGroupBox = QtWidgets.QGroupBox(self)
        frameGroupBox.setGeometry(QtCore.QRect(10, 125, 510, 70))
        frameGroupBox.setTitle("Frame Range")

        self.currentFrameRadio = QtWidgets.QRadioButton(frameGroupBox)
        self.currentFrameRadio.setGeometry(QtCore.QRect(10, firstY, 100, 15))
        self.currentFrameRadio.setText("Current Frame")
        self.currentFrameRadio.clicked.connect(self.clickedOptionRadiobutton)
        self.currentFrameRadio.setChecked(True)
        threeY = 50

        self.frameRangeRadio = QtWidgets.QRadioButton(frameGroupBox)
        self.frameRangeRadio.setGeometry(QtCore.QRect(10, threeY, 100, 15))
        self.frameRangeRadio.setChecked(mel.eval('optionVar -query "dxExportFrameMode";') == "sgnrangetime")
        self.frameRangeRadio.setText("Start/End Frame")
        self.frameRangeRadio.clicked.connect(self.clickedOptionRadiobutton)

        self.minFrameLineEdit = QtWidgets.QLineEdit(frameGroupBox)
        self.minFrameLineEdit.setGeometry(QtCore.QRect(115, threeY - 5, 60, 20))
        self.minFrameLineEdit.setText(str(int(mel.eval('playbackOptions -q -min;'))))
        doubleValidator = QtGui.QIntValidator(self)
        self.minFrameLineEdit.setValidator(doubleValidator)
        self.minFrameLineEdit.setMaxLength(4)

        self.maxFrameLineEdit = QtWidgets.QLineEdit(frameGroupBox)
        self.maxFrameLineEdit.setGeometry(QtCore.QRect(180, threeY - 5, 60, 20))
        self.maxFrameLineEdit.setText(str(int(mel.eval('playbackOptions -q -max;'))))
        self.maxFrameLineEdit.setValidator(doubleValidator)
        self.maxFrameLineEdit.setMaxLength(4)

        self.publishBtn = QtWidgets.QPushButton(self)
        self.publishBtn.setGeometry(QtCore.QRect(215, 210, 100, 40))
        self.publishBtn.setText("Publish")
        self.publishBtn.clicked.connect(self.publishClick)

        workspacePath = cmds.workspace(q=True, rd=True)

        if workspacePath.startswith('/netapp/dexter/show'):
            workspacePath = workspacePath.replace('/netapp/dexter/show', '/show')

        if "/show" in workspacePath:
            try:
                index = self.projectComboBox.findText(workspacePath.split('/')[2])
                self.projectComboBox.setCurrentIndex(index)
                if workspacePath.split('/')[3] == 'asset':
                    self.assetRadio.setChecked(True)
                else:
                    self.shotRadio.setChecked(True)
                self.clickedRadioButton()
                self.assetLineEdit.setText(workspacePath.split('/')[5])
            except:
                pass

        self.clickedOptionRadiobutton()
        self.updatePreview()

        self.process = False

    def selectDirPath(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setMinimumSize(800, 400)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        result = dialog.exec_()
        if result == 1:
            print(dialog.selectedFiles())
            dirPath = dialog.selectedFiles()[-1]
            basePath = os.path.basename(dialog.selectedFiles()[-1]) + '.json'
            self.previewLineEdit.setText(os.path.join(dirPath, basePath))
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.red)
            self.previewLineEdit.setPalette(palette)
            self.isWriteDB = False

    def clickedRadioButton(self):
        if self.assetRadio.isChecked() == True:
            self.assetLabel.setText("asset")
        elif self.shotRadio.isChecked() == True:
            self.assetLabel.setText("shot")

        self.updatePreview()

    def clickedPubDevCheckBox(self):
        if self.devCheck.isChecked() == True:
            self.pubdev = "dev"
        else:
            self.pubdev = "pub"

        self.updatePreview()

    def clickedOptionRadiobutton(self):
        self.minFrameLineEdit.setEnabled(not self.currentFrameRadio.isChecked())
        self.maxFrameLineEdit.setEnabled(not self.currentFrameRadio.isChecked())

    def publishClick(self):
        print("publishClick")

        dataType = "model"
        if self.team == 'lighting':
            dataType = 'lighting'
        elif self.team == "layout":
            dataType = 'layout'

        print(self.previewLineEdit.text().split('/%s' % dataType)[0])

        if self.isWriteDB == False and self.previewLineEdit.text():
            result = cmds.confirmDialog(title='Warning!',
                               message="information is not saved\nDo you want to proceed?\nOK : Save information\nCancel : Don't Save information",
                               icon='warning',
                               button=['OK', 'CANCEL'])
            if result == "OK":
                self.isWriteDB = True
        else:
            if not os.path.exists(self.previewLineEdit.text().split('/%s' % dataType)[0]):
                # cmds.confirmDialog(title='Warning!',
                #                    message="don't search directory.\ncheck publish path please",
                #                    icon='warning',
                #                    button=['OK'])
                # return
                os.makedirs(self.previewLineEdit.text().split('/%s' % dataType)[0])

        self.process = True
        self.close()

    def updatePreview(self, value=None):
        filePath = "/show/{0}/{1}/".format(self.projectComboBox.currentText(), self.assetLabel.text())
        if self.assetLabel.text() == "asset":
            filePath += "env/{0}/".format(self.assetLineEdit.text())
        else:
            shotSplit = self.assetLineEdit.text().split('_')
            filePath += "{0}/{1}/".format(shotSplit[0], self.assetLineEdit.text())

        if self.team == "asset":
            filePath += "model/{0}/envlayout/".format(self.pubdev)
        elif self.team == "lighting":
            filePath += "lighting/{0}/Assembly/".format(self.pubdev)
        elif self.team == "layout":
            filePath += "layout/{0}/data/".format(self.pubdev)

        if self.dataLineEdit.text() == "":
            filePath += "//"
        else:
            filePath += "{0}/".format(self.dataLineEdit.text())

        fileName = "{0}_assembly_v{1}.json".format(self.assetLineEdit.text(), self.versionLineEdit.text())

        print(filePath + fileName)

        self.previewLineEdit.setText(filePath + fileName)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        self.previewLineEdit.setPalette(palette)
        self.isWriteDB = True

def exportAssemblyFileDialog():
    # fn = cmds.fileDialog2( fm=0,
    #                        cap='ZENV and LAYOUT export to Assembly',
    #                        okc='export',
    #                        fileFilter='JSON (*.json)',
    #                        ocr='dxsgnExportFrame_UICreate',
    #                        oin='dxsgnExportFrame_UIInit',
    #                        ocm='dxsgnExportFrame_UICommit' )

    assemblyUI = AssemblyPubUI()
    assemblyUI.exec_()

    if assemblyUI.process == False:
        return

    fn = assemblyUI.previewLineEdit.text()

    if (assemblyUI.currentFrameRadio.isChecked() == True):
        mode = 'sgncurrenttime'
    else:
        mode = 'sgnrangetime'

    start = int(assemblyUI.minFrameLineEdit.text()) - 1
    end = int(assemblyUI.maxFrameLineEdit.text()) + 1

    if not fn:
        return

    # mode  = cmds.optionVar( q='dxExportFrameMode' )
    # start = int( cmds.optionVar(q='dxExportFrameStart') ) -1
    # end   = int( cmds.optionVar(q='dxExportFrameEnd') ) +1

    current = int(cmds.currentTime(q=True))

    files = None

    # asbClass = sgAssembly.AssemblyExport(fn[0], 1, 1)
    # asbClass.doIt()

    if mode == 'sgncurrenttime':
        asbClass = sgAssembly.AssemblyExport(fn, current, current)
        asbFiles, abcFiles = asbClass.doIt()
    else:
        asbClass = sgAssembly.AssemblyExport(fn, start, end)
        asbFiles, abcFiles = asbClass.doIt()

    if asbFiles == None and abcFiles == None:
        return

    # insert db
    # ppRulebook = rulebook.Coder()
    PROJECT = ""
    ASSETNAME = ""
    outputPath = ""
    try:
        if "dev" in fn or assemblyUI.isWriteDB == False:
            return

        # MEMBER QUERY
        query_members = {}
        query_members['api_key'] = API_KEY
        # query_members['code'] = 'minchul.kim'
        # query_members['code'] = 'dayoung.lee'
        query_members['code'] = getpass.getuser()

        if getpass.getuser() == "rmantd":
            query_members['code'] = 'sanghun.kim'

        info = requests.get("http://{0}/dexter/search/user.php".format(dxConfig.getConf("TACTIC_IP")),
                            params=query_members).json()

        team = (info['department'].split(' ')[0]).lower()

        print(team)

        # write log in db [ 2017.04.19 by daeseok.chae ]
        # ppRulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_publish.yaml")

        outputPath = os.path.dirname(fn)

        if outputPath.startswith('/netapp/dexter/show'):
            outputPath = outputPath.replace('/netapp/dexter/show', '/show')

        productName = 'root'

        if 'shot' in outputPath:
            productName = 'shot'

        splitOutpuPath = outputPath.split('/')

        # decodingRule = ppRulebook._child[team].env_assembly.decode(outputPath, product_name=productName)

        PROJECT = splitOutpuPath[2]

        if productName == 'shot':
            SHOT = splitOutpuPath[5]
            SEQUENCE = splitOutpuPath[4]
        else:
            TYPE = splitOutpuPath[4]
            ASSETNAME = splitOutpuPath[5]

        task = team
        dataType = "assembly"

        pubFiles = {}

        pubFiles['json_file'] = [fn]

        for file in asbFiles:
            if not pubFiles.has_key("assembly"):
                pubFiles['assembly'] = []
            pubFiles['assembly'].append(file)
        for file in abcFiles:
            if not pubFiles.has_key("file"):
                pubFiles['file'] = []
            pubFiles['file'].append(file)

        record = {"show": PROJECT,
                  "data_type": dataType,
                  "tags": tag_parser.run(outputPath),
                  "task": task,
                  "artist": getpass.getuser(),
                  "enabled": True,
                  "time": datetime.datetime.now().isoformat(),
                  "task_publish": {},
                  "files": pubFiles,
                  "maya_version": "maya2_2017"
                  }

        if splitOutpuPath[-2] != "envlayout":
            record['task_publish']['data_type'] = splitOutpuPath[-2]

        # record Shot
        if productName == 'shot':
            record['shot'] = SHOT
            record['version'] = getPubVersion(show=PROJECT,
                                              task=task,
                                              data_type=dataType,
                                              shot=SHOT)
        # record other
        else:
            record['version'] = getPubVersion(show=PROJECT,
                                              task=task,
                                              data_type=dataType,
                                              asset_name=ASSETNAME)

            record['asset_type'] = TYPE
            record['asset_name'] = ASSETNAME

        COLLNAME = PROJECT
        client = MongoClient(DBIP)
        database = client[DBNAME]
        dbColl = database[COLLNAME]

        dbColl.insert_one(record)

        print("success db write", record)

        cmds.confirmDialog(title='Success!',
                           message="Export Success",
                           icon='info',
                           button=['OK'])
        return

    except Exception as e:
        COLLNAME = "puberror"
        dbName = "test"
        client = MongoClient(DBIP)
        database = client[dbName]
        dbColl = database[COLLNAME]

        record = {"user": getpass.getuser(),
                  "errorMsg": str(e),
                  "time": datetime.datetime.now().isoformat(),
                  "project": PROJECT,
                  "asset": ASSETNAME,
                  "type": "assembly",
                  "outputpath": outputPath,
                  "maya_version": "maya2_2017"}

        dbColl.insert_one(record)

        cmds.confirmDialog(title='fail!',
                           message="db Write Fail\nlog checking please",
                           icon='warning',
                           button=['OK'])
        return


# -------------------------------------------------------------------------------
#
#	Alembic
#
# -------------------------------------------------------------------------------
#	Export
def exportAlembicFileDialog():
    fn = cmds.fileDialog2(fm=3,
                          cap='Export Alembic (Select Directory)',
                          okc='export',
                          ocr='dxsgnAbcExport_UICreate',
                          oin='dxsgnAbcExport_UIInit',
                          ocm='dxsgnAbcExport_UICommit')
    if not fn:
        return

    fmode = cmds.optionVar(q='dxExportFrameMode')
    fstart = cmds.optionVar(q='dxExportFrameStart')
    fend = cmds.optionVar(q='dxExportFrameEnd')
    fjust = cmds.optionVar(q='dxJustFrame')
    fstep = cmds.optionVar(q='dxFrameStep')
    selm = cmds.optionVar(q='dxSelectionMode')

    nodes = list()
    # print(fmode, fstart, fend, fjust, fstep, selm)
    if selm == 'sgnall':
        cmds.select(cl=True)
    else:
        nodes = cmds.ls(sl=True, type=['dxRig', 'dxComponent'])

    currentf = int(cmds.currentTime(q=True))
    if fmode == 'sgncurrenttime':
        start = currentf
        end = currentf
        step = 1.0
    else:
        start = int(fstart)
        end = int(fend)
        step = float(fstep)

    abcClass = sgAlembic.CacheExport(
        FilePath=fn[0], Nodes=nodes,
        Start=start, End=end, Step=step, Just=bool(fjust))
    abcClass.doIt()


#	Merge
def mergeAlembicFileDialog():
    fn = cmds.fileDialog2(fm=1,
                          ff='Alembic (*.abc)',
                          cap='Alembic Merge Import',
                          okc='import')
    if not fn:
        return

    mgClass = sgAlembic.CacheMerge(fn[0])


# -------------------------------------------------------------------------------
#
#	ZENN
#
# -------------------------------------------------------------------------------
def zennStrandsViewerDialog():
    fn = cmds.fileDialog2(fm=3,
                          cap='Select ZennCache Directory',
                          okc='import')
    if not fn:
        return
    zennGroup = sgZenn.zennStrandsViewer(fn[0])

    geoGroup = os.path.basename(fn[0])
    geoGroup += '_rig_GRP'
    if cmds.objExists(geoGroup):
        conGroup = cmds.listRelatives(geoGroup, p=True)
        if conGroup:
            cmds.parent(zennGroup, conGroup[0])
            cmds.setAttr('%s.t' % zennGroup, 0, 0, 0)
            cmds.setAttr('%s.r' % zennGroup, 0, 0, 0)
            cmds.setAttr('%s.s' % zennGroup, 1, 1, 1)
            # connect initScale
            if cmds.attributeQuery('initScale', n=conGroup[0], ex=True):
                for i in ['sx', 'sy', 'sz']:
                    cmds.connectAttr('%s.initScale' % conGroup[0], '%s.%s' % (zennGroup, i))
    cmds.select(cl=True)


# #-------------------------------------------------------------------------------
# #
# #	ZENV
# #
# #-------------------------------------------------------------------------------
# def exportZEnvToAssemblyFileDialog():
#     fn = cmds.fileDialog2( fm=2,
#                            cap='Export ZENV to Assembly (Select Directory)',
#                            okc='export' )
#     if not fn:
#         return
#
#     sgZEnv.exportAbc( fn[0] )



# -------------------------------------------------------------------------------
#
#	ZENN
#
# -------------------------------------------------------------------------------
def exportZennToAssemblyFileDialog():
    fn = cmds.fileDialog2(fm=2,
                          cap='Export ZENN to Assembly (Select Directory)',
                          okc='export',
                          ocr='dxsgnAbcExport_UICreate',
                          oin='dxsgnAbcExport_UIInit',
                          ocm='dxsgnAbcExport_UICommit')
    if not fn:
        return

    mode = cmds.optionVar(q='dxExportFrameMode')
    start = int(cmds.optionVar(q='dxExportFrameStart')) - 1
    end = int(cmds.optionVar(q='dxExportFrameEnd')) + 1
    step = float(cmds.optionVar(q='dxFrameStep'))
    selm = cmds.optionVar(q='dxSelectionMode')

    current = int(cmds.currentTime(q=True))

    if selm == 'sgnall':
        nodes = cmds.ls(type='ZN_FeatherInstance')
    else:
        nodes = cmds.ls(sl=True, type='ZN_FeatherInstance')

    if mode == 'sgncurrenttime':
        rstart = current
        rend = current
    else:
        rstart = start
        rend = end

    zclass = sgZenn.ZennPointsExport(fn[0], nodes)
    zclass.m_start = rstart
    zclass.m_end = rend
    zclass.m_step = step
    zclass.doIt()


# -------------------------------------------------------------------------------
#
#	WORLD ANIM
#
# -------------------------------------------------------------------------------
def worldAnimImportDialog():
    sel = cmds.ls(sl=1)
    if sel:
        sel = sel[0]
    else:
        sel = cmds.group(n='world_import', em=True, w=True)

    fn = cmds.fileDialog2(fm=1,
                          ff='Alembic (*.wrd *.abc) ',
                          cap='Import World Anim',
                          okc='import')
    if not fn:
        return
    sgCommon.import_worldAlembic(sel, None, fn[0])


def worldAnimExportDialog():
    sel = cmds.ls(sl=1)[0]
    attrList = _CON_MAP['dexter']['attrs']  # ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    conList = _CON_MAP['dexter']['nodes']  # ['place_CON', 'direction_CON', 'move_CON']

    if sel.find(':') > -1:
        outConList = sgAlembic.get_rigConObjects(sel, conList)
    else:
        nodes = cmds.listRelatives(sel, p=True, f=True)
        outConList = nodes[0].split('|')[1:]

    fn = cmds.fileDialog2(fm=3,
                          cap='Export World Anim',
                          okc='Export')

    if not fn:
        return

    outFile = os.path.join(fn[0], sel + '.wrd')
    start = cmds.playbackOptions(q=True, min=True)
    end = cmds.playbackOptions(q=True, max=True)

    sgCommon.export_worldAlembic(outConList, outConList[0], start, end, 1, outFile)
