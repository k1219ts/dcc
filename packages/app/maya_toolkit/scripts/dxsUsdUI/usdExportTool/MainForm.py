# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#
#   daeseok.chae rmantd
#
#	2018.04.02
#   2018.05.18 - subprocess style change. stderr, messagebox
#   2018.06.03 - command change. mayapy -> App mayapy
#   2019.06.14 - add $DEVELOPER_LOCATION for debug
#
#-------------------------------------------------------------------------------

import os, sys
import string
import site
import subprocess
import getpass
import datetime
import traceback
import pprint

# IP config
import dxConfig

DB_IP = dxConfig.getConf('DB_IP')

# Mongo DB
import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)

# QT
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

# USD EXPORT TOOL
import usdCommonSetup
from ui.usdExportUI import Ui_Form
from sceneGraphItem import sceneGraphItem
from sceneGraphItem import categoryItem

import json

ScriptRoot = os.path.dirname( os.path.abspath(__file__) )

# BATCHSCENESCRIPT = '/netapp/backstage/pub/bin/App mayapy -v {MAYAVER} --dev {SCRIPTPATH}/apps/Maya/toolkits/dxsUsd/BatchScene.py'
BATCHSCENESCRIPT = '/backstage/bin/DCC rez-env {REZ_RESOLVED} -- mayapy {SCRIPTPATH}/apps/Maya/toolkits/dxsUsd/BatchScene.py'
ROOTPATH = os.getenv('DEVELOPER_LOCATION')
if ROOTPATH and ROOTPATH in ScriptRoot:
    pass
else:
    ROOTPATH = '/backstage'
# BATCHSCENESCRIPT = '/netapp/backstage/pub/bin/App mayapy -v {MAYAVER} /netapp/backstage/pub/apps/Maya/toolkits/dxsUsd/BatchScene.py'

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.connections()
        self.styleSetting()
        self.defaultSetting()

        self.ui.machineType_comboBox.setCurrentIndex(0)
        self.ui.stepFrame_lineEdit.setText('1.0')

        # GET TASK LIST #
        self.getShowList()
        self.getTaskNames()
        self.getWorkCodes()
        try:
            self.loadUserSetup()
        except:
            pass

    def connections(self):
        self.ui.doExport_pushButton.clicked.connect(self.doExport)
        self.ui.findExportFile_pushButton.clicked.connect(self.selectExportFile)
        self.ui.findOutDir_pushButton.clicked.connect(self.selectExportDir)
        self.ui.findOutDir_pushButton.clicked.connect(self.selectExportFile)
        self.ui.openExportFile_pushButton.clicked.connect(self.openDirectory)
        self.ui.openOutDir_pushButton.clicked.connect(self.openDirectory)
        self.ui.exportFile_lineEdit.textChanged.connect(lambda :self.setSceneGraph(self.ui.exportFile_lineEdit.text().replace(".mb", ".json")))
        self.ui.only_zenn_checkBox.clicked.connect(lambda :self.setSceneGraph(self.ui.exportFile_lineEdit.text().replace(".mb", ".json")))
        self.ui.only_bake_checkBox.clicked.connect(lambda: self.setSceneGraph(self.ui.exportFile_lineEdit.text().replace(".mb", ".json")))

        # TASK LIST #
        self.ui.showName_comboBox.activated.connect(self.getTaskNames)
        self.ui.workType_comboBox.activated.connect(self.getTaskNames)
        self.ui.workName_comboBox.activated.connect(self.getWorkCodes)
        self.ui.workSubName_comboBox.activated.connect(self.getWorkSubName)
        self.ui.findShotNum_lineEdit.textChanged.connect(self.findWorkSubName)

    def styleSetting(self):
        imagePath = '%s/ui/pxr_usd_w.png'%ScriptRoot
        image = QtGui.QPixmap(imagePath).scaled(30, 30, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
        self.ui.titleLogo_label.setPixmap(image)
        imagePath_folder = '%s/ui/folder.png'%ScriptRoot
        self.ui.openExportFile_pushButton.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))
        self.ui.openOutDir_pushButton.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))
        comboBoxStyle = '''
                        QComboBox QAbstractItemView::item { min-height: 30px; min-width: 120px;}
                        background-color: rgb(90,90,90);
                        color : white;
                        '''
        self.ui.machineType_comboBox.setView(QtWidgets.QListView())
        self.ui.machineType_comboBox.setStyleSheet(comboBoxStyle)
        comboBoxStyle1= '''color: white;'''
        self.ui.showName_comboBox.setStyleSheet(comboBoxStyle1)
        self.ui.workName_comboBox.setStyleSheet(comboBoxStyle1)
        self.ui.workSubName_comboBox.setStyleSheet(comboBoxStyle1)
        self.ui.findShotNum_lineEdit.setStyleSheet(comboBoxStyle1)

    def defaultSetting(self):
        self.ui.userName_label.setText("user:" + getpass.getuser())
        self.ui.startFrame_lineEdit.setText('0')
        self.ui.endFrame_lineEdit.setText('0')
        self.ui.stepFrame_lineEdit.setText('0.0')

    def selectExportFile(self):
        dirtext = ""
        if self.sender().objectName() == "findExportFile_pushButton":
            dirtext = self.ui.exportFile_lineEdit.text()

        if os.path.exists(dirtext):
            dialog = FindFileDialog(self, "find export file", dirtext)
            result = dialog.exec_()
            if result == 1:
                path = dialog.selectedFiles()[-1]
                self.ui.exportFile_lineEdit.setText(path)
                jsonFile = path.replace(".mb", ".json")
                self.setSceneGraph(jsonFile)

    def setSceneGraph(self, jsonFile):
        # tree widget clear
        while self.ui.sceneGraphTreeWidget.topLevelItemCount() > 0:
            self.ui.sceneGraphTreeWidget.takeTopLevelItem(0)
            treeitem = self.ui.sceneGraphTreeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()

        if not os.path.exists(jsonFile) or not os.path.isfile(jsonFile):
            return

        splitPath = jsonFile.split('/')
        showIndex = splitPath.index("show")
        if showIndex == -1:
            return
        show = splitPath[showIndex + 1]
        seq = splitPath[showIndex + 3]
        shot = splitPath[showIndex + 4]

        isOnlyZenn = self.ui.only_zenn_checkBox.isChecked()
        camEnabled = True

        nameSpaceErrorList = []

        with open(jsonFile, "r") as f:
            sceneGraph = json.load(f)
            if sceneGraph.has_key('rezResolve'):
                self.rezResolveConfig = sceneGraph['rezResolve']
            else:
                self.rezResolveConfig = 'baselib-1.0 maya-2018 usd_maya-19.05 zelos-3.0.0708 tane-1.0.0702'

                # self.jsonFrame = sceneGraph['frameRange']
            if sceneGraph.has_key("task") and "hairSim" in sceneGraph['task']:
                camEnabled = False

            if not sceneGraph.has_key("mayaVersion"):
                self.mayaVersion = "2017"
            else:
                self.mayaVersion = sceneGraph["mayaVersion"]

            if len(sceneGraph['camera']) >= 1:
                camItem = categoryItem(self.ui.sceneGraphTreeWidget, 'cam')
                for node in sceneGraph['camera']:
                    sceneGraphItem(camItem, show, seq, shot, "cam", node, exportDisable = camEnabled)

            if len(sceneGraph["layout"]) >= 1:
                layItem = categoryItem(self.ui.sceneGraphTreeWidget, 'layout')
                # sceneGraphItem(layItem, sceneGraph['layout'])
                for node in sceneGraph['layout']:
                    # node[1] = 1     # set is always export.
                    sceneGraphItem(layItem, show, seq, shot, "set", node)

            if len(sceneGraph["geoCache"]) >= 1:
                geoItem = categoryItem(self.ui.sceneGraphTreeWidget, 'geo')
                # sceneGraphItem(geoItem, sceneGraph['geoCache'])
                for node in sceneGraph['geoCache']:
                    if len(node[0].split(":")) > 2:
                        nameSpaceErrorList.append(node[0])
                        continue
                    sceneGraphItem(geoItem, show, seq, shot, "ani", node, isOnlyZenn=isOnlyZenn, exportDisable = camEnabled)

            if sceneGraph.has_key('crowd') and len(sceneGraph["crowd"]) >= 1:
                geoItem = categoryItem(self.ui.sceneGraphTreeWidget, 'crowd')
                # sceneGraphItem(geoItem, sceneGraph['geoCache'])
                for node in sceneGraph['crowd']:
                    sceneGraphItem(geoItem, show, seq, shot, "crowd", node, isOnlyBake=self.ui.only_bake_checkBox.isChecked())

            # if len(sceneGraph["components"]) >= 1:
            #     geoItem = categoryItem(self.ui.sceneGraphTreeWidget, 'geo', camEnabled)
            #     # sceneGraphItem(geoItem, sceneGraph['geoCache'])
            #     for node in sceneGraph['components']:
            #         sceneGraphItem(geoItem, show, seq, shot, "ani", node, True, isOnlyZenn=False, exportDisable = camEnabled)

            if sceneGraph.has_key('sim') and len(sceneGraph["sim"]) >= 1:
                if sceneGraph.has_key('zenn') and len(sceneGraph["zenn"]) >= 1 and (camEnabled == False):
                    simItem = categoryItem(self.ui.sceneGraphTreeWidget, 'hairSim')
                    category = 'zenn'
                else:
                    simItem = categoryItem(self.ui.sceneGraphTreeWidget, 'sim')
                    category = 'sim'
                # sceneGraphItem(geoItem, sceneGraph['geoCache'])
                for node in sceneGraph['sim']:
                    if len(node[0].split(":")) > 2:
                        nameSpaceErrorList.append(node[0])
                        continue
                    sceneGraphItem(simItem, show, seq, shot, category, node, isOnlyZenn=isOnlyZenn)

            # if sceneGraph.has_key('hairSim') and len(sceneGraph["hairSim"]) >= 1:
            #     simItem = categoryItem(self.ui.sceneGraphTreeWidget, 'hairSim')
            #     # sceneGraphItem(geoItem, sceneGraph['geoCache'])
            #     for node in sceneGraph['hairSim']:
            #         sceneGraphItem(simItem, show, shot, "hairSim", node, isOnlyZenn=isOnlyZenn)

            if nameSpaceErrorList:
                print nameSpaceErrorList
                text = "\n".join(nameSpaceErrorList)
                text += "\ntoo many namespace, please clean up namespace"
                showfinishedDialog("info", text)

            self.ui.sceneGraphTreeWidget.expandAll()

        # outDir = '/show/%s_pub/shot/%s/%s' % (show, shot.split("_")[0], shot)
        # self.ui.outDir_lineEdit.setText(outDir)

        showDir= usdCommonSetup.GetShowDir(show)
        outDir = '{SHOWDIR}/shot/{SEQ}/{SHOT}'.format(SHOWDIR=showDir, SEQ=seq, SHOT=shot)
        self.ui.outDir_lineEdit.setText(outDir)


    def selectExportDir(self):
        dirtext = ""
        if self.sender().objectName() == "findOutDir_pushButton":
            dirtext = self.ui.outDir_lineEdit.text()

        if os.path.exists(dirtext):
            dialog = FindFileDialog(self, "find export directory", dirtext)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            result = dialog.exec_()
            if result == 1:
                path = dialog.selectedFiles()[-1]
                self.ui.outDir_lineEdit.setText(path)

    def openDirectory(self):
        dirtext = ""
        if self.sender().objectName() == "openExportFile_pushButton":
            if os.path.isdir(self.ui.exportFile_lineEdit.text()):
                dirtext = self.ui.exportFile_lineEdit.text()
            else:
                dirtext = os.path.dirname(self.ui.exportFile_lineEdit.text())
        elif self.sender().objectName() == "openOutDir_pushButton":
            dirtext = self.ui.outDir_lineEdit.text()

        if os.path.exists(dirtext):
            subprocess.Popen(['xdg-open', str(dirtext)])

    def sgItemToString(self, item):
        data = dict()
        # print '# item :', item.exportCheckBox.text()
        for index in range(item.childCount()):
            visible  = item.child(index).exportCheckBox.isChecked()
            nodeName = item.child(index).exportCheckBox.text()
            version  = item.child(index).versionLineEdit.text()
            # print '#\t', visible, nodeName, version
            if visible:
                if not data.has_key(version):
                    data[version] = list()
                data[version].append(nodeName)
        if data:
            opts = list()
            for v in data:
                opts.append('{VER}={NODES}'.format(VER=v, NODES=string.join(data[v], ',')))
            return string.join(opts, ';')

    def getExportOption(self):
        opts = ""
        srcfile = self.ui.exportFile_lineEdit.text()
        outdir = self.ui.outDir_lineEdit.text()
        start = self.ui.startFrame_lineEdit.text()
        end = self.ui.endFrame_lineEdit.text()
        step = self.ui.stepFrame_lineEdit.text()
        rigVerUp = self.ui.rigVersionUp_checkBox.isChecked()
        zenncache  = self.ui.zenn_checkBox.isChecked()
        onlyzenn = self.ui.only_zenn_checkBox.isChecked()
        onlyBake = self.ui.only_bake_checkBox.isChecked()
        dontParallelExport = self.ui.dontParallelCheckBox.isChecked()

        if srcfile:
            if outdir:
                opts += " --outdir %s" % outdir
            opts += " --mayaver %s" % self.mayaVersion

            for index in range(self.ui.sceneGraphTreeWidget.topLevelItemCount()):
                item = self.ui.sceneGraphTreeWidget.topLevelItem(index)

                # TODO: visible is checked from checkbox
                if item.exportCheckBox.isChecked(): # if visible true
                    if item.exportCheckBox.text() == "cam": # camera
                        # has camera
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --camera "%s"' % optstr

                    if item.exportCheckBox.text() == "layout": # layout
                        # has layout
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --layout "%s"' % optstr

                    if item.exportCheckBox.text() == "geo": # geoCache
                        # has geoCache
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --mesh "%s"' % optstr

                    if item.exportCheckBox.text() == "sim": # simulation
                        # has geoCache
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --sim "%s"' % optstr

                    if item.exportCheckBox.text() == "hairSim": # simulation
                        # has geoCache
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --hairSim "%s"' % optstr

                    if item.exportCheckBox.text() == "crowd": # simulation
                        # has crowd scenes
                        optstr = self.sgItemToString(item)
                        if optstr: opts += ' --crowd "%s"' % optstr

            # frame
            opts += " --fr %s %s" %(start, end)
            opts += " --step %s" % step
            if rigVerUp:
                opts += " --rigUpdate"
            if onlyzenn:
                opts += ' --onlyzenn'
            elif zenncache:
                opts += ' --zenn'
            if onlyBake:
                opts += ' --crowdbake'
            if dontParallelExport:
                opts += ' --serial'
        else:
            print 'No export file name. Export cancelled'

        return opts

    def doExport(self):
        optList = []
        opts = self.getExportOption()
        if not opts:
            return

        message_type = 'info'
        text = ""
        if self.ui.machineType_comboBox.currentText() == 'LOCAL':
            opts += " --host local"
            text  = " Export Completed "
            if ".mb" in os.path.splitext(self.ui.exportFile_lineEdit.text())[-1]:
                srcfile = self.ui.exportFile_lineEdit.text()
                exopts = opts + " --srcfile %s" % srcfile
                optList.append(exopts)

                errorText = ''
                for exopts in optList:
                    print '# Debug opts : %s' % exopts
                    command = BATCHSCENESCRIPT.format(REZ_RESOLVED=self.rezResolveConfig, SCRIPTPATH=ROOTPATH)
                    command+= exopts

                    # p = subprocess.Popen(command, shell=True)
                    # p.wait()

                    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    while p.poll() == None:
                        output = p.stdout.readline()
                        if output:
                            print output.strip()
                        if output.startswith("AssertionError:"):
                            errorText += output.replace("AssertionError: ", "")
                            message_type = 'critical'

            else:
                message_type = "critical"
                errorText = "Scene file check please "
                print "# Scene file check please."

        elif self.ui.machineType_comboBox.currentText() == 'TRACTOR':
            opts += " --host spool"
            text  = " Spool Completed "

            srcfile = self.ui.exportFile_lineEdit.text()
            exopts = opts + " --srcfile %s" % srcfile

            # command = BATCHSCENESCRIPT.format(MAYAVER=self.mayaVersion, SCRIPTPATH=ROOTPATH)
            command = BATCHSCENESCRIPT.format(REZ_RESOLVED=self.rezResolveConfig, SCRIPTPATH=ROOTPATH)
            command+= exopts
            print command
            p = subprocess.Popen(command, shell=True)
            p.wait()
            # print command

        if message_type == 'critical':
            text = errorText
        showfinishedDialog(message_type, text)

        self.saveUserSetup()

    '''
    ### USER SETUP
    '''
    def saveUserSetup(self):
        print 'save setup!'
        data = { 'user': getpass.getuser(),
                 'show': self.ui.showName_comboBox.currentIndex(),
                 'type': self.ui.workType_comboBox.currentIndex(),
                 'seq' : self.ui.workName_comboBox.currentIndex(),
                 'shot': self.ui.workSubName_comboBox.currentIndex(),
                 'time': datetime.datetime.now().isoformat()
                 }

        db = client['PUBLISH']
        coll = db['usdExport_user']
        coll.insert(data)

    def loadUserSetup(self):
        print 'load setup!'
        username = getpass.getuser()
        db = client['PUBLISH']
        coll = db['usdExport_user']
        result = coll.find({'user':username}).sort( 'time', pymongo.DESCENDING ).limit(1)

        if result.count() > 0:
            data = result[0]
            self.ui.showName_comboBox.setCurrentIndex( int(data['show']) )
            self.ui.workType_comboBox.setCurrentIndex( int(data['type']) )
            self.getTaskNames()
            self.ui.workName_comboBox.setCurrentIndex( int(data['seq']) )
            self.getWorkCodes()
            self.getWorkSubName()
            self.ui.workSubName_comboBox.setCurrentIndex( int(data['shot']) )
            self.getWorkSubName()

    '''
    ### SHOW LIST ###
    '''
    def getShowList(self):
        titleList =[]
        self.ui.showName_comboBox.clear()
        for i in os.listdir('/show'):
            if not i.startswith('.') and not "_pub" in i:
                titleList.append(i.upper())

        self.ui.showName_comboBox.addItems( sorted(titleList) )

    def getCurrentPath(self):
        showName = workType = workName = workSubName = ""
        showName = self.ui.showName_comboBox.currentText().lower()
        workType = self.ui.workType_comboBox.currentText()
        workName = self.ui.workName_comboBox.currentText()
        workSubName = self.ui.workSubName_comboBox.currentText()
        path =  os.path.join( '/show/', showName, workType, workName, workSubName)
        return path

    def getFolders(self, path):
        folders = []
        if os.path.exists(path):
            for i in os.listdir( path ):
                if not i.startswith('.'):
                    folders.append(i)
        return folders

    def getTaskNames(self):
        self.ui.workName_comboBox.clear()
        self.ui.workSubName_comboBox.clear()
        path = self.getCurrentPath()
        self.ui.exportFile_lineEdit.setText(path)
        taskList = self.getFolders(path)
        self.ui.workName_comboBox.addItems( sorted(taskList) )

    def getWorkCodes(self):
        self.ui.workSubName_comboBox.clear()
        path = self.getCurrentPath()
        self.ui.exportFile_lineEdit.setText(path)
        workCodes = self.getFolders(path)
        self.ui.workSubName_comboBox.addItems( sorted(workCodes) )

    def getWorkSubName(self):
        path = self.getCurrentPath()
        self.ui.exportFile_lineEdit.setText(path)

    def findWorkSubName(self):
        text = self.ui.findShotNum_lineEdit.text()
        index = self.ui.workSubName_comboBox.findText(text, QtCore.Qt.MatchContains)
        self.ui.workSubName_comboBox.setCurrentIndex(index)

    def closeEvent(self, event):
        self.saveUserSetup()
        self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

### DIALOGUE ###
def showfinishedDialog(type='info', text='Export Completed'):
    dialog = QtWidgets.QMessageBox()
    dialog.setStyleSheet('background-color: rgb(70, 70, 70); color: white;')
    if type == 'critical':
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
    else:
        dialog.setIcon(QtWidgets.QMessageBox.Information)
    dialog.setText(text)
    dialog.setWindowTitle('MESSAGE')
    dialog.addButton('OK',QtWidgets.QMessageBox.YesRole)
    dialog.move(QtWidgets.QDesktopWidget().availableGeometry().center())
    dialog.show()
    dialog.exec_()

class FindFileDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, windowName='', dirPath='' ):
        QtWidgets.QFileDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setDirectory(dirPath)
        self.setMinimumSize(1200, 800)


class FindDirectoryDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, windowName='', dirPath='' ):
        QtWidgets.QFileDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setDirectory(dirPath)
        self.setMinimumSize(1200, 800)
