# -*- coding: utf-8 -*-
from PySide2 import QtWidgets, QtCore

import sys, os, json, requests, pprint, platform
import nuke, nukescripts
import DXRulebook.Interface as rb
rb.Reload()

import pymongo
from pymongo import MongoClient
import getpass
from dxConfig import dxConfig

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'

from ui_cameraSearch import Ui_Form


def chkPlatform():
    # print('platform:', platform.system())
    return platform.system()

def resolvePath(path):
    if 'Darwin' == chkPlatform():
        if not os.path.isfile(path):
            path = '/opt/' + path
    elif 'Windows' == chkPlatform():
        if not os.path.isfile(path):
            path = 'M:' + path.replace('/show/', '/')
    # print('resolvePath:', path)
    return path


class CameraSearch(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CameraSearch, self).__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('Matchmove Search')

        self.widgetFont = self.font()
        self.widgetFont.setPointSize(12)
        self.setFont(self.widgetFont)

        self.API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
        param = {}
        param['api_key'] = self.API_KEY
        param['category'] = 'Active'
        param['status'] = 'in_progres'

        projects = requests.get("http://10.0.0.51/dexter/search/project.php",
                                params=param).json()

        self.ui.cameraTree.setColumnCount(5)
        self.ui.cameraTree.headerItem().setText(0, 'Camera File')
        self.ui.cameraTree.headerItem().setText(1, 'Type')
        self.ui.cameraTree.headerItem().setText(2, 'Team')
        self.ui.cameraTree.headerItem().setText(3, 'Version')
        self.ui.cameraTree.headerItem().setText(4, 'Over Scan')
        self.ui.cameraTree.headerItem().setText(5, 'Stereo')
        self.ui.cameraTree.headerItem().setText(6, 'Time')

        self.ui.cameraTree.header().resizeSection(0, 350)
        self.ui.cameraTree.header().resizeSection(1, 80)
        self.ui.cameraTree.header().resizeSection(2, 100)
        self.ui.cameraTree.header().resizeSection(3, 60)
        self.ui.cameraTree.header().resizeSection(6, 200)
        param['category'] = 'Active'

        self.ui.distortTree.setColumnCount(4)
        self.ui.distortTree.headerItem().setText(0, 'Distortion File')
        self.ui.distortTree.headerItem().setText(1, 'Version')
        self.ui.distortTree.headerItem().setText(2, 'Plate')
        self.ui.distortTree.headerItem().setText(3, 'Time')
        self.ui.distortTree.header().resizeSection(0, 350)
        self.ui.distortTree.header().resizeSection(1, 80)
        self.ui.distortTree.header().resizeSection(2, 100)
        self.ui.distortTree.header().resizeSection(3, 60)

        self.boolColor = {True: QtCore.Qt.green, False: QtCore.Qt.red,
                          '?':QtCore.Qt.black}

        path = nuke.root().name()
        if 'Root' not in path:
            coder = rb.Coder()
            arg = coder.D.SHOW.Decode(path)
            self.showName = arg.show
        else:
            self.showName = os.getenv('SHOW')

        self.titleDic = {}
        for i in sorted(projects, key=lambda k:k['title']):
            self.ui.prjComboBox.addItem(i['title'])
            self.titleDic[i['title']] = i

            lowTitle = i['title'].lower()
            if self.showName in lowTitle:
                self.ui.prjComboBox.setCurrentText(i['title'])
                index = self.ui.prjComboBox.currentIndex()
                self.setSeq(index)

        self.ui.prjComboBox.insertItem(0, 'None')
        self.titleDic['None'] = {'name':'None'}

        self.ui.prjComboBox.currentIndexChanged.connect(self.setSeq)
        self.ui.shotPushButton.clicked.connect(self.searchShot)
        self.ui.shotLineEdit.returnPressed.connect(self.searchShot)
        self.ui.buttonBox.accepted.connect(self.makeScene)
        self.ui.buttonBox.rejected.connect(self.close)

        # DEBUG
        # self.ui.prjComboBox.setCurrentIndex(0)
        # print ('showName : {0}'.format(self.showName))
        self.ui.prjComboBox.addItem(u'파이프라인 (pipe)')
        self.titleDic[u'파이프라인 (pipe)'] = {'name':'pipe'}
        # self.ui.seqComboBox.setCurrentIndex(1)
        # self.ui.shotLineEdit.setText('DZR_0820')

    def getShotPath(self):
        prjData = self.titleDic[self.ui.prjComboBox.currentText()]
        project = prjData['name']
        shotDir = resolvePath('/show/%s/_2d/shot' % project)
        if not os.path.isdir(shotDir):
            shotDir = resolvePath('/show/%s/shot' % project)
        return shotDir

    def getProject(self):
        prjData = self.titleDic[self.ui.prjComboBox.currentText()]
        project = prjData['name']
        return project

    def setSeq(self, index):
        self.ui.seqComboBox.clear()
        shotDir = self.getShotPath()
        if not os.path.isdir(shotDir):
            return
        seqs = sorted([i for i in os.listdir(shotDir) if (not(i.startswith('.')))])

        self.ui.seqComboBox.clear()
        self.ui.seqComboBox.addItems(seqs)

    def searchShot(self):
        project = self.getProject()
        seq = self.ui.seqComboBox.currentText()
        shot = self.ui.shotLineEdit.text()
        if not('_' in shot) and shot.isalnum():
            shot = '%s_%s' % (seq, str(shot))

        print(project, seq, shot)

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[project]
        camResult = coll.find({'data_type':'camera',
                               'show':project,
                               'shot':shot}).sort('version',pymongo.DESCENDING)

        distResult = coll.find({'data_type':'distortion',
                                'show':project,
                                'shot':shot}).sort('version',pymongo.DESCENDING)


        self.ui.cameraTree.clear()
        for i in camResult:
            #pprint.pprint(i)
            cameraPath = resolvePath(i['files']['camera_path'][0])
            camType = str(i['task_publish']['plateType'])
            team = str(i['task'])
            try:
                overScan = i['task_publish']['overscan']
            except:
                overScan = '?'
            version = str(i['version'])
            try:
                stereo = i['task_publish']['isStereo']
            except:
                stereo = '?'
            pubTime = i['time']

            item = CameraItem(self.ui.cameraTree)
            item.setText(0, os.path.basename(cameraPath))
            item.setText(1, camType)
            item.setText(2, team)
            item.setText(3, version)
            item.setText(4, str(overScan))
            item.setForeground(4, self.boolColor[overScan])
            item.setText(5, str(stereo))
            item.setForeground(5, self.boolColor[stereo])
            item.setText(6, pubTime.split('.')[0])
            item.setDBData(i)
        try:
            self.ui.cameraTree.setCurrentItem(self.ui.cameraTree.topLevelItem(0))
        except:
            pass

        self.ui.distortTree.clear()
        for i in distResult:
            #pprint.pprint(i)
            distPath = resolvePath(i['files']['path'][0])
            version = str(i['version'])
            pubTime = i['time']
            plate = i['task_publish']['plateType']

            item = DistortionItem(self.ui.distortTree)
            item.setText(0, os.path.basename(distPath))
            item.setText(1, version)
            item.setText(2, plate)
            item.setText(3, pubTime.split('.')[0])

            item.setDBData(i)
        try:
            self.ui.distortTree.setCurrentItem(self.ui.distortTree.topLevelItem(0))
        except:
            pass

    def addOverScanReformat(self, DB, scanlineNode, mmvFormat):
        if DB['task_publish'].get('overscan_value'):
            overscan_value = float(DB['task_publish']['overscan_value'])
        elif DB['task_publish'].get('overscanSize'):
            overscan_value = float(DB['task_publish']['overscanSize'])
        else:
            overscan_value = 1.08

        print('overscan:', overscan_value)

        # Reformat1 #
        #################################
        reformat1 = nuke.nodes.Reformat()
        reformat1['type'].setValue('to box')
        if DB['task_publish'].get('renderWidth'):
            reformat1['box_width'].setValue(int(DB['task_publish']['renderWidth']) * overscan_value)
        elif DB['task_publish'].get('resWidth'):
            reformat1['box_width'].setValue(int(DB['task_publish']['resWidth']) * overscan_value)
        else:
            reformat1['box_width'].setValue(int(DB['task_publish']['render_width']) * overscan_value)
        reformat1.setXYpos(scanlineNode.xpos(), scanlineNode.ypos() + 30)
        reformat1.setInput(0, scanlineNode)

        # Reformat2 #
        #################################
        reformat2 = nuke.nodes.Reformat()
        reformat2['type'].setValue('to format')
        reformat2['format'].setValue(mmvFormat)
        reformat2['resize'].setValue('none')
        reformat2['pbb'].setValue(True)
        reformat2.setInput(0, reformat1)
        reformat2.setXYpos(reformat1.xpos(), reformat1.ypos() + 30)
        bottomNode = reformat2

        return bottomNode

    def makeScene(self):
        camItem = self.ui.cameraTree.selectedItems()
        distItem = self.ui.distortTree.selectedItems()

        backdropList = []
        bottomNode = None
        reformatSwitch = False

        if camItem:
            camData = camItem[0].getDBData()
            mmvFormat = None
            try:
                mmvFormat = nuke.addFormat('%s %s' % (camData['task_publish']['render_width'],
                                                      camData['task_publish']['render_height']))
            except:
                if camData['task_publish'].get('renderWidth'):
                    mmvFormat = nuke.addFormat('%s %s' % (camData['task_publish']['renderWidth'],
                                                          camData['task_publish']['renderHeight']))
            if mmvFormat:
                nuke.root()['format'].setValue(mmvFormat)

            # 1 CAMERA
            cameraFile = resolvePath(camData['files']['camera_path'][0])
            cameraNode = nuke.createNode('Camera2', 'file {%s} read_from_file True' % cameraFile)
            backdropList.append(cameraNode)
            # 2 SCENE
            sceneNode = nuke.nodes.Scene()
            sceneNode.setXYpos(cameraNode.xpos()+150, cameraNode.ypos())
            backdropList.append(sceneNode)

            # 3 SCANLINE RENDER
            scanlineNode = nuke.nodes.ScanlineRender()
            backdropList.append(scanlineNode)
            scanlineNode.setInput(1, sceneNode)
            scanlineNode.setInput(2, cameraNode)

            scanlineNode.setXYpos(sceneNode.xpos(), sceneNode.ypos()+100)
            bottomNode = scanlineNode

            # 4 OVERSCAN REFORMAT
            try:
                if camData['task_publish']['overscan']:
                    bottomNode = self.addOverScanReformat(camData, scanlineNode, mmvFormat)
                    reformatSwitch = True
            except Exception as e:
                print(e)


            # 5 CHECKER BOARD
            checkerBoard = nuke.nodes.CheckerBoard2()
            if mmvFormat:
                checkerBoard['format'].setValue(mmvFormat)
            checkerBoard.setXYpos(sceneNode.xpos(), sceneNode.ypos()-200)
            backdropList.append(checkerBoard)

            # 6 READGEO
            geoKeyList = ['camera_geo_path', 'camera_asset_geo_path']
            readGeo = None

            for key in geoKeyList:
                if camData['files'].get(key):
                    for geoFile in camData['files'][key]:
                        readGeo = nuke.createNode('ReadGeo2', 'file {%s}' % resolvePath(geoFile))
                        readGeo['disable'].setValue(True)
                        readGeo.setInput(0, checkerBoard)
                        sceneNode.setInput(sceneNode.inputs(), readGeo)
                        readGeo.setXYpos(readGeo.xpos()+50, sceneNode.ypos() - 100)
                        backdropList.append(readGeo)

        if distItem:
            distData = distItem[0].getDBData()

            if not reformatSwitch and distData['task_publish']['overscanSize']:
                bottomNode = self.addOverScanReformat(distData, scanlineNode, mmvFormat)

            nukescripts.clear_selection_recursive()
            beforeDistor = nuke.allNodes()
            distorPath = resolvePath(distData['files']['path'][0])
            nuke.scriptReadFile(distorPath)
            afterDistor = nuke.allNodes()
            distorNodes = list(set(afterDistor) - set(beforeDistor))

            for index, distorNode in enumerate(distorNodes):
                # if 'Transform_FilmBack_' in distorNode.name():
                #     trNode = distorNode
                # elif distorNode.knob('output'):
                #     distorNode['output'].setValue('Redistort')
                if distorNode.knob('direction'):
                    distorNode['direction'].setValue('distort')
                try:
                    distorNode.setXYpos(scanlineNode.xpos() + (index * (-100)),
                                        scanlineNode.ypos() + 100)
                except:
                    pass

                backdropList.append(distorNode)

            distorNode.setInput(0, bottomNode)

        nukescripts.clear_selection_recursive()
        for i in backdropList:
            i['selected'].setValue(True)

        bdNode = nukescripts.autoBackdrop()
        bdNode['bdheight'].setValue(bdNode['bdheight'].value() + 50)
        bdNode['bdwidth'].setValue(bdNode['bdwidth'].value() + 150)
        if camItem:
            bdNode['label'].setValue(camItem[0].text(0))
        bdNode['note_font_size'].setValue(16)


class DBItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        self.dbData = None

    def setDBData(self, data):
        self.dbData = data

    def getDBData(self):
        return self.dbData

class CameraItem(DBItem):
    def __init__(self, parent=None):
        super(CameraItem, self).__init__(parent)

class DistortionItem(DBItem):
    def __init__(self, parent=None):
        super(DistortionItem, self).__init__(parent)


class FileItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)
        self.absPath = ''

    def setAbsPath(self, filepath):
        self.absPath = filepath

    def getAbsPath(self):
        return self.absPath



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ms = CameraSearch()
    ms.show()
    sys.exit(app.exec_())
