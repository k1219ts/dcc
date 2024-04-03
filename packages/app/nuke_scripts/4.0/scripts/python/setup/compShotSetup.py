# -*- coding: utf-8 -*-
import re, os, sys, json, getpass,requests
from imp import reload
from datetime import datetime
import DXRulebook.Interface as rb

import nuke, nukescripts

from PySide2 import QtWidgets, QtCore, QtGui
import pymongo
from pymongo import MongoClient
# from tactic_client_lib import TacticServerStub

from dxConfig import dxConfig

from python import nukeCommon as comm

from . import ui_compShotSetup
reload(ui_compShotSetup)

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'

API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
tactic_ip = dxConfig.getConf('TACTIC_IP')

taskServer = "http://%s/dexter/search/task.php" % (tactic_ip)
prjServer = "http://%s/dexter/search/project.php" % (tactic_ip)
seqServer = "http://%s/dexter/search/sequence.php" % (tactic_ip)
shotServer = "http://%s/dexter/search/shot.php" %(tactic_ip)

class CompShotSetup(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_compShotSetup.Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle("Comp Shot Setup")

        ##################################
        # Tactic Requests API Variable
        self.prjDict = self.getPrjItem()
        ##################################
        self.ui.ctx_comboBox.addItems(['comp','keying','remove','roto','retime','precomp','ocula',
                                       'lighting','censorship','fx','merge',u'직접 입력'])
        ###################################
        self.task = self.getUserTask()
        ########################################################################

        grpStyleSheet = """
        QGroupBox {
        margin-top: 10;
        border: 1px solid rgb(0,0,0);
        spacing: 1;
        }
        QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
        }
        """

        self.ui.setup_GRP.setStyleSheet(grpStyleSheet)
        self.ui.mmv_GRP.setStyleSheet(grpStyleSheet)
        self.ui.plate_GRP.setStyleSheet(grpStyleSheet)

        ##################
        ### SIGNALS ####

        # ### setting PRJ ###
        param = {}
        param['api_key'] = API_KEY

        projects0 = requests.get(prjServer,
                                params={'api_key' : API_KEY,
                                        'category' : 'Active'}).json()

        self.showName = os.getenv('SHOW')
        self.titleDic = {}
        for i in sorted(projects0, key=lambda k:k['title']):

            # self.ui.prj_comboBox.addItem(i['title'])
            self.titleDic[i['title']] = i
            lowTitle = i['title'].lower()
            reTitle = lowTitle.replace(')',"")
            splitPath = reTitle.split('(')
            splitTitle = splitPath[-1]

            if self.showName == splitTitle:
                self.ui.prj_comboBox.setCurrentText(i['title'])
                index = self.ui.prj_comboBox.currentIndex()
                self.getSeqItem(index)

        self.ui.prj_comboBox.currentIndexChanged.connect(self.getSeqItem)
        self.ui.ctx_comboBox.currentIndexChanged.connect(self.setCtx)

        self.ui.seq_listWidget.itemSelectionChanged.connect(self.getShotItem)

        self.ui.shot_listWidget.itemSelectionChanged.connect(self.getMatchMove)
        self.ui.shot_listWidget.itemSelectionChanged.connect(self.getPlate)

        self.ui.filter_exist_checkBox.stateChanged.connect(self.hideExistPath)
        self.ui.filter_job_checkBox.stateChanged.connect(self.filterMyJobs)
        self.ui.plateCheck.toggled.connect(self.checkallPlate)

        self.ui.cancel_button.clicked.connect(self.close)
        self.ui.ok_button.clicked.connect(self.scriptClear)
        self.ui.ok_button.clicked.connect(self.makeDir)
        self.ui.ok_button.clicked.connect(self.makeMatchmove)
        self.ui.ok_button.clicked.connect(self.makePlate)
        self.ui.ok_button.clicked.connect(self.makeDiscription)
        self.ui.ok_button.clicked.connect(self.projectSetting)
        self.ui.ok_button.clicked.connect(self.zoom)
        self.ui.ok_button.clicked.connect(self.close)

    def getUserTask(self):
        userTask = {}
        prjCode = []
        seq = []
        shot = []
        userName = getpass.getuser()
        userInfo = requests.get(taskServer, params = {'api_key'  : API_KEY,
                                                      'login'    : userName}).json()
        for i in userInfo:
            prjCode.append(i['project_code'])

            coder = rb.Coder()
            try:
                argv = coder.N.SHOTNAME.Decode(i['extra_code'])
                if argv.seq:
                    seq.append(argv.seq)
            except:
                seq.append(i['category_code'])

            shot.append(i['extra_code'])

        userTask['prjCode'] = set(prjCode)
        userTask['seq'] = set(tuple(seq))
        print (userTask['seq'])
        userTask['shot'] = set(shot)

        return userTask

    def getPrjItem(self):
        # self.ui.prj_comboBox.addItem('None')
        prjDict = {}
        prjInfo = requests.get(prjServer, params={'api_key'  : API_KEY,
                                                  'category' : 'Active'}).json()
        for i in prjInfo:
            code = i['code']  # show71
            name = i['name']  # asd01
            description = i['description']  # 아스달연대기

            if code != 'test' and code != 'testshot':
                self.ui.prj_comboBox.addItem(description)
                ##################################################
                prjDict[description] = code  # 아스달연대기 : show71
                prjDict[code] = name  # show71    : asd01

        return prjDict

    def getSeqItem(self, idx):
        ###############################
        # Clear Sequence List / Shot List
        self.ui.seq_listWidget.clear()
        self.ui.shot_listWidget.clear()
        ################################

        prjName = self.ui.prj_comboBox.itemText(idx)
        prjCode = self.prjDict[prjName]

        seqInfo = requests.get(seqServer, params={'api_key'      : API_KEY,
                                                  'project_code' : prjCode}).json()
        ################################################
        for seq in seqInfo:
            seqItem = ListItem(self.ui.seq_listWidget, seq)

            if self.ui.filter_job_checkBox.isChecked():
                if seq not in self.task['seq']:
                    seqItem.setFlagsDisable()

    def getShotItem(self):
        if self.ui.seq_listWidget.selectedItems():
            ###############################
            # Clear Shot List
            self.ui.shot_listWidget.clear()
            ###############################
            description = self.ui.prj_comboBox.currentText()
            prjCode = self.prjDict[description]  # show71
            # print (prjCode)
            prjDir = self.prjDict[prjCode]  # asd01
            seqName = self.ui.seq_listWidget.selectedItems()[0].text()

            #######################################
            coder = rb.Coder()
            argv = {'show': prjDir,
                    'pub': '_2d',
                    'seq': seqName}
            shotPath = coder.D.Encode(**argv)
            for shot in sorted(os.listdir(shotPath)):
                if '_' in shot:
                    shotItem = ShotListItem(self.ui.shot_listWidget, shot)
                    shotItem.setExistsPath(prjDir, seqName, shot)

                    # Hide Exist Path
                    if self.ui.filter_exist_checkBox.isChecked():
                        if shotItem.isExists == True:
                            shotItem.setHidden(True)
                    else:
                        if shotItem.isExists == True:
                            shotItem.setHidden(False)

                    if self.ui.filter_job_checkBox.isChecked():
                        if shot not in self.task['shot']:
                            shotItem.setFlagsDisable()
                else:
                    pass

            ShotAddButton(self.ui.shot_listWidget, seqName)

        else:
            self.ui.shot_listWidget.clear()

    def setCtx(self, idx):
        ctx = self.ui.ctx_comboBox.itemText(idx)
        if ctx == u'직접 입력':
            self.ui.ctx_lineEdit.setEnabled(True)
        else:
            self.ui.ctx_lineEdit.clear()
            self.ui.ctx_lineEdit.setEnabled(False)

    def getMatchMove(self):
        if self.ui.shot_listWidget.selectedItems():
            self.ui.cam_treeWidget.clear()
            self.ui.dist_treeWidget.clear()
            ######################################
            description = self.ui.prj_comboBox.currentText()
            prjCode = self.prjDict[description]
            prjName = self.prjDict[prjCode]
            shot = self.ui.shot_listWidget.selectedItems()[0].text()
            ######################################

            client = MongoClient(DB_IP)
            db = client[DB_NAME]
            coll = db[prjName]

            camData = coll.find({'data_type':'camera',
                                 'show'     : prjName,
                                 'shot'     : shot}).sort('version',pymongo.DESCENDING)
            distData = coll.find({'data_type':'distortion',
                                 'show'      : prjName,
                                 'shot'      : shot}).sort('version',pymongo.DESCENDING)

            ##################################################

            for i in camData:
                camPath = i['files']['camera_path'][0]
                camFile = os.path.basename(camPath)
                camType = str(i['task_publish']['plateType'])
                team = str(i['task'])
                camVer = str(i['version'])
                camOverScan = str(i['task_publish']['overscan'])
                camStereo = str(i['task_publish']['stereo'])
                camTime = i['time']
                date = camTime.split('T')[0]
                time = (camTime.split('T')[1]).split('.')[0]

                camItem = CamTreeItem(self.ui.cam_treeWidget)
                camItem.setDBdata(i)
                #################################################
                camItem.setText(0, camFile)
                camItem.setText(1, camType)
                camItem.setText(2, team)
                camItem.setText(3, 'v' + camVer.zfill(3))
                camItem.setText(4, camOverScan)
                camItem.setText(5, camStereo)
                camItem.setText(6, date + ' ' + time)

            for i in distData:
                distPath = i['files']['path'][0]
                # distPath = i['files']['nuke11_path'][0]
                distFile = os.path.basename(distPath)
                distVer = str(i['version'])
                distPlate = i['task_publish']['plateType']
                distTime = i['time']
                date = distTime.split('T')[0]
                time = (distTime.split('T')[1]).split('.')[0]

                distItem = DistTreeItem(self.ui.dist_treeWidget)
                distItem.setDBdata(i)
                ######################################################
                distItem.setText(0, distFile)
                distItem.setText(1, 'v' + distVer.zfill(3))
                distItem.setText(2, distPlate)
                distItem.setText(3, date + ' ' + time)
        else:
            self.ui.cam_treeWidget.clear()
            self.ui.dist_treeWidget.clear()

    def getPlate(self):
        if self.ui.shot_listWidget.selectedItems():
            self.ui.plate_treeWidget.clear()
            ######################################
            description = self.ui.prj_comboBox.currentText()
            prjCode = self.prjDict[description]  # show71
            prjName = self.prjDict[prjCode]
            seq = self.ui.seq_listWidget.selectedItems()[0].text()
            shot = self.ui.shot_listWidget.selectedItems()[0].text()
            ######################################
            basePath = '/show/%s/_2d/shot/%s/%s/plates/' %(prjName, seq, shot)
            if os.path.isdir(basePath) == True:
                for plateType in sorted(os.listdir(basePath)):
                    typeItem = PlateTreeItem(self.ui.plate_treeWidget)
                    typeItem.addCheckBox(0)
                    typeItem.checkBox.setText(plateType)
                    typeItem.setExpanded(True)

                    plateTypePath = os.path.join(basePath, plateType)
                    if os.path.isdir(plateTypePath) == True:
                        for ver in os.listdir(plateTypePath):

                            verItem = PlateTreeItem(typeItem)
                            verItem.addCheckBox(1)
                            verItem.checkBox.setText(ver)

                            filePath = os.path.join(plateTypePath, ver)
                            ##################################
                            verItem.setAbsPath(filePath)
                            ##################################
                            ts = os.path.getctime(filePath)
                            verTime = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                            verItem.setText(2, verTime)
        else:
            self.ui.plate_treeWidget.clear()

    def checkallPlate(self, checkState):
        for idx in range(self.ui.plate_treeWidget.topLevelItemCount()):
            rootItem = self.ui.plate_treeWidget.topLevelItem(idx)
            if checkState == True:
                rootItem.checkBox.setChecked(True)
            else:
                rootItem.checkBox.setChecked(False)

    def filterMyJobs(self, checkState):
        if checkState == 2:
            ###########
            for idx in range(self.ui.prj_comboBox.count()):
                if idx != 0 :
                    description = self.ui.prj_comboBox.itemText(idx)
                    prjCode = self.prjDict[description]
                    if prjCode not in self.task['prjCode']:
                        self.ui.prj_comboBox.model().item(idx).setEnabled(False)
                        if self.ui.prj_comboBox.currentIndex() == idx:
                            self.ui.prj_comboBox.setCurrentIndex(0)
            ###########
            for idx in range(self.ui.seq_listWidget.count()):
                seq = self.ui.seq_listWidget.item(idx)
                if seq.text() not in self.task['seq']:
                    seq.setFlagsDisable()
                    if seq.isSelected:
                        seq.setSelected(False)
            ############
            for idx in range(self.ui.shot_listWidget.count()):
                shot = self.ui.shot_listWidget.item(idx)
                if shot.text() not in self.task['shot']:
                    shot.setFlagsDisable()
                    if shot.isSelected:
                        shot.setSelected(False)

        else:
            for idx in range(self.ui.prj_comboBox.count()):
                self.ui.prj_comboBox.model().item(idx).setEnabled(True)

            for idx in range(self.ui.seq_listWidget.count()):
                self.ui.seq_listWidget.item(idx).setFlagsEnable()

            for idx in range(self.ui.shot_listWidget.count()):
                self.ui.shot_listWidget.item(idx).setFlagsEnable()

    def hideExistPath(self):
        for idx in range(self.ui.shot_listWidget.count()):
            item = self.ui.shot_listWidget.item(idx)
            if self.ui.filter_exist_checkBox.isChecked():
                if item.isExists == True:
                    item.setHidden(True)
                    if item.isSelected() == True:
                        item.setSelected(False)
            else:
                if item.isExists == True:
                    item.setHidden(False)

    def makeMatchmove(self):
        mmvBdList = []
        bottomNode = None
        reformatSwitch = False

        selectedCam = self.ui.cam_treeWidget.selectedItems()
        selectedDist = self.ui.dist_treeWidget.selectedItems()

        # Set Format in Project Setting
        ################################
        if selectedCam:
            camDB = selectedCam[0].getDBdata()
            mmvFormat = None
            try:
                mmvFormat = nuke.addFormat('%s %s' % (camDB['task_publish']['render_width'], camDB['task_publish']['render_height']))
            except:
                if camDB['task_publish'].get('renderWidth'):
                    mmvFormat = nuke.addFormat('%s %s' % (camDB['task_publish']['renderWidth'], camDB['task_publish']['renderHeight']))
            if mmvFormat:
                nuke.root()['format'].setValue(mmvFormat)

            # Scence Node
            ##############
            sceneNode = nuke.nodes.Scene()
            sceneNode.setXYpos(0, 0)
            mmvBdList.append(sceneNode)

            # Camera Node
            ####################
            camPath = camDB['files']['camera_path'][0]
            camNode = nuke.createNode('Camera2', 'file {%s} read_from_file True' % camPath)
            camNode.setXYpos(sceneNode.xpos() - 150, sceneNode.ypos())
            mmvBdList.append(camNode)

            # CheckerBoard
            ###############
            checkerBoard = nuke.nodes.CheckerBoard2()
            if mmvFormat:
                checkerBoard['format'].setValue(mmvFormat)
            checkerBoard.setXYpos(sceneNode.xpos(), sceneNode.ypos() - 200)
            mmvBdList.append(checkerBoard)

            # ReadGeo
            #########
            geoKeyList = ['camera_geo_path', 'camera_asset_geo_path']
            for idx, key in enumerate(geoKeyList):
                if camDB['files'].get(key):
                    for geoFile in camDB['files'][key]:
                        readGeo = nuke.createNode('ReadGeo2', 'file {%s}' % geoFile)
                        readGeo['disable'].setValue(True)
                        readGeo.setInput(0, checkerBoard)
                        sceneNode.setInput(sceneNode.inputs(), readGeo)
                        readGeo.setXYpos((sceneNode.xpos() - 100) + (idx*50), sceneNode.ypos() - 100)
                        mmvBdList.append(readGeo)

            # ScanlineRender Node
            #######################
            scanlineNode = nuke.nodes.ScanlineRender()
            scanlineNode.setInput(1, sceneNode)

            # prat2 layout camera offset setting
            # if 'prat2' == camDB['show'] and 'layout' == camDB['task_publish']['plateType']:
            #     offset = prat2LayoutOffset.getOffsetRange('prat2', camDB['shot'])
            #
            #     if offset:
            #         timeOffset = nuke.createNode('TimeOffset', 'time_offset -%s' % offset)
            #         timeOffset.setInput(0, camNode)
            #         scanlineNode.setInput(2, timeOffset)
            # else:
            scanlineNode.setInput(2, camNode)
            scanlineNode.setXYpos(sceneNode.xpos(), sceneNode.ypos() + 100)
            mmvBdList.append(scanlineNode)
            bottomNode = scanlineNode

            # Overscan Reformat
            ####################
            if camDB['task_publish']['overscan']:
                bottomNode = self.addOverScanReformat(camDB, scanlineNode, mmvFormat, mmvBdList)
                reformatSwitch = True

        # Distortion NK File
        #####################
        if selectedDist:
            distDB = selectedDist[0].getDBdata()
            distPath = distDB['files']['path'][0]

            if not reformatSwitch and distDB['task_publish']['overscanSize']:
                bottomNode = self.addOverScanReformat(distDB, scanlineNode, mmvFormat, mmvBdList)

            beforeDist = nuke.allNodes()
            nuke.nodePaste(distPath)
            afterDist = nuke.allNodes()

            distNodes = list(set(afterDist) - set(beforeDist))
            trNode = None
            for idx, distNode in enumerate(distNodes):
                # if 'Transform_FilmBack_' in distNode.name():
                #     trNode = distNode
                # elif distNode.knob('output'):
                #     distNode['output'].setValue('Redistort')
                # if trNode and selectedCam:
                #     trNode.setInput(0, bottomNode)

                if distNode.knob('direction'):
                    distNode['direction'].setValue('distort')
                    if selectedCam:
                        distNode.setXYpos(bottomNode.xpos() + (idx * (-100)), bottomNode.ypos() + 30)
                        distNode.setInput(0, bottomNode)
                    else:
                        distNode.setXYpos(100 + (idx * (-100)), 0)
                mmvBdList.append(distNode)

        # Backdrop Node
        ########################################
        nukescripts.clear_selection_recursive()

        if mmvBdList:
            for i in mmvBdList:
                i['selected'].setValue(True)

            mmvBdNode = nukescripts.autoBackdrop()
            mmvBdNode['name'].setValue('MatchmoveBackdrop')
            mmvBdNode['bdheight'].setValue(mmvBdNode['bdheight'].value() + 30)
            mmvBdNode['bdwidth'].setValue(mmvBdNode['bdwidth'].value() + 20)
            mmvBdNode['label'].setValue('Matchmove')
            mmvBdNode['note_font_size'].setValue(30)
            mmvBdNode['tile_color'].setValue(1401174527)

    def addOverScanReformat(self, DB, scanlineNode, mmvFormat, mmvBdList):
        if DB['task_publish'].get('overscan_value'):
            overscan_value = float(DB['task_publish']['overscan_value'])
        elif DB['task_publish'].get('overscanSize'):
            overscan_value = float(DB['task_publish']['overscanSize'])
        else:
            overscan_value = 1.08

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
        mmvBdList.append(reformat1)

        # Reformat2 #
        #################################
        reformat2 = nuke.nodes.Reformat()
        reformat2['type'].setValue('to format')
        reformat2['format'].setValue(mmvFormat)
        reformat2['resize'].setValue('none')
        reformat2['pbb'].setValue(True)
        reformat2.setInput(0, reformat1)
        reformat2.setXYpos(reformat1.xpos(), reformat1.ypos() + 30)
        mmvBdList.append(reformat2)
        bottomNode = reformat2

        return bottomNode

    def makePlate(self):
        plateBdList = []
        description = self.ui.prj_comboBox.currentText()
        prjCode = self.prjDict[description]  # show71
        prjName = self.prjDict[prjCode]  # asd01
        #################################################
        for idx in range(self.ui.plate_treeWidget.topLevelItemCount()):
            rootItem = self.ui.plate_treeWidget.topLevelItem(idx)
            for subIdx in range(rootItem.childCount()):
                subItem = rootItem.child(subIdx)
                if subItem.checkBox.isChecked():
                    filePath = str(subItem.getAbsPath())
                    beforeRead = nuke.allNodes()

                    nuke.tcl('drop', filePath)
                    for i in nuke.allNodes('Read'):
                        if i.error():
                            print('deleteNode:', i.name(), i['file'].value())
                            nuke.delete(i)

                    afterRead = nuke.allNodes()
                    plateReads = list(set(afterRead) - set(beforeRead))
                    nukescripts.clear_selection_recursive()

                    for plate in plateReads:
                        plateBdList.append(plate)

        nukescripts.clear_selection_recursive()

        if plateBdList:
            # Set Position & SelectAll
            for idx, plateRead in enumerate(plateBdList):
                sceneNode = nuke.toNode('Scene1')
                mmvBd = nuke.toNode('MatchmoveBackdrop')
                if sceneNode:
                    plateRead.setXYpos((sceneNode.xpos() + 200) + (idx * 100), sceneNode.ypos())
                elif mmvBd:
                    plateRead.setXYpos((mmvBd.xpos() + 200) + (idx * 100), mmvBd.ypos())
                else:
                    plateRead.setXYpos(100 + (idx *100), 0)

            for plateRead in plateBdList:
                plateRead['selected'].setValue(True)

            # Create BackdropNode
            plateBdNode = nukescripts.autoBackdrop()
            plateBdNode['name'].setValue('PlateBackdrop')
            plateBdNode['bdheight'].setValue(plateBdNode['bdheight'].value() + 50)
            plateBdNode['label'].setValue('Plate')
            plateBdNode['note_font_size'].setValue(30)
            plateBdNode['tile_color'].setValue(948866560)

    def makeDiscription(self):
        description = self.ui.prj_comboBox.currentText()
        prjCode = self.prjDict[description]  # show71
        shotNum = self.ui.shot_listWidget.selectedItems()[0].text()

        info = requests.get(shotServer, params={'api_key'      : API_KEY,
                                                'project_code' : prjCode,
                                                'code': shotNum}).json()

        firstFrame = info[0]['frame_in']
        lastFrame = info[0]['frame_out']
        description_vfx = info[0]['description_vfx']
        if description_vfx:
            description_vfx = description_vfx.replace('<', '[')
            description_vfx = description_vfx.replace('>', ']')

        description_vfxdetail = info[0]['description_vfxdetail']
        if description_vfxdetail:
            description_vfxdetail = description_vfxdetail.replace('<', '[')
            description_vfxdetail = description_vfxdetail.replace('>', ']')

        noteText  = '<font size=10 color="red">Frame Range : %s-%s </font><br/>' % (firstFrame, lastFrame)
        noteText += '<font size=8 color="blue">description : <br/>%s</font><br/>' % (info[0]['description'])
        noteText += '<font size=8 color="green">vfx description : <br/>%s</font><br/>' % (description_vfx)
        noteText += '<font size=8 color="purple">vfx detail : <br/>%s</font><br/>' % (description_vfxdetail)
        noteText += '<br><br><br>'

        note = nuke.createNode("StickyNote")
        note['label'].setValue(noteText)
        mmvBd = nuke.toNode('MatchmoveBackdrop')
        plateBd = nuke.toNode('PlateBackdrop')
        width = note.screenWidth()

        if mmvBd:
            note.setXYpos(mmvBd.xpos() - 1300, mmvBd.ypos())
        elif plateBd:
            note.setXYpos(plateBd.xpos() - 1300, plateBd.ypos())
        else:
            note.setXYpos(-800, 0)

        # ProjectSetting #
        nuke.knob("root.first_frame", str(firstFrame))
        nuke.knob("root.last_frame", str(lastFrame))
        nuke.knob("root.lock_range", "1")

    def makeDir(self):
        description = self.ui.prj_comboBox.currentText()
        prjCode = self.prjDict[description] # show71
        prjDir = self.prjDict[prjCode]  # asd01
        seqCode = self.ui.seq_listWidget.selectedItems()[0].text()
        shotNum = self.ui.shot_listWidget.selectedItems()[0].text()

        # Get Context #
        ctx = self.ui.ctx_comboBox.currentText()
        if ctx == u'직접 입력':
            ctx = self.ui.ctx_lineEdit.text()

        coder = rb.Coder()
        pattern = coder.Rulebook().flag['ver'].pattern
        verDigit = int(re.search(r'(?<={)\d+(?=})', pattern).group())
        ver = 'v%s' % (str(1).zfill(verDigit))

        if not rb.MatchFlag('ver', ver):
            ver = ver.upper()
            if not rb.MatchFlag('ver', ver):
                raise 'check rulebook ver!!!'

        argv = {'show': prjDir,
                'pub': '_2d',
                'task': ctx,
                'ver': ver}
        argv.update(coder.N.SHOTNAME.Decode(shotNum))
        basePath = coder.D.SHOT.Encode(**argv)

        # Save Nukescripts #
        if ctx == 'fx':
            scriptPath = os.path.join(basePath, 'fx/precomp/nuke')
        elif ctx == 'lighting':
            ctx = 'lgt'
            scriptPath = os.path.join(basePath, 'lighting/precomp/nuke')
        else:
            argv['departs'] = 'CMP'
            basePath = coder.D.WORKS.Encode(**argv)
            scriptPath = os.path.join(basePath, seqCode, shotNum, ctx)
        fileName = coder.F.NUKE.BASE.Encode(**argv).replace('LD', ctx) # task 직접입력 시 룰북에서 인코딩이 안되 기본값인 LD로 지정되는데 이를 직접 입력한 ctx로 대체한다

        if not os.path.exists(scriptPath):
            os.makedirs(scriptPath)

        filePath = os.path.join(scriptPath, fileName)
        nuke.scriptSaveAs(filePath)
        nukescripts.clear_selection_recursive()

        print("save path :", filePath)


    def projectSetting(self):
        description = self.ui.prj_comboBox.currentText()
        prjCode = self.prjDict[description]  # show71
        prjName = self.prjDict[prjCode]  # asd01

        configData = comm.getDxConfig()
        if configData['colorSpace'].get('ACES'):
            nuke.root().knob('colorManagement').setValue('OCIO')
            print('Set colorManagement: OCIO')
        else:
            nuke.root().knob('colorManagement').setValue('Nuke')
            print('Set colorManagement: Nuke Default')

        # viewer setting
        allViewer = nuke.allNodes('Viewer')
        if allViewer:
            for i in allViewer:
                i['viewerProcess'].setValue(configData['colorSpace']['display'])
                print('Set viewerProcess: %s' % configData['colorSpace']['display'])
        else:
            if configData['colorSpace'].get('ACES'):
                nuke.createNode('Viewer').knob('viewerProcess').setValue('Rec.709 (ACES)')
                print('Set viewerProcess: Rec.709 (ACES)')
            elif configData['colorSpace'].get('in') == 'Cineon':
                nuke.createNode('Viewer').knob('viewerProcess').setValue('rec.709')
                print('Set viewerProcess: rec709')
            elif configData['colorSpace'].get('in') == 'rec709':
                nuke.createNode('Viewer').knob('viewerProcess').setValue('rec.709')
                print('Set viewerProcess: rec709')
            else:
                nuke.createNode('Viewer').knob('viewerProcess').setValue('sRGB')
                print('Set viewerProcess: sRGB')

        # readNode setting
        allRead = nuke.allNodes('Read')
        for i in allRead:
            if 'plates' in i['file'].value():
                i['colorspace'].setValue(str(configData['colorSpace']['in']))

        res = '%s %s' % (str(configData['delivery']['resolution'][0]), str(configData['delivery']['resolution'][1]))
        prjFormat = nuke.addFormat(res)
        nuke.root()['format'].setValue(prjFormat)
        nuke.root()['fps'].setValue(configData['delivery']['fps'])

        # showSetting (CDL, LUT, etc...)
        settingFile = os.path.join('/show', prjName, '_config', 'nuke', 'scripts', 'importShowSetting.py')
        if os.path.isfile(settingFile):
            print('settingFile:', settingFile)
            print('shot:', self.ui.shot_listWidget.selectedItems()[0].text())

            import importShowSetting
            reload(importShowSetting)
            importShowSetting.doIt(self.ui.shot_listWidget.selectedItems()[0].text())

        nuke.scriptSave()

    def scriptClear(self):
        nuke.scriptClear()

    def zoom(self):
        nuke.selectAll()
        nuke.zoomToFitSelected()
        nukescripts.clear_selection_recursive()

########################################################################################################################

class ListItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent, item):
        super(ListItem, self).__init__(parent)
        self.setText(item)

    def setFlagsDisable(self):
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEnabled)

    def setFlagsEnable(self):
        self.setFlags(self.flags() |QtCore.Qt.ItemIsEnabled)


class ShotAddButton(QtWidgets.QListWidgetItem):
    def __init__(self, parent, seqName):
        super(ShotAddButton, self).__init__(parent)

        self.seq = seqName
        self.isExists = ''

    def __hash__(self):
        return hash(self.seqName)

        widget = QtWidgets.QWidget()
        self.button = QtWidgets.QToolButton()
        self.textLine = QtWidgets.QLineEdit()
        self.button.setText('+')
        buttonFont = QtGui.QFont()
        buttonFont.setPointSize(5)
        buttonFont.setBold(True)

        self.button.setFont(buttonFont)
        self.button.setIconSize(QtCore.QSize(10, 10))

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.textLine)
        self.layout.addWidget(self.button)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)

        widget.setLayout(self.layout)
        self.listWidget().setItemWidget(self, widget)

        self.button.clicked.connect(self.addShotItem)

    def setExistsPath(self, prjName, seq, shot):
        compPath = '/show/%s/_2d/shot/%s/%s/comp' % (prjName, seq, shot)
        self.isExists = os.path.exists(compPath)

    def getExistsPath(self):
        return self.isExists

    def setFlagsDisable(self):
        pass

    def setFlagsEnable(self):
        pass

    def addShotItem(self):
        text = self.textLine.text()
        itemCount = self.listWidget().count() - 1

        if text.isdigit():
            if len(text) == 4:
                num = text

        elif '_' in text:
            if len(text.split('_')[1]) == 4:
                num = text.split('_')[1]

        shotItem = self.seq + '_' + num
        search = self.listWidget().findItems(shotItem, QtCore.Qt.MatchExactly)

        if not search:
            self.listWidget().insertItem(itemCount, shotItem)
            self.listWidget().item(itemCount).setSelected(True)


class ShotListItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent, item):
        super(ShotListItem, self).__init__(parent)
        # widget = ShotWidget()
        item = self.setText(item)
        self.isExists = ''

    def setExistsPath(self, prjName, seq, shot):
        compPath = '/show/%s/_2d/shot/%s/%s/comp' % (prjName, seq, shot)
        self.isExists = os.path.exists(compPath)

    def getExistsPath(self):
        return self.isExists

    def setFlagsDisable(self):
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsEnabled)

    def setFlagsEnable(self):
        self.setFlags(self.flags() | QtCore.Qt.ItemIsEnabled)


class CamTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent):
        super(CamTreeItem, self).__init__(parent)
        self.camDBdata = None

    def setDBdata(self, data):
        self.camDBdata = data

    def getDBdata(self):
        return self.camDBdata


class DistTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent):
        super(DistTreeItem, self).__init__(parent)
        self.distDBdata = None

    def setDBdata(self, data):
        self.distDBdata = data

    def getDBdata(self):
        return self.distDBdata


class PlateTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent):
        super(PlateTreeItem, self).__init__(parent)

        self.checkBox = QtWidgets.QCheckBox()
        self.checkBox.setChecked(True)
        self.checkBox.toggled.connect(self.setChildChecked)

        self.absPath = ''

    def addCheckBox(self, idx):
        self.treeWidget().setItemWidget(self, idx, self.checkBox)

    def setChildChecked(self, checkState):
        if checkState == True:
            for i in range(self.childCount()):
                self.child(i).checkBox.setChecked(True)
        else:
            for i in range(self.childCount()):
                self.child(i).checkBox.setChecked(False)

    def setAbsPath(self, path):
        self.absPath = path

    def getAbsPath(self):
        return self.absPath



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ce = CompShotSetup()
    ce.show()
    sys.exit(app.exec_())
