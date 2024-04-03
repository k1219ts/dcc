#encoding=utf-8
import os, sys, site, logging, requests
import dxConfig
import getpass
import maya.cmds as cmds
import maya.mel as mel
from Qt_Ani.Qt import QtCore, QtGui, QtWidgets
import mayaUi;reload(mayaUi)
import redshift_Submit
reload(redshift_Submit)
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RedshiftOptionWindow(mayaUi.RedshiftWindow):
    def __init__(self, parent=None):
        super(RedshiftOptionWindow, self).__init__(parent)
        self.settingFirst()
        self.addSequencerList()
        self.addCameraList()
        self.connectSignals()
        self.checkSequencer()

    def connectSignals(self):
        self.useSequencer_checkBox.stateChanged.connect(self.useSequencer)
        self.render_Btn.clicked.connect(self.render)
        self.cancel_Btn.clicked.connect(self.close)
        self.openSequencerWindow_Btn.clicked.connect(self.openSequencerWindow)
        self.image_combo.currentIndexChanged.connect(self.outputImageSetting)
        self.resolution_txt.returnPressed.connect(self.setResolution)
        self.speed_checkBox.stateChanged.connect(self.exrChange)
        self.set_Btn.clicked.connect(self.openRenderWindow)

    def settingFirst(self):
        self.checkRender()
        self.maxActive_spinBox.setValue(3)#Max Active set(rnd)
        min, max, by = self.getFrame()
        self.startFrame_lineEdit.setText(str(min))
        self.endFrame_lineEdit.setText(str(max))
        self.byFrame_lineEdit.setText(str(by))
        self.comboList()
        self.speed = 'no'

    def checkRender(self, currentScene = None):
    # 현재 씬 위치 체크 & redshift 플러그인 체크와 랜더러 변경
        currentScene = cmds.file(q=True, sn=True)
        current_renderer = cmds.getAttr('defaultRenderGlobals.currentRenderer')
        renderer = 'redshift'
        # 씬 파일이 로컬에 있거나 씬 파일 이름이 untitled 인 경우 취소 처리
        if not currentScene or currentScene.find('C:') != -1 or currentScene.find('/home/') != -1:
            messageBox('Save Scene and retry please (Null or Not show path) !!', 'warning', ['OK'])
            sys.exit()
        else:
        # redshift 플러그인 로드 랜더러 변경
            if not current_renderer == renderer:
                try:
                    self.loadMayaPlugin('%s4maya' % renderer)
                    self.loadRenderPlugin(renderer)
                    # cmds.setAttr("redshiftOptions.imageFormat", 4)
                    self.show()
                except:
                    messageBox('RedShift Maya Plugin Not Loading !!', 'warning', ['OK'])
            else:
                self.show()

    def loadMayaPlugin(self, p):
    # redshift 플러그인을 강제 로딩하고 현재의 렌더러를 강제 변경 처리
        if not cmds.pluginInfo(p, q=True, l=True):
            cmds.loadPlugin(p)

    def loadRenderPlugin(self, p):
    # redshift로 현재의 렌더러를 강제 변경 처리 & 시퀀스가 존재할경우 옵션설정 스킵이나 리넘버
        cmds.setAttr('defaultRenderGlobals.currentRenderer', l=False)
        cmds.setAttr('defaultRenderGlobals.currentRenderer', p, type='string')
        mel.eval('loadPreferredRenderGlobalsPreset("%s");' % p)
        cmds.setAttr('defaultRenderGlobals.modifyExtension', 0)  # renumber frames 설정 비활성 처리

    def getFrame(self):
    # render frame get
        min = cmds.playbackOptions(q=True, min=True)
        max = cmds.playbackOptions(q=True, max=True)
        by = cmds.playbackOptions(q=True, by=True)
        return min, max, by

    def comboList(self):
    # render mov size, output sequence image type set, teamname get, resolution get
        resolution = ['hd_720', 'hd_1080']
        images = ['iff', 'exr', 'png', 'tga', 'jpg', 'tif']
        self.resolution_combo.setEnabled(False)
        self.resolution_combo.addItems(resolution)
        self.image_combo.addItems(images)
        self.userTeam()
        self.setImageFormat()
        self.getResolution()

    def getResolution(self):# 1280x720, 1920x1080
    # resolution get (aniteam fixed resolution)
        height = cmds.getAttr("defaultResolution.h")
        width = cmds.getAttr("defaultResolution.w")
        self.resolution_txt.setText('%s X %s'%(width, height))
        if self.userteam == 'Ani' or height == 720:
            self.resolution_combo.setCurrentIndex(0)
        else:
            self.resolution_combo.setCurrentIndex(1)

    def setResolution(self):
    # resolution combo list change >> render setting change
        res = self.resolution_txt.text().split('X')
        print res
        width = int(res[0])
        height = int(res[1])
        cmds.setAttr("defaultResolution.h", height)
        cmds.setAttr("defaultResolution.w", width)
        self.getResolution()

    def setImageFormat(self):
    # output sequence image type >> render setting
        imageformat = cmds.getAttr("redshiftOptions.imageFormat")
        self.image_combo.setCurrentIndex(imageformat)

    def openRenderWindow(self):
    # render setting window open
        if cmds.window("unifiedRenderGlobalsWindow", exists=True):
            cmds.deleteUI("unifiedRenderGlobalsWindow")
        mel.eval('unifiedRenderGlobalsWindow;')

    def exrChange(self):
    # speed check >> output sequence image exr change
        self.image_combo.setCurrentIndex(1)
        self.outputImageSetting()
        state = self.speed_checkBox.isChecked()
        if  state:
            self.speed = 'ok'
        else:
            self.speed = 'no'

    def outputImageSetting(self):
    # output combo list change >> render setting change
        format = self.image_combo.currentIndex()
        cmds.setAttr("redshiftOptions.imageFormat", format)

    def userTeam(self):
    # user team query
        users = getpass.getuser()
        params = {}
        params['api_key'] = API_KEY
        params['code'] = users
        infos = requests.get("http://%s/dexter/search/user.php" % (dxConfig.getConf('TACTIC_IP')),
                             params=params).json()
        self.userteam = infos['department']  # Ani,Layout
        if self.userteam.startswith('Ani'):
            self.userteam = 'Ani'
        elif self.userteam.startswith('Layout'):
            self.userteam = 'Layout'

    def checkSequencer(self):
    # sequence shot usage check
        shots = cmds.sequenceManager(listShots=True)
        if not shots:
            self.useSequencer_checkBox.setChecked(False)
        else:
            for shot in shots:
                isMute = cmds.shot(shot, q=True, mute=True)
                if not isMute:
                    self.useSequencer_checkBox.setChecked(True)

    def useSequencer(self):
    # use sequencer shot checkbox on/off
        state = self.useSequencer_checkBox.isChecked()
        if state:
            self.sequencer_tableWidget.setEnabled(True)
            self.camera_tableWidget.setEnabled(False)
            self.frameRangeSet(0)
            self.checkReset(self.camera_tableWidget)
            self.shotCheckstateChange()
        else:
            self.sequencer_tableWidget.setEnabled(False)
            self.camera_tableWidget.setEnabled(True)
            self.frameRangeSet(1)
            self.checkReset(self.sequencer_tableWidget)
            self.setRenderable()

    def checkReset(self, object):
    # use sequencer change >> shot, sequencer check box reset
        rowCount = object.rowCount()
        for row in range(rowCount):
            camNameCheckBox = object.cellWidget(row, 0)
            camNameCheckBox.setChecked(False)

    def frameRangeSet(self, ck = None):
    # use sequencer change >> framerange display setenabled check
        self.startFrame_lineEdit.setEnabled(ck)
        self.endFrame_lineEdit.setEnabled(ck)
        self.byFrame_lineEdit.setEnabled(ck)

    def addSequencerList(self):
    # sequencer list display
        shots = cmds.sequenceManager(listShots=True)
        if not shots:# shot == null sequencer table enabled
            self.sequencer_tableWidget.setEnabled(False)
            self.useSequencer_checkBox.setEnabled(False)
            self.useSequencer_checkBox.setChecked(False)
            self.frameRangeSet(1)
        else:
            self.camera_tableWidget.setEnabled(False)
            self.sequencer_tableWidget.setRowCount(len(shots))
            self.useSequencer_checkBox.setChecked(True)
            self.frameRangeSet(0)
            for i, shot in enumerate(shots):
                isShotMuted = not cmds.shot(shot, q=True, mute=True)
                shotCam = cmds.shot(shot, q=True, currentCamera=True)
                startTime = cmds.shot(shot, q=True, sequenceStartTime=True)
                endTime = cmds.shot(shot, q=True, sequenceEndTime=True)

                shotCheckBox = QtWidgets.QCheckBox(shot)
                startTimeWidget = QtWidgets.QLineEdit()
                startTimeWidget.setText(str(startTime))
                endTimeWidget = QtWidgets.QLineEdit()
                endTimeWidget.setText(str(endTime))
                startTimeWidget.setEnabled(False)
                endTimeWidget.setEnabled(False)
                shotCheckBox.setChecked(isShotMuted)

                self.sequencer_tableWidget.setCellWidget(i, 0, shotCheckBox)
                self.sequencer_tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(shotCam))
                self.sequencer_tableWidget.setCellWidget(i, 2, startTimeWidget)
                self.sequencer_tableWidget.setCellWidget(i, 3, endTimeWidget)
                shotCheckBox.stateChanged.connect(self.shotCheckstateChange)
            self.sequencer_tableWidget.resizeRowsToContents()

    def addCameraList(self):
    # cameralist display
        cameratype = ['Standard', 'Standard', 'Fisheye', 'Spherical', 'Cylindrical', 'Stereo Spherical']
        # 0 = stan 1 = X   2=fisheye
        cams = cmds.listCameras()
        self.camera_tableWidget.setRowCount(len(cams))
        min, max, by = self.getFrame()

        for i, item in enumerate(cams):
            isRenderable = cmds.getAttr(item + ".renderable")
            camtype = cmds.getAttr('%s.rsCameraType' % item)

            cameraCheckbox = QtWidgets.QCheckBox(item)
            camtypeWidget = QtWidgets.QLineEdit()
            camtypeWidget.setText(cameratype[camtype])
            startTimeWidget = QtWidgets.QLineEdit()
            startTimeWidget.setText(str(min))
            endTimeWidget = QtWidgets.QLineEdit()
            endTimeWidget.setText(str(max))
            startTimeWidget.setEnabled(False)
            endTimeWidget.setEnabled(False)

            self.camera_tableWidget.setCellWidget(i, 0, cameraCheckbox)
            self.camera_tableWidget.setCellWidget(i, 1, camtypeWidget)
            self.camera_tableWidget.setCellWidget(i, 2, startTimeWidget)
            self.camera_tableWidget.setCellWidget(i, 3, endTimeWidget)
            if isRenderable:
                cameraCheckbox.setChecked(True)
            cameraCheckbox.stateChanged.connect(self.setRenderable)
        self.camera_tableWidget.resizeRowsToContents()

    def shotCheckstateChange(self):
    # shot list render list check >> shot mute, unmute setting
        rowCount = self.sequencer_tableWidget.rowCount()
        for row in range(rowCount):
            shotCheckBox = self.sequencer_tableWidget.cellWidget(row, 0)
            state = shotCheckBox.isChecked()
            if state:
                self.muteTrack(shot=str(shotCheckBox.text()), mute=False)
            else:
                self.muteTrack(shot=str(shotCheckBox.text()), mute=True)

    def setRenderable(self):
    # camera list render list check >> camera renderable on/off setting
        rowCount = self.camera_tableWidget.rowCount()
        for row in range(rowCount):
            camNameCheckBox = self.camera_tableWidget.cellWidget(row, 0)
            state = camNameCheckBox.isChecked()
            cam = str(camNameCheckBox.text())
            if state and cam:
                cmds.setAttr(cam + ".renderable", 1)
                logger.debug(u'Set renderable : {0}, ON'.format(cam))
            else:
                cmds.setAttr(cam + ".renderable", 0)
                logger.debug(u'Set renderable : {0}, OFF'.format(cam))

    def muteTrack(self, shot, mute=True):
    # shot mute setting
        if mute:
            logger.debug(u'Mute shot : {shot}'.format(shot=shot))
        else:
            logger.debug(u'Unmute shot : {shot}'.format(shot=shot))
        cmds.shot(shot, e=True, mute=mute)

    def getOptions(self):
    # setting list options list set
        options = dict()
        options["limitTagOption"] = list()
        options["startTime"] = float(self.startFrame_lineEdit.text())
        options["endTime"] = float(self.endFrame_lineEdit.text())
        options["byFrame"] = float(self.byFrame_lineEdit.text())
        options["chunkSize"] = int(self.chunkSize_spinBox.value())
        options["maxActive"] = int(self.maxActive_spinBox.value())

        state = self.useSequencer_checkBox.isChecked()
        options["useShot"] = state

        useCacheFarm = self.cache_checkBox.isChecked()
        useGpuFarm = self.gpu_checkBox.isChecked()
        useUserFarm = self.user_checkBox.isChecked()

        if useCacheFarm:
            options["limitTagOption"].append('gpu_cache')
        if useGpuFarm:
            options["limitTagOption"].append('gpu_render')
        if useUserFarm:
            options["limitTagOption"].append('gpu_user')
        return options

    def render(self):
    # directTractorSubmit list send
        options = self.getOptions()
        framesOption = '{0}/{1}/{2}'.format(options["startTime"],
                                            options["endTime"],
                                            options["byFrame"])

        kwargs = {'frames':framesOption,
                  'limitTag':options["limitTagOption"],
                  'chunkSize':options["chunkSize"]}
        if options["maxActive"]:
            kwargs.update({'maxActive':options["maxActive"]})

        kwargs['speed'] = self.speed
        kwargs['team'] = self.userteam

        optionString = u'Render option >> {}'
        logger.debug(optionString.format(kwargs))
        #{'frames': '0.004-0.04/1.0', 'maxActive': 3, 'limitTag': ['gpu_render'], 'speed': 'no', 'chunkSize': 6}
        redshift_Submit.LayoutRedshift(kwargs)
        self.close()

    def openSequencerWindow(self):
    # seqencer window open
        mel.eval("SequenceEditor")

def messageBox(messages='information message',
# error message box
               icons= 'warning',  # warning, question, information, critical
               buttons=['OK', 'Cancel']):
    try:
        redshift_version = str(cmds.pluginInfo('redshift4maya', q=True, v=True))#'redshift-2.5.21-2017'
    except Exception, e:  # 플러그인 로딩이 안되는 경우 기본 버전 처리
        redshift_version = '2.5.52'
    titles = 'Layout for Redshift-%s'%redshift_version
    msg = '%s    ' % messages
    bgcolor = [0.9, 0.6, 0.6]
    cmds.confirmDialog(title=titles, message=msg,
                       messageAlign='center', icon=icons,
                       button=buttons, backgroundColor=bgcolor)

def showUI():
    global lay_UI
    lay_UI = RedshiftOptionWindow()
    lay_UI.setObjectName('layout_redshift')

def closeUI():
    lay_UI.close()



