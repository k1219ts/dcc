# -*- coding: utf-8 -*-
import os
import getpass
import logging
import subprocess
import distutils.util
import maya.OpenMaya as OpenMaya
import maya.OpenMayaUI as OpenMayaUI
import maya.cmds as cmds
import maya.mel as mel
from HUD import HUDmodules;reload(HUDmodules)
import ANI_common
import optionvar
import core.mayaMov as mayaMov;reload(mayaMov)
import utils
reload(utils)
from PySide2 import QtCore, QtGui, QtWidgets
from resource.ui_ddpm import Ui_DDPM
import shiboken2
import rigPubPreview.rigPubStatus as rpp;reload(rpp)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
rcss = open(os.sep.join([CURRENT_DIR, 'resource', 'css', 'ddpm.css']), 'r').read()
rcss = rcss.replace("ICON_DIRNAME", os.sep.join([CURRENT_DIR, 'resource', 'icons']))

SHOW_PATH = '/show'

_CODEC = ['h264', 'h265']

_RESOLUTION = {'From Window': None,
               'Render Setting': None,
               'Fit width 1920': None,
               'Fit width 1280': None,
               'Fit height 720': None,
               'Fit height 1080': None,
               'hd720': [1280, 720],
               'hd1080': [1920, 1080],
               'From Project': None,
               'Custom': None
               }
_DEPARTMENTS = {'Animation': ['Blocking',
                              'Detail',
                              'Facial',
                              'Final'],
                'Creature': ['Rigging',
                             'Simulation',
                             'Finalize',
                             'RigPub']
                }

_TIMEMAP = {"game": 15, "film": 24, "pal" : 25, "ntsc" : 30, "show": 48, "palf" : 50, "ntscf": 60}

def getMayaWindow():
    ptr = OpenMayaUI.MQtUtil.mainWindow()
    return shiboken2.wrapInstance( long(ptr), QtWidgets.QWidget )

class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        optionvar.initialize()
        self.ui = Ui_DDPM()
        self.ui.setupUi(self)

        self.setWindowTitle('Dexter - Maya Movie Maker')
        self.setStyleSheet(rcss)
        self.move(QtCore.QPoint(1920 / 2, 500))
        self.ui.redshift_Btn.setIcon(QtGui.QIcon(
            QtGui.QPixmap(CURRENT_DIR + "/resource/icons/redshift_tractorSubmit32.png"))
        )
        self.ui.openDir_Btn.setIcon(QtGui.QIcon(
            QtGui.QPixmap(CURRENT_DIR + "/resource/icons/folder.png"))
        )
        self.ui.play_Btn.setIcon(QtGui.QIcon(
            QtGui.QPixmap(CURRENT_DIR + "/resource/icons/branch_closed.png"))
        )
        self.ui.refresh_Btn.setIcon(QtGui.QIcon(
            QtGui.QPixmap(CURRENT_DIR + "/resource/icons/refresh.png"))
        )
        self.ui.status_comboBox.setEnabled(False)
        self.ui.progress_comboBox.setEnabled(False)
        self.setupUi()
        self.connectSignals()
        self.mayaCallbacks()
        logger.debug(u'Main Window Initialized.')

    def connectSignals(self):
        logger.debug(u'Connect Ui Signals')
        self.ui.hudOn_Btn.clicked.connect(self.createHud)
        self.ui.hudOff_Btn.clicked.connect(HUDmodules.mg_removeHUD)
        self.ui.depart_comboBox.currentIndexChanged.connect(self.departChange)
        self.ui.prvSizePreset_comboBox.currentIndexChanged.connect(self.updatePreviewSize)
        self.ui.projectList_comboBox.currentIndexChanged.connect(self.updatePreviewSize)
        self.ui.camSeq_checkBox.stateChanged.connect(self.updateFrameRange)
        self.ui.browse_Btn.clicked.connect(self.browse)
        self.ui.openDir_Btn.clicked.connect(self.openDir)
        self.ui.play_Btn.clicked.connect(self.play)
        self.ui.fcheck_Btn.clicked.connect(self.fcheck)
        self.ui.blueMatte_Btn.clicked.connect(self.blueMatte)
        self.ui.createMovie_Btn.clicked.connect(lambda: self.make(type='movie'))
        self.ui.createSequence_Btn.clicked.connect(lambda: self.make(type='sequence'))
        # self.ui.redshift_Btn.clicked.connect(self.redshiftPopup)
        self.ui.refresh_Btn.clicked.connect(self.refreshUi)
        self.ui.close_Btn.clicked.connect(self.close)

    def mayaCallbacks(self):
        self.sceneSavedCallback = OpenMaya.MEventMessage.addEventCallback("SceneSaved", self.refreshUi)
        self.sceneOpenedCallback = OpenMaya.MEventMessage.addEventCallback("SceneOpened", self.refreshUi)

    def setupUi(self):
        logger.debug(u'Setup UI Elements')
        prv_size_preset = _RESOLUTION.keys()
        prv_size_preset.sort()
        userName = cmds.optionVar(q='M3_User')
        sceneFile = utils.getPath(type='file')
        outputPath = utils.getPath(type='folder')
        depart = cmds.optionVar(q='M3_depart')
        codec = cmds.optionVar(q='M3_codec')
        prv_size_current = cmds.optionVar(q='M3_prvPreset')
        project = cmds.optionVar(q='M3_project')

        # movie name for playblast command
        if outputPath == "Select file path":
            self.ui.movie_name = outputPath
        else:
            self.ui.movie_name = os.path.join(outputPath, sceneFile) + ".mov"
        self.ui.artistName_lineEdit.setText(userName)
        self.ui.sceneName_lineEdit.setText(sceneFile)
        self.ui.depart_comboBox.addItems(_DEPARTMENTS.keys())
        self.ui.depart_comboBox.setCurrentText(depart)
        self.ui.progress_comboBox.addItems([str(i) for i in range(10, 110, 10)])
        self.ui.codec_comboBox.addItems(_CODEC)
        self.ui.codec_comboBox.setCurrentIndex(0)
        self.ui.prvSizePreset_comboBox.addItems(prv_size_preset)
        self.ui.prvSizePreset_comboBox.setCurrentText(prv_size_current)
        self.ui.projectList_comboBox.addItems(ANI_common.getListDirs(SHOW_PATH))
        self.ui.projectList_comboBox.setCurrentText(project)
        self.ui.output_lineEdit.setText(self.ui.movie_name)

        self.ui.addStamp_checkBox.setChecked(False)
        self.ui.addStamp_checkBox.setEnabled(False)

        self.ui.autoHud_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_autoHUD')))
        )
        self.ui.rmSeq_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_removeSequencer')))
        )
        self.ui.offScreen_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_offScreenVal')))
        )
        self.ui.camSeq_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_cameraSequencer')))
        )
        self.ui.addStamp_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_addStemp')))
        )
        self.ui.addCompRetime_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_addCompRetime')))
        )
        self.ui.showImageplane_checkBox.setChecked(
            bool(distutils.util.strtobool(cmds.optionVar(q='M3_showImageplane')))
        )
        self.updatePreviewSize()
        self.updateFrameRange()
        self.departChange()

    def getOptionsFromUi(self):
        options = dict()
        options['artist'] = str(self.ui.artistName_lineEdit.text())
        options['scene'] = str(self.ui.sceneName_lineEdit.text())
        options['depart'] = str(self.ui.depart_comboBox.currentText())
        options['status'] = str(self.ui.status_comboBox.currentText())
        options['progress'] = str(self.ui.progress_comboBox.currentText())
        options['auto_hud'] = str(self.ui.autoHud_checkBox.isChecked())
        options['codec'] = str(self.ui.codec_comboBox.currentText())
        if self.ui.prvSizeW_lineEdit.text():
            options['width'] = int(self.ui.prvSizeW_lineEdit.text())
        else:
            options['width'] = 0
        if self.ui.prvSizeH_lineEdit.text():
            options['height'] = int(self.ui.prvSizeH_lineEdit.text())
        else:
            options['height'] = 0
        options['wh'] = [options['width'], options['height']]
        options['seqTime'] = self.ui.camSeq_checkBox.isChecked()
        options['compRetime'] = self.ui.addCompRetime_checkBox.isChecked()
        options['startTime'] = int(float(self.ui.startFrame_lineEdit.text()))
        options['endTime'] = int(float(self.ui.endFrame_lineEdit.text()))
        options['sizePreset'] = str(self.ui.prvSizePreset_comboBox.currentText())
        options['project'] = str(self.ui.projectList_comboBox.currentText())
        options['remove_sequence'] = self.ui.rmSeq_checkBox.isChecked()
        options['offScreen'] = self.ui.offScreen_checkBox.isChecked()
        options['camera_sequencer'] = self.ui.camSeq_checkBox.isChecked()
        options['addstemp'] = self.ui.addStamp_checkBox.isChecked()
        options['addCompRetime'] = self.ui.addCompRetime_checkBox.isChecked()
        options['showImgeplane'] = self.ui.showImageplane_checkBox.isChecked()
        options['outpath'] = str(self.ui.output_lineEdit.text())
        try:
            options['seqStartTime'] = int(float(self.ui.seqStartFrame_lineEdit.text()))
            options['seqEndTime'] = int(float(self.ui.seqEndFrame_lineEdit.text()))
        except Exception as e:
            logger.debug(u'[Exception] : {0}'.format(e))
        return options

    def closeEvent(self, event):
        options = self.getOptionsFromUi()
        cmds.optionVar(sv=('M3_User', options['artist']))
        cmds.optionVar(sv=('M3_depart', options['depart']))
        cmds.optionVar(sv=('M3_codec', options['codec']))
        cmds.optionVar(iv=('M3_width', options['width']))
        cmds.optionVar(iv=('M3_height', options['height']))
        cmds.optionVar(sv=('M3_autoHUD', options['auto_hud']))
        cmds.optionVar(sv=('M3_removeSequencer', options['remove_sequence']))
        cmds.optionVar(sv=('M3_offScreenVal', options['offScreen']))
        cmds.optionVar(sv=('M3_cameraSequencer', options['camera_sequencer']))
        cmds.optionVar(sv=('M3_addStemp', options['addstemp']))
        cmds.optionVar(sv=('M3_addCompRetime', options['addCompRetime']))
        cmds.optionVar(sv=('M3_showImageplane', options['showImgeplane']))
        cmds.optionVar(sv=('M3_prvPreset', options['sizePreset']))
        cmds.optionVar(sv=('M3_project', options['project']))
        logger.debug(u'[{0}] Ui option saved'.format(event))
        if self.ui.autoHud_checkBox.isChecked():
            HUDmodules.mg_removeHUD()

        try:
            global _win
            OpenMaya.MEventMessage.removeCallback(self.sceneSavedCallback)
            OpenMaya.MEventMessage.removeCallback(self.sceneOpenedCallback)
            del _win
            logger.debug(u'Callback Removed')
        except:
            logger.exception(u'Failed to remove callback')
            raise

    def sizeEditable(self, edit=True):
        if edit:
            self.ui.prvSizeW_lineEdit.setEnabled(True)
            self.ui.prvSizeH_lineEdit.setEnabled(True)
        else:
            self.ui.prvSizeW_lineEdit.setEnabled(False)
            self.ui.prvSizeH_lineEdit.setEnabled(False)

    def updatePreviewSize(self):
        sizePreset = str(self.ui.prvSizePreset_comboBox.currentText())
        if sizePreset:
            size = _RESOLUTION[sizePreset]
        else:
            size = None
        logger.debug(u'Update preview size : {}'.format(sizePreset))
        if sizePreset == 'From Project':
            project = str(self.ui.projectList_comboBox.currentText())
            self.sizeEditable(edit=True)
            self.ui.projectList_comboBox.setEnabled(True)
            camInfo = utils.getProjectInfo(project=project)
            if not camInfo:
                logger.debug(u'No camera info file for : {}'.format(project))
                self.ui.prvSizeW_lineEdit.clear()
                self.ui.prvSizeH_lineEdit.clear()
                return
            try:
                previewSize = camInfo['CAMERAS']['FullCG']['preview_size']
                self.ui.prvSizeW_lineEdit.setText(str(previewSize[0]))
                self.ui.prvSizeH_lineEdit.setText(str(previewSize[1]))
            except:
                pass
        else:
            self.ui.projectList_comboBox.setEnabled(False)
        if size:
            self.sizeEditable(edit=True)
            self.ui.prvSizeW_lineEdit.setText(str(size[0]))
            self.ui.prvSizeH_lineEdit.setText(str(size[1]))
        elif sizePreset == 'From Window':
            self.ui.prvSizeW_lineEdit.clear()
            self.ui.prvSizeH_lineEdit.clear()
            self.sizeEditable(edit=False)
        elif sizePreset == 'Render Setting':
            self.sizeEditable(edit=True)
            renderSizeW = cmds.getAttr("defaultResolution.width")
            renderSizeH = cmds.getAttr("defaultResolution.height")
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))
        elif sizePreset == 'Fit width 1920':
            self.sizeEditable(edit=True)
            renderSizeW = 1920
            renderSizeH = int(1920 * cmds.getAttr("defaultResolution.height") / cmds.getAttr("defaultResolution.width"))
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))
        elif sizePreset == 'Fit width 1280':
            self.sizeEditable(edit=True)
            renderSizeW = 1280
            renderSizeH = int(1280 * cmds.getAttr("defaultResolution.height") / cmds.getAttr("defaultResolution.width"))
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))
        elif sizePreset == 'Fit height 720':
            self.sizeEditable(edit=True)
            renderSizeW = int(720 * cmds.getAttr("defaultResolution.width") / cmds.getAttr("defaultResolution.height"))
            renderSizeH = 720
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))
        elif sizePreset == 'Fit height 1080':
            self.sizeEditable(edit=True)
            renderSizeW = int(1080 * cmds.getAttr("defaultResolution.width") / cmds.getAttr("defaultResolution.height"))
            renderSizeH = 1080
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))
        elif sizePreset == 'Custom':
            self.sizeEditable(edit=True)
            renderSizeW = cmds.optionVar(q='M3_width')
            renderSizeH = cmds.optionVar(q='M3_height')
            self.ui.prvSizeW_lineEdit.setText(str(renderSizeW))
            self.ui.prvSizeH_lineEdit.setText(str(renderSizeH))

    def updateFrameRange(self):
        self.ui.startFrame_lineEdit.setText(str(utils.getFrameRange("start")))
        self.ui.endFrame_lineEdit.setText(str(utils.getFrameRange("end")))
        if self.ui.camSeq_checkBox.isChecked():
            sequencer_list = cmds.ls(type='sequencer')
            if not sequencer_list:
                QtWidgets.QMessageBox.warning(self, 'waring!', 'No Camera Sequencer')
                self.ui.camSeq_checkBox.setChecked(False)
                return
            self.ui.seqStartFrame_lineEdit.setEnabled(True)
            self.ui.seqEndFrame_lineEdit.setEnabled(True)
            self.ui.seqStartFrame_lineEdit.setText(str(utils.getFrameRange("start", sequencer=True)))
            self.ui.seqEndFrame_lineEdit.setText(str(utils.getFrameRange("end", sequencer=True)))
        else:
            self.ui.seqStartFrame_lineEdit.clear()
            self.ui.seqEndFrame_lineEdit.clear()
            self.ui.seqStartFrame_lineEdit.setEnabled(False)
            self.ui.seqEndFrame_lineEdit.setEnabled(False)

    def refreshUi(self, *args, **kwargs):
        sceneFile = utils.getPath(type='file')
        outputPath = utils.getPath(type='folder')
        minTime = utils.getFrameRange(type='start')
        maxTime = utils.getFrameRange(type='end')

        # movie name for playblast command
        if outputPath == "Select file path":
            movie_name = outputPath
        else:
            movie_name = os.path.join(outputPath, sceneFile) + ".mov"
        self.ui.sceneName_lineEdit.setText(sceneFile)
        self.ui.output_lineEdit.setText(movie_name)
        self.ui.startFrame_lineEdit.setText(str(minTime))
        self.ui.endFrame_lineEdit.setText(str(maxTime))
        self.ui.artistName_lineEdit.setText(getpass.getuser())

    def departChange(self):
        self.ui.status_comboBox.clear()
        currentDepart = str(self.ui.depart_comboBox.currentText())
        if not currentDepart:
            return
        if currentDepart == "Animation":
            self.ui.status_comboBox.setEnabled(False)
        else:
            self.ui.status_comboBox.setEnabled(True)
            status = _DEPARTMENTS[currentDepart]
            self.ui.status_comboBox.addItems(status)

    def createHud(self):
        artist = str(self.ui.artistName_lineEdit.text())
        sceneName = str(self.ui.sceneName_lineEdit.text())
        status = str(self.ui.status_comboBox.currentText())
        progress = str(self.ui.progress_comboBox.currentText())
        HUDmodules.mg_CreateHUD(artistName=artist,
                                sceneName=sceneName,
                                status=status,
                                progress=progress,
                                fontSize='large')

    def browse(self):
        startPath = str(self.ui.output_lineEdit.text())
        if startPath == "Select file path":
            startPath = "untitled"

        fileName = QtWidgets.QFileDialog.getSaveFileName(None,
                                                         'save filename',
                                                         startPath)[0]
        if len(fileName):
            if not fileName.endswith(".mov"):
                self.movie_name = fileName
                fileName += '.mov'
            else:
                self.movie_name = os.path.splitext(fileName)[0]
            self.ui.output_lineEdit.setText(str(fileName))

    def openDir(self):
        mov = str(self.ui.output_lineEdit.text())
        movDir = os.path.dirname(mov)
        p = subprocess.Popen('/usr/bin/nautilus {0}'.format(movDir), shell=True)

    def play(self):
        movie = str(self.ui.output_lineEdit.text())
        logger.debug(u'Play Movie File : {}'.format(movie))
        env = mayaMov.getEnv()
        subprocess.Popen('ffplay-dd {0}'.format(movie), shell=True)
        # try:
        #     subprocess.Popen('/usr/local/bin/pdplayer {0}'.format(movie), env=env, shell=True)
        # except:
        #     subprocess.Popen('ffplay-dd {0}'.format(movie), shell=True)

    def fcheck(self):
        fromWindow = False
        options = self.getOptionsFromUi()
        options['outpath'] = None

        if self.ui.autoHud_checkBox.isChecked():
            self.createHud()
        if self.ui.prvSizePreset_comboBox.currentText() == 'From Window':
            fromWindow = True

        movClass = mayaMov.MakeMovie()
        movClass.options = options
        movClass.playblast(parentWidget=self, fromWindow=fromWindow)

        if self.ui.autoHud_checkBox.isChecked():
            HUDmodules.mg_removeHUD()

    def blueMatte(self):
        utils.mayaBackgroundEdit(rgb=[0, 0, 1], offimp=True)

    def modelPanelShow(self):
        try:
            panel = cmds.getPanel(wf=True)
            cmds.modelEditor(panel, e=True, imagePlane=True)
            return True
        except:
            return True

    def make(self, type):
        """

        :param type: 'movie' or 'sequence'
        """
        if self.ui.showImageplane_checkBox.isChecked():
            states = self.modelPanelShow()
            if not states:
                return
        fromWindow = False
        addStamp = False
        options = self.getOptionsFromUi()
        options.update({'script_path': CURRENT_DIR})
        options.update({'mayafile': str(cmds.file(q=True, sn=True))})

        # 편집실에 aniPreview 보낼 때 맞추기 위해 24프레임은 23.98로 고정
        # 하지만 실제 mov걸 때 show _config가 있다면 그게 우선이 된다.
        fps = int(mel.eval('currentTimeUnitToFPS'))
        if 24 == fps:
            options.update({'fps': '23.976'})

        if self.ui.autoHud_checkBox.isChecked():
            self.createHud()
        if self.ui.prvSizePreset_comboBox.currentText() == 'From Window':
            fromWindow = True
        if self.ui.addStamp_checkBox.isChecked():
            addStamp = True

        # soundItem = cmds.ls(type = 'audio')
        # if soundItem:
        #     options.update({"sound":soundItem[0]})

        #rigPubPreview
        if options["status"] == "RigPub":
            rigmsg = QtWidgets.QMessageBox.question(
                self,
                unicode("실행주의", 'utf-8'),
                unicode("리그펍 후에 사용해주세요. 되돌릴 수 없습니다. 사용하시겠습니까?", 'utf-8'),
                QtWidgets.QMessageBox.Ok,
                QtWidgets.QMessageBox.Cancel)
            if rigmsg == QtWidgets.QMessageBox.Ok:
                rpp.DoIt()
                maxFrame = cmds.playbackOptions(max=True, q=True)
                options.update({"endTime": maxFrame})
                minFrame = cmds.playbackOptions(min=True, q=True)
                options.update({"startTime": minFrame})

        movClass = mayaMov.MakeMovie(options)
        movClass.playblast(
            parentWidget=self,
            fromWindow=fromWindow,
            type=type,
            addstamp=addStamp
        )
        if self.ui.autoHud_checkBox.isChecked():
            HUDmodules.mg_removeHUD()

    # def redshiftPopup(self):
    #     redshiftmaya.main.showUI()


def showUI():
    global _win
    try:
        _win.close()
    except:
        pass
    _win = MainWindow()
    _win.show()
    _win.resize(400, 200)
    return _win
