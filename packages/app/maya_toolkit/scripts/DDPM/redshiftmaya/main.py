#encoding=utf-8

import os
import sys
import site
import logging
import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMaya as OpenMaya
from Qt_Ani.Qt import QtCore, QtGui, QtWidgets, QtCompat
import DDPM.utils;reload(DDPM.utils)
import mayaUi;reload(mayaUi)

#redshift_script_path = '%s/apps/redshift2/scripts/tractorSpool' % os.getenv('BACKSTAGE_PATH')
redshift_script_path = '%s/apps/Redshift/scripts/tractorSpool' % os.getenv('BACKSTAGE_PATH')
site.addsitedir(redshift_script_path)

import redshift_directTractorSubmit
reload(redshift_directTractorSubmit)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)

class RedshiftOptionWindow(mayaUi.RedshiftWindow):
    def __init__(self, parent=None):
        super(RedshiftOptionWindow, self).__init__(parent)
        self.connectSignals()
        self.checkSequencer()
        self.addSequencerList()
        self.setFrameRange(sequence_time=True, byFrame=1)
        self.addCameraList()
        self.mayaSequencerCallback()

    def fontChange(self, widget, pointSize):
        if sys.platform != "darwin":
            fontPath = os.sep.join([CURRENT_DIR, 'resource', 'font', "OpenSans-Regular.ttf"])
            fontId = QtGui.QFontDatabase.addApplicationFont(fontPath)
            if fontId is not -1:
                family = QtGui.QFontDatabase.applicationFontFamilies(fontId)
                font = QtGui.QFont(family[0])
                font.setPointSize(pointSize)
                widget.setFont(font)

    def connectSignals(self):
        self.useSequencer_checkBox.stateChanged.connect(self.useSequencer)
        self.render_Btn.clicked.connect(self.render)
        self.cancel_Btn.clicked.connect(self.close)
        # self.reload_Btn.clicked.connect(self.reloadSequencer)
        self.openSequencerWindow_Btn.clicked.connect(self.openSequencerWindow)

    def mayaSequencerCallback(self):
        self.selectionChangedCallback = OpenMaya.MEventMessage.addEventCallback("SelectionChanged",
                                                                                self.reloadSequencer)
        self.timeChangedCallback = OpenMaya.MEventMessage.addEventCallback("timeChanged",
                                                                           self.reloadSequencer)
        self.playbackRangeSliderChangedCallback = OpenMaya.MEventMessage.addEventCallback("playbackRangeSliderChanged",
                                                                                          self.reloadSequencer)

    def removeMayaCallbacks(self):
        OpenMaya.MMessage.removeCallback(self.selectionChangedCallback)
        OpenMaya.MMessage.removeCallback(self.timeChangedCallback)
        OpenMaya.MMessage.removeCallback(self.playbackRangeSliderChangedCallback)


    def closeEvent(self, event):
        try:
            OpenMaya.MMessage.removeCallback(self.selectionChangedCallback)
            OpenMaya.MMessage.removeCallback(self.timeChangedCallback)
            OpenMaya.MMessage.removeCallback(self.playbackRangeSliderChangedCallback)
            logger.debug(u'Callback removed')
        except:
            sys.stderr.write("Failed to remove callback\n")
            raise

    def openSequencerWindow(self):
        mel.eval("SequenceEditor")

    def setFrameRange(self, sequence_time, byFrame):
        if sequence_time:
            DDPM.utils.cleanupSequencer()
            sequencers = cmds.ls(type='sequencer')
            shots = cmds.sequenceManager(listShots=True)
            if not sequencers and not shots:
                sequence_time = False
            elif sequencers and not shots:
                sequence_time = False

        startTime = DDPM.utils.getFrameRange(type='start', sequencer=sequence_time)
        endTime = DDPM.utils.getFrameRange(type='end', sequencer=sequence_time)
        self.startFrame_lineEdit.setText(str(startTime))
        self.endFrame_lineEdit.setText(str(endTime))
        if byFrame:
            self.byFrame_lineEdit.setText(str(byFrame))

    def checkSequencer(self):
        shots = cmds.sequenceManager(listShots=True)
        if not shots:
            self.useSequencer_checkBox.setChecked(False)
            return

        for shot in shots:
            isMute = cmds.shot(shot, q=True, mute=True)
            if not isMute:
                self.useSequencer_checkBox.setChecked(True)
                return

    def reloadSequencer(self, *args, **kwargs):
        self.sequencer_tableWidget.clear()
        isSequencerExsist = self.addSequencerList()
        #self.checkSequencer()
        if not isSequencerExsist:
            return
        self.useSequencer()

    def useSequencer(self):
        state = self.useSequencer_checkBox.isChecked()
        if state:
            self.sequencer_tableWidget.clear()
            self.sequencer_tableWidget.setEnabled(True)
            self.camera_frame.setEnabled(False)
            self.addSequencerList()
        else:
            self.sequencer_tableWidget.setEnabled(False)
            self.camera_frame.setEnabled(True)
            for row in range(self.sequencer_tableWidget.rowCount()):
                shotCheckBox = self.sequencer_tableWidget.cellWidget(row, 0)
                if shotCheckBox: shotCheckBox.setChecked(False)

        rowCount = self.sequencer_tableWidget.rowCount()
        for row in range(rowCount):
            shotCheckBox = self.sequencer_tableWidget.cellWidget(row, 0)
            if not shotCheckBox:
                break
            shot = str(shotCheckBox.text())
            isMute = not cmds.shot(shot, q=True, mute=True)
            shotCheckBox.setChecked(isMute)
        self.setFrameRange(sequence_time=state, byFrame=1)


    def addSequencerList(self):
        shots = cmds.sequenceManager(listShots=True)
        self.sequencer_tableWidget.setHorizontalHeaderLabels(['Shot', 'Camera', 'Start', 'End'])
        if not shots:
            return False
        self.sequencer_tableWidget.setRowCount(len(shots))
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
        cams = cmds.listCameras()
        self.camera_tableWidget.setRowCount(len(cams))
        startTime = DDPM.utils.getFrameRange(type='start')
        endTime = DDPM.utils.getFrameRange(type='end')

        for i, item in enumerate(cams):
            isRenderable = cmds.getAttr(item + ".renderable")
            cameraCheckbox = QtWidgets.QCheckBox(item)
            startTimeWidget = QtWidgets.QLineEdit()
            startTimeWidget.setText(str(startTime))
            endTimeWidget = QtWidgets.QLineEdit()
            endTimeWidget.setText(str(endTime))
            startTimeWidget.setEnabled(False)
            endTimeWidget.setEnabled(False)
            self.camera_tableWidget.setCellWidget(i, 0, cameraCheckbox)
            self.camera_tableWidget.setCellWidget(i, 1, startTimeWidget)
            self.camera_tableWidget.setCellWidget(i, 2, endTimeWidget)
            if isRenderable:
                cameraCheckbox.setChecked(True)
            cameraCheckbox.stateChanged.connect(self.setRenderable)

        self.camera_tableWidget.resizeRowsToContents()


    def sequencerStateChange(self, *args, **kwargs):
        self.sequencer_tableWidget.clear()
        self.addSequencerList()
        rowCount = self.sequencer_tableWidget.rowCount()
        for row in range(rowCount):
            shotCheckBox = self.sequencer_tableWidget.cellWidget(row, 0)
            shot = str(shotCheckBox.text())
            isMute = cmds.shot(shot, q=True, mute=True)
            shotCheckBox.setChecked(isMute)


    def shotCheckstateChange(self):
        self.removeMayaCallbacks()

        rowCount = self.sequencer_tableWidget.rowCount()
        for row in range(rowCount):
            shotCheckBox = self.sequencer_tableWidget.cellWidget(row, 0)
            state = shotCheckBox.isChecked()
            if state:
                self.muteTrack(shot=str(shotCheckBox.text()), mute=False)
            else:
                self.muteTrack(shot=str(shotCheckBox.text()), mute=True)
        self.mayaSequencerCallback()


    def setRenderable(self):
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
        if mute:
            logger.debug(u'Mute shot : {shot}'.format(shot=shot))
        else:
            logger.debug(u'Unmute shot : {shot}'.format(shot=shot))
        cmds.shot(shot, e=True, mute=mute)


    def getOptions(self):
        """Get redshift render options from ui

        :rtype: dict
        """
        if self.speed_checkBox.isChecked():
            ck = 1
        else:
            ck = 0
        options = dict()
        options["limitTagOption"] = list()
        options["startTime"] = float(self.startFrame_lineEdit.text())
        options["endTime"] = float(self.endFrame_lineEdit.text())
        options["byFrame"] = float(self.byFrame_lineEdit.text())
        options["chunkSize"] = int(self.chunkSize_spinBox.value())
        options["maxActive"] = int(self.maxActive_spinBox.value())
        options["speedcheck"] = int(ck)

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
        options = self.getOptions()
        framesOption = '{0}-{1}/{2}'.format(options["startTime"],
                                            options["endTime"],
                                            options["byFrame"])

        kwargs = {'frames':framesOption,
                  'limitTag':options["limitTagOption"],
                  'chunkSize':options["chunkSize"]}
        if options["maxActive"]:
            kwargs.update({'maxActive':options["maxActive"]})

        if self.speed_checkBox.isChecked() == True:
            kwargs['speed'] = 'ok'
        else:
            kwargs['speed'] = 'no'

        optionString = u'Render option >> {}'
        logger.debug(optionString.format(kwargs))
        redshift_directTractorSubmit.redshiftSubmit(**kwargs)
        self.close()
     
def showUI():
    global lay_UI
    lay_UI = RedshiftOptionWindow()
    lay_UI.show()


