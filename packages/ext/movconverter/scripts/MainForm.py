# -*- coding: utf-8 -*-
import os, sys
import json
import subprocess
import getpass

# QT
from PySide2 import QtWidgets, QtGui, QtCore

# STYLESHEET
import qdarkstyle

from ui.ui_movConverter import Ui_MainWindow
import mcCommon as mc

import tractor.api.author as author

TRACTOR_IP = '10.0.0.25'
PORT = 80


class MainForm(QtWidgets.QMainWindow):
    def __init__(self, parent):
        QtWidgets.QMainWindow.__init__(self, parent)

        self.input = ''
        self.output = ''

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        imgPath = os.path.dirname(os.path.realpath(__file__))+'/resources/dexter_studios_logo_dark.png'
        self.ui.logo_label.setPixmap(QtGui.QPixmap(imgPath))
        darkStyleSheet = qdarkstyle.load_stylesheet()
        self.setStyleSheet(darkStyleSheet)

        # macOS: TRACTOR only
        if 'Darwin' == mc.chkPlatform():
            idx = self.ui.machineType_comboBox.findText('LOCAL')
            self.ui.machineType_comboBox.removeItem(idx)

        if sys.argv:
            for arg in sys.argv:
                if '.py' in arg:
                    continue
                self.input = os.path.abspath(arg)
            self.output = mc.getOutputPath(self.input)

            self.ui.input_lineEdit.setText(self.input)
            self.ui.output_lineEdit.setText(self.output)

        codec = mc.loadCodecConfig(self.input)
        for codecName in codec.keys():
            self.ui.codec_comboBox.addItem(codecName)
        currentIdx = self.ui.codec_comboBox.findText('h265')
        self.ui.codec_comboBox.setCurrentIndex(currentIdx)

        self.ui.findInput_pushButton.clicked.connect(self.findInputDialog)
        self.ui.findOutput_pushButton.clicked.connect(self.findOutputDialog)
        self.ui.convert_pushButton.clicked.connect(self.doIt)

    def doIt(self):
        self.input = self.ui.input_lineEdit.text()
        self.output = self.ui.output_lineEdit.text()
        self.remakeMov = self.ui.remakeMov_checkBox.isChecked()
        user = getpass.getuser()

        if '' == self.input or '' == self.output:
            self.messagePopup('input, output 경로를 확인 해 주세요.')
            return

        codec = self.ui.codec_comboBox.currentText()
        cmdType = self.ui.machineType_comboBox.currentText()

        result = QtWidgets.QMessageBox.information(self, 'movConverter',
                                                   '{CODEC} codec으로 convert 하겠습니까?'.format(CODEC=codec),
                                                   QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        if result != QtWidgets.QMessageBox.Ok:
            self.messagePopup('취소하였습니다.')
            return

        progress = 0
        self.ui.progressBar.setValue(progress)

        cmd = ['{DCCPROC}', 'rez-env', 'ffmpeg_toolkit', '--']
        cmd += ['ffmpeg_converter', '-c', codec]
        cmd += ['-r', '{FPS}']
        cmd += ['-i', '{INPUT}', '-o', '{OUTPUT}']
        cmd += ['-au']
        cmd += ['-u', '{USER}']
        cmd = ' '.join(cmd)

        if not '.mov' in self.output:
            expDir = self.output
        else:
            expDir = os.path.dirname(self.output)

        if not os.path.isdir(expDir):
            os.mkdir(expDir)

        if 'LOCAL' == cmdType:
            if os.path.isdir(self.input):
                movCount = len(os.walk(self.input).next()[2])
                per = 100 / movCount
                for idx, mov in enumerate(os.walk(self.input).next()[2]):
                    self.ui.info_label.setText('converting... %s [%s / %s]' % \
                                               (mov, idx+1, movCount))

                    inputMov = os.path.join(self.input, mov)
                    outputMov = os.path.join(self.output, mov)

                    if not os.path.isfile(inputMov) or mov.startswith('.'):
                        continue

                    metadata = mc.getSrcSeq(inputMov)
                    fps = metadata['fps']

                    if self.remakeMov and metadata.has_key('srcSeq'):
                        inputMov = metadata['srcSeq']
                        if metadata.has_key('artist'):
                            user = metadata['artist']

                    command = cmd.format(DCCPROC=os.environ['DCCPROC'], FPS=fps,
                                         USER=user, INPUT=inputMov, OUTPUT=outputMov)
                    run = subprocess.Popen(command, shell=True)
                    run.wait()

                    progress+=per
                    self.ui.progressBar.setValue(progress)

                else:
                    self.ui.progressBar.setValue(100)

            elif os.path.isfile(self.input):
                self.ui.info_label.setText('converting... %s [1 / 1]' % \
                                           (os.path.basename(self.input)))

                metadata = mc.getSrcSeq(self.input)
                fps = metadata['fps']

                if self.remakeMov and metadata.has_key('srcSeq'):
                    self.input = metadata['srcSeq']
                    if metadata.has_key('artist'):
                        user = metadata['artist']

                command = cmd.format(DCCPROC=os.environ['DCCPROC'], FPS=fps,
                                     USER=user, INPUT=self.input, OUTPUT=self.output)
                run = subprocess.Popen(command, shell=True)
                run.wait()

                self.ui.progressBar.setValue(100)

            self.messagePopup('완료되었습니다.')

        elif 'TRACTOR' == cmdType:
            job = author.Job()
            job.title = 'mov codec Convert'
            job.comment = ''
            job.metadata = ''
            job.service = 'Cache'
            job.maxactive = 0
            job.tier = 'cache'
            job.tags = ['']
            job.projects = ['export']
            job.priority = 100

            MainJobTask = author.Task(title='movConverter')

            if os.path.isdir(self.input):
                for idx, mov in enumerate(os.walk(self.input).next()[2]):

                    inputMov = os.path.join(self.input, mov)
                    outputMov = os.path.join(self.output, mov)

                    if not os.path.isfile(inputMov) or mov.startswith('.'):
                        continue

                    metadata = mc.getSrcSeq(inputMov)
                    fps = metadata['fps']

                    if self.remakeMov and metadata.has_key('srcSeq'):
                        inputMov = metadata['srcSeq']
                        if metadata.has_key('artist'):
                            user = metadata['artist']

                    command = cmd.format(DCCPROC='/backstage/dcc/DCC', FPS=fps,
                                         INPUT=mc.resolvePath(inputMov),
                                         OUTPUT=mc.resolvePath(outputMov),
                                         USER=user).split(' ')
                    movTask = author.Task(title=str(mov))
                    movTask.addCommand(author.Command(argv=command, service='Cache'))
                    MainJobTask.addChild(movTask)

                    job.addChild(MainJobTask)

            elif os.path.isfile(self.input):
                metadata = mc.getSrcSeq(self.input)
                fps = metadata['fps']

                if self.remakeMov and metadata.has_key('srcSeq'):
                    self.input = metadata['srcSeq']
                    if metadata.has_key('artist'):
                        user = metadata['artist']

                command = cmd.format(DCCPROC='/backstage/dcc/DCC', FPS=fps,
                                     INPUT=mc.resolvePath(self.input),
                                     OUTPUT=mc.resolvePath(self.output),
                                     USER=user).split(' ')
                movTask = author.Task(title=str(os.path.basename(self.input)))
                movTask.addCommand(author.Command(argv=command, service='Cache'))
                MainJobTask.addChild(movTask)

                job.addChild(MainJobTask)

            author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT,
                                        user=getpass.getuser(), debug=True)
            job.spool()
            print job.asTcl()
            author.closeEngineClient()

            self.messagePopup('TRACTOR(%s) job spool 완료.' % TRACTOR_IP)

    def findInputDialog(self):
        if not '.mov' in self.input:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Find mov Directory", self.input, QtWidgets.QFileDialog.ShowDirsOnly)
        else:
            path = QtWidgets.QFileDialog.getSaveFileName(self, 'save filename', self.input)[0]
        if path:
            self.input = path
            self.output = mc.getOutputPath(self.input)

            self.ui.input_lineEdit.setText(self.input)
            self.ui.output_lineEdit.setText(self.output)

    def findOutputDialog(self):
        if not '.mov' in self.output:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Find mov Directory", self.output, QtWidgets.QFileDialog.ShowDirsOnly)
        else:
            path = QtWidgets.QFileDialog.getSaveFileName(self, 'save filename', self.output)[0]
        if path:
            self.output = path
            self.ui.output_lineEdit.setText(self.output)

    def setOutputPath(self, input):
        self.input = input
        self.output = mc.getOutputPath(input)
        self.ui.output_lineEdit.setText(self.output)

    def messagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, 'movConverter', msg, QtWidgets.QMessageBox.Ok)
