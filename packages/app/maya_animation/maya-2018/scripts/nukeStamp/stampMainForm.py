# -*- coding: utf-8 -*-
import os
import getpass
import subprocess
import requests
import dxConfig
import DXRulebook.Interface as rb
import maya.cmds as cmds

from PySide2 import QtWidgets, QtGui, QtCore

import ui_nukeStamp
reload(ui_nukeStamp)

import tractor.api.author as author

TRACTOR_IP = '10.0.0.25'
PORT = 80


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_nukeStamp.Ui_MainWindow()
        self.ui.setupUi(self)

        # get scene path
        self.scene = cmds.file(q=True, sn=True)
        self.sceneDir = os.path.dirname(self.scene)

        if not self.scene:
            self.messagePopup('Scene 정보를 가져올 수 없습니다.\n다시 실행 해 주세요.')
        else:
            coder = rb.Coder()
            self.argv = coder.F.MAYA.Decode(os.path.basename(self.scene))

            shotName = self.argv.seq + '_' + self.argv.shot
            fileName, comments = self.getTacticNote(shotName)

            if fileName:
                self.ui.filename_textEdit.setText(fileName.strip())
                self.ui.comments_textEdit.setText(comments.strip())
            else:
                self.ui.filename_textEdit.setText('파일명을 입력하세요.')
                self.ui.comments_textEdit.setText('Veloz에 comments 내용이 없습니다. 컨펌 코멘트를 입력하세요.')

        self.ui.movFile_find_pushButton.clicked.connect(self.openDialog_selectMovFile)
        self.ui.export_pushButton.clicked.connect(self.doIt)

    def getTacticNote(self, shot):
        API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
        note = {}
        note['api_key'] = API_KEY
        note['project_code'] = 'show124' #koz
        note['code'] = shot
        note['process'] = 'publish'

        infos = requests.get("http://%s/dexter/search/note.php" %(dxConfig.getConf('TACTIC_IP')), params=note).json()
        if infos:
            try:
                note = infos[0]['note']
                start = note.index('[')
                end = note.index(']')
                filename, comments = note[start+1:end].split(':')
                return filename, comments
            except:
                return None, None
        else:
            return None, None

    def doIt(self):
        movPath = self.ui.mov_textEdit.text()
        fileName = self.ui.filename_textEdit.text()
        comments = self.ui.comments_textEdit.text()

        impPath = ''
        for imp in cmds.ls(type='imagePlane'):
            img = cmds.getAttr(imp + '.imageName')
            if 'imageplane' in img:
                if os.path.isdir(os.path.dirname(img)):
                    impPath = img
                    break

        if not movPath:
            self.messagePopup('movPath를 입력 해 주세요.')
            return

        if not os.path.isdir(movPath):
            self.messagePopup('SEQ image 경로를 확인 해 주세요.')
            return

        result = QtWidgets.QMessageBox.information(self, 'nukeStamp',
                                                   '진행하겠습니까?',
                                                   QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.Cancel)
        if result != QtWidgets.QMessageBox.Ok:
            # self.messagePopup('취소하였습니다.')
            return

        command = ['/backstage/dcc/DCC', 'rez-env', 'nuke-12.2.4', 'ffmpeg_toolkit', '--show', 'koz', '--']
        command += ['nukeX', '-i', '-t', '%s/nukeStamp_koz.py' % os.path.dirname(os.path.abspath(__file__))]
        command += ['--jpgPath', movPath]
        if impPath:
            command += ['--impPath', os.path.dirname(impPath)]
        command += ['--fileName', fileName]
        command += ['--comments', '"' + comments + '"']

        print 'nuke_command:', command

        # if '7' in os.environ['REZ_CENTOS_MAJOR_VERSION']:
        #     run = subprocess.Popen(command, shell=True)
        # else:
        #     run = subprocess.Popen(command, shell=True, env={'USER': getpass.getuser()})
        # run.wait()

        job = author.Job()
        job.title = '(MOV) [koz]nukeStamp: %s' % str(fileName)
        job.comment = ''
        job.metadata = ''
        job.service = 'Cache'
        job.maxactive = 0
        job.tier = 'cache'
        job.tags = ['']
        job.projects = ['export']
        job.priority = 100

        MainJobTask = author.Task(title='nukeStamp: %s' % str(fileName))

        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        movTask = author.Task(title=str(fileName))
        movTask.addCommand(author.Command(argv=command, service='Cache'))
        MainJobTask.addChild(movTask)

        job.addChild(MainJobTask)

        author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT,
                                    user=getpass.getuser(), debug=True)

        job.spool()
        print job.asTcl()
        author.closeEngineClient()

        self.messagePopup('TRACTOR(%s) job spool 완료.' % TRACTOR_IP)

    def openDialog_selectMovFile(self):
        previewDir = self.sceneDir.replace('scenes', 'preview')
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Find Plate Directory", previewDir, QtWidgets.QFileDialog.ShowDirsOnly)

        if path:
            self.ui.mov_textEdit.setText(path)

    def messagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, 'nukeStamp', msg, QtWidgets.QMessageBox.Ok)
