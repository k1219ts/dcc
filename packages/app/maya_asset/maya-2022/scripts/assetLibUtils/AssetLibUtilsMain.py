# -*- coding: utf-8 -*-
import os
import socket
import getpass
import datetime

import maya.cmds as cmds

import tractor.api.author as author

from PySide2 import QtWidgets, QtGui, QtCore

import ui_assetLibUtils
reload(ui_assetLibUtils)

import texAssign as txAssign
reload(txAssign)

TRACTOR_IP = '10.0.0.25'
PORT = 80

tempDir = '/knot/show/asset/tmp'
# fileName = 'test'

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = ui_assetLibUtils.Ui_AssetLibUtils()
        self.ui.setupUi(self)

        # get scene path
        scene = os.path.basename(cmds.file(q=True, sn=True))
        scenePath = os.path.join(tempDir, scene)

        renderPath = os.path.dirname(cmds.file(q=True, sn=True)).replace('/model', '/preview')
        txPath = os.path.dirname(cmds.file(q=True, sn=True)).replace('/model', '/texture')

        self.ui.scene_lineEdit.setText(scenePath.strip())
        self.ui.renderpath_lineEdit.setText(renderPath.strip())
        self.ui.txPath_lineEdit.setText(txPath.strip())

        self.ui.scene_pushButton.clicked.connect(self.openDialog_selectFile)
        self.ui.renderpath_pushButton.clicked.connect(self.openDialog_selectRenderPath)
        self.ui.txPath_pushButton.clicked.connect(self.openDialog_selectTexturePath)
        self.ui.reload_pushButton.clicked.connect(self.getCameraList)
        self.ui.render_pushButton.clicked.connect(self.sendTractor)
        self.ui.assign_pushButton.clicked.connect(self.assignTexture)

        self.getCameraList()

        startFrame = str(int(cmds.playbackOptions(q=True, min=True)))
        endFrame = str(int(cmds.playbackOptions(q=True, max=True)))
        self.ui.frameIn_lineEdit.setText(startFrame)
        self.ui.frameOut_lineEdit.setText(endFrame)

    def getIpAddress(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # GOOGLE PUBLIC DNS SERVER
        ip_addr = s.getsockname()[0]
        return ip_addr

    def getCameraList(self):
        self.ui.camera_comboBox.clear()
        self.ui.camera_comboBox.addItem("Select Camera")

        allCam = cmds.ls(cameras=True)
        camList = []

        for c in allCam:
            camList.append(cmds.listRelatives(c, type="transform", parent=True)[0])

        for c in cmds.listCameras(p=False, o=True):
            try:
                camList.remove(c)
            except:
                print "Removing Othographic Cameras Failed."

        self.ui.camera_comboBox.addItems(camList)

    def sendTractor(self):
        startFrame = str(self.ui.frameIn_lineEdit.text())
        endFrame = str(self.ui.frameOut_lineEdit.text())
        camera = self.ui.camera_comboBox.currentText()

        if camera == 'Select Camera':
            self.messagePopup('카메라를 선택해 주세요.')
            return

        result = QtWidgets.QMessageBox.information(self, "확인", 'frameRange %s - %s\n렌더를 진행하겠습니까?' % (startFrame, endFrame),
                                                   QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.No)
        if result != QtWidgets.QMessageBox.Ok:
            return

        # Scene File Naming
        sceneFile = self.ui.scene_lineEdit.text()
        tempScene = os.path.join(tempDir, sceneFile)
        print 'tempScene:', tempScene

        renderDir = self.ui.renderpath_lineEdit.text()

        # Save As
        if not os.path.isfile(tempScene):
            cmds.file(tempScene, pr=True, typ='mayaBinary', options='v=0;', ea=True, f=True)

        # make Job
        job = author.Job()
        job.title = '(V-RAY) ' + str(os.path.basename(tempScene))
        job.comment = 'RenderFile: %s' % str(tempScene)
        job.metadata = ''
        job.service = 'VRAY'
        job.maxactive = 1
        job.tier = 'cache'
        job.tags = ['']
        job.projects = ['export']
        job.priority = 100

        MainJobTask = author.Task(title="main")

        cmd = ['/backstage/apps/bladeControl/nc_to_exe/eog_command.sh']
        cmd += [renderDir, self.getIpAddress()]
        ncCmd  = author.Command(argv=cmd)
        MainJobTask.addCommand(ncCmd)

        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')  # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')  # Done

        renderCmd = ['/backstage/dcc/DCC', 'rez-env']
        for package in os.environ['REZ_USED_RESOLVE'].split():
            if 'centos' not in package:
                renderCmd += [package]
        # renderCmd += ['--show', show]
        renderCmd += ['--', 'Render']
        renderCmd += ['-cam', camera]
        renderCmd += ['-rd', renderDir]
        renderCmd += ['-s', str(startFrame)]
        renderCmd += ['-e', str(endFrame)]
        # renderCmd += ['-im', fileName]
        renderCmd += [str(tempScene)]

        # print 'renderCmd:', renderCmd

        renderTask = author.Task(title=str(os.path.basename(tempScene)))
        renderTask.addCommand(author.Command(argv=renderCmd, service='Cache'))
        renderTask.addCommand(author.Command(argv='/usr/bin/rm ' + str(tempScene), service='Cache'))
        MainJobTask.addChild(renderTask)

        job.addChild(MainJobTask)

        author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT,
                                    user=getpass.getuser(), debug=True)

        job.spool()
        print job.asTcl()
        author.closeEngineClient()

        self.messagePopup('TRACTOR(%s) job spool 완료.' % TRACTOR_IP)

    def assignTexture(self):
        texturePath = self.ui.txPath_lineEdit.text()

        if not os.path.exists(texturePath):
            self.messagePopup('%s\n텍스쳐 경로를 확인 해 주세요.' % str(texturePath))
            return

        for node in cmds.ls(dag=1, type="surfaceShape"):
            txAssign.AssignShaderlegacy(node = node,
                                        texdir = texturePath,
                                        scene = cmds.file(q=True, sn=True))
        else:
            self.messagePopup('shader Assign 완료.')

    def openDialog_selectFile(self):
        scenePath = self.ui.scene_lineEdit.text()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "임시 저장 씬 경로 설정", scenePath, QtWidgets.QFileDialog.ShowDirsOnly)

        if path:
            self.ui.scene_lineEdit.setText(path)

    def openDialog_selectRenderPath(self):
        renderPath = self.ui.renderpath_lineEdit.text()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "렌더 경로 설정", renderPath, QtWidgets.QFileDialog.ShowDirsOnly)

        if path:
            self.ui.renderpath_lineEdit.setText(path)

    def openDialog_selectTexturePath(self):
        texturePath = self.ui.txPath_lineEdit.text()
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "텍스쳐 경로 설정", texturePath, QtWidgets.QFileDialog.ShowDirsOnly)

        if path:
            self.ui.txPath_lineEdit.setText(path)

    def messagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, 'AssetLibUtils', msg, QtWidgets.QMessageBox.Ok)
