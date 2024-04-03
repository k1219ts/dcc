#coding:utf-8

##########################################
__author__  = 'daeseok.chae @ DexterStudios CGSUP'
__date__ = '2020.09.18'
__comment__ = 'maya to katana render controller'
__windowName__ = "MTOK - 2.0"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

import maya.cmds as cmds
import maya.api.OpenMayaUI as OpenMayaUI
import maya.api.OpenMaya as OpenMaya

from MainFormUI import Ui_Form
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2 import QtGui

import socket
import os
import shutil
import time
currentDir = os.path.dirname(__file__)

import DXUSD_MAYA.Message as msg
import DXUSD_MAYA.Model as MDL
import DXUSD_MAYA.Groom as GRM
import DXUSD_MAYA.Clip as CLP
import DXUSD_MAYA.MUtils as mutls


def getMayaWindow():
    '''
    get Maya Window Process
    :return: Maya window Process
    '''
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QMainWindow)
    except:
        return None


def GetTempAssetPath():
    current = cmds.file(q=True, sn=True)
    if current:
        current = os.path.dirname(current)
    else:
        current = cmds.workspace(q=True, rd=True)
    rootDir = os.path.join(current, 'tmp')
    return str(rootDir)


def TempAssetExport(tmpDir=None, show=None, clear=True):
    if not tmpDir:
        tmpDir = GetTempAssetPath()

    selected = cmds.ls(sl=True)

    cmds.waitCursor(state=True)

    startTime = time.time()

    # try:
    # model
    if selected:
        nodes = cmds.ls(['|*_model_GRP', '|*_model_*_GRP'], sl=True, r=True)
    else:
        nodes = cmds.ls(['|*_model_GRP', '|*_model_*_GRP'], r=True)

    if nodes:
        for node in nodes:
            if mutls.GetViz(cmds.ls(node, l=True)[0]):
                MDL.mtkExport(nodes=[node], customdir=tmpDir)

    # groom
    if selected:
        rootNode = cmds.ls('|*', type='dxBlock', sl=True, r=True)
    else:
        rootNode = cmds.ls('|*', type='dxBlock', r=True)
    if rootNode or cmds.objExists('ZN_ExportSet'):
        if not rootNode:
            rootNode = cmds.ls(['|*_model_GRP', '|*_model_*_GRP'])
        GRM.mtkExport(node=rootNode[0], customdir=tmpDir)

    # clip
    if selected:
        rootNode = cmds.ls('|*', type='dxRig', sl=True, r=True)
    else:
        rootNode = cmds.ls('|*', type='dxRig', r=True)
    if rootNode:
        for node in rootNode:
            if mutls.GetViz(cmds.ls(node, l=True)[0]):
                CLP.mtkRigExport(node=node, customdir=tmpDir)


    endTime = time.time()
    OpenMaya.MGlobal.displayInfo('# Result : TempAssetExport finished!')
    msg.debug('# Compute Time :', '%.3f sec' % (endTime - startTime))

    tmpFile = os.path.join(tmpDir, 'asset', 'asset.usd')
    cmds.waitCursor(state=False)
    return tmpFile


class dxsMTKMain(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle(__windowName__)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.connectPixmap = QtGui.QPixmap("%s/resources/Circle03-Green.png" % currentDir)
        self.dontConnectPixmap = QtGui.QPixmap("%s/resources/Circle04-DarkRed.png" % currentDir)

        self.sock = None

        self.ui.connectBtn.clicked.connect(self.connectKatana)
        self.ui.exportTempCacheBtn.clicked.connect(self.exportTempCacheClicked)
        self.ui.clearCacheBtn.clicked.connect(self.clearCacheClicked)
        self.ui.liveRenderBtn.clicked.connect(self.liveRenderClicked)
        self.ui.previewRenderBtn.clicked.connect(self.previewRenderClicked)
        self.ui.cancelRenderBtn.clicked.connect(self.cancelRenderClicked)

        # self.ui.dirPathEdit.setText("%s/tmp" % os.environ['HOME'])
        self.ui.dirPathEdit.setText(GetTempAssetPath())

        # TEST
        self.ui.testSendBtn.clicked.connect(self.testSendBtnClicked)

        # Member Variables
        self.viewPanel = None
        self.transformCallback = None
        self.tmpCachefile = None
        self.isCached = False

    def getApiObject(self, name):
        sels = OpenMaya.MGlobal.getSelectionListByName(name)
        return sels.getDependNode(0)

    def getViewPanelCamera(self):
        focusPanel = cmds.getPanel(withFocus=True)
        if focusPanel and focusPanel.find('modelPanel') > -1:
            self.viewPanel = focusPanel
        if not self.viewPanel:
            OpenMaya.MGlobal.displayError('Not found viewPanel')
            return
        return cmds.modelPanel(self.viewPanel, q=True, cam=True)


    def getActiveViewCamera(self):
        self.clearCallback()
        camera = self.getViewPanelCamera()
        if not camera:
            OpenMaya.MGlobal.displayError('Not found camera!')
            return
        obj = self.getApiObject(camera)
        self.transformCallback = OpenMaya.MNodeMessage.addNodeDirtyCallback(obj, self.cameraTransformChanged)

    def cameraTransformChanged(self, *args, **kwargs):
        # print '>>', args
        dagFn = OpenMaya.MFnDagNode(args[0])
        camera = dagFn.fullPathName()
        # print camera
        translate = cmds.getAttr('%s.translate' % camera)[0]
        rotate    = cmds.getAttr('%s.rotate' % camera)[0]
        # print translate, rotate
        if not self.sock:
            return
        command = 'import MtoK;'
        command+= ' MtoK.SetCameraTransform('
        command+= '%f, %f, %f' % (translate[0], translate[1], translate[2])
        command+= ',%f, %f, %f);' % (rotate[0], rotate[1], rotate[2])
        self.sock.send(command)


    def liveRenderClicked(self):
        self.getActiveViewCamera()
        if not self.sock:
            return
        command = 'import MtoK; MtoK.LiveRender();'
        self.sock.send(command)


    def previewRenderClicked(self):
        camera = self.getViewPanelCamera()
        if not camera:
            return

        if cmds.nodeType(camera) == "camera":
            camera = cmds.listRelatives(camera, p = True)[0]

        translate= cmds.xform(camera, q = True, ws = True, translation = True)
        rotate   = cmds.xform(camera, q = True, ws = True, rotation = True)
        scale    = cmds.xform(camera, q = True, ws = True, scale = True)

        # horizontalFieldOfView (angle)
        fov = cmds.camera(camera, hfv=True, q=True)
        # horizontalFilmAperture (inch)
        hfa = cmds.camera(camera, hfa=True, q=True)
        # verticalFilmAperture (inch)
        vfa = cmds.camera(camera, vfa=True, q=True)

        if not self.sock:
            OpenMaya.MGlobal.displayError('Not connected Katana!')
            return

        command = 'import MtoK;'
        command+= ' MtoK.SetCameraBase(%f, %f, %f);' % (fov, hfa, vfa)
        command+= ' MtoK.SetCameraTransform('
        command+= '%f, %f, %f' % (translate[0], translate[1], translate[2])
        command+= ',%f, %f, %f);' % (rotate[0], rotate[1], rotate[2])

        # 2DPanZoom
        if cmds.getAttr('%s.panZoomEnabled' % camera) and cmds.getAttr('%s.renderPanZoom' % camera):
            pan  = cmds.getAttr('%s.pan' % camera)[0]
            zoom = cmds.getAttr('%s.zoom' % camera)
            command+= ' MtoK.SetCameraPanZoom(%f, %f, %f, %f);' % (pan[0], pan[1], zoom, hfa)

        if self.tmpCachefile:
            command += ' MtoK.SetCacheFile("%s");' % self.tmpCachefile

        command+= ' MtoK.PreviewRender();'
        self.sock.send(command)
        # print('>>>', command)
        OpenMaya.MGlobal.displayInfo('# Result : send message success!')


    def cancelRenderClicked(self):
        if not self.sock:
            OpenMaya.MGlobal.displayError('Not connected Katana!')
            return
        self.clearCallback()
        command = 'import MtoK; MtoK.CancelRender();'
        self.sock.send(command)

    def clearCacheClicked(self):
        tmpDir = str(self.ui.dirPathEdit.text())
        if os.path.exists(os.path.join(tmpDir, "cache.usd")):
            shutil.rmtree(tmpDir)
        OpenMaya.MGlobal.displayInfo('# Result : Clear tmpDir!')


    #---------------------------------------------------------------------------
    # CACHE OUT
    #---------------------------------------------------------------------------
    def exportTempCacheClicked(self):
        tmpDir = str(self.ui.dirPathEdit.text())
        clear = False if self.isCached else True
        self.tmpCachefile = TempAssetExport(tmpDir=tmpDir, clear=clear)
        self.isCached = True


    #---------------------------------------------------------------------------
    # CONNECT KATANA
    #---------------------------------------------------------------------------
    def connectKatana(self):
        if self.sock:
            msg = cmds.confirmDialog(
                title='disconnect?', message='do you want disconnect from katana?',
                icon='warning', button=['OK', "CANCEL"]
            )
            if msg == "OK":
                self.sock.close()
                self.sock = None
                self.ui.statusLabel.setPixmap(self.dontConnectPixmap)
                self.ui.connectBtn.setText("Connect")
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(("localhost", 7031))

            if self.sock:
                self.ui.statusLabel.setPixmap(self.connectPixmap)
                self.ui.connectBtn.setText("Disconnect")
            else:
                self.ui.statusLabel.setPixmap(self.dontConnectPixmap)
                self.ui.connectBtn.setText("Connect")
        except Exception as e:
            cmds.confirmDialog(
                title='Katana!', message='failed connect.',
                icon='warning', button=['OK']
            )
            self.sock = None
        # sock.send("print 'Hello World!'")
        # sock.send("NodegraphAPI.CreateNode( 'Merge', NodegraphAPI.GetRootNode() )")


    #---------------------------------------------------------------------------
    # CLOSE
    #---------------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def clearCallback(self):
        if self.transformCallback:
            try:
                OpenMayaUI.MUiMessage.removeCallback(self.transformCallback)
            except:
                pass

    def closeEvent(self, event):
        print '# MtoK : Close event'
        # clean up callback
        self.clearCallback()

    ####################### TEMP #######################
    def testSendBtnClicked(self):
        if not self.sock:
            return

        command = str(self.ui.testCommandEdit.text())
        self.sock.send(command)


def main():
    if cmds.window(__windowName__, exists=True, q=True):
        cmds.deleteUI(__windowName__)

    window = dxsMTKMain()
    window.setObjectName(__windowName__)
    window.show()
