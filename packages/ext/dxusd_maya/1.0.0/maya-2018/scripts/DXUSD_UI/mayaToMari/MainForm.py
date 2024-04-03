#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2019.01.08'
__comment__ = 'maya to mari object sender'
__windowName__ = "dxsMTM"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

import maya.cmds as cmds
import maya.api.OpenMayaUI as OpenMayaUI
import maya.api.OpenMaya as OpenMaya

from MainFormUI import Ui_Form

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

import socket
import os
currentDir = os.path.dirname(__file__)

import dxExportMesh


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

def writeUserMel(*args):
    #create userSetup Mel to keep port open over maya sessions
    scriptsPath = cmds.internalVar(userScriptDir=True)
    name_of_file = "userSetup.mel"
    completeName = os.path.join(scriptsPath, name_of_file)

    appendText='commandPort -n "localhost:6010" -sourceType "python";'
    try:
        with open(completeName, "a+") as myfile:
            appendTextExists = False
            lines = myfile.readlines()
            #getting rid of '\n'
            lines = map(lambda s: s.strip(), lines)
            #Seek in the lines of the file userSetup.mel for the command that you want to append.
            for line in lines:
                if line == appendText:
                    appendTextExists = True

            #Append the command to the userSetup.mel
            if appendTextExists != True:
                myfile.write(appendText+'\n')
    except:
        pass

class dxsMTMMain(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.connectPixmap = QtGui.QPixmap("%s/resources/Circle03-Green.png" % currentDir)
        self.dontConnectPixmap = QtGui.QPixmap("%s/resources/Circle04-DarkRed.png" % currentDir)

        self.mariHost = "localhost"
        self.mariPort = 6100

        if cmds.commandPort("%s:%s" % (self.mariHost, self.mariPort), q=True):
            cmds.commandPort(name="%s:%s" % (self.mariHost, self.mariPort), cl=True)
            cmds.commandPort(name="%s:%s" % (self.mariHost, self.mariPort), sourceType="python")
            writeUserMel()

        self.sock = None

        # self.ui.connectBtn.clicked.connect(self.connectKatana)
        self.ui.sendObjBtn.clicked.connect(self.exportTmpObj)
        # self.ui.findPathBtn.clicked.connect(self.exportTempCacheClicked)

        self.connectMari()

    #---------------------------------------------------------------------------
    # CACHE OUT
    #---------------------------------------------------------------------------
    def exportTmpObj(self):
        self.sendMsg('print ("hello World!")\x04')

        scenePath = cmds.file( q=True, sn=True )
        if not scenePath:
            scenePath = cmds.workspace(q=True, rd=True)

        if scenePath.startswith('/netapp/dexter/show'):
            scenePath = scenePath.replace('/netapp/dexter/show', '/show')

        splitScenePath = scenePath.split("/")
        show = "unknown"
        assetName = self.ui.assetNamePath.text()
        if "show" in splitScenePath:
            show = splitScenePath[splitScenePath.index("show") + 1]

        expOutDir = os.path.join(os.environ['HOME'], "dxsMTM", show, assetName)

        outputFile = os.path.join(expOutDir, "%s_tx.abc" % assetName)
        expMesh = dxExportMesh.ExportMesh(outputFile, cmds.ls(sl=True, l=True))
        # expMesh.mesh_export(maya=False, abc=False, tx=True)

        expMesh.uvClass = dxExportMesh.UVLayOut()
        txlayout = expMesh.uvClass.layoutInfo()
        txindexs = []
        for layer in txlayout:
            txindexs.append(txlayout[layer]['txindex'][0])
        txindexs = list(set(txindexs))

        # texture objects
        for i in txlayout:
            for o in txlayout[i]['members']:
                expMesh.textureObjects += cmds.listRelatives(o, f=True, p=True)

        if len(txindexs) > 1 or len(txlayout.keys()) == 1:
            file_txabc = outputFile
            if not os.path.exists(os.path.dirname(file_txabc)):
                os.makedirs(os.path.dirname(file_txabc))

            # texture uv
            expMesh.alembic_tex_export(file_txabc, cmds.ls(sl=True, l=True))
            # texture uvSetsself.sendMsg
            if expMesh.uvClass.uvSets:
                for u in expMesh.uvClass.uvSets:
                    # uvSet
                    for i in expMesh.uvClass.uvSets[u]:
                        cmds.polyUVSet(i, currentUVSet=True, uvSet=u)
                    fn = file_txabc.replace('_tx.abc', '_%s.abc' % u)
                    expMesh.alembic_tex_export(fn, expMesh.uvClass.uvSets[u])

                # undo uvSet
                for u in expMesh.uvClass.uvSets:
                    for i in expMesh.uvClass.uvSets[u]:
                        cmds.polyUVSet(i, currentUVSet=True, uvSet='map1')
            # attributes
            file_txattr = file_txabc.replace('.abc', '.json')
            log = expMesh.texturelayerinfo_export(file_txattr)
        else:
            cmds.confirmDialog("don't have displayLayer.")

        # Initial the creation project in Mari.
        if self.ui.newRadio.isChecked():
            self.sendMsg('dxMtm.setupProject("%s", "%s", "%s")\x04' % (show, assetName, outputFile))
            # mari.close()
        # elif self.ui.addObjRadio.isChecked():
        #     self.sendMsg('dxMtm.addObject("%s")\x04' % outputFile)
        elif self.ui.addVerRadio.isChecked():
            self.sendMsg('dxMtm.addVersion("%s", "%s")\x04' % (outputFile, self.ui.versionEdit.text()))

    def sendMsg(self, cmd):
        mari = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mari.connect((self.mariHost, self.mariPort))
        mari.send(cmd)
        print "connect ok"
        mari.close()

    #---------------------------------------------------------------------------
    # CONNECT MARI
    #---------------------------------------------------------------------------
    def connectMari(self):
        try:
            self.sendMsg('print ("establishing connection with mGo Maya...")\x04')
            self.ui.statusLabel.setPixmap(self.connectPixmap)
        except:
            self.ui.statusLabel.setPixmap(self.dontConnectPixmap)
            self.ui.projectGrpBox.setTitle("Project")
            return

        scenePath = cmds.file(q=True, sn=True)
        if not scenePath:
            scenePath = cmds.workspace(q=True, rd=True)

        if scenePath.startswith('/netapp/dexter/show'):
            scenePath = scenePath.replace('/netapp/dexter/show', '/show')

        splitScenePath = scenePath.split("/")

        if "show" in splitScenePath:
            show = splitScenePath[splitScenePath.index("show") + 1]
        else:
            show = "unknown"

        self.ui.projectGrpBox.setTitle("Project : %s" % show)

    #---------------------------------------------------------------------------
    # CLOSE
    #---------------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        print '# MtoM : Close event'
        # clean up callback
        # self.clearCallback()

def main():
    if cmds.window(__windowName__, exists=True, q=True):
        cmds.deleteUI(__windowName__)

    window = dxsMTMMain()
    window.setObjectName(__windowName__)
    window.show()
