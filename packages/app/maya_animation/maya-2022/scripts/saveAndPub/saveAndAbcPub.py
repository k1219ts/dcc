# encoding=utf-8
# !/usr/bin/env python

# -------------------------------------------------------------------------------
#
#   Dexter Rigging Team
#
#		taehoon.kim
#
#	2016.08.23
# -------------------------------------------------------------------------------
import os
import string
import shutil, errno
import json
from PySide2 import QtCore, QtWidgets, QtGui, load_ui
import subprocess
import maya.cmds as cmds
import maya.mel as mm
import getpass

import ANI_common
reload(ANI_common)
import GH_RefGpuSwitchUI_dexcmd.GH_RefGpuSwitchUI_dexcmd as GHrgs #reload(GHrgs)
#import dexcmd.alembicBatchExport as alembicBatchExport
import saveAndPub.tractorSpool as tractorSpool
# reload(tractorSpool)
import tactic_checkin
#reload(tactic_checkin)

currentpath = os.path.abspath(__file__)
UIROOT = os.path.dirname(currentpath)
uiFile = os.path.join(UIROOT, 'ui/saveAndAbcPub.ui')
PREPUB_PREFIX = "INPROGRESS"
SHOW_PATH = "/show"
#SHOW_PATH = '/home/gyeongheon.jeong/maya/projects/show'
STATUS = ['Approved', 'Review', 'In-Progress' ]

def hconv(text):
    return unicode(text, 'utf-8')

class saveAndAbcPub(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(saveAndAbcPub, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.copyProcess = QtCore.QProcess()
        self.ui.setWindowTitle('Save And Alembic Pub Tool')
        #txtBrwsrFont = self.textBrowser.font()
        #txtBrwsrFont.setPointSize(9)
        #self.textBrowser.setFont(txtBrwsrFont)
        #self.textBrowser.setHidden(True)
        self.ui.refresh_Button.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(UIROOT, 'res/cache.png'))))
        self.ui.browse_Button.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(UIROOT, 'res/folder_open.png'))))
        self.ui.showList_comboBox.setEnabled(False)
        self.ui.seqList_comboBox.setEnabled(False)
        self.ui.shotList_listWidget.setEnabled(False)
        self.ui.sceneList_listWidget.setEnabled(False)
        self.ui.showList_comboBox.addItems(ANI_common.getListDirs(SHOW_PATH))
        self.ui.status_comboBox.addItems(STATUS)
        self.ui.password_lineEdit.setEchoMode(QtGui.QLineEdit.Password)
        #self.ui.bakeConstraint_checkBox.setChecked(False)
        #self.ui.bakeConstraint_checkBox.setEnabled(False)
        self.ui.cleanup_checkBox.setChecked(True)
        self.ui.cleanup_checkBox.setEnabled(True)

        if cmds.optionVar(ex="tacticUser"):
            self.ui.user_lineEdit.setText(cmds.optionVar(q="tacticUser"))
        else:
            self.ui.user_lineEdit.setText(getpass.getuser())
            cmds.optionVar(sv=("tacticUser", getpass.getuser()))

        if cmds.optionVar(ex="tacticPassword"):
            self.ui.password_lineEdit.setText(cmds.optionVar(q="tacticPassword"))

        self.updateData()
        self.connectSignals()

    def connectSignals(self):
        self.ui.save_pushButton.clicked.connect(self.save_run)
        self.ui.radioButton_fromCurrent.toggled.connect(self.radioBtnToggle)
        self.ui.showList_comboBox.currentIndexChanged.connect(self.reloadSeqList)
        self.ui.seqList_comboBox.currentIndexChanged.connect(self.reloadShotList)
        self.ui.shotList_listWidget.itemClicked.connect(self.reloadSceneList)
        self.ui.savePublish_pushButton.clicked.connect(self.copy_publish)
        self.ui.browse_Button.clicked.connect(self.browseSceneFolders)
        self.ui.RGSR_pushButton.clicked.connect(self.referenceGPU_switch)
        self.ui.connect(self.copyProcess, QtCore.SIGNAL('finished(int, QProcess::ExitStatus)'),
                     self.finishProcess)
        self.copyProcess.readyReadStandardOutput.connect(self.readOutPut)
        self.copyProcess.readyReadStandardError.connect(self.readError)
        self.ui.user_lineEdit.textChanged.connect(self.loginChanged)
        self.ui.password_lineEdit.textChanged.connect(self.loginChanged)
        self.ui.login_pushButton.clicked.connect(self.tacticLogin)

    def tacticLogin(self):
        username =  str(self.ui.user_lineEdit.text())
        password = str(self.ui.password_lineEdit.text())
        self.checkIn = tactic_checkin.tactic_checkin()
        result = self.checkIn.tacticLogin(user=username, password=password)

        if result:
            self.ui.login_groupBox.setHidden(True)

    def loginChanged(self):
        user = str(self.ui.user_lineEdit.text())
        password = str(self.ui.password_lineEdit.text())

        cmds.optionVar(sv=("tacticUser", user))
        cmds.optionVar(sv=("tacticPassword", password))

    def readOutPut(self):
        #self.ui.textBrowser.append(unicode(self.copyProcess.readAllStandardOutput(), 'utf-8'))
        str = unicode(self.copyProcess.readAllStandardOutput(), 'utf-8')
        print str
        #mm.eval('print "{:s}";'.format(str))

    def readError(self):
        #self.ui.textBrowser.append(unicode(self.copyProcess.readAllStandardError(), 'utf-8'))
        str = unicode(self.copyProcess.readAllStandardError(), 'utf-8')
        print str
        #mm.eval('print "{:s}";'.format(str))

    def finishProcess(self, exitCode, status):
        #self.ui.textBrowser.append("exit code : {0}, exit status : {1}".format(exitCode, status))
        print "exit code : {0}, exit status : {1}".format(exitCode, status)
        QtGui.QMessageBox.information(self, "finish", "")

    def reloadSeqList(self):
        project = str(self.ui.showList_comboBox.currentText())
        try:
            self.showPath = os.sep.join([SHOW_PATH,project, "shot"])
            seqList = ANI_common.getListDirs(self.showPath)
            seqList.sort()
            self.ui.seqList_comboBox.clear()
            self.ui.seqList_comboBox.addItems(seqList)
        except OSError:
            pass

    def reloadShotList(self):
        sequence = str(self.ui.seqList_comboBox.currentText())
        try:
            self.seqPath = os.sep.join([self.showPath, sequence])
            shotList = ANI_common.getListDirs(self.seqPath)
            shotList.sort()
            self.ui.shotList_listWidget.clear()
            self.ui.shotList_listWidget.addItems(shotList)
        except OSError:
            pass

    def reloadSceneList(self):
        shots = self.ui.shotList_listWidget.selectedItems()
        self.ui.sceneList_listWidget.clear()
        self.prePubScene = dict()

        for shot in shots:
            try:
                self.shotPath = os.sep.join([self.seqPath, str(shot.text()),
                                             "ani", "pub", "scenes"])
                sceneList = ANI_common.getListDirs(self.shotPath, isFile=True, fileType=".mb")
                for s in sceneList:
                    if s.find("INPROGRESS") != -1:
                        self.prePubScene[s] = self.shotPath
                        # {"test_shot_INPROGRESS.mb":".../ani/pub/scenes"}
            except OSError:
                pass
        self.ui.sceneList_listWidget.addItems(self.prePubScene.keys())

    def browseFolder(self, _dir):
        os.system('/usr/bin/nautilus %s &' % _dir)

    def browseSceneFolders(self):
        if self.prePubScene.values():
            for dir in self.prePubScene.values():
                self.browseFolder(dir)

    def updateData(self):
        currentScene = cmds.file(q=1, sn=1)

        try:
            self.currentSceneDir = os.sep.join( currentScene.split(os.sep)[:-1] )
            self.scenefile = cmds.file(q=1, sn=1).replace("/dev/", "/pub/")
            pathSplit = self.scenefile.split(os.sep)
            self.sceneDir = os.sep.join(pathSplit[:-1])
            self.pubDir = os.sep.join(pathSplit[:-2])
            self.shotNumDir = os.sep.join(pathSplit[:-4])
            self.sceneName = pathSplit[-1]
            self.showName =ANI_common.getShowShot(self.scenefile)[0]
            self.shotName = ANI_common.getShowShot(self.scenefile)[1]

            self.PubName = "_".join(self.sceneName.split("_")[:4])
            self.prePubName = "_".join([self.PubName, PREPUB_PREFIX])

            print "-------------------------------------------------------" * 5
            print "Find Maya File : \n" + os.sep.join([self.sceneDir, self.prePubName + ".mb"])
            print "-------------------------------------------------------" * 5

            if not os.path.exists(os.sep.join([self.sceneDir, self.prePubName + ".mb"])):
                self.prePubName = "No Published file"
                self.PubName = "Save & Publish First"

        except:
            self.PubName = "None"
            self.prePubName = "None"

        self.ui.label_OriName.setText(self.prePubName + ".mb")
        self.ui.label_NewName.setText(self.PubName + ".mb")

    def radioBtnToggle(self):
        if self.ui.radioButton_fromCurrent.isChecked():
            self.updateData()
            self.ui.showList_comboBox.setEnabled(False)
            self.ui.seqList_comboBox.setEnabled(False)
            self.ui.shotList_listWidget.setEnabled(False)
            self.ui.sceneList_listWidget.setEnabled(False)
        else:
            self.ui.label_OriName.setText("")
            self.ui.label_NewName.setText("")
            self.ui.showList_comboBox.setEnabled(True)
            self.ui.seqList_comboBox.setEnabled(True)
            self.ui.shotList_listWidget.setEnabled(True)
            self.ui.sceneList_listWidget.setEnabled(True)

    def backgroundMaya(self, filename):
        virEnv = os.environ.copy()

        command = ['{0}/backGroundMaya'.format(UIROOT), '{0}'.format(filename)]

        p = subprocess.Popen(command, env=virEnv, shell=False)
        p.communicate()

    def save_run(self):
        curSceneName = cmds.file(q=True, sn=True)
        filePathList = curSceneName.split(os.sep)
        showIndex = filePathList.index("shot") + 4

        fileName = os.path.basename(curSceneName)
        # [u'OPN', u'0820', u'ani', u'v01']
        fileNameList = os.path.splitext(fileName)[0].split('_')[:4]

        # versionF = "_".join(fileNameList) # OPN_0820_ani_v01
        versionF = "_".join(fileNameList + [PREPUB_PREFIX])  # OPN_0820_ani_v01_INPROGRESS
        mainPath = os.sep.join(filePathList[:showIndex + 1]).replace("dev", "pub")
        mayaFileName = os.sep.join([mainPath, "scenes", versionF + ".mb"])
        dataPath = os.sep.join([mainPath, "data"])
        cameraFileName = os.sep.join([dataPath, "cam", versionF])
        logFileName = os.sep.join([dataPath, versionF + ".json"])

        timewarpNode = None
        saveFileDialog = QtGui.QInputDialog()
        saveFiles, ok = saveFileDialog.getText(self, "Save As", "Enter New Filename",
                                               QtGui.QLineEdit.Normal, fileName)

        ls_animLYR = cmds.ls(type='animLayer')

        if ls_animLYR:
            mm.eval('source buildSetAnimLayerMenu;')
            mm.eval('selectLayer("BaseAnimation");')

        if ok:
            if self.ui.retimeData_checkBox.isChecked():
                self.alembicStepVal = ANI_common.ExportSeqRetime(sim=True, path=mainPath,
                                                                 sceneName=versionF)
                print '# to Rigging'
            else:
                self.alembicStepVal, timewarpNode = ANI_common.ExportSeqRetime(sim=False, path=mainPath,
                                                                               sceneName=versionF)
                print '# to Render'

            print "# Retime Max Scale : {}".format(self.alembicStepVal)
            print "# Applied TimeWarp Node : {}\n".format(timewarpNode)

            newfileName = os.sep.join( [self.currentSceneDir, saveFiles] )
            if os.path.exists(newfileName):
                msg = QtGui.QMessageBox.warning(self, "Warning!",
                                          hconv("같은 이름의 파일이 있습니다. 덮어쓰시겠습니까?"),
                                          QtGui.QMessageBox.Ok, QtGui.QMessageBox.Cancel)
                if msg != QtGui.QMessageBox.Ok: return

            cmds.file(rn=newfileName)
            renamedFile = cmds.file(save=True)
            print '# File saved as : %s' % renamedFile

            # Spool
            self.tractorSpoolRun(curSceneName=curSceneName, cameraFileName=cameraFileName,
                                 logFileName=logFileName, mainPath=mainPath,
                                 mayaFileName=mayaFileName)

            if timewarpNode:
                cmds.delete(timewarpNode)

            if self.ui.tractorOpen_checkBox.isChecked():
                FF = subprocess.Popen(['firefox', '10.0.0.30'])


    def editJsonValue(self, jsonDic=dict(), key=str()):
        value = jsonDic[key]
        newValue = value
        if isinstance(value, unicode):
            value = str(value)
            if value.find("_INPROGRESS") != -1:
                newValue = value.replace("_INPROGRESS", "")

        print 'Change value : "{value}"   To \n\t"{newValue}"'.format(value=value, newValue=newValue)
        return newValue

    def editJson(self, jsonDic):
        if isinstance(jsonDic, dict):
            for key in jsonDic:
                newKey = key

                if isinstance(jsonDic[key], list):
                    newkeyList = list()
                    for i in jsonDic[key]:
                        if i.find("_INPROGRESS") != -1:
                            i_edit = i.replace("_INPROGRESS", "")
                            newkeyList.append(i_edit)
                            print 'Change value : "{value}"   To \n\t"{newValue}"'.format(value=i, newValue=i_edit)
                        else:
                            newkeyList.append(i)
                    jsonDic[key] = newkeyList
                else:
                    if key.find("_INPROGRESS") != -1:
                        newKey = key.replace("_INPROGRESS", "")
                        jsonDic[newKey] = jsonDic.pop(key)
                        print 'Change value : "{value}"   To \n\t"{newValue}"'.format(value=key, newValue=newKey)

                    value = self.editJsonValue(jsonDic, newKey)
                    jsonDic[newKey] = value
                    self.editJson(jsonDic[newKey])

        return jsonDic

    def jsonCopy(self, jsonFile, newJsonFile):
        with open(jsonFile, 'r') as f:
            Hjson = json.load(f)

        NewJsonDic = self.editJson(Hjson)

        with open(newJsonFile, 'w') as f:
            json.dump(NewJsonDic, f, indent=4)
            f.close()

    def copy_publish(self):
        if self.ui.radioButton_fromCurrent.isChecked():
            project_code = self.checkIn.getProjectCode(self.showName)  # ie. show1

            infoDic = self.copyfiles(prepubScene=self.prePubName,
                                     pubdir=self.pubDir,
                                     shotnumdir=self.shotNumDir,
                                     scenedir=self.sceneDir)
            shotName = ANI_common.getShowShot(infoDic['scenePath'])[1]
            note = self.checkIn.createNote(infoDic)
            self.checkIn.checkin(user=str(self.ui.user_lineEdit.text()), project_code=project_code,
                                 shotName=shotName, checkin_file=infoDic["previewPath"],
                                 note=note, status=str(self.ui.status_comboBox.currentText()))
        else:
            for prepubScene in self.prePubScene.keys(): # WEP_1220_ani_v01_INPROGRESS.mb
                prepubScenePath = self.prePubScene[prepubScene] # ".../pub/scenes"
                prepubPath = os.sep.join(prepubScenePath.split(os.sep)[:-1]) # ".../pub"
                shotNumPath = os.sep.join(prepubScenePath.split(os.sep)[:-3]) # ".../WEP_1220

                infoDic = self.copyfiles(prepubScene=os.path.splitext(prepubScene)[0], # WEP_1220_ani_v01_INPROGRESS
                                         pubdir=prepubPath,
                                         shotnumdir=shotNumPath,
                                         scenedir=prepubScenePath)

                showName = ANI_common.getShowShot(infoDic['scenePath'])[0]
                shotName = ANI_common.getShowShot(infoDic['scenePath'])[1]
                project_code = self.checkIn.getProjectCode(showName)  # ie. show1

                note = self.checkIn.createNote(infoDic)
                self.checkIn.checkin(user=str(self.ui.user_lineEdit.text()), project_code=project_code,
                                     shotName=shotName, checkin_file=infoDic["previewPath"],
                                     note=note, status=str(self.ui.status_comboBox.currentText()))

    def copyfiles(self, prepubScene=None, pubdir=None, shotnumdir=None, scenedir=None):
        '''
        copy [INPROGRESS] files and rename
        scene path          : /show/kfyg/shot/CHS/CHS_0465/ani/pub/scenes/CHS_0465_ani_v01_INPROGRESS.mb
        cache path          : /show/kfyg/shot/CHS/CHS_0465/ani/pub/data/geoCache/CHS_0465_ani_v01_INPROGRESS
        cam path            : /show/kfyg/shot/CHS/CHS_0465/ani/pub/data/cam/CHS_0465_ani_v01_INPROGRESS
        retime path         : /show/kfyg/shot/CHS/CHS_0465/ani/pub/data/retime/CHS_0465_ani_v01_INPROGRESS_retime.json
        json path           : /show/kfyg/shot/CHS/CHS_0465/ani/pub/data/CHS_0465_ani_v01_INPROGRESS.json
        preview path        : /show/kfyg/shot/CHS/CHS_0465/ani/pub/data/preview/CHS_0465_ani_v01_INPROGRESS_STAMP.mov
        zenn path           : /show/kfyg/shot/CHS/CHS_0465/hair/pub/data/zenn/CHS_0465_ani_hair_v01_INPROGRESS
        zenn json path      : /show/kfyg/shot/CHS/CHS_0465/hair/pub/data/CHS_0465_ani_hair_v01_INPROGRESS.json
        zenn preview path   : /show/kfyg/shot/CHS/CHS_0465/hair/pub/data/preview/CHS_0465_ani_v01_INPROGRESS_STAMP.mov
        '''

        env = os.environ.copy()
        envList = []
        for key, value in env.iteritems():
            temp = "{0}={1}".format(key, value)
            envList.append(temp)

        self.copyProcess.setEnvironment(envList)

        sceneName = prepubScene
        hairSceneNameList = prepubScene.split("_")
        aniIndex = hairSceneNameList.index("ani")
        hairSceneNameList.insert(aniIndex + 1, "hair")
        hairSceneName = string.join(hairSceneNameList, "_")

        pubDataPath = os.sep.join([pubdir, "data"])
        hairDataPath = os.sep.join([shotnumdir, "hair", "pub", "data"])

        pathDict = dict()
        pathDict[sceneName] = dict()

        pathDict[sceneName]["scenePath"] = os.sep.join([scenedir, (sceneName + ".mb")])
        pathDict[sceneName]["cachePath"] = os.sep.join([pubDataPath, "geoCache", sceneName])
        pathDict[sceneName]["camPath"] = os.sep.join([pubDataPath, "cam", sceneName])
        pathDict[sceneName]["retimePath"] = os.sep.join([pubDataPath, "retime", (sceneName + "_retime" + ".json")])
        pathDict[sceneName]["jsonPath"] = os.sep.join([pubDataPath, (sceneName + ".json")])
        pathDict[sceneName]["previewPath"] = os.sep.join([pubDataPath, "preview", (sceneName + "_STAMP.mov")])

        pathDict[sceneName]["zennPath"] = os.sep.join([hairDataPath, "zenn", hairSceneName])
        pathDict[sceneName]["zennJsonPath"] = os.sep.join([hairDataPath, (hairSceneName + ".json")])
        pathDict[sceneName]["zennPreviewPath"] = os.sep.join([hairDataPath, "preview", (hairSceneName + "_STAMP.mov")])

        #self.copyProcess.start("/usr/bin/bash")

        print "\n\n"
        for key_A in pathDict.keys():
            for key_B in pathDict[key_A].keys():
                if os.path.exists(pathDict[key_A][key_B]):
                    oldFile = pathDict[key_A][key_B]
                    newFile = pathDict[key_A][key_B].replace( "_"+PREPUB_PREFIX, "")

                    resultTxt = "----------------------------------" * 10 + "\n"
                    resultTxt +=  " # Copy Files and Folders\n"
                    resultTxt += " {path} :\n".format(path=key_B.upper())
                    resultTxt += "      {oldFile}\n".format(oldFile=oldFile)
                    resultTxt += " TO :\n"
                    resultTxt += "      {newFile}\n".format(newFile=newFile)
                    resultTxt += "----------------------------------" *10 + "\n\n"

                    #self.ui.textBrowser.append(resultTxt)
                    print resultTxt

                    addnum = 1
                    revPath = newFile

                    while os.path.exists(revPath):
                        if os.path.isfile(revPath):
                            revPath = os.path.splitext(newFile)[0]
                            revPath += "_rev_{:02d}".format( addnum )
                            revPath += os.path.splitext(newFile)[1]
                        else:
                            revPath = string.join(newFile.split("_")[:-1], "_")
                            revPath += "_rev_{:02d}".format( addnum )
                        addnum += 1

                    if not newFile == revPath:
                        os.rename(newFile, revPath)

                    if oldFile.endswith(".json"):
                        self.jsonCopy(oldFile, newFile)
                    else:
                        if os.path.isdir(oldFile):
                            if key_B == "camPath":
                                if not os.path.exists(newFile):
                                    os.makedirs(newFile)

                                listCamFiles = os.listdir(oldFile)

                                for cam in listCamFiles:
                                    oldCamName = os.sep.join( [oldFile, cam] )
                                    newCamName = os.sep.join( [newFile, cam.replace("_" + PREPUB_PREFIX, "")] )
                                    arg = 'rsync -av --progress {0} {1}\n'.format(oldCamName, newCamName)
                                    p = subprocess.Popen(arg, shell=True)
                                    p.communicate()
                            else:
                                #arg = 'rsync -av --progress {0}/ {1}/\n'.format(oldFile, newFile)
                                arg = 'gnome-terminal'
                                arg += ' -e "rsync -av --progress {0}/ {1}/"\n'.format(oldFile, newFile)
                                p = subprocess.Popen(arg, shell=True)
                                p.communicate()
                        else:
                            arg = 'rsync -av --progress {0} {1}\n'.format(oldFile, newFile)
                            #arg = 'gnome-terminal'
                            #arg += ' -e "rsync -av --progress {0} {1}"\n'.format(oldFile, newFile)
                            p = subprocess.Popen(arg, shell=True)
                            #p.wait()
                            p.communicate()
                        #self.copyProcess.write(arg)
                        #self.copyProcess.waitForFinished()
                    if key_B == "previewPath":
                        pathDict[key_A][key_B] = newFile
                else:
                    resultTxt = "# Failed To Copy File.\n"
                    resultTxt += "# {}\n".format( pathDict[key_A][key_B] )
                    resultTxt += "# '{}' Not exists.\n\n".format( key_B )
                    print resultTxt
        return pathDict[sceneName]

        #QtGui.QMessageBox.information(self, "finish", "")
        #self.copyProcess.closeWriteChannel()
        #self.copyProcess.waitForFinished()

    def tractorSpoolRun(self, curSceneName, mayaFileName, mainPath, cameraFileName, logFileName):
        #filePathList = curSceneName.split(os.sep)

        if not os.path.exists(os.sep.join(mayaFileName.split(os.sep)[:-1])):
            os.makedirs(os.sep.join(mayaFileName.split(os.sep)[:-1]))

        shutil.copy2(curSceneName, mayaFileName)
        # /show/kfyg/shot/OPN/OPN_0820/ani/dev/data/OPN_0820_ani_v01.json

        RN_node = cmds.ls(rn=True, type='dxRig')
        '''
        meshTypeList = []
        if self.ui.mesh_render_checkBox.isChecked():
            meshTypeList.append('render')
        elif self.ui.mesh_mid_checkBox.isChecked():
            meshTypeList.append('mid')
        elif self.ui.mesh_low_checkBox.isChecked():
            meshTypeList.append('low')
        elif self.ui.mesh_simulation_checkBox.isChecked():
            meshTypeList.append('sim')
        meshType = string.join(meshTypeList, ",")
        '''
        meshType = 'render,mid,low,sim'
        strRN = string.join(RN_node, ',')

        # Spool
        if self.ui.separate_checkBox.isChecked():
            self.characteSeparateState = True
        else:
            self.characteSeparateState = False

        if self.ui.nextHairTask_checkBox.isChecked():
            self.nextHairTaskState = True
        else:
            self.nextHairTaskState = False

        cleanupQ = self.ui.cleanup_checkBox.isChecked()

        if cleanupQ:
            try:
                mm.eval('selectLayer("BaseAnimation");')
            except:
                pass
            self.backgroundMaya(mayaFileName)

        tractorSpool.tractorSpool(file=mayaFileName, outPath=mainPath, camPath=cameraFileName,
                                  absPath=False, step=self.alembicStepVal, mesh=True,
                                  meshType=meshType, node=strRN, camera=True,
                                  preview=True, stamp=True, zenn=True, layout=True,
                                  nextHairTask=self.nextHairTaskState,
                                  host='tractor', separate=self.characteSeparateState, revision=False,
                                  offLogWrite=False, logfile=logFileName, dbinsert=True,
                                  cleanup=False)

    def referenceGPU_switch(self):
        GHrgs.GH_RefGpuSwitchUI().DoSwitch()

    def openFolder(self):
        subprocess.Popen(['xdg-open', self.ui.alembicPath_lineEdit.text()])


def OPEN():
    global win
    try:
        win.close()
    except:
        pass

    win = saveAndAbcPub()
    win.ui.show()
    win.ui.resize(500, 700)