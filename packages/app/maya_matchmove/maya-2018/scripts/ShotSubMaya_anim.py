
#
# import modules
import os
import sys
import glob
import json

from PySide2 import QtCore, QtGui, QtUiTools, QtWidgets
import ui_shotsub_anim_widget
reload(ui_shotsub_anim_widget)

import maya.cmds as mc
import maya.mel as mm

import shotDB_common
import CameraAsset

import pymongo
from pymongo import MongoClient
DB_IP = "10.0.0.12:27017, 10.0.0.13:27017"
DB_NAME = 'PIPE_PUB'
#
# predefined variables
SHOWROOT = "/show"

windowObject = "shotsub v1.0"
dockMode = False


#
# main class
class ShotSubMaya_anim(QtWidgets.QDialog):
    def __init__(self, parent=shotDB_common.get_maya_window()):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = ui_shotsub_anim_widget.Ui_shotsub_anim_Dialog()
        self.ui.setupUi(self)

        self.setObjectName(windowObject)

        self.initGUI()
        self.connectSlot()


    def initGUI(self):
        # initialize show/seq/shot
        self.showList = shotDB_common.get_dir_list(SHOWROOT)
        self.showList = shotDB_common.reorder_list(self.showList, "prat")
        self.ui.showUI.addItems(self.showList)


    def connectSlot(self):
        # self.ui.showUI.currentIndexChanged.connect(self.reloadShow)
        # self.ui.seqUI.currentIndexChanged.connect(self.reloadSeq)
        # self.ui.shotUI.currentIndexChanged.connect(self.reloadShot)
        # self.ui.plateTypeUI.currentIndexChanged.connect(self.changePlateType)
        # self.ui.sceneUI.currentIndexChanged.connect(self.reloadSceneInfo)
        self.ui.showUI.activated.connect(self.reloadShow)
        self.ui.seqUI.activated.connect(self.reloadSeq)
        self.ui.shotUI.activated.connect(self.reloadShot)
        self.ui.plateTypeUI.activated.connect(self.changePlateType)
        self.ui.sceneUI.activated.connect(self.reloadSceneInfo)
        self.ui.doIt.clicked.connect(self.importScene)

    def reloadShow(self):
        self.show = str(self.ui.showUI.currentText())
        self.seqList = shotDB_common.get_dir_list(os.path.join(SHOWROOT, self.show, "shot"))
        self.ui.seqUI.clear()
        self.ui.seqUI.addItems(self.seqList)

        self.reloadSeq()


    def reloadSeq(self):
        self.seq = str(self.ui.seqUI.currentText())
        self.shotList = shotDB_common.get_dir_list(os.path.join(SHOWROOT, self.show, "shot", self.seq))
        self.ui.shotUI.clear()
        self.ui.shotUI.addItems(self.shotList)

        self.reloadShot()


    def reloadShot(self):
        self.reloadPlateType()


    def reloadPlateType(self):
        self.shot = str(self.ui.shotUI.currentText())
        self.plateTypeList = shotDB_common.get_dir_list(os.path.join(SHOWROOT,self.show, "shot", self.seq, self.shot, "plates"))
        if self.plateTypeList[0].startswith == "No ":
            self.plateTypeList = ["No plates"]
        else:
            self.plateTypeList = shotDB_common.reorder_list(self.plateTypeList, "main")

        self.ui.plateTypeUI.clear()
        self.ui.plateTypeUI.addItems(self.plateTypeList)

        self.reloadScene()

    def reloadScene(self):
        print "def reloadScene"
        self.plateType = str(self.ui.plateTypeUI.currentText())
        self.scenePath = os.path.join(SHOWROOT,self.show, "shot", self.seq, self.shot, "matchmove", "pub", "scenes")

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[self.show]

        result = coll.find({'show':self.show,
                            'shot':self.shot,
                            'task':'matchmove',
                            'task_publish.plateType':self.plateType
                            })
        self.ui.sceneUI.clear()

        self.sceneDic = {}
        if result.count():
            for i in result:
                filepath = i['files']['maya_pub_file'][0]
                filename = os.path.basename(filepath)
                self.ui.sceneUI.addItem(filename)
                i['source'] = 'DB'
                self.sceneDic[filename] = i
                self.sdb_publisher = i["artist"]
                self.sdb_startFrame = i["task_publish"]["startFrame"]
                self.sdb_endFrame = i["task_publish"]["endFrame"]
                self.sdb_renderWidth = i["task_publish"]["renderWidth"]
                self.sdb_renderHeight = i["task_publish"]["renderHeight"]
                self.sdb_overscan = i["task_publish"]["overscan"]
                self.sdb_stereo = i["task_publish"]["isStereo"]
                self.sdb_mayaCam = i["files"]["camera_path"][0]
                self.sdb_mayaScene = i["files"]["camera_asset_geo_path"]
                self.sdb_plate = i["task_publish"]["plateType"]

        elif shotDB_common.get_file_list(self.scenePath, filename=self.shot+"_"+self.plateType, ext="*.mb"):
            self.sceneList = shotDB_common.get_file_list(self.scenePath,
                                                         filename=self.shot + "_" + self.plateType,
                                                         ext="*.mb")
            for scene in self.sceneList:

                self.readJSON(scene)
            self.ui.sceneUI.addItems(self.sceneList)

        else:
            self.ui.sceneUI.addItem("No pub scenes in Database")
        self.reloadSceneInfo()

    def changePlateType(self):
        self.reloadScene()

    def clearSceneInfoUI(self):
        self.ui.publisherUI.clear()
        self.ui.verUI.clear()
        self.ui.dateUI.clear()
        self.ui.startFrameUI.clear()
        self.ui.endFrameUI.clear()
        self.ui.renderHeightUI.clear()
        self.ui.renderWidthUI.clear()
        self.ui.overscanUI.setCheckState(QtCore.Qt.Unchecked)
        self.ui.stereoUI.setCheckState(QtCore.Qt.Unchecked)

    def reloadSceneInfo(self):
        self.scene = str(self.ui.sceneUI.currentText())
        if self.scene.startswith("No "):
            self.clearSceneInfoUI()
            return 0
        sceneInfo = self.sceneDic[self.scene]

        if sceneInfo['source'] == 'DB':
            self.ui.publisherUI.setText(sceneInfo['artist'])
            self.ui.verUI.setText(str(sceneInfo['version']))
            self.ui.dateUI.setText(sceneInfo['time'].split('.')[0])
            #self.ui.commentUI.setText(self.sdb_comment)
            self.ui.startFrameUI.setText(str(sceneInfo['task_publish']['startFrame']))
            self.ui.endFrameUI.setText(str(sceneInfo['task_publish']['endFrame']))
            self.ui.renderWidthUI.setText(str(sceneInfo['task_publish']['renderWidth']))
            self.ui.renderHeightUI.setText(str(sceneInfo['task_publish']['renderHeight']))

            if sceneInfo['task_publish']['overscan']:
                self.ui.overscanUI.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.overscanUI.setCheckState(QtCore.Qt.Unchecked)

            if sceneInfo['task_publish']['isStereo']:
                self.ui.stereoUI.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.stereoUI.setCheckState(QtCore.Qt.Unchecked)

        elif sceneInfo['source'] == 'JSON':
            self.ui.publisherUI.setText(self.sdb_publisher)
            self.ui.verUI.setText(self.sdb_version)
            self.ui.dateUI.setText(self.sdb_date)
            #self.commentUI.setText(self.sdb_comment)
            self.ui.startFrameUI.setText(self.sdb_startFrame)
            self.ui.endFrameUI.setText(self.sdb_endFrame)
            self.ui.renderWidthUI.setText(self.sdb_renderWidth)
            self.ui.renderHeightUI.setText(self.sdb_renderHeight)

            if self.sdb_overscan:
                self.ui.overscanUI.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.overscanUI.setCheckState(QtCore.Qt.Unchecked)
            if self.sdb_stereo:
                self.ui.stereoUI.setCheckState(QtCore.Qt.Checked)
            else:
                self.ui.stereoUI.setCheckState(QtCore.Qt.Unchecked)

    def readJSON(self, sceneName):
        self.sdb_publisher = "No shotdb data"
        self.sdb_version = ""
        self.sdb_date = ""
        self.sdb_comment = ""
        self.sdb_version = ""
        self.sdb_startFrame = ""
        self.sdb_endFrame = ""
        self.sdb_renderWidth = ""
        self.sdb_renderHeight = ""
        self.sdb_overscan = False
        self.sdb_stereo = False
        shotdbName = sceneName.replace(".mb", ".shotdb")
        self.jsonPath = os.path.join(self.scenePath, shotdbName)

        #sceneName = sceneName.replace(".mb", ".shotdb")
        scenePath = os.path.join(self.scenePath, sceneName)

        print self.jsonPath
        if os.path.isfile(self.jsonPath):
            f = open(self.jsonPath, "r")
            j = json.load(f)
            j['source'] = 'JSON'
            #self.sceneDic[filename] = i
            self.sceneDic[sceneName] = j
            self.sdb_publisher = j["user"]
            self.sdb_date = j["date"]
            self.sdb_startFrame = j["startFrame"]
            self.sdb_endFrame = j["endFrame"]
            self.sdb_renderWidth = j["renderWidth"]
            self.sdb_renderHeight = j["renderHeight"]
            self.sdb_overscan = j["overscan"]
            self.sdb_stereo = j["stereo"]
            self.sdb_mayaCam = j["mayaCam"]
            self.sdb_mayaScene = j["mayaScene"]
            self.sdb_plate = j["plate"]



    def getUIData(self):
        self.show = str(self.ui.showUI.currentText())
        self.seq = str(self.ui.seqUI.currentText())
        self.shot = str(self.ui.shotUI.currentText())
        self.plateType = str(self.ui.plateTypeUI.currentText())
        self.scene = str(self.ui.sceneUI.currentText())

        self.startFrame = float(self.ui.startFrameUI.text())
        self.endFrame = float(self.ui.endFrameUI.text())
        self.renderWidth = int(self.ui.renderWidthUI.text())
        self.renderHeight = int(self.ui.renderHeightUI.text())

        self.stereo = self.ui.stereoUI.checkState()
        self.overscan = self.ui.overscanUI.checkState()
        self.layoutCamera = self.ui.layoutCameraUI.checkState()


    def importScene(self):
        self.getUIData()
        self.scene = str(self.ui.sceneUI.currentText())
        if self.scene.startswith("No "):
            return 0
        sceneInfo = self.sceneDic[self.scene]
        print "sceneInfo!!", sceneInfo

        if self.show.startswith("select "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select show first!", QtWidgets.QMessageBox.Ok)
            return False

        if self.show.startswith("No ") or self.seq.startswith("No ") or self.shot.startswith("No ") or self.plateType.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "There is no publish folder. Ask to your TD.", QtWidgets.QMessageBox.Ok)
            return False

        if self.plateType.startswith("select ") or self.plateType.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select shot!", QtWidgets.QMessageBox.Ok)
            return False

        if self.scene.startswith("select ") or self.scene.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select plate type!", QtWidgets.QMessageBox.Ok)
            return False

        # set render size and duration.
        mc.setAttr("defaultResolution.width", self.renderWidth)
        mc.setAttr("defaultResolution.height", self.renderHeight)
        mc.setAttr("defaultResolution.deviceAspectRatio", float(self.renderWidth)/float(self.renderHeight))
        mc.setAttr("defaultRenderGlobals.animation", 1)
        mc.setAttr("defaultRenderGlobals.extensionPadding", 4)
        mc.setAttr("defaultRenderGlobals.startFrame", self.startFrame)
        mc.setAttr("defaultRenderGlobals.endFrame", self.endFrame)
        mc.playbackOptions(ast=self.startFrame, aet=self.endFrame, min=self.startFrame, max=self.endFrame)
        mc.currentTime(self.startFrame)

        # import matchmove scene as reference.
        SDBNode = "%s_%s_matchmove_SDBNode"%(self.shot, self.plateType)
        self.sceneFile = os.path.join(self.scenePath, self.scene)

        try:
            mc.file(self.sceneFile,
                    type="mayaBinary",
                    i=False,
                    reference=True,
                    groupLocator=False,
                    groupReference=True,
                    groupName=SDBNode,
                    loadReferenceDepth="all",
                    sharedNodes="renderLayersByName",
                    preserveReferences=True,
                    mergeNamespacesOnClash=True,
                    namespace="%s_%s_matchmove"%(self.shot, self.plateType),
                    options="v=0")

            mc.xform(SDBNode,
                    translation=(0.000000000000000, 0.000000000000000, 0.000000000000000),
                    scale=(1.000000000000000, 1.000000000000000, 1.000000000000000),
                    rotation=(-0.000000000000000, 0.000000000000000, 0.000000000000000),
                    zeroTransformPivots=True,
                    rotateOrder="zxy")
        except:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Cannot load matchmove scene.", QtWidgets.QMessageBox.Ok)

        # add attributes to SDBNode
        try:
            mc.addAttr(SDBNode, longName="publisher", niceName="publisher", dataType="string")
            mc.setAttr(SDBNode+".publisher", self.sdb_publisher, type="string", lock=True)
            #mc.setAttr(SDBNode + ".publisher", sceneInfo['artist'], type="string", lock=True)

            mc.addAttr(SDBNode, longName="stereo", niceName="stereo", dataType="string")
            mc.setAttr(SDBNode+".stereo", self.sdb_stereo, type="string", lock=True)
            # mc.setAttr(SDBNode + ".stereo", sceneInfo['task_publish']['isStereo'], type="string", lock=True)

            mc.addAttr(SDBNode, longName="from", niceName="from", dataType="string")
            #mc.setAttr(SDBNode + ".from", sceneInfo['files']['maya_pub_file'][0], type="string", lock=True)
            mc.setAttr(SDBNode+".from", self.sdb_mayaScene, type="string", lock=True)

            if sceneInfo['source'] == 'DB':
                mc.addAttr(SDBNode, longName="database", niceName="database", dataType="string")
                mc.setAttr(SDBNode+".database", sceneInfo['show'], type="string", lock=True)

                mc.addAttr(SDBNode, longName="objectId", niceName="objectId", dataType="string")
                mc.setAttr(SDBNode+".objectId", sceneInfo['_id'], type="string", lock=True)

            elif sceneInfo['source'] == 'JSON':
                mc.addAttr(SDBNode, longName="json", niceName="json", dataType="string")
                mc.setAttr(SDBNode+".json", self.jsonPath, type="string", lock=True)

            mc.addAttr(SDBNode, longName="overscan", niceName="overscan", dataType="string")
            mc.setAttr(SDBNode+".overscan", self.sdb_overscan, type="string", lock=True)
            mc.addAttr(SDBNode, longName="plate", niceName="plate", dataType="string")
            mc.setAttr(SDBNode+".plate", self.sdb_plate, type="string", lock=True)
            mc.addAttr(SDBNode, longName="startFrame", niceName="startFrame", dataType="string")
            mc.setAttr(SDBNode+".startFrame", self.sdb_startFrame, type="string", lock=True)
            mc.addAttr(SDBNode, longName="endFrame", niceName="endFrame", dataType="string")
            mc.setAttr(SDBNode+".endFrame", self.sdb_endFrame, type="string", lock=True)
            mc.addAttr(SDBNode, longName="renderWidth", niceName="renderWidth", dataType="string")
            mc.setAttr(SDBNode+".renderWidth", self.sdb_renderWidth, type="string", lock=True)
            mc.addAttr(SDBNode, longName="renderHeight", niceName="renderHeight", dataType="string")
            mc.setAttr(SDBNode+".renderHeight", self.sdb_renderHeight, type="string", lock=True)
            mc.addAttr(SDBNode, longName="date", niceName="date", dataType="string")
            mc.setAttr(SDBNode+".date", mc.date(), type="string", lock=True)
            mc.addAttr(SDBNode, longName="aniSubUser", niceName="aniSubUser", dataType="string")
            mc.setAttr(SDBNode+".aniSubUser", os.getenv("USER"), type="string", lock=True)
        except:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Cannot write scene info to SDBNode.", QtWidgets.QMessageBox.Ok)

        # import matchmove camera as layout camera

        if self.layoutCamera == 2:
            if sceneInfo['source'] == 'DB':
                camPath = sceneInfo['files']['camera_path'][0]
                camFileType = 'Alembic'
            elif sceneInfo['source'] == 'JSON':
                camPath = sceneInfo['mayaCam']
                camFileType = 'mayaBinary'
            try:
                mc.file(camPath,
                        type=camFileType,
                        i=True,
                        reference=False,
                        groupLocator=False,
                        groupReference=False,
                        loadReferenceDepth="all",
                        sharedNodes="renderLayersByName",
                        preserveReferences=True,
                        mergeNamespacesOnClash=True,
                        renameAll=True,
                        renamingPrefix="layoutCamera",
                        options="v=0")
            except:
                QtWidgets.QMessageBox.warning(self, "Warning!", "Cannot load camera file.", QtWidgets.QMessageBox.Ok)

        #QtGui.QMessageBox.information(self, "Information", "Subscribing Success!", QtGui.QMessageBox.Ok)
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Subscribing Success!")
        msgBox.setInformativeText("Do you want to close the window?")
        msgBox.setStandardButtons(QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Yes)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Yes)
        ret = msgBox.exec_()
        if ret == QtWidgets.QMessageBox.Yes:
            shotsub.close()
        else:
            pass
            QtWidgets.QMessageBox.information(self, "Information", "Please, DO NOT MOVE pivot point of \"%s\"."%SDBNode,
                                              QtWidgets.QMessageBox.Ok)

#
# main function
def shotSubMaya():
    global shotsub
    try:
        shotsub.close()
    except:
        pass

    shotsub = ShotSubMaya_anim()

    if sys.platform != "darwin":
        fontPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "OpenSans-Regular.ttf")
        fontId = QtGui.QFontDatabase.addApplicationFont(fontPath)
        if fontId is not -1:
            family = QtGui.QFontDatabase.applicationFontFamilies(fontId)
            font = QtGui.QFont(family[0])
            font.setPointSize(9)
            shotsub.setFont(font)

    if dockMode:
        mc.dockControl(label=window_object, area="right", content=shotsub(), allowedArea=["left", "right"])
    else:
        shotsub.show()
        shotsub.resize(461, 530)
