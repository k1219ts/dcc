#
# import modules
import os
import sys
import glob
import json
import re

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
from pymongo import MongoClient

from PySide2 import QtWidgets, QtCompat

import maya.cmds as mc

import MATCHMOVE.shotDB_common
import MATCHMOVE.CameraAsset

# predefined variables
SHOWROOT = "/show"
UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shotsub_anim_widget.ui")

windowObject = "shotsub v1.0"
dockMode = False


def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

# main class
class ShotSubMaya_anim(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ShotSubMaya_anim, self).__init__(parent)
        ui = QtCompat.loadUi(UIFILE, self)
        setup_ui(ui, self)

        self.initGUI()
        self.connectSlot()

    def initGUI(self):
        # initialize show/seq/shot
        self.showList = MATCHMOVE.shotDB_common.get_dir_list(SHOWROOT)
        self.showList = MATCHMOVE.shotDB_common.reorder_list(self.showList, "prat")
        self.showUI.addItems(self.showList)

    def connectSlot(self):
        self.showUI.currentIndexChanged.connect(self.reloadShow)
        self.seqUI.currentIndexChanged.connect( self.reloadSeq)
        self.shotUI.currentIndexChanged.connect( self.reloadShot)
        self.plateTypeUI.currentIndexChanged.connect(self.changePlateType)
        self.sceneUI.currentIndexChanged.connect(self.reloadSceneInfo)
        self.doIt.clicked.connect(self.importScene)

    def reloadShow(self):
        self.show = str(self.showUI.currentText())
        self.seqList = MATCHMOVE.shotDB_common.get_dir_list(os.path.join(SHOWROOT, self.show, "shot"))
        self.seqUI.clear()
        self.seqUI.addItems(self.seqList)

        self.reloadSeq()

    def reloadSeq(self):
        self.seq = str(self.seqUI.currentText())
        self.shotList = MATCHMOVE.shotDB_common.get_dir_list(os.path.join(SHOWROOT, self.show, "shot", self.seq))
        self.shotUI.clear()
        self.shotUI.addItems(self.shotList)

        self.reloadShot()

    def reloadShot(self):
        self.reloadPlateType()

    def reloadPlateType(self):
        self.shot = str(self.shotUI.currentText())
        self.plateTypeList = MATCHMOVE.shotDB_common.get_dir_list(
            os.path.join(SHOWROOT, self.show, "shot", self.seq, self.shot, "plates"))
        self.plateTypeList.insert(0, 'layout')
        if self.plateTypeList and self.plateTypeList[0].startswith == "No ":
            self.plateTypeList = ["No plates"]
        else:
            self.plateTypeList = MATCHMOVE.shotDB_common.reorder_list(self.plateTypeList, "main")

        self.plateTypeUI.clear()
        self.plateTypeUI.addItems(self.plateTypeList)

        self.reloadScene()

    def changePlateType(self):
        self.reloadScene()

    def reloadScene(self):
        self.plateType = str(self.plateTypeUI.currentText())
        self.scenePath = os.path.join(SHOWROOT, self.show, "shot", self.seq, self.shot, "matchmove", "pub", "scenes")
        self.sceneList = MATCHMOVE.shotDB_common.get_file_list(self.scenePath,
                                                               filename=self.shot + "_" + self.plateType,
                                                               ext="*.mb")

        for ver in glob.glob(os.path.join(self.scenePath, 'v[0-9]*')):
            versionDir = os.path.join(self.scenePath, ver)
            vmb = []
            for f in os.listdir(versionDir):
                if (f.endswith('.mb')) and (self.plateType in f):
                    vmb.append(os.path.join(os.path.basename(ver), f))

            self.sceneList += vmb

        if self.sceneList[0].startswith == "No ":
            self.sceneList = ["No pub scenes"]
        try:
            self.sceneList.sort(reverse=True)
        except:
            pass

        self.sceneUI.clear()
        self.sceneUI.addItems(self.sceneList)

        self.reloadSceneInfo()

    def reloadSceneInfo(self):
        self.scene = str(self.sceneUI.currentText())
        if self.scene.startswith("No "):
            return 0
        self.readJSON(self.scene)

        self.publisherUI.setText(self.sdb_publisher)
        self.verUI.setText(self.sdb_version)
        self.dateUI.setText(self.sdb_date)
        self.commentUI.setText(self.sdb_comment)
        self.startFrameUI.setText(self.sdb_startFrame)
        self.endFrameUI.setText(self.sdb_endFrame)
        self.renderWidthUI.setText(self.sdb_renderWidth)
        self.renderHeightUI.setText(self.sdb_renderHeight)

        if self.sdb_overscan:
            self.overscanUI.setChecked(True)
        else:
            self.overscanUI.setChecked(False)
        if self.sdb_stereo:
            self.stereoUI.setChecked(True)
        else:
            self.stereoUI.setChecked(False)

    def readJSON(self, sceneName):
        if re.search('^v[0-9]+', sceneName):
            # VERSION 2
            client = MongoClient(DB_IP)
            db = client['PIPE_PUB']
            coll = db[self.show]
            pubFile = os.path.join(self.scenePath, self.scene)
            j = coll.find_one({'files.maya_pub_file':pubFile})
            if j:
                self.sdb_publisher = j["artist"]
                self.sdb_startFrame = str(j['task_publish']["startFrame"])
                self.sdb_endFrame = str(j['task_publish']["endFrame"])
                self.sdb_renderWidth = str(j['task_publish']["render_width"])
                self.sdb_renderHeight = str(j['task_publish']["render_height"])
                self.sdb_overscan = str(j['task_publish']["overscan"])
                self.sdb_stereo = str(j['task_publish']["stereo"])

                #self.sdb_mayaCam = j["mayaCam"]
                self.sdb_mayaScene = pubFile
                self.sdb_plate = j['task_publish']["plateType"]


        else:
            # VERSION 1
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

            sceneName = sceneName.replace(".mb", ".shotdb")
            self.jsonPath = os.path.join(self.scenePath, sceneName)
            if os.path.isfile(self.jsonPath):
                f = open(self.jsonPath, "r")
                j = json.load(f)

                self.sdb_publisher = j["user"]
                # self.sdb_version = j["ver"]
                # self.sdb_date = j["date"]
                # self.sdb_comment = j["comment"]
                # self.sdb_version = j["ver"]
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
        self.show = str(self.showUI.currentText())
        self.seq = str(self.seqUI.currentText())
        self.shot = str(self.shotUI.currentText())
        self.plateType = str(self.plateTypeUI.currentText())
        self.scene = str(self.sceneUI.currentText())

        self.startFrame = float(self.startFrameUI.text())
        self.endFrame = float(self.endFrameUI.text())
        self.renderWidth = int(self.renderWidthUI.text())
        self.renderHeight = int(self.renderHeightUI.text())

        self.stereo = self.stereoUI.checkState()
        self.overscan = self.overscanUI.checkState()
        self.layoutCamera = self.layoutCameraUI.checkState()

    def importScene(self):
        self.getUIData()

        if self.show.startswith("select "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select show first!", QtWidgets.QMessageBox.Ok)
            return False

        if self.show.startswith("No ") or self.seq.startswith("No ") or self.shot.startswith(
                "No ") or self.plateType.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "There is no publish folder. Ask to your TD.",
                                      QtWidgets.QMessageBox.Ok)
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
        mc.setAttr("defaultResolution.deviceAspectRatio", float(self.renderWidth) / float(self.renderHeight))
        mc.setAttr("defaultRenderGlobals.animation", 1)
        mc.setAttr("defaultRenderGlobals.extensionPadding", 4)
        mc.setAttr("defaultRenderGlobals.startFrame", self.startFrame)
        mc.setAttr("defaultRenderGlobals.endFrame", self.endFrame)
        mc.playbackOptions(ast=self.startFrame, aet=self.endFrame, min=self.startFrame, max=self.endFrame)
        mc.currentTime(self.startFrame)

        # import matchmove scene as reference.
        SDBNode = "%s_%s_matchmove_SDBNode" % (self.shot, self.plateType)
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
                    namespace="%s_%s_matchmove" % (self.shot, self.plateType),
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
            mc.setAttr(SDBNode + ".publisher", self.sdb_publisher, type="string", lock=True)
            mc.addAttr(SDBNode, longName="stereo", niceName="stereo", dataType="string")
            mc.setAttr(SDBNode + ".stereo", self.sdb_stereo, type="string", lock=True)
            mc.addAttr(SDBNode, longName="from", niceName="from", dataType="string")
            mc.setAttr(SDBNode + ".from", self.sdb_mayaScene, type="string", lock=True)
            mc.addAttr(SDBNode, longName="json", niceName="json", dataType="string")
            mc.setAttr(SDBNode + ".json", self.jsonPath, type="string", lock=True)
            mc.addAttr(SDBNode, longName="overscan", niceName="overscan", dataType="string")
            mc.setAttr(SDBNode + ".overscan", self.sdb_overscan, type="string", lock=True)
            mc.addAttr(SDBNode, longName="plate", niceName="plate", dataType="string")
            mc.setAttr(SDBNode + ".plate", self.sdb_plate, type="string", lock=True)
            mc.addAttr(SDBNode, longName="startFrame", niceName="startFrame", dataType="string")
            mc.setAttr(SDBNode + ".startFrame", self.sdb_startFrame, type="string", lock=True)
            mc.addAttr(SDBNode, longName="endFrame", niceName="endFrame", dataType="string")
            mc.setAttr(SDBNode + ".endFrame", self.sdb_endFrame, type="string", lock=True)
            mc.addAttr(SDBNode, longName="renderWidth", niceName="renderWidth", dataType="string")
            mc.setAttr(SDBNode + ".renderWidth", self.sdb_renderWidth, type="string", lock=True)
            mc.addAttr(SDBNode, longName="renderHeight", niceName="renderHeight", dataType="string")
            mc.setAttr(SDBNode + ".renderHeight", self.sdb_renderHeight, type="string", lock=True)
            mc.addAttr(SDBNode, longName="date", niceName="date", dataType="string")
            mc.setAttr(SDBNode + ".date", mc.date(), type="string", lock=True)
            mc.addAttr(SDBNode, longName="aniSubUser", niceName="aniSubUser", dataType="string")
            mc.setAttr(SDBNode + ".aniSubUser", os.getenv("USER"), type="string", lock=True)
        except:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Cannot write scene info to SDBNode.", QtWidgets.QMessageBox.Ok)

        # import matchmove camera as layout camera

        if self.layoutCamera == 2:
            try:
                mc.file(self.sdb_mayaCam,
                        type="mayaBinary",
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

        # QtWidgets.QMessageBox.information(self, "Information", "Subscribing Success!", QtWidgets.QMessageBox.Ok)
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
        QtWidgets.QMessageBox.information(self, "Information", "Please, DO NOT MOVE pivot point of \"%s\"." % SDBNode,
                                      QtWidgets.QMessageBox.Ok)

    # def resizeEvent(self, event):
    #     # print event.size()
    #     QtWidgets.QDialog.resizeEvent(self, event)


#
# main function
def shotSubMaya():
    global shotsub
    try:
        shotsub.close()
    except:
        pass

    shotsub = ShotSubMaya_anim()
    shotsub.show()
    shotsub.resize(461, 530)
