import os
import sys
import glob
import json

from PySide2 import QtCore, QtGui, QtUiTools, QtWidgets
import ui_shotpub_anim_widget
reload(ui_shotpub_anim_widget)

import maya.cmds as mc

import shotDB_common
from CameraAssetAnim import *

#
# predefined variables
SHOWROOT = "/show"

windowObject = "shotpub v1.3"
dockMode = False

#
# main class
class ShotPubMaya_anim(QtWidgets.QDialog):
    def __init__(self, parent=shotDB_common.get_maya_window()):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = ui_shotpub_anim_widget.Ui_shotpub_camera_Dialog()
        self.ui.setupUi(self)

        self.setObjectName(windowObject)

        self.initGUI()
        self.connectSlot()


    def initGUI(self):
        # initialize show/seq/shot
        self.showList = shotDB_common.get_dir_list(SHOWROOT)
        self.showList = shotDB_common.reorder_list(self.showList, "prat")
        self.ui.showUI.addItems(self.showList)
        self.ui.userUI.setText(os.getenv("USER"))

        # initialize frame range
        self.start_frame = str(mc.playbackOptions(q=True, min=True))
        self.end_frame = str(mc.playbackOptions(q=True, max=True))
        self.render_width = str(mc.getAttr("defaultResolution.width"))
        self.render_height = str(mc.getAttr("defaultResolution.height"))
        self.ui.startFrameUI.setText(self.start_frame)
        self.ui.endFrameUI.setText(self.end_frame)
        self.ui.renderWidthUI.setText(self.render_width)
        self.ui.renderHeightUI.setText(self.render_height)

        # initialize cameras
        self.reloadCam()

        # initialize SDBNode
        self.sdbList = []
        for i in mc.ls(transforms=True):
            if i.endswith("SDBNode"):
                self.sdbList.append(i)
        self.ui.sdbNodeUI.addItems(self.sdbList)

    def connectSlot(self):
        QtCore.QObject.connect(self.ui.showUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadShow)
        QtCore.QObject.connect(self.ui.seqUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadSeq)
        QtCore.QObject.connect(self.ui.shotUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadShot)
        QtCore.QObject.connect(self.ui.stereoUI, QtCore.SIGNAL("stateChanged(int)"), self.setStereo)
        QtCore.QObject.connect(self.ui.reloadUI, QtCore.SIGNAL("clicked()"), self.reloadCam)
        QtCore.QObject.connect(self.ui.doUI, QtCore.SIGNAL("clicked()"), self.preparePub)


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
        self.ui.plateTypeUI.addItems(["layout"])
        self.ui.plateTypeUI.addItems(self.plateTypeList)


    def reloadCam(self):
        self.ui.leftCamUI.clear()
        self.ui.leftCamUI.addItem("select camera")
        self.ui.rightCamUI.clear()
        self.ui.rightCamUI.addItem("select camera")
        self.all_cam_list = mc.ls(cameras=True)
        self.cam_list = []

        for c in self.all_cam_list:
            self.cam_list.append(mc.listRelatives(c, type="transform", parent=True)[0])

        for c in mc.listCameras(p=False, o=True):
            try:
                self.cam_list.remove(c)
            except:
                print "Removing Othographic Cameras Failed."

        self.ui.leftCamUI.addItems(self.cam_list)
        self.ui.rightCamUI.addItems(self.cam_list)


    def setStereo(self):
        # if stereo camera checkbox is on, set second camera to enable.
        if self.ui.stereoUI.checkState() == 0:
            self.ui.leftCamLabel.setText("Camera")
            self.ui.rightCamLabel.setText("Only Stereo")
            self.ui.rightCamUI.setEnabled(0)
            self.ui.rightCamUI.setEnabled(0)
        else:
            self.ui.leftCamLabel.setText("Left Camera")
            self.ui.rightCamLabel.setText("Right Camera")
            self.ui.rightCamUI.setEnabled(1)
            self.ui.rightCamUI.setEnabled(1)


    def getUIData(self):
        self.show = str(self.ui.showUI.currentText())
        self.seq = str(self.ui.seqUI.currentText())
        self.shot = str(self.ui.shotUI.currentText())
        self.user = str(self.ui.userUI.text())
        self.dept = str(self.ui.deptUI.currentText())

        self.startFrame = float(self.ui.startFrameUI.text())
        self.endFrame = float(self.ui.endFrameUI.text())
        self.renderWidth = int(self.ui.renderWidthUI.text())
        self.renderHeight = int(self.ui.renderHeightUI.text())
        self.fullCg = self.ui.fullCGUI.checkState()

        self.leftCam = str(self.ui.leftCamUI.currentText())
        self.rightCam = str(self.ui.rightCamUI.currentText())
        self.plateType = str(self.ui.plateTypeUI.currentText())

        self.bakeCam = self.ui.bakeCamUI.checkState()
        self.abcCam = self.ui.abcCamUI.checkState()
        self.fbxCam = self.ui.fbxCamUI.checkState()
        self.stereo = self.ui.stereoUI.checkState()
        self.renewIP = self.ui.renewIPUI.checkState()

    def preparePub(self):
        self.getUIData()

        if self.show.startswith("select ") or self.seq.startswith("select ") or self.shot.startswith("select "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select show first!", QtWidgets.QMessageBox.Ok)
            return False

        if self.show.startswith("No ") or self.seq.startswith("No ") or self.shot.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "There is no publish folder. Ask to your TD.", QtWidgets.QMessageBox.Ok)
            return False

        if self.leftCam == "select camera":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Left Camera!", QtWidgets.QMessageBox.Ok)
            return False

        if not self.leftCam in mc.listCameras():
            QtWidgets.QMessageBox.warning(self, "Warning!", "No Left Camera!\nPlease, Reload Camera.", QtWidgets.QMessageBox.Ok)
            return False

        if self.stereo == 2:
            if self.rightCam == "select camera":
                QtWidgets.QMessageBox.warning(self, "Warning!", "Select Right Camera!", QtWidgets.QMessageBox.Ok)
                return False
            else:
                if not self.rightCam in mc.listCameras():
                    QtWidgets.QMessageBox.warning(self, "Warning!", "No Right Camera!\nPlease, Reload Camera.", QtWidgets.QMessageBox.Ok)
                    return False
            if self.leftCam == self.rightCam:
                QtWidgets.QMessageBox.warning(self, "Warning!", "Left and Right Camera are Equal!", QtWidgets.QMessageBox.Ok)
                return False

        if self.plateType.startswith("select ") or self.plateType.startswith("No "):
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select shot!", QtWidgets.QMessageBox.Ok)
            return False

#               if self.fullCg == 0:
#                       if not shotDB_common.checkImagePlanePath() and self.fullCg == 0:
#                               QtGui.QMessageBox.warning(self, "Warning!", "Imageplane is not in matchmove publish folder!", QtGui.QMessageBox.Ok)
#                               return False

        self.publish()


    def publish(self):
        self.cam_list = [self.leftCam]
        if self.stereo == 2:
            self.cam_list.append(self.rightCam)

        pubAnimScene = CameraAssetAnim(
                shot_info=
                        {
                                "root":SHOWROOT,
                                "show":self.show,
                                "seq":self.seq,
                                "shot":self.shot,
                                "user":self.user,
                                "dept":self.dept
                        },
                scene_options=
                        {
                                "start_frame":self.start_frame,
                                "end_frame":self.end_frame,
                                "render_width":self.render_width,
                                "render_height":self.render_height,
                                "sdb_list":self.sdbList,
                                "full_cg":self.fullCg
                        },
                cam_options=
                        {
                                "cam_list":self.cam_list,
                                "stereo":self.stereo,
                                "bake_cam":self.bakeCam,
                                "plate_type":self.plateType,
                                "abc_cam":self.abcCam,
                                "fbx_cam":self.fbxCam,
                                "renew_ip":self.renewIP
                        }
        )

        pubState = pubAnimScene.publish()

        if pubState == "DONE":
            QtWidgets.QMessageBox.information(self, "Information", "Published!", QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Publish Failed!", QtWidgets.QMessageBox.Ok)
        shotpub.close()


    # def resizeEvent(self, event):
    #     print event.size()
    #     QtWidgets.QDialog.resizeEvent(self, event)


#
# main function
def shotPubMaya():
    global shotpub
    try:
        shotpub.close()
    except:
        pass

    shotpub = ShotPubMaya_anim()

    if sys.platform != "darwin":
        fontPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "OpenSans-Regular.ttf")
        fontId = QtGui.QFontDatabase.addApplicationFont(fontPath)
        if fontId is not -1:
            family = QtGui.QFontDatabase.applicationFontFamilies(fontId)
            font = QtGui.QFont(family[0])
            font.setPointSize(9)
            shotpub.setFont(font)

    if dockMode:
        mc.dockControl(label=window_object, area="right", content=shotpub(), allowedArea=["left", "right"])
    else:
        shotpub.show()
        shotpub.resize(469, 661)
