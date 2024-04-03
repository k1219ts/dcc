#
# import modules
import os
import sys
import glob
import json

#from PyQt4 import QtCore
#from PyQt4 import QtGui
#from PyQt4 import uic

from PySide import QtCore, QtGui
import pysideuic
import maya.cmds as mc

import MATCHMOVE.shotDB_common
#from CameraAssetAnim import *
from MATCHMOVE.CameraAssetAnim import *

#
# predefined variables
SHOWROOT = "/show"
UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shotpub_anim_widget.ui")

windowObject = "shotpub v1.3"
dockMode = False


def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' %form_class]
        base_class = eval('%s' % widget_class)

    return form_class, base_class

formclass, baseclass = loadUiType(UIFILE)

# main class
class ShotPubMaya_anim(formclass, baseclass):
    def __init__(self, parent=MATCHMOVE.shotDB_common.get_maya_window()):
        QtGui.QDialog.__init__(self, parent)
        self.setObjectName(windowObject)

        self.initGUI()
        self.connectSlot()

    def initGUI(self):
        # initialize show/seq/shot
        self.showList = MATCHMOVE.shotDB_common.get_dir_list(SHOWROOT)
        self.showList = MATCHMOVE.shotDB_common.reorder_list(self.showList, "prat")
        self.showUI.addItems(self.showList)
        self.userUI.setText(os.getenv("USER"))

        # initialize frame range
        self.start_frame = str(mc.playbackOptions(q=True, min=True))
        self.end_frame = str(mc.playbackOptions(q=True, max=True))
        self.render_width = str(mc.getAttr("defaultResolution.width"))
        self.render_height = str(mc.getAttr("defaultResolution.height"))
        self.startFrameUI.setText(self.start_frame)
        self.endFrameUI.setText(self.end_frame)
        self.renderWidthUI.setText(self.render_width)
        self.renderHeightUI.setText(self.render_height)

        # initialize cameras
        self.reloadCam()

        # initialize SDBNode
        self.sdbList = []
        for i in mc.ls(transforms=True):
            if i.endswith("SDBNode"):
                self.sdbList.append(i)
        self.sdbNodeUI.addItems(self.sdbList)

    def connectSlot(self):
        QtCore.QObject.connect(self.showUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadShow)
        QtCore.QObject.connect(self.seqUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadSeq)
        QtCore.QObject.connect(self.shotUI, QtCore.SIGNAL("currentIndexChanged(int)"), self.reloadShot)
        QtCore.QObject.connect(self.stereoUI, QtCore.SIGNAL("stateChanged(int)"), self.setStereo)
        QtCore.QObject.connect(self.reloadUI, QtCore.SIGNAL("clicked()"), self.reloadCam)
        QtCore.QObject.connect(self.doUI, QtCore.SIGNAL("clicked()"), self.preparePub)

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
        if self.plateTypeList[0].startswith == "No ":
            self.plateTypeList = ["No plates"]
        else:
            self.plateTypeList = MATCHMOVE.shotDB_common.reorder_list(self.plateTypeList, "main")
        self.plateTypeUI.clear()
        self.plateTypeUI.addItems(["layout"])
        self.plateTypeUI.addItems(self.plateTypeList)

    def reloadCam(self):
        self.leftCamUI.clear()
        self.leftCamUI.addItem("select camera")
        self.rightCamUI.clear()
        self.rightCamUI.addItem("select camera")
        self.all_cam_list = mc.ls(cameras=True)
        self.cam_list = []

        for c in self.all_cam_list:
            self.cam_list.append(mc.listRelatives(c, type="transform", parent=True)[0])

        for c in mc.listCameras(p=False, o=True):
            try:
                self.cam_list.remove(c)
            except:
                print "Removing Othographic Cameras Failed."

        self.leftCamUI.addItems(self.cam_list)
        self.rightCamUI.addItems(self.cam_list)

    def setStereo(self):
        # if stereo camera checkbox is on, set second camera to enable.
        if self.stereoUI.checkState() == 0:
            self.leftCamLabel.setText("Camera")
            self.rightCamLabel.setText("Only Stereo")
            self.rightCamUI.setEnabled(0)
            self.rightCamUI.setEnabled(0)
        else:
            self.leftCamLabel.setText("Left Camera")
            self.rightCamLabel.setText("Right Camera")
            self.rightCamUI.setEnabled(1)
            self.rightCamUI.setEnabled(1)

    def getUIData(self):
        self.show = str(self.showUI.currentText())
        self.seq = str(self.seqUI.currentText())
        self.shot = str(self.shotUI.currentText())
        self.user = str(self.userUI.text())
        self.dept = str(self.deptUI.currentText())

        self.startFrame = float(self.startFrameUI.text())
        self.endFrame = float(self.endFrameUI.text())
        self.renderWidth = int(self.renderWidthUI.text())
        self.renderHeight = int(self.renderHeightUI.text())
        self.fullCg = self.fullCGUI.checkState()

        self.leftCam = str(self.leftCamUI.currentText())
        self.rightCam = str(self.rightCamUI.currentText())
        self.plateType = str(self.plateTypeUI.currentText())

        self.bakeCam = self.bakeCamUI.checkState()
        self.abcCam = self.abcCamUI.checkState()
        self.fbxCam = self.fbxCamUI.checkState()
        self.stereo = self.stereoUI.checkState()
        self.renewIP = self.renewIPUI.checkState()

    def preparePub(self):
        self.getUIData()

        if self.show.startswith("select ") or self.seq.startswith("select ") or self.shot.startswith("select "):
            QtGui.QMessageBox.warning(self, "Warning!", "Select show first!", QtGui.QMessageBox.Ok)
            return False

        if self.show.startswith("No ") or self.seq.startswith("No ") or self.shot.startswith("No "):
            QtGui.QMessageBox.warning(self, "Warning!", "There is no publish folder. Ask to your TD.",
                                      QtGui.QMessageBox.Ok)
            return False

        if self.leftCam == "select camera":
            QtGui.QMessageBox.warning(self, "Warning!", "Select Left Camera!", QtGui.QMessageBox.Ok)
            return False

        if not self.leftCam in mc.listCameras():
            QtGui.QMessageBox.warning(self, "Warning!", "No Left Camera!\nPlease, Reload Camera.", QtGui.QMessageBox.Ok)
            return False

        if self.stereo == 2:
            if self.rightCam == "select camera":
                QtGui.QMessageBox.warning(self, "Warning!", "Select Right Camera!", QtGui.QMessageBox.Ok)
                return False
            else:
                if not self.rightCam in mc.listCameras():
                    QtGui.QMessageBox.warning(self, "Warning!", "No Right Camera!\nPlease, Reload Camera.",
                                              QtGui.QMessageBox.Ok)
                    return False
            if self.leftCam == self.rightCam:
                QtGui.QMessageBox.warning(self, "Warning!", "Left and Right Camera are Equal!", QtGui.QMessageBox.Ok)
                return False

        if self.plateType.startswith("select ") or self.plateType.startswith("No "):
            QtGui.QMessageBox.warning(self, "Warning!", "Select shot!", QtGui.QMessageBox.Ok)
            return False

        #		if self.fullCg == 0:
        #			if not MATCHMOVE.shotDB_common.checkImagePlanePath() and self.fullCg == 0:
        #				QtGui.QMessageBox.warning(self, "Warning!", "Imageplane is not in matchmove publish folder!", QtGui.QMessageBox.Ok)
        #				return False

        self.publish()

    def publish(self):
        self.cam_list = [self.leftCam]
        if self.stereo == 2:
            self.cam_list.append(self.rightCam)

        pubAnimScene = MATCHMOVE.CameraAssetAnim(
            shot_info=
            {
                "root": SHOWROOT,
                "show": self.show,
                "seq": self.seq,
                "shot": self.shot,
                "user": self.user,
                "dept": self.dept
            },
            scene_options=
            {
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "render_width": self.render_width,
                "render_height": self.render_height,
                "sdb_list": self.sdbList,
                "full_cg": self.fullCg
            },
            cam_options=
            {
                "cam_list": self.cam_list,
                "stereo": self.stereo,
                "bake_cam": self.bakeCam,
                "plate_type": self.plateType,
                "abc_cam": self.abcCam,
                "fbx_cam": self.fbxCam,
                "renew_ip": self.renewIP
            }
        )

        pubState = pubAnimScene.publish()

        if pubState == "DONE":
            QtGui.QMessageBox.information(self, "Information", "Published!", QtGui.QMessageBox.Ok)
        else:
            QtGui.QMessageBox.critical(self, "Error", "Publish Failed!", QtGui.QMessageBox.Ok)
        shotpub.close()

    def resizeEvent(self, event):
        print event.size()
        QtGui.QDialog.resizeEvent(self, event)


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
