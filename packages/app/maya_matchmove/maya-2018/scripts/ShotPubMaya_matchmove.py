import os
import sys
import glob
from PySide2 import QtCore, QtGui, QtUiTools, QtWidgets

import maya.cmds as mc

import shotDB_common
from CameraAsset import CameraAsset, SgCameraMMV

SHOW_ROOT = "/show"
#from ui_shotpub_matchmove_widget import Ui_shotpub_camera_Dialog
import ui_shotpub_matchmove_widget
reload(ui_shotpub_matchmove_widget)
import pymongo
from pymongo import MongoClient
DB_IP = "10.0.0.12:27017, 10.0.0.13:27017"
DB_NAME = 'PIPE_PUB'

try:
    f = open('/home/%s/.mmvuser' % os.getenv('USER'))
    USER = (f.readline()).rstrip('\n')
    f.close()
except:
    USER = os.getenv("USER")

window_object = "shotpub v1.3(20140716)"
dock_mode = False


class ShotPubMaya_matchmove(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = ui_shotpub_matchmove_widget.Ui_shotpub_camera_Dialog()
        self.ui.setupUi(self)
        self.setObjectName(window_object)

        self.init_GUI()
        self.init_CurrentPath()
        self.connect_slot()
        # self.change_shot()
        self.update_scene_version()

    def init_GUI(self):
        # initialize show/seq/shot
        self.show_list = shotDB_common.get_dir_list(SHOW_ROOT)
        self.show_list = shotDB_common.reorder_list(self.show_list, "mmv")
        self.seq_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, self.show_list[0], "shot"))

        self.shot_list = shotDB_common.get_dir_list(os.path.join(SHOW_ROOT,
                                                                 self.show_list[0],
                                                                 "shot",
                                                                 self.seq_list[0]))

        self.ui.pubTarget_show.addItems(self.show_list)

        self.ui.user_user.setText(USER)

        # initialize plate type
        self.plate_type_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, self.show_list[0], "shot", self.seq_list[0],
                         self.shot_list[0], "plates"))
        if self.plate_type_list[0] == "No Dirs":
            self.plate_type_list = ["main"]
        else:
            self.plate_type_list = shotDB_common.reorder_list(self.plate_type_list, "main")
        self.ui.camOptions_plateList.addItems(self.plate_type_list)

        # initialize frame range
        self.start_frame = str(mc.playbackOptions(q=True, min=True))
        self.end_frame = str(mc.playbackOptions(q=True, max=True))
        self.render_width = str(mc.getAttr("defaultResolution.width"))
        self.render_height = str(mc.getAttr("defaultResolution.height"))

        self.ui.sceneOptions_startFrame.setText(self.start_frame)
        self.ui.sceneOptions_endFrame.setText(self.end_frame)
        self.ui.sceneOptions_renderWidth.setText(self.render_width)
        self.ui.sceneOptions_renderHeight.setText(self.render_height)

        # initialize cameras
        self.reload_cam()

        # initialize stereo
        if len(mc.fileInfo("stereo", query=True)) != 0:
            if mc.fileInfo("stereo", query=True)[0] == "true":
                self.ui.camOptions_stereo.setCheckState(2)
                self.set_stereo()

        # initialize overscan
        if len(mc.fileInfo("overscan", query=True)) != 0:
            if mc.fileInfo("overscan", query=True)[0] == "true":
                self.ui.camOptions_overscan.setCheckState(QtCore.Qt.Checked)
                #self.ui.camOptions_overscan.setCheckState(2)

        self.ui.camOptions_abcCam.setChecked(True)
        self.ui.camOptions_abcCam.setEnabled(False)
        self.ui.sceneOptions_abc.setChecked(True)
        self.ui.sceneOptions_abc.setEnabled(False)


    def init_CurrentPath(self):
        currentfile = mc.file(q=True, sn=True)
        if not currentfile:
            return
        pathsource = currentfile.split('/')
        if not 'show' in pathsource:
            return

        current_show = pathsource[pathsource.index('show') + 1]
        self.ui.pubTarget_show.setCurrentIndex(self.show_list.index(current_show) + 1)
        self.ui.pubTarget_seq.clear()
        self.seq_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, current_show, 'shot'))
        self.ui.pubTarget_seq.addItems(self.seq_list)
        if not 'shot' in pathsource:
            return
        current_seq = pathsource[pathsource.index('shot') + 1]
        self.ui.pubTarget_seq.setCurrentIndex(self.seq_list.index(current_seq))
        self.ui.pubTarget_shot.clear()
        self.shot_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, current_show, 'shot', current_seq))
        self.ui.pubTarget_shot.addItems(self.shot_list)
        current_shot = pathsource[pathsource.index(current_seq) + 1]
        self.ui.pubTarget_shot.setCurrentIndex(self.shot_list.index(current_shot))

        self.ui.camOptions_plateList.clear()
        plate_type_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, current_show, 'shot', current_seq,
                         current_shot, 'plates'))
        if plate_type_list[0] == 'No Dirs':
            print 'No Plates'
            plate_type_list = ['main']
        else:
            plate_type_list = shotDB_common.reorder_list(plate_type_list, 'main')
        self.ui.camOptions_plateList.addItems(plate_type_list)

    def connect_slot(self):
        QtCore.QObject.connect(self.ui.pubTarget_show,
                               QtCore.SIGNAL("currentIndexChanged(int)"),
                               self.change_show)
        QtCore.QObject.connect(self.ui.pubTarget_seq,
                               QtCore.SIGNAL("currentIndexChanged(int)"),
                               self.change_seq)
        QtCore.QObject.connect(self.ui.pubTarget_shot,
                               QtCore.SIGNAL("currentIndexChanged(int)"),
                               self.change_shot)

        QtCore.QObject.connect(self.ui.camOptions_stereo,
                               QtCore.SIGNAL("stateChanged(int)"),
                               self.set_stereo)
        QtCore.QObject.connect(self.ui.camOptions_reload,
                               QtCore.SIGNAL("clicked()"), self.reload_cam)

        QtCore.QObject.connect(self.ui.camOptions_plateList,
                               QtCore.SIGNAL("currentIndexChanged(int)"),
                               self.plate_updated)

        QtCore.QObject.connect(self.ui.doPub_do, QtCore.SIGNAL("clicked()"),
                               self.check_pub_data)

    # def getPubVersion(self, show, seq, shot, data_type):
    #     client = MongoClient(DB_IP)
    #     db = client[DB_NAME]
    #     coll = db[show]
    #     recentDoc = coll.find_one({'show': show,
    #                                'shot': shot,
    #                                'data_type': data_type},
    #                               sort=[('time', pymongo.DESCENDING)])
    #     if recentDoc:
    #         return recentDoc['version']
    #     else:
    #         return 0

    def set_stereo(self):
        # if stereo camera checkbox is on, set second camera to enable.
        if self.ui.camOptions_stereo.checkState() == 0:
            self.ui.camOptions_leftCam.setText("Camera")
            self.ui.camOptions_rightCam.setText("Only Stereo")
            self.ui.camOptions_rightCam.setEnabled(0)
            self.ui.camOptions_rightCamList.setEnabled(0)
        else:
            self.ui.camOptions_leftCam.setText("Left Camera")
            self.ui.camOptions_rightCam.setText("Right Camera")
            self.ui.camOptions_rightCam.setEnabled(1)
            self.ui.camOptions_rightCamList.setEnabled(1)

    def change_show(self):
        self.show = str(self.ui.pubTarget_show.currentText())

        self.ui.pubTarget_seq.clear()
        self.seq_list = shotDB_common.get_dir_list(os.path.join(SHOW_ROOT, self.show, "shot"))
        self.ui.pubTarget_seq.addItems(self.seq_list)

        self.change_seq()

    def change_seq(self):
        self.show = str(self.ui.pubTarget_show.currentText())
        self.seq = str(self.ui.pubTarget_seq.currentText())

        self.ui.pubTarget_shot.clear()
        self.shot_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, self.show, "shot", self.seq))
        self.ui.pubTarget_shot.addItems(self.shot_list)

        self.change_shot()

    def change_shot(self):
        self.change_plate_type()
        # ------------------------------------------------------------------------------
        self.update_scene_version()

    def update_scene_version(self):
        # curFileName = os.path.splitext(os.path.basename(mc.file(q=True, sn=True)))[0]
        # scene_ver = 1
        # for flag in curFileName.split('_'):
        #     if flag.startswith('v') and flag[1:].isdigit():
        #         scene_ver = int(flag[1:])
        # self.ui.ver_spin.setValue(scene_ver)
        #def getPubVersion(self, show, seq, shot, data_type):
        recentVer = shotDB_common.getPubVersion(show=str(self.ui.pubTarget_show.currentText()),
                                                seq= str(self.ui.pubTarget_seq.currentText()),
                                                shot=str(self.ui.pubTarget_shot.currentText()),
                                                data_type='camera'
                                                )
        self.ui.ver_spin.setValue(recentVer + 1)


    def plate_updated(self):
        self.update_scene_version()

    def change_plate_type(self):
        # if show/seq/shot changed, reload plate type.
        self.show = str(self.ui.pubTarget_show.currentText())
        self.seq = str(self.ui.pubTarget_seq.currentText())
        self.shot = str(self.ui.pubTarget_shot.currentText())

        self.ui.camOptions_plateList.clear()

        self.plate_type_list = shotDB_common.get_dir_list(
            os.path.join(SHOW_ROOT, self.show, "shot", self.seq, self.shot,
                         "plates"))
        if self.plate_type_list[0] == "No Dirs":
            print "No Plates."
            self.plate_type_list = ["main"]
        else:
            self.plate_type_list = shotDB_common.reorder_list(self.plate_type_list, "main")
        self.ui.camOptions_plateList.addItems(self.plate_type_list)

    def reload_cam(self):
        self.ui.camOptions_leftCamList.clear()
        self.ui.camOptions_leftCamList.addItem("Select Camera")
        self.ui.camOptions_rightCamList.clear()
        self.ui.camOptions_rightCamList.addItem("Select Camera")

        self.all_cam_list = mc.ls(cameras=True)
        self.cam_list = []

        for c in self.all_cam_list:
            self.cam_list.append(
                mc.listRelatives(c, type="transform", parent=True)[0])

        for c in mc.listCameras(p=False, o=True):
            try:
                self.cam_list.remove(c)
            except:
                print "Removing Othographic Cameras Failed."

        self.ui.camOptions_leftCamList.addItems(self.cam_list)
        self.ui.camOptions_rightCamList.addItems(self.cam_list)

    def get_data_from_UI(self):
        self.show = str(self.ui.pubTarget_show.currentText())
        self.seq = str(self.ui.pubTarget_seq.currentText())
        self.shot = str(self.ui.pubTarget_shot.currentText())
        self.user = str(self.ui.user_user.text())
        self.version = self.ui.ver_spin.value()


        self.start_frame = str(self.ui.sceneOptions_startFrame.text())
        self.end_frame = str(self.ui.sceneOptions_endFrame.text())
        self.render_width = str(self.ui.sceneOptions_renderWidth.text())
        self.render_height = str(self.ui.sceneOptions_renderHeight.text())
        self.maya_scene = self.ui.sceneOptions_maya.checkState()
        #self.fbx_scene = self.ui.sceneOptions_fbx.checkState()
        #self.export_geo_loc = self.ui.sceneOptions_geo_loc.checkState()
        self.abc_scene = self.ui.sceneOptions_abc.checkState()
        self.retime_scene = self.ui.sceneOptions_retime.checkState()

        self.left_cam = str(self.ui.camOptions_leftCamList.currentText())
        self.right_cam = str(self.ui.camOptions_rightCamList.currentText())
        self.stereo = self.ui.camOptions_stereo.checkState()
        self.plate_type = str(self.ui.camOptions_plateList.currentText())

        self.overscan = self.ui.camOptions_overscan.checkState()
        #self.add_key = self.ui.camOptions_mbKey.checkState()
        self.lock_cam = self.ui.camOptions_lockCam.checkState()
        self.iplane = self.ui.camOptions_iplane.checkState()
        #self.maya_cam = self.ui.camOptions_mayaCam.checkState()
        #self.fbx_cam = self.ui.camOptions_fbxCam.checkState()
        self.alembic_cam = self.ui.camOptions_abcCam.checkState()

    def check_pub_data(self):
        self.get_data_from_UI()

        if self.show == "select show":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Show first!",
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.show == "No Dirs" or self.seq == "No Dirs" or self.shot == "No Dirs":
            QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "There is no publish folder.",
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.user == "":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Type Your Name!",
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.left_cam == "Select Camera" and self.right_cam == "Select Camera":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Camera!",
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.left_cam == "Select Camera":
            QtWidgets.QMessageBox.warning(self, "Warning!", "Select Left Camera!",
                                          QtWidgets.QMessageBox.Ok)
            return False

        if self.left_cam == self.right_cam:
            QtWidgets.QMessageBox.warning(self, "Warning!",
                                      "Left and Right Camera are Equal!",
                                      QtWidgets.QMessageBox.Ok)
            return False

        if not ("_" + self.plate_type + "_") in self.left_cam:
            QtWidgets.QMessageBox.warning(self, "Warning!", "Check Plate Type!",
                                      QtWidgets.QMessageBox.Ok)
            return False

        if self.stereo == 2:
            if not ("_" + self.plate_type + "_") in self.right_cam:
                QtWidgets.QMessageBox.warning(self, "Warning!", "Check Plate Type!",
                                          QtWidgets.QMessageBox.Ok)
                return False

        if not self.left_cam in mc.listCameras():
            QtWidgets.QMessageBox.warning(self, "Warning!",
                                      "No Left Camera!\nPlease, Reload Camera.",
                                      QtWidgets.QMessageBox.Ok)
            return False

        if self.stereo == 2:
            if self.right_cam == "Select Camera":
                QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "Select Right Camera!",
                                          QtWidgets.QMessageBox.Ok)
                return False
            else:
                if not self.right_cam in mc.listCameras():
                    QtWidgets.QMessageBox.warning(self, "Warning!",
                                              "No Right Camera!\nPlease, Reload Camera.",
                                              QtWidgets.QMessageBox.Ok)
                    return False

        if not self.shot in self.left_cam:
            QtWidgets.QMessageBox.warning(self, "Warning!",
                                      "Check Show/Seq/Show to publish!",
                                      QtWidgets.QMessageBox.Ok)
            return False

        if self.stereo == 2:
            if not self.shot in self.right_cam:
                QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "Check Show/Seq/Show to publish!",
                                          QtWidgets.QMessageBox.Ok)
                return False

        if self.stereo == 2:
            if not "left" in self.left_cam:
                QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "Check Left Camera Orientation!",
                                          QtWidgets.QMessageBox.Ok)
                return False
            elif not "right" in self.right_cam:
                QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "Check Right Camera Orientation!",
                                          QtWidgets.QMessageBox.Ok)
                return False

        # if self.fbx_cam == 2:
        #     try:
        #         mc.loadPlugin("fbxmaya.so", qt=True)
        #     except:
        #         QtWidgets.QMessageBox.warning(self, "Warning!",
        #                                   "FBX Plugin was cannot Found.",
        #                                   QtWidgets.QMessageBox.Ok)
        #         return False

        if self.alembic_cam == 2:
            try:
                mc.loadPlugin("AbcExport.so", qt=True)
            except:
                QtWidgets.QMessageBox.warning(self, "Warning!",
                                          "Alembic Plugin was cannot Found.",
                                          QtWidgets.QMessageBox.Ok)
                return False

        self.publish()

    def publish(self):
        print "start publish"

        self.cam_list = [self.left_cam]
        if self.right_cam != "Select Camera":
            self.cam_list.append(self.right_cam)

        pub_cam = CameraAsset(
            shot_info=
            {
                "root": SHOW_ROOT,
                "show": self.show,
                "seq": self.seq,
                "shot": self.shot,
                "user": self.user,
                "version": self.version
            },
            scene_options=
            {
                "start_frame": self.start_frame,
                "end_frame": self.end_frame,
                "render_width": self.render_width,
                "render_height": self.render_height,
                "maya_scene": self.maya_scene,
                #"fbx_scene": self.fbx_scene,
                #"export_geo_loc": self.export_geo_loc,
                "abc_scene": self.abc_scene,
                "retime_scene": self.retime_scene
            },
            cam_options=
            {
                "cam_list": self.cam_list,
                "stereo": self.stereo,
                "plate_type": self.plate_type,
                "overscan": self.overscan,
                #"add_key": self.add_key,
                "iplane": self.iplane,
                "lock_cam": self.lock_cam,
                #"maya_cam": self.maya_cam,
                "abc_cam": self.alembic_cam,
                #"fbx_cam": self.fbx_cam
            }
        )

        pub_result = pub_cam.publish_cam()

        if pub_result:
            QtWidgets.QMessageBox.information(self, "Information", "Published!",
                                          QtWidgets.QMessageBox.Ok)
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Publish Failed!",
                                       QtWidgets.QMessageBox.Ok)
        shotpub.close()

    def resizeEvent(self, event):
        print event.size()
        QtWidgets.QDialog.resizeEvent(self, event)


def shotpub_maya():
    global shotpub
    try:
        shotpub.close()
    except:
        pass
    mayaParent = shotDB_common.get_maya_window()
    shotpub = ShotPubMaya_matchmove(mayaParent)

    if sys.platform != "darwin":
        font_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 "OpenSans-Regular.ttf")
        font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
        if font_id is not -1:
            family = QtGui.QFontDatabase.applicationFontFamilies(font_id)
            font = QtGui.QFont(family[0])
            font.setPointSize(9)
            shotpub.setFont(font)

    if dock_mode:
        mc.dockControl(label=window_object, area="right", content=ShotPub(),
                       allowedArea=["left", "right"])
    else:
        shotpub.show()

        widgetWidth = 469
        widgetHeight = 653

        parentWidth = mayaParent.size().width()
        parentHeight = mayaParent.size().height()

        shotpub.resize(widgetWidth, widgetHeight)
        # shotpub.move((parentWidth/2.0) -(widgetWidth/2.0),
        #              (parentHeight/2.0) - (widgetHeight/2.0))
