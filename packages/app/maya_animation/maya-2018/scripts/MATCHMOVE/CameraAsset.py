import os
import traceback
import json

from MATCHMOVE.shotDB_common import *

from PySide2 import QtWidgets

import maya.cmds as mc
import maya.mel as mm


class CameraAsset(object):
    def __init__(self, shot_info={}, scene_options={}, cam_options={}):

        self.shot_info = shot_info
        self.scene_options = scene_options
        self.cam_options = cam_options
        self.shot_path = os.path.join(
            self.shot_info["root"],
            self.shot_info["show"],
            "shot",
            self.shot_info["seq"],
            self.shot_info["shot"]
        )
        if self.cam_options["stereo"] == 2:
            self.cam_orientation = "_left"
        else:
            self.cam_orientation = ""

    def publish_cam(self):
        try:
            # export scene.
            self.matchmove_pub_path = os.path.join(self.shot_path, "matchmove", "pub",
                                                   "scenes")  # /show/prat/shot/SHS/SHS_0420/matchmove/pub/scenes/
            self.pub_filename = "%s_%s_matchmove" % (
            self.shot_info["shot"], self.cam_options["plate_type"])  # result: ELX_0100_main_matchmove

            if self.cam_options["iplane"] == 2:
                renewImagePlane()

            for c in self.cam_options["cam_list"]:
                camera_transform = c
                camera_shape = mc.listRelatives(camera_transform, allDescendents=True, type="camera")[0]

                if self.cam_options["lock_cam"] == 2:
                    mc.setAttr(camera_transform + ".tx", l=True)
                    mc.setAttr(camera_transform + ".ty", l=True)
                    mc.setAttr(camera_transform + ".tz", l=True)
                    mc.setAttr(camera_transform + ".rx", l=True)
                    mc.setAttr(camera_transform + ".ry", l=True)
                    mc.setAttr(camera_transform + ".rz", l=True)
                    mc.setAttr(camera_transform + ".sx", l=True)
                    mc.setAttr(camera_transform + ".sy", l=True)
                    mc.setAttr(camera_transform + ".sz", l=True)
                    mc.setAttr(camera_shape + ".focalLength", l=True)
                    mc.setAttr(camera_shape + ".horizontalFilmAperture", l=True)
                    mc.setAttr(camera_shape + ".verticalFilmAperture", l=True)
                    print "lock ok"

                if self.cam_options["add_key"] == 2:
                    print "TODO: add keys for motion blur."
                    print "add_keys ok"

                if self.cam_options["stereo"] == 2:
                    self.cam_orientation = "_right"

            if self.scene_options["maya_scene"] == 2:
                make_dir(self.matchmove_pub_path)
                self.scene_ver = get_final_ver(self.matchmove_pub_path, self.pub_filename, "*.mb")
                if self.scene_ver == 0:
                    print "There is no any Maya Scene file. V01 will be published."
                # QtGui.QMessageBox.information(None, "Information", "There is no any Maya Scene file.\nV01 will be published.", QtGui.QMessageBox.Ok)
                self.pub_maya_scene_path = os.path.join(self.matchmove_pub_path,
                                                        self.pub_filename + "_v%.2d" % (self.scene_ver + 1) + ".mb")
                try:
                    mc.file(rename=self.pub_maya_scene_path)
                    mc.file(save=True)
                    print "Matchmove Scene Published! %s" % self.pub_maya_scene_path
                    QtWidgets.QMessageBox.information(None, "Information",
                                                  "Matchmove Scene Published!\n\n%s" % self.pub_maya_scene_path,
                                                      QtWidgets.QMessageBox.Ok)
                except:
                    QtWidgets.QMessageBox.critical(None, "Error",
                                               "Matchmove Scene Failed!\n\n%s" % self.pub_maya_scene_path,
                                                   QtWidgets.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False

            # hide all objects.
            for panName in mc.getPanel(all=True):
                if 'modelPanel' in panName: mc.isolateSelect(panName, state=1)
            ###

            if self.scene_options["fbx_scene"] == 2:
                make_dir(self.matchmove_pub_path)
                self.pub_fbx_scene_path = os.path.join(self.matchmove_pub_path,
                                                       self.pub_filename + "_v%.2d" % (self.scene_ver + 1) + ".fbx")
                try:
                    print "going into fbx_scene."
                    mm.eval('FBXExportFileVersion "FBX201200"')
                    mm.eval('FBXExportInAscii -v false')
                    mm.eval('FBXExportBakeComplexAnimation -v true')
                    mm.eval('FBXExport -f "%s"' % self.pub_fbx_scene_path)
                    print "FBX Scene Published! %s" % self.pub_fbx_scene_path
                # QtGui.QMessageBox.information(None, "Information", "FBX Scene Published!\n\n%s"%self.pub_fbx_scene_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error",
                                               "Exporting FBX Scene Failed!\n\n%s" % self.pub_fbx_scene_path,
                                               QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False

            if self.scene_options["abc_scene"] == 2:
                make_dir(self.matchmove_pub_path)
                self.pub_abc_scene_path = os.path.join(self.matchmove_pub_path,
                                                       self.pub_filename + "_v%.2d" % (self.scene_ver + 1) + ".abc")
                try:
                    print "going into abc_scene."
                    abc_job = "-frameRange %s %s -uv -file %s" % (
                    self.scene_options["start_frame"], self.scene_options["end_frame"], self.pub_abc_scene_path)
                    mc.AbcExport(j=abc_job)
                    print "Alembic Scene Published! %s" % self.pub_abc_scene_path
                # QtGui.QMessageBox.information(None, "Information", "Alembic Scene Published!\n\n%s"%self.pub_abc_scene_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error", "Exporting Alembic Scene Failed!", QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False
            print "publishing scene: OK."

            # export camera.
            self.camera_pub_path = os.path.join(self.shot_path, "camera", "pub",
                                                "scenes")  # /show/prat/shot/SHS/SHS_0420/camera/pub/scenes/
            self.pub_filename = "%s_%s_camera" % (
            self.shot_info["shot"], self.cam_options["plate_type"])  # result: ELX_0100_main_matchmove

            mc.select(clear=True)
            for c in self.cam_options["cam_list"]:
                mc.select(c, add=True)

            if self.cam_options["maya_cam"] == 2:
                self.cam_ver = get_final_ver(self.camera_pub_path, self.pub_filename, "*.mb")
                if self.cam_ver == 0:
                    print "There is no any Maya Camera file. V01 will be published."
                # QtGui.QMessageBox.information(None, "Information", "There is no any Maya Camera file.\nV01 will be published.", QtGui.QMessageBox.Ok)
                make_dir(self.camera_pub_path)
                self.pub_maya_cam_path = os.path.join(self.camera_pub_path,
                                                      self.pub_filename + "_v%.2d" % (self.cam_ver + 1) + ".mb")
                try:
                    mc.file(self.pub_maya_cam_path, op="v=0", typ="mayaBinary", preserveReferences=True,
                            constructionHistory=True, channels=True, constraints=False, expressions=False, shader=False,
                            exportSelected=True, exportAll=False, force=True)
                    print "Maya Camera Published! %s" % self.pub_maya_cam_path
                # QtGui.QMessageBox.information(None, "Information", "Maya Camera Published!\n\n%s"%self.pub_maya_cam_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error", "Exporting Maya Camera Failed!", QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False

            if self.cam_options["fbx_cam"] == 2:
                make_dir(self.camera_pub_path)
                self.pub_fbx_cam_path = os.path.join(self.camera_pub_path,
                                                     self.pub_filename + "_v%.2d" % (self.cam_ver + 1) + ".fbx")
                try:
                    print "going into fbx_cam."
                    mc.loadPlugin("fbxmaya.so", qt=True)
                    mm.eval('FBXExportFileVersion "FBX201200"')
                    mm.eval('FBXExportInAscii -v false')
                    mm.eval('FBXExportBakeComplexAnimation -v true')
                    mm.eval('FBXExport -f "%s" -s' % self.pub_fbx_cam_path)
                    print "FBX Camera Published! %s" % self.pub_fbx_cam_path
                # QtGui.QMessageBox.information(None, "Information", "FBX Camera Published!\n\n%s"%self.pub_fbx_cam_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error", "Exporting FBX Camera Failed!", QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False

            if self.cam_options["abc_cam"] == 2:
                make_dir(self.camera_pub_path)
                self.pub_abc_cam_path = os.path.join(self.camera_pub_path,
                                                     self.pub_filename + "_v%.2d" % (self.cam_ver + 1) + ".abc")
                try:
                    print "going into abc_cam."
                    abc_job = ""
                    for c in self.cam_options["cam_list"]:
                        abc_job += "-root %s " % mc.ls(c, long=True)[0]

                    mc.loadPlugin("AbcExport.so", qt=True)
                    abc_job += "-frameRange %s %s -file %s" % (
                    self.scene_options["start_frame"], self.scene_options["end_frame"], self.pub_abc_cam_path)
                    print abc_job
                    mc.AbcExport(j=abc_job)
                    print "Alembic Camera Published! %s" % self.pub_abc_cam_path
                # QtGui.QMessageBox.information(None, "Information", "Alembic Camera Published!\n\n%s"%self.pub_abc_cam_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error", "Exporting Alembic Camera Failed!", QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False
            print "publishing cameras: OK."

            self.write_scene_json()
            self.write_cam_json()

            mc.select(clear=True)
            mc.file(save=True)
            return "DONE"
        except:
            var = traceback.format_exc()
            print var
            return None, var
        finally:
            # show all objects.
            for panName in mc.getPanel(all=True):
                if 'modelPanel' in panName: mc.isolateSelect(panName, state=0)
            ###

    def write_scene_json(self):
        self.scene_json = dict()
        self.scene_json["show"] = self.shot_info["show"]
        self.scene_json["seq"] = self.shot_info["seq"]
        self.scene_json["shot"] = self.shot_info["shot"]
        self.scene_json["plate"] = self.cam_options["plate_type"]
        self.scene_json["user"] = self.shot_info["user"]
        self.scene_json["task"] = "matchmove"
        self.scene_json["date"] = mc.date()
        self.scene_json["version"] = self.scene_ver
        self.scene_json["startFrame"] = self.scene_options["start_frame"]
        self.scene_json["endFrame"] = self.scene_options["end_frame"]
        self.scene_json["renderWidth"] = self.scene_options["render_width"]
        self.scene_json["renderHeight"] = self.scene_options["render_height"]

        self.scene_json["leftCamera"] = self.cam_options["cam_list"][0]
        if self.cam_options["stereo"] == 2:
            self.scene_json["rightCamera"] = self.cam_options["cam_list"][1]
            self.scene_json["stereo"] = True
        else:
            self.scene_json["rightCamera"] = ""
            self.scene_json["stereo"] = False

        if self.cam_options["overscan"] == 2:
            self.scene_json["overscan"] = True
        else:
            self.scene_json["overscan"] = False

        if self.scene_options["maya_scene"] == 2:
            self.scene_json["mayaScene"] = self.pub_maya_scene_path
        else:
            self.scene_json["mayaScene"] = False

        if self.scene_options["fbx_scene"] == 2:
            self.scene_json["fbxScene"] = self.pub_fbx_scene_path
        else:
            self.scene_json["fbxScene"] = False

        if self.scene_options["abc_scene"] == 2:
            self.scene_json["abcScene"] = self.pub_abc_scene_path
        else:
            self.scene_json["abcScene"] = False

        if self.scene_options["retime_scene"] == 2:
            self.scene_json["retimeScene"] = True
        else:
            self.scene_json["retimeScene"] = False

        if self.cam_options["maya_cam"] == 2:
            self.scene_json["mayaCam"] = self.pub_maya_cam_path
        else:
            self.scene_json["mayaCam"] = False

        if self.cam_options["fbx_cam"] == 2:
            self.scene_json["fbxCam"] = self.pub_fbx_cam_path
        else:
            self.scene_json["fbxCam"] = False

        if self.cam_options["abc_cam"] == 2:
            self.scene_json["abcCam"] = self.pub_abc_cam_path
        else:
            self.scene_json["abcCam"] = False

        filepath = self.pub_maya_scene_path.replace(".mb", ".shotdb")
        try:
            f = open(filepath, "w")
            f.write(json.dumps(self.scene_json, sort_keys=True, indent=4, separators=(",", ":")))
        except:
            QtGui.QMessageBox.critical(None, "Error", "Writting Scene JSON Failed!", QtGui.QMessageBox.Ok)
            print traceback.format_exc()
            return False
        finally:
            f.close()

    def write_cam_json(self):
        self.cam_json = dict()
        self.cam_json["show"] = self.shot_info["show"]
        self.cam_json["seq"] = self.shot_info["seq"]
        self.cam_json["shot"] = self.shot_info["shot"]
        self.cam_json["from"] = self.pub_maya_scene_path
        self.cam_json["user"] = self.shot_info["user"]
        self.cam_json["task"] = "matchmove"
        self.cam_json["date"] = mc.date()
        self.cam_json["version"] = self.cam_ver
        self.cam_json["startFrame"] = self.scene_options["start_frame"]
        self.cam_json["endFrame"] = self.scene_options["end_frame"]
        self.cam_json["renderWidth"] = self.scene_options["render_width"]
        self.cam_json["renderHeight"] = self.scene_options["render_height"]
        self.cam_json["leftCamera"] = self.cam_options["cam_list"][0]
        if self.cam_options["stereo"] == 2:
            self.cam_json["rightCamera"] = self.cam_options["cam_list"][1]
            self.cam_json["stereo"] = True
        else:
            self.cam_json["rightCamera"] = ""
            self.cam_json["stereo"] = False

        if self.cam_options["overscan"] == 2:
            self.cam_json["overscan"] = True
        else:
            self.cam_json["overscan"] = False

        if self.scene_options["retime_scene"] == 2:
            self.cam_json["retimeScene"] = True
        else:
            self.cam_json["retimeScene"] = False

        filepath = self.pub_maya_cam_path.replace(".mb", ".shotdb")
        try:
            f = open(filepath, "w")
            f.write(json.dumps(self.cam_json, sort_keys=True, indent=4, separators=(",", ":")))
        except:
            QtGui.QMessageBox.critical(None, "Error", "Writting Camera JSON Failed!", QtGui.QMessageBox.Ok)
            print traceback.format_exc()
            return False
        finally:
            f.close()
