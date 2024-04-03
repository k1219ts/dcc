import os
import traceback
import json

from MATCHMOVE.shotDB_common import *
#from PyQt4 import QtGui
from PySide import QtGui

import maya.cmds as mc
import maya.mel as mm


class CameraAssetAnim(object):
    def __init__(self, shot_info={}, scene_options={}, cam_options={}):

        self.shot_info = shot_info
        self.scene_options = scene_options
        self.sdb_list = self.scene_options["sdb_list"]
        self.cam_options = cam_options
        self.shot_path = os.path.join(
            self.shot_info["root"],
            self.shot_info["show"],
            "shot",
            self.shot_info["seq"],
            self.shot_info["shot"]
        )

    def publish(self):
        try:
            self.cam_orientation = "_left"
            #
            #
            # export scene to pub/scenes folder.
            if self.shot_info["dept"] == "animation":
                self.anim_pub_path = os.path.join(self.shot_path, "ani", "pub",
                                                  "scenes")  # /show/prat/shot/SHS/SHS_0420/ani/pub/scenes/
                self.pub_filename = "%s_layout" % self.shot_info["shot"]  # result: ELX_0100_layout
            elif self.shot_info["dept"] == "matchmove":
                self.anim_pub_path = os.path.join(self.shot_path, "layout", "pub",
                                                  "scenes")  # /show/prat/shot/SHS/SHS_0420/ani/pub/scenes/
                self.pub_filename = "%s_layout" % self.shot_info["shot"]  # result: ELX_0100_layout

            # renew imageplane
            if self.scene_options["full_cg"] == 2:
                print "full cg shot. 1"
            elif self.cam_options["renew_ip"] == 2:
                self.renewIpList = renewImagePlane()
            else:
                print "no renewip. 2"

            # bake camera.
            self.tempCamList = []
            for c in self.cam_options["cam_list"]:
                if not self.cam_options["stereo"] == 2:
                    self.cam_orientation = ""

                camera_transform = c
                camera_shape = mc.listRelatives(camera_transform, allDescendents=True, type="camera")[0]

                tmp_camera = mc.duplicate(camera_transform, un=True, rc=True)

                for i in range(100):
                    unlockAttr(tmp_camera[0] + ".tx")
                    unlockAttr(tmp_camera[0] + ".ty")
                    unlockAttr(tmp_camera[0] + ".tz")
                    unlockAttr(tmp_camera[0] + ".rx")
                    unlockAttr(tmp_camera[0] + ".ry")
                    unlockAttr(tmp_camera[0] + ".rz")
                    unlockAttr(tmp_camera[0] + ".sx")
                    unlockAttr(tmp_camera[0] + ".sy")
                    unlockAttr(tmp_camera[0] + ".sz")

                p_constraint = mc.parentConstraint(camera_transform, tmp_camera[0])

                # hide all objects.
                for panName in mc.getPanel(all=True):
                    if 'modelPanel' in panName: mc.isolateSelect(panName, state=1)

                try:
                    mc.parent(tmp_camera, world=True)
                    if self.cam_options["bake_cam"] == 2:
                        mc.bakeResults(tmp_camera,
                                       simulation=True,
                                       t=(self.scene_options["start_frame"], self.scene_options["end_frame"]),
                                       sb=True,
                                       at=["tx", "ty", "tz", "rx", "ry", "rz"],
                                       hi="below")
                except:
                    print "Camera is already in worldspace. It does not need to bake."

                # show all objects.
                for panName in mc.getPanel(all=True):
                    if 'modelPanel' in panName: mc.isolateSelect(panName, state=0)

                mc.delete(p_constraint)
                self.tempCamList.append(
                    mc.rename(tmp_camera[0], "%s_layoutCamera%s" % (self.shot_info["shot"], self.cam_orientation)))

                # I think this is not a good idea.
                for dup in tmp_camera:
                    if "imagePlane" in dup:
                        try:
                            if mc.nodeType(dup) == "transform":
                                mc.delete(dup)
                                print "deleted", dup
                        except:
                            pass

                self.cam_orientation = "_right"

            self.cam_options["cam_list"] = self.tempCamList
            #
            for c in self.cam_options["cam_list"]:
                camera_transform = c
                camera_shape = mc.listRelatives(camera_transform, allDescendents=True, type="camera")[0]

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
            # mc.setAttr(camera_shape+".panZoomEnabled", 0)

            # if self.cam_options["add_key"] == 2:
            # 	print "TODO: add keys for motion blur."
            # 	print "add_keys ok"

            make_dir(self.anim_pub_path)
            self.scene_ver = get_final_ver(self.anim_pub_path, self.pub_filename, "*.mb")
            if self.scene_ver == 0:
                print "There is no any Maya Scene file. V01 will be published."
            # QtGui.QMessageBox.information(None, "Information", "There is no any Maya Scene file.\nV01 will be published.", QtGui.QMessageBox.Ok)
            self.pub_maya_scene_path = os.path.join(self.anim_pub_path,
                                                    self.pub_filename + "_v%.2d" % (self.scene_ver + 1) + ".mb")
            try:
                mc.file(rename=self.pub_maya_scene_path)
                mc.file(save=True)
                print "Animation Scene Published! %s" % self.pub_maya_scene_path
                QtGui.QMessageBox.information(None, "Information",
                                              "Animation Scene Published!\n\n%s" % self.pub_maya_scene_path,
                                              QtGui.QMessageBox.Ok)
            except:
                QtGui.QMessageBox.critical(None, "Error", "Animation Scene Failed!\n\n%s" % self.pub_maya_scene_path,
                                           QtGui.QMessageBox.Ok)
                print traceback.format_exc()
                return False
            print "publishing ani scene: OK."

            # hide all objects.
            for panName in mc.getPanel(all=True):
                if 'modelPanel' in panName: mc.isolateSelect(panName, state=1)
            # convert imageplane to polyplane.
            if self.scene_options["full_cg"] == 2:
                self.polyIpList = []
                print "full cg shot. 2"
            else:
                self.polyIpList = convertPolyImagePlane(self.renewIpList)

            #
            #
            # export camera to camera folder.
            self.camera_pub_path = os.path.join(self.shot_path, "camera", "pub",
                                                "scenes")  # /show/prat/shot/SHS/SHS_0420/camera/pub/scenes/
            self.pub_filename = "%s_layout_camera" % (self.shot_info["shot"])  # result: ELX_0100_layout_camera

            mc.select(clear=True)
            for c in self.cam_options["cam_list"]:
                mc.select(c, add=True)
            if len(self.polyIpList) > 0:
                for i in self.polyIpList:
                    mc.select(i, add=True)

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
                    if len(self.polyIpList) > 0:
                        for i in self.polyIpList:
                            abc_job += "-root %s " % mc.ls(i, long=True)[0]

                    mc.loadPlugin("AbcExport.so", qt=True)
                    abc_job += "-uvWrite -frameRange %s %s -file %s" % (
                    self.scene_options["start_frame"], self.scene_options["end_frame"], self.pub_abc_cam_path)
                    print abc_job
                    mc.AbcExport(j=abc_job)
                    print "Alembic Camera Published! %s" % self.pub_abc_cam_path
                # QtGui.QMessageBox.information(None, "Information", "Alembic Camera Published!\n\n%s"%self.pub_abc_cam_path, QtGui.QMessageBox.Ok)
                except:
                    QtGui.QMessageBox.critical(None, "Error", "Exporting Alembic Camera Failed!", QtGui.QMessageBox.Ok)
                    print traceback.format_exc()
                    return False

            if self.scene_options["full_cg"] == 2:
                print "full cg shot. 3"
            else:
                mc.delete(self.polyIpList)
                for r in self.renewIpList:
                    mc.delete(mc.listRelatives(r, parent=True))

            try:
                mc.file(save=True)
                print "publishing cameras: OK."
            except:
                pass

            self.write_scene_json()
            self.write_cam_json()

            mc.select(clear=True)
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
        sceneJSON = dict()
        sceneJSON["show"] = self.shot_info["show"]
        sceneJSON["seq"] = self.shot_info["seq"]
        sceneJSON["shot"] = self.shot_info["shot"]
        sceneJSON["plate"] = self.cam_options["plate_type"]
        sceneJSON["user"] = self.shot_info["user"]
        sceneJSON["department"] = self.shot_info["dept"]
        sceneJSON["date"] = mc.date()
        sceneJSON["version"] = str(self.scene_ver + 1)
        sceneJSON["startFrame"] = self.scene_options["start_frame"]
        sceneJSON["endFrame"] = self.scene_options["end_frame"]
        sceneJSON["renderWidth"] = self.scene_options["render_width"]
        sceneJSON["renderHeight"] = self.scene_options["render_height"]

        sceneJSON["leftCamera"] = self.cam_options["cam_list"][0]
        if self.cam_options["stereo"] == 2:
            sceneJSON["rightCamera"] = self.cam_options["cam_list"][1]
            sceneJSON["stereo"] = True
        else:
            sceneJSON["rightCamera"] = ""
            sceneJSON["stereo"] = False

        sceneJSON["mayaScene"] = self.pub_maya_scene_path
        sceneJSON["mayaCam"] = self.pub_maya_cam_path

        if self.cam_options["fbx_cam"] == 2:
            sceneJSON["fbxCam"] = self.pub_fbx_cam_path
        else:
            sceneJSON["fbxCam"] = False

        if self.cam_options["abc_cam"] == 2:
            sceneJSON["abcCam"] = self.pub_abc_cam_path
        else:
            sceneJSON["abcCam"] = False

        if self.scene_options["full_cg"] == 2:
            sceneJSON["fullCG"] = True
        else:
            sceneJSON["fullCG"] = False

        filepath = self.pub_maya_scene_path.replace(".mb", ".shotdb")
        try:
            f = open(filepath, "w")
            f.write(json.dumps(sceneJSON, sort_keys=True, indent=4, separators=(",", ":")))
        except:
            QtGui.QMessageBox.critical(None, "Error", "Writting Scene JSON Failed!", QtGui.QMessageBox.Ok)
            print traceback.format_exc()
            return False
        finally:
            f.close()

    def write_cam_json(self):
        camJSON = dict()
        camJSON["show"] = self.shot_info["show"]
        camJSON["seq"] = self.shot_info["seq"]
        camJSON["shot"] = self.shot_info["shot"]
        camJSON["from"] = self.pub_maya_scene_path
        camJSON["user"] = self.shot_info["user"]
        camJSON["department"] = self.shot_info["dept"]
        camJSON["date"] = mc.date()
        camJSON["version"] = str(self.cam_ver + 1)
        camJSON["startFrame"] = self.scene_options["start_frame"]
        camJSON["endFrame"] = self.scene_options["end_frame"]
        camJSON["renderWidth"] = self.scene_options["render_width"]
        camJSON["renderHeight"] = self.scene_options["render_height"]
        camJSON["leftCamera"] = self.cam_options["cam_list"][0]
        if self.cam_options["stereo"] == 2:
            camJSON["rightCamera"] = self.cam_options["cam_list"][1]
            camJSON["stereo"] = True
        else:
            camJSON["rightCamera"] = ""
            camJSON["stereo"] = False

        if self.scene_options["full_cg"] == 2:
            camJSON["fullCG"] = True
        else:
            camJSON["fullCG"] = False

        #
        # export sdb_list
        sdbCamList = []
        if len(self.sdb_list) > 0:
            for sdb in self.sdb_list:
                jsonPath = mc.getAttr(sdb + ".json")
                if os.path.isfile(jsonPath):
                    fsdb = open(jsonPath, "r")
                    jsdb = json.load(fsdb)
                    fsdb.close()
                    mayaCamPath = os.path.split(jsdb["mayaCam"])[1]
                    # export layout transform
                    self.write_layout_json(sdb)
                    #
                    mayaCamPath = mayaCamPath.replace(".mb", ".layout")
                    sdbCamList.append(mayaCamPath)
                else:
                    QtGui.QMessageBox.critical(None, "Error", "Reading Camera JSON Failed!", QtGui.QMessageBox.Ok)
        else:
            print "no sdblist."
        camJSON["sdbList"] = sdbCamList
        ###################################

        filepath = self.pub_maya_cam_path.replace(".mb", ".shotdb")
        try:
            f = open(filepath, "w")
            f.write(json.dumps(camJSON, sort_keys=True, indent=4, separators=(",", ":")))
        except:
            QtGui.QMessageBox.critical(None, "Error", "Writting Camera JSON Failed!", QtGui.QMessageBox.Ok)
            print traceback.format_exc()
            return False
        finally:
            f.close()

    def write_layout_json(self, SDBNode):
        mmv_json = mc.getAttr(SDBNode + ".json")
        if os.path.isfile(mmv_json):
            f = open(mmv_json, "r")
            j = json.load(f)
            f.close()
        else:
            QtGui.QMessageBox.critical(None, "Error", "Reading Matchmove JSON Failed!", QtGui.QMessageBox.Ok)

        layoutPath = j["mayaCam"].replace(".mb", ".layout")
        layoutJSON = dict()
        lt = dict()

        try:
            sframe = int(float(self.scene_options["start_frame"]))
            eframe = int(float(self.scene_options["end_frame"]))
            for i in range(sframe, eframe):
                lt[i] = mc.xform(SDBNode, query=True, matrix=True, worldSpace=True)
        except:
            pass

        layoutJSON["from"] = self.pub_maya_scene_path
        roo = mc.getAttr(SDBNode + ".rotateOrder")
        if roo == 0:
            layoutJSON["rotateOrder"] = "xyz"
        elif roo == 1:
            layoutJSON["rotateOrder"] = "yzx"
        elif roo == 2:
            layoutJSON["rotateOrder"] = "zxy"
        elif roo == 3:
            layoutJSON["rotateOrder"] = "xzy"
        elif roo == 4:
            layoutJSON["rotateOrder"] = "yxz"
        elif roo == 5:
            layoutJSON["rotateOrder"] = "zyx"
        layoutJSON["layoutTranform"] = lt
        layoutJSON["mayaScene"] = j["mayaScene"]
        layoutJSON["mayaCam"] = j["mayaCam"]
        layoutJSON["show"] = j["show"]
        layoutJSON["seq"] = j["seq"]
        layoutJSON["shot"] = j["shot"]
        layoutJSON["overscan"] = j["overscan"]
        layoutJSON["plate"] = j["plate"]

        f = open(layoutPath, "w")
        f.write(json.dumps(layoutJSON, sort_keys=True, indent=4, separators=(",", ":")))
        f.close()
