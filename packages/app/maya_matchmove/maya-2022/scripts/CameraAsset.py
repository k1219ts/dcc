import os, getpass, datetime #
import traceback
import json

#from shotDB_common import *
import shotDB_common

from dxname import rulebook
from dxname import tag_parser

try:
    from PyQt4 import QtGui
except:
    from PySide2 import QtGui

import maya.cmds as mc
import maya.mel as mm
import sgCamera
import dplCommon
import sgCommon
import sgAnimation

import maya.cmds as cmds

import pymongo
from pymongo import MongoClient
DB_IP = "10.0.0.12:27017, 10.0.0.13:27017"
DB_NAME = 'PIPE_PUB'

def offsetKey():
    cam = cmds.ls(cameras=True)
    transform = []
    for i in cam:
        transform = cmds.listRelatives(i, p=True)

        startOffsetKey(transform[0], "tx", 1)
        startOffsetKey(transform[0], "ty", 1)
        startOffsetKey(transform[0], "tz", 1)

        startOffsetKey(transform[0], "rx", 1)
        startOffsetKey(transform[0], "ry", 1)
        startOffsetKey(transform[0], "rz", 1)

        startOffsetKey(transform[0], "sx", 1)
        startOffsetKey(transform[0], "sy", 1)
        startOffsetKey(transform[0], "sz", 1)

        startOffsetKey(i, "focalLength", 1)

        endOffsetKey(transform[0], "tx", 1)
        endOffsetKey(transform[0], "ty", 1)
        endOffsetKey(transform[0], "tz", 1)

        endOffsetKey(transform[0], "rx", 1)
        endOffsetKey(transform[0], "ry", 1)
        endOffsetKey(transform[0], "rz", 1)

        endOffsetKey(transform[0], "sx", 1)
        endOffsetKey(transform[0], "sy", 1)
        endOffsetKey(transform[0], "sz", 1)

        endOffsetKey(i, "focalLength", 1)


def endOffsetKey(name, attr, offset):
    camera = []
    tmp = []
    nType = ""
    con = 0
    endFrame = []
    endV = 0.0
    offsetV = 0.0
    value = 0.0
    node = name + "." + attr

    con = cmds.connectionInfo(node, id=True)
    if con == 1:
        tmp = cmds.listConnections(node)
        nType = cmds.nodeType(tmp[0])
        # print nType
        if nType.count("anim") > 0:
            endFrame = cmds.keyframe(node, q=True, lsl=True)
            endV = cmds.getAttr(node, t=endFrame[0])
            offsetV = cmds.getAttr(node, t=endFrame[0] - offset)
            value = endV - offsetV
            # print endFrame[0], endV, offsetV, value
            cmds.setKeyframe(node, itt="spline", ott="spline",
                             t=endFrame[0] + offset, at=attr, v=endV + value)


def startOffsetKey(name, attr, offset):
    camera = []
    tmp = []
    nType = ""
    con = 0
    startFrame = []
    startV = 0.0
    offsetV = 0.0
    value = 0.0
    node = name + "." + attr
    print node

    con = cmds.connectionInfo(node, id=True)
    print con
    if con == 1:
        tmp = cmds.listConnections(node)
        nType = cmds.nodeType(tmp[0])
        # print nType
        if nType.count("anim") > 0:
            startFrame = cmds.keyframe(node, q=True, a=True)
            startV = cmds.getAttr(node, t=startFrame[0])
            offsetV = cmds.getAttr(node, t=startFrame[0] - offset)
            value = startV - offsetV
            # print startFrame[0], startV, offsetV, value
            cmds.setKeyframe(node, itt="spline", ott="spline",
                             t=startFrame[0] - offset, at=attr,
                             v=startV - value)


# TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
"""
class SgCameraMMV(sgCamera.ExportCamera, object):
    def __init__(self, Path=None, Start=None, End=None, Step=1.0,
                 nameRoot=None, dbRecord=None):

        super(SgCameraMMV, self).__init__(Path=Path, Start=Start, End=End, Step=Step)
        self.nameRoot = nameRoot
        self.dbRecord = dbRecord

        #self.maya_scene_path = self.nameRoot.camera.matchmove.product['maya_pub_file']
        self.export_geo_loc = False

        self.target_cameras = []
        self.out_camera_path = self.nameRoot.camera.matchmove.product['camera_path']
        self.out_camgeo_path = self.nameRoot.camera.matchmove.product['camera_geo_path']
        self.out_camloc_path = self.nameRoot.camera.matchmove.product['camera_loc_path']

        self.out_imgplane_path = self.nameRoot.camera.matchmove.product['imageplane_path']
        self.out_imgplane_json_path = self.nameRoot.camera.matchmove.product['imageplane_json_path']
        self.out_panzoom_path = os.path.splitext(self.nameRoot.camera.matchmove.product['panzoom_json_path'])[0]

    def getCameras(self):
        for camera in self.target_cameras:
            cShape = mc.listRelatives(camera, f=1)[0]
            cTrans = mc.listRelatives( cShape, p=True, f=True )[0]
            self.m_exportCameras.append( cTrans )
            if mc.getAttr( '%s.renderable' % cShape ):
                trans = mc.listRelatives( cShape, p=True )[0]
                name = trans.split(':')[-1]
                self.m_logDict['render_camera'].append( name )

    def export_abc(self):
        # TO EXPORT ADDITIONAL ABC AT ONCE
        start = self.m_start - 1
        end = self.m_end + 1

        # SET NAME
        abcFile = self.out_camera_path

        options = '-v -j "-ef -worldSpace -fr %s %s -s %s' % (
        start, end, self.m_step)

        for i in self.m_createCameras:
            options += ' -rt %s' % i
        options += ' -f %s"' % abcFile
        self.dbRecord['files']['camera_path'] = [self.out_camera_path]

        # EXPORT IMAGE PLANES
        if self.m_polyPlanes:
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            for i in self.m_polyPlanes:
                options += ' -rt %s' % i
            options += ' -f %s"' % self.out_imgplane_path
            self.dbRecord['files']['imageplane_path'] = [self.out_imgplane_path]

        # CAM_GEO / CAM_LOC
        if mc.ls('cam_geo'):
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            options += ' -rt cam_geo'
            options += ' -f %s"' % self.out_camgeo_path
            self.dbRecord['files']['camera_geo_path'] = [self.out_camgeo_path]

        if mc.ls('cam_loc'):
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            options += ' -rt cam_loc'
            options += ' -f %s"' % self.out_camloc_path
            self.dbRecord['files']['camera_loc_path'] = [self.out_camloc_path]

        if self.export_geo_loc:
            # ASSET GEO / ASSET LOC
            assetGeoList = mc.ls('*_geo')
            assetLocList = mc.ls('*_loc')
            if 'cam_geo' in assetGeoList:
                assetGeoList.remove('cam_geo')
            if 'cam_loc' in assetLocList:
                assetLocList.remove('cam_loc')

            if assetGeoList:
                self.dbRecord['files']['camera_asset_geo_path'] = []

                for geo in assetGeoList:
                    options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % geo
                    self.nameRoot.camera.matchmove.flag['ASSET_GEO'] = geo
                    geoPath = self.nameRoot.camera.matchmove.product['camera_asset_geo_path']
                    options += ' -f %s"' % geoPath
                    self.dbRecord['files']['camera_asset_geo_path'].append(geoPath)

            if assetLocList:
                self.dbRecord['files']['camera_asset_loc_path'] = []
                for loc in assetLocList:
                    options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % loc
                    self.nameRoot.camera.matchmove.flag['ASSET_LOC'] = loc
                    locPath = self.nameRoot.camera.matchmove.product['camera_asset_loc_path']
                    options += ' -f %s"' % locPath
                    self.dbRecord['files']['camera_asset_loc_path'].append(locPath)

        print options

        mm.eval('AbcExport %s' % options)
        self.m_logDict['abc_camera'] = abcFile

    def export_file(self):
        if not os.path.exists(self.m_Path):
            os.makedirs(self.m_Path)

        # if self.m_enable_maya:
        #     self.export_maya(self.maya_scene_path)

        self.export_abc()

        #pzFile = os.path.join(self.m_Path, '%s_camera.panzoom' % self.m_baseName)
        # sgCamera.exportPanZoom(self.out_panzoom_path, self.m_exportCameras,
        #               self.m_start, self.m_end,self.m_username)
        if sgCamera.export_2DPanZoom(self.out_panzoom_path, self.m_exportCameras,
                                     self.m_start, self.m_end,self.m_username):
            self.dbRecord['files']['panzoom_json_path'] = [self.out_panzoom_path+'.json']
            self.dbRecord['files']['panzoom_nuke_path'] = [self.out_panzoom_path + '.nk']

        # imageplane
        if self.m_imagePlane:
            #fn = os.path.join(self.m_Path, '%s_camera.imageplane' % self.m_baseName)
            dplCommon.writeJsonLog(File=self.out_imgplane_json_path,
                                   Data={'ImagePlane': self.m_imagePlane},
                                   Context=mc.file(q=True, sn=True),
                                   User=self.m_username)
            self.dbRecord['files']['imageplane_json_path'] = [self.out_imgplane_json_path]

    # def export_maya(self, mayaFile):
    #     mc.select(self.m_createCameras + self.m_polyPlanes)
    #     mc.file(mayaFile, force=True, options='v=0;', typ='mayaBinary', pr=False,
    #               es=True)
    #     self.m_logDict['maya_camera'] = mayaFile
    #
    #     # FOR DB
    #     self.dbRecord['files']['maya_dev_file'] = [mc.file(q=True, sn=True)]
    #     self.dbRecord['files']['maya_pub_file'] = [mayaFile]

    def getRecord(self):
        return self.dbRecord
"""
class SgCameraMMV(sgCamera.ExportCamera, object):
    def __init__(self, Path=None, Start=None, End=None, Step=1.0,
                 nameRoot=None, dbRecord=None):

        super(SgCameraMMV, self).__init__(Path=Path, Start=Start, End=End, Step=Step)
        self.nameRoot = nameRoot
        self.dbRecord = dbRecord

        self.export_geo_loc = False

        self.camera_pub_path = os.path.join(Path, 'camera','pub','scenes')
        self.mmv_pub_path = os.path.join(Path, 'matchmove','pub','scenes')
        self.mmv_dev_path = os.path.join(Path, 'matchmove', 'dev', 'scenes')

        self.target_cameras = []

        self.out_camera_path = os.path.join(self.camera_pub_path,
                                            os.path.basename(self.nameRoot.camera.matchmove.product['camera_path']))

        self.out_camgeo_path = os.path.join(self.mmv_pub_path,
                                            os.path.basename(self.nameRoot.camera.matchmove.product['camera_geo_path']))

        self.out_camloc_path = os.path.join(self.mmv_pub_path,
                                            os.path.basename(self.nameRoot.camera.matchmove.product['camera_loc_path']))

        self.out_imgplane_path = os.path.join(self.mmv_pub_path,
                                              os.path.basename(self.nameRoot.camera.matchmove.product['imageplane_path']))

        self.out_imgplane_json_path = os.path.join(self.mmv_pub_path,
                                                   os.path.basename(self.nameRoot.camera.matchmove.product['imageplane_json_path']))
        self.out_panzoom_path = os.path.join(self.mmv_pub_path,
                                             os.path.splitext(os.path.basename(self.nameRoot.camera.matchmove.product['panzoom_json_path']))[0]
                                             )

    def getCameras(self):
        for camera in self.target_cameras:
            cShape = mc.listRelatives(camera, f=1)[0]
            cTrans = mc.listRelatives( cShape, p=True, f=True )[0]
            self.m_exportCameras.append( cTrans )
            if mc.getAttr( '%s.renderable' % cShape ):
                trans = mc.listRelatives( cShape, p=True )[0]
                name = trans.split(':')[-1]
                self.m_logDict['render_camera'].append( name )

    def export_abc(self):
        # TO EXPORT ADDITIONAL ABC AT ONCE
        start = self.m_start - 1
        end = self.m_end + 1

        # SET NAME
        abcFile = self.out_camera_path

        options = '-v -j "-ef -worldSpace -fr %s %s -s %s' % (
        start, end, self.m_step)

        for i in self.m_createCameras:
            options += ' -rt %s' % i
        options += ' -f %s"' % abcFile
        self.dbRecord['files']['camera_path'] = [self.out_camera_path]

        # EXPORT IMAGE PLANES
        if self.m_polyPlanes:
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            for i in self.m_polyPlanes:
                options += ' -rt %s' % i
            options += ' -f %s"' % self.out_imgplane_path
            self.dbRecord['files']['imageplane_path'] = [self.out_imgplane_path]

        # CAM_GEO / CAM_LOC
        if mc.ls('cam_geo'):
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            options += ' -rt cam_geo'
            options += ' -f %s"' % self.out_camgeo_path
            self.dbRecord['files']['camera_geo_path'] = [self.out_camgeo_path]

        if mc.ls('cam_loc'):
            options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
            options += ' -rt cam_loc'
            options += ' -f %s"' % self.out_camloc_path
            self.dbRecord['files']['camera_loc_path'] = [self.out_camloc_path]

        if self.export_geo_loc:
            # ASSET GEO / ASSET LOC
            assetGeoList = mc.ls('*_geo')
            assetLocList = mc.ls('*_loc')
            if 'cam_geo' in assetGeoList:
                assetGeoList.remove('cam_geo')
            if 'cam_loc' in assetLocList:
                assetLocList.remove('cam_loc')

            if assetGeoList:
                self.dbRecord['files']['camera_asset_geo_path'] = []

                for geo in assetGeoList:
                    options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % geo
                    self.nameRoot.camera.matchmove.flag['ASSET_GEO'] = geo
                    geoPath = self.nameRoot.camera.matchmove.product['camera_asset_geo_path']
                    # TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
                    geoPath = os.path.join(self.mmv_pub_path, os.path.basename(geoPath))
                    #
                    options += ' -f %s"' % geoPath
                    self.dbRecord['files']['camera_asset_geo_path'].append(geoPath)

            if assetLocList:
                self.dbRecord['files']['camera_asset_loc_path'] = []

                for loc in assetLocList:
                    options += ' -j "-uv -worldSpace -fr %s %s' % (start, end)
                    options += ' -rt %s' % loc
                    self.nameRoot.camera.matchmove.flag['ASSET_LOC'] = loc
                    locPath = self.nameRoot.camera.matchmove.product['camera_asset_loc_path']
                    # TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
                    locPath = os.path.join(self.mmv_pub_path, os.path.basename(locPath))
                    #
                    options += ' -f %s"' % locPath
                    self.dbRecord['files']['camera_asset_loc_path'].append(locPath)

        print options

        mm.eval('AbcExport %s' % options)
        self.m_logDict['abc_camera'] = abcFile

    def export_file(self):
        if not os.path.exists(self.m_Path):
            os.makedirs(self.m_Path)

        # if self.m_enable_maya:
        #     self.export_maya(self.maya_scene_path)

        self.export_abc()

        #pzFile = os.path.join(self.m_Path, '%s_camera.panzoom' % self.m_baseName)
        # sgCamera.exportPanZoom(self.out_panzoom_path, self.m_exportCameras,
        #               self.m_start, self.m_end,self.m_username)
        if sgCamera.export_2DPanZoom(self.out_panzoom_path, self.m_exportCameras,
                                     self.m_start, self.m_end,self.m_username):
            self.dbRecord['files']['panzoom_json_path'] = [self.out_panzoom_path+'.json']
            self.dbRecord['files']['panzoom_nuke_path'] = [self.out_panzoom_path + '.nk']

        # imageplane
        if self.m_imagePlane:
            #fn = os.path.join(self.m_Path, '%s_camera.imageplane' % self.m_baseName)
            dplCommon.writeJsonLog(File=self.out_imgplane_json_path,
                                   Data={'ImagePlane': self.m_imagePlane},
                                   Context=mc.file(q=True, sn=True),
                                   User=self.m_username)
            self.dbRecord['files']['imageplane_json_path'] = [self.out_imgplane_json_path]

    def getRecord(self):
        return self.dbRecord


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

        self.dxnameRoot = rulebook.Coder()
        self.dxnameRoot.load_rulebook('/backstage/libs/python_lib/dxname/name_for_publish.yaml')
        self.dxnameRoot.flag['PROJECT'] = self.shot_info["show"]
        self.dxnameRoot.flag['SEQUENCE'] = self.shot_info["seq"]
        self.dxnameRoot.flag['SHOT'] = self.shot_info["shot"]

        self.version = shotDB_common.getPubVersion(show = self.shot_info['show'],
                                                   seq= self.shot_info['seq'],
                                                   shot= self.shot_info['shot'],
                                                   data_type='camera',
                                                   plateType=self.cam_options["plate_type"]) + 1

        self.dxnameRoot.flag['VER'] = 'v'+str(self.version).zfill(2)
        self.dxnameRoot.camera.matchmove.flag['PLATE'] = self.cam_options['plate_type']

        self.dbRecord = {}
        self.dbRecord['show'] = self.shot_info["show"]
        self.dbRecord['sequence'] = self.shot_info["seq"]
        self.dbRecord['shot'] = self.shot_info["shot"]
        self.dbRecord['task'] = 'matchmove'
        self.dbRecord['data_type'] = 'camera'
        self.dbRecord['artist'] = getpass.getuser()
        self.dbRecord['tags'] = tag_parser.run(self.dxnameRoot.camera.matchmove.product['maya_pub_file'])
        self.dbRecord['time'] = datetime.datetime.now().isoformat()
        self.dbRecord['files'] = {}
        self.dbRecord['version'] = self.version

    def publish_cam(self):

        # offsetKey
        try:
            offsetKey()
        except:
            # maybe camera locked??
            pass

        # RENEW IMAGEPLANE
        if self.cam_options["iplane"] == 2:
            shotDB_common.renewImagePlane()

        # LOCK CAMERA
        self.lock_camera()

        # TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
        """
        # MAKE ROOT FOLDER IF NOT EXISTS
        pubRoot = self.dxnameRoot.camera.matchmove.product['root']
        if not(os.path.exists(pubRoot)):
            os.makedirs(pubRoot)
        
        sg = SgCameraMMV(Path=pubRoot, nameRoot=self.dxnameRoot,
                         dbRecord=self.dbRecord)        
        sg.nameRoot = self.dxnameRoot
        """

        shotPath = self.dxnameRoot.product['shot_path']
        sg = SgCameraMMV(Path=shotPath, nameRoot=self.dxnameRoot,
                         dbRecord=self.dbRecord)

        sg.target_cameras = self.cam_options["cam_list"]
        sg.export_geo_loc = self.scene_options['abc_scene']
        sg.doIt()

        self.dbRecord = sg.getRecord()

        # MAYA SCENE FILE SAVE AS TO PUB DIR
        if self.scene_options["maya_scene"]:
            maya_scene_path = self.dxnameRoot.camera.matchmove.product['maya_pub_file']
            # TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
            maya_scene_path = os.path.join(shotPath, 'matchmove', 'pub','scenes',
                                           os.path.basename(maya_scene_path))
            #
            mc.file(maya_scene_path, ea=True, options='v=0;', typ='mayaBinary',
                    pr=True)

            self.dbRecord['files']['maya_dev_file'] = [mc.file(q=True, sn=True)]
            self.dbRecord['files']['maya_pub_file'] = [maya_scene_path]

        # EXPORT RIG KEY JSON IF EXISTS
        if cmds.ls(type='dxRig', rn=True):
            self.dbRecord['files']['camera_asset_key_path'] = []

            for assetKey in cmds.ls(type='dxRig', rn=True):
                # [SINGLE RIG], JSON FILE NAME -> sgAnimation.write()
                self.dxnameRoot.camera.matchmove.flag['ASSET_KEY'] = assetKey
                rigJson = self.dxnameRoot.camera.matchmove.product['camera_asset_key_path']
                # TODO: PUBLISH FROM DXNAME / PIPELINE 2.0 PHASE 2
                rigJson = os.path.join(shotPath, 'matchmove', 'pub', 'scenes',
                                       os.path.basename(rigJson))
                #
                sgAnimation.write([assetKey], rigJson)
                self.dbRecord['files']['camera_asset_key_path'].append(rigJson)

        # EXPORT JSON
        self.write_scene_json()
        self.write_cam_json()

        taskDic = {}
        taskDic["plateType"] = self.cam_options["plate_type"]
        taskDic["startFrame"] = float(self.scene_options["start_frame"])
        taskDic["endFrame"] = float(self.scene_options["end_frame"])
        taskDic["renderWidth"] = self.scene_options["render_width"]
        taskDic["renderHeight"] = self.scene_options["render_height"]

        if self.scene_options["retime_scene"] == 2:
            taskDic["retimeScene"] = True
        else:
            taskDic["retimeScene"] = False

        if self.cam_options["overscan"] == 2:
            taskDic["overscan"] = True
        else:
            taskDic["overscan"] = False

        if self.cam_options["stereo"] == 2:
            taskDic["isStereo"] = True
        else:
            taskDic["isStereo"] = False


        self.dbRecord['task_publish'] = taskDic
        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[self.shot_info["show"]]
        result = coll.insert_one(self.dbRecord)
        print "db insert : ", result
        mc.select(clear=True)
        # TODO: SHOULD RECORD DATABASE
        # TODO: COLLECT GENERATED FILE FROM SGCAMERA
        # TODO: MAYBE MAKE JSON FILE OF RESULT LIKE scene_json cam_json

        return True

    def lock_camera(self):
        for c in self.cam_options["cam_list"]:
            camera_transform = c
            camera_shape = mc.listRelatives(camera_transform,
                                            allDescendents=True,
                                            type="camera")[0]

            if self.cam_options["lock_cam"] == 2:
                mc.setAttr(camera_transform+".tx", l=True)
                mc.setAttr(camera_transform+".ty", l=True)
                mc.setAttr(camera_transform+".tz", l=True)
                mc.setAttr(camera_transform+".rx", l=True)
                mc.setAttr(camera_transform+".ry", l=True)
                mc.setAttr(camera_transform+".rz", l=True)
                mc.setAttr(camera_transform+".sx", l=True)
                mc.setAttr(camera_transform+".sy", l=True)
                mc.setAttr(camera_transform+".sz", l=True)
                mc.setAttr(camera_shape+".focalLength", l=True)
                mc.setAttr(camera_shape+".horizontalFilmAperture", l=True)
                mc.setAttr(camera_shape+".verticalFilmAperture", l=True)
                print "lock ok"

    def write_scene_json(self):
        self.scene_json = dict()
        self.scene_json["show"] = self.shot_info["show"]
        self.scene_json["seq"] = self.shot_info["seq"]
        self.scene_json["shot"] = self.shot_info["shot"]
        self.scene_json["plate"] = self.cam_options["plate_type"]
        self.scene_json["user"] = self.shot_info["user"]
        self.scene_json["task"] = "matchmove"
        self.scene_json["date"] = mc.date()
        #self.scene_json["version"] = self.scene_ver
        self.scene_json['version'] = self.shot_info["version"]
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

        if self.dbRecord['files']['camera_asset_geo_path']:
            self.scene_json['abc_scene'] = self.dbRecord['files']['camera_asset_geo_path']
        else:
            self.scene_json['abc_scene'] = []

        if self.scene_options["retime_scene"] == 2:
            self.scene_json["retimeScene"] = True
        else:
            self.scene_json["retimeScene"] = False

        if self.cam_options["abc_cam"] == 2:
            self.scene_json["abcCam"] = self.dbRecord['files']['camera_path'][0]
        else:
            self.scene_json["abcCam"] = False

        filepath = self.dbRecord['files']['maya_pub_file'][0].replace(".mb", ".shotdb")
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
        #self.cam_json["from"] =  self.dxnameRoot.camera.matchmove.product['maya_pub_file']
        self.cam_json["from"] = self.dbRecord['files']['maya_pub_file'][0]
        self.cam_json["user"] = self.shot_info["user"]
        self.cam_json["task"] = "matchmove"
        self.cam_json["date"] = mc.date()
        self.cam_json["version"] = self.shot_info["version"]
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


        #filepath =  self.dxnameRoot.camera.matchmove.product['maya_pub_file'].replace(".mb", ".shotdb")
        filepath = self.dbRecord['files']['camera_path'][0].replace(".abc", ".shotdb")

        try:
            f = open(filepath, "w")
            f.write(json.dumps(self.cam_json, sort_keys=True, indent=4, separators=(",", ":")))
        except:
            QtGui.QMessageBox.critical(None, "Error", "Writting Camera JSON Failed!", QtGui.QMessageBox.Ok)
            print traceback.format_exc()
            return False
        finally:
            f.close()
