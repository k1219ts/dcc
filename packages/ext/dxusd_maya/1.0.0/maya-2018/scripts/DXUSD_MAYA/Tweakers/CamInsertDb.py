#coding:utf-8
from __future__ import print_function
import os, datetime
import requests

import pymongo
from pymongo import MongoClient

import dxConfig
from dxname import tag_parser

import maya.cmds as cmds
from pxr import Sdf, Usd, UsdGeom

from DXUSD.Tweakers.Tweaker import Tweaker, ATweaker
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD_MAYA.Message as msg

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'


class ACamInsertDb(ATweaker):
    def __init__(self, **kwargs):
        ATweaker.__init__(self, **kwargs)

        for geom in self.geomfiles:
            if 'ani' in geom:
                self.desc = 'fullCg'
                break

            try:
                self.F.USD.SetDecode(utl.BaseName(geom))
                if self.desc:
                    break
            except:
                pass

    def Treat(self):
        return var.SUCCESS

class CamInsertDb(Tweaker):
    ARGCLASS = ACamInsertDb

    def getUserInfo(self, user):
        params = {}
        params['api_key'] = 'c70181f2b648fdc2102714e8b5cb344d'
        params['code'] = user

        try:
            infos = requests.get("http://%s/dexter/search/user.php" %(dxConfig.getConf('TACTIC_IP')),
                                                                      params = params).json()
            return infos['department'].split(' ')[0]
        except:
            return 'cam'

    # def getLatestPubVersion(self, show, seq, shot, data_type,plateType=None):
    #     client = MongoClient(DB_IP)
    #     db = client[DB_NAME]
    #     coll = db[show]
    #     if plateType:
    #         recentDoc = coll.find_one({'show': show,
    #                                    'shot': shot,
    #                                    'data_type': data_type,
    #                                    'task_publish.plateType':plateType},
    #                                   sort=[('version', pymongo.DESCENDING)])
    #     else:
    #         recentDoc = coll.find_one({'show': show,
    #                                    'shot': shot,
    #                                    'data_type': data_type},
    #                                   sort=[('version', pymongo.DESCENDING)])
    #
    #     if recentDoc:
    #         return recentDoc['version']
    #     else:
    #         return 0

    def getDbRecord(self):
        #################### DB RECORD & NAMING MUDULE ####################
        # version = self.getLatestPubVersion(show=self.arg.show,
        #                               seq=self.arg.seq,
        #                               shot=self.arg.shot,
        #                               data_type='camera',
        #                               plateType=self.arg.desc) + 1

        version = utl.VerAsInt(self.arg.ver)
        print('#### version:', version)

        dbRecord = {'show': self.arg.show,
                    'sequence': self.arg.seq,
                    'shot': '%s_%s' % (self.arg.seq, self.arg.shot),
                    'task': self.getUserInfo(self.arg.user),
                    'version': version,
                    'data_type': 'camera',
                    'time': datetime.datetime.now().isoformat(),
                    'artist': self.arg.user,
                    'files': {'camera_path': [],
                              'imageplane_path': [],
                              'camera_geo_path': [],
                              'camera_loc_path': [],
                              'camera_asset_geo_path': [],
                              'camera_asset_loc_path': [],
                              'camera_asset_key_abc_path': [],
                              'camera_asset_key_path': []
                             },
                     'enabled':True,
                     'sub_cameras': [],
                     'sub_camera_id': [],
                     'task_publish': {'camera_type': 'camera',
                                      'plateType': self.arg.desc,
                                      'stereo': self.arg.isStereo,
                                      'render_width': str(cmds.getAttr("defaultResolution.width")),
                                      'render_height': str(cmds.getAttr("defaultResolution.height")),
                                      'dx_camera': [],
                                      'camera_only': False,
                                      'startFrame': self.arg.frameRange[0],
                                      'endFrame': self.arg.frameRange[1],
                                      'overscan': False
                                      }
                     }

        dbRecord['files']['maya_dev_file'] = [self.arg.scene]
        dbRecord['files']['maya_pub_file'] = [self.arg.pubScene]

        overscanValue = cmds.fileInfo("overscan_value", query=True)
        if overscanValue and overscanValue != '1.0':
            dbRecord['task_publish']['overscan'] = True
            dbRecord['task_publish']['overscan_value'] = overscanValue[0]

        mainDxnode = []
        for cam in self.arg.maincam:
            fullPath = cmds.ls(cam, l=True)
            dbRecord['task_publish']['dx_camera'].append(fullPath)
            dxnode = list(filter(None, fullPath[0].split('|')))[0]

        for geom in self.arg.geomfiles:
            geom = geom.replace('.usd', '.abc')
            if not geom in dbRecord['files']['camera_path']:
                dbRecord['files']['camera_path'].append(geom)

        if not dbRecord['task_publish']['camera_only']:
            for imp in self.arg.imgPlanefiles:
                imp = imp.replace('.usd', '.abc')
                dbRecord['files']['imageplane_path'].append(imp)

            for dxNode in self.arg.dummyfiles.keys():
                dummys = self.arg.dummyfiles[dxNode] + self.arg.dummyAbc[dxNode]
                for dummy in dummys:
                    dummy = dummy.replace('.usd', '.abc')

                    kinds = ''
                    if not 'dxCamera' in dummy:
                        kinds = 'asset_'

                    if '.geom' in dummy:
                        dbRecord['files']['camera_%sgeo_path' % (kinds)].append(dummy)
                    else:
                        dbRecord['files']['camera_%sloc_path' % (kinds)].append(dummy)

        try:
            dbRecord['files']['imageplane_json_path'] = [self.arg.impAttrfile]
            dbRecord['argv'] = self.arg
        except:
            pass

        return dbRecord

    def DoIt(self):
        dbRecord = self.getDbRecord()

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[dbRecord['show']]

        condition = {'data_type': dbRecord['data_type'],
                     'shot': dbRecord['shot'],
                     'version': dbRecord['version'],
                     'task_publish.plateType': dbRecord['task_publish']['plateType']}

        find = coll.find_one(condition)
        if find:    coll.update_one(condition, {'$set': dbRecord})
        else:       coll.insert_one(dbRecord)

        msg.debug('#'*50)
        msg.debug(dbRecord)
        msg.debug('#'*50)

        return var.SUCCESS
