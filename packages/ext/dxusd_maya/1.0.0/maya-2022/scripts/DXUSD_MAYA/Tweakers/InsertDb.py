#coding:utf-8
from __future__ import print_function
import os, datetime
import requests
import getpass
import pprint

import pymongo
from pymongo import MongoClient

import dxConfig
from dxname import tag_parser
from tactic_client_lib import TacticServerStub

import maya.cmds as cmds
from pxr import Sdf, Usd, UsdGeom

from DXUSD.Tweakers.Tweaker import Tweaker, ATweaker
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD_MAYA.Message as msg

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'PIPE_PUB'
API_KEY = "c70181f2b648fdc2102714e8b5cb344d"


class AInsertDb(ATweaker):
    def __init__(self, **kwargs):
        ATweaker.__init__(self, **kwargs)

    def Treat(self):
        return var.SUCCESS

class InsertDb(Tweaker):
    ARGCLASS = AInsertDb

    def showNameToCode(self, show):
        showCode = None
        if 'pipe' in show:
            showCode = 'testshot'
        elif 'cdh1' in  show:
            show = 'cdh'

        params = {'api_key': API_KEY,
                  'name': show,
                  'status' : 'in_progres'}
        infos = requests.get("http://%s/dexter/search/project.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

        if infos:
            showCode = infos[0]['code']

        return showCode

    def getTaskArtist(self):
        artist = ''
        prjCode = self.showNameToCode(self.arg.show)
        if 'shot' in self.arg:          extraCode = self.arg.N.SHOTNAME
        else:                           extraCode = self.arg.asset

        if   'rig' in self.arg.task:    context = 'creature/rigging'
        elif 'groom' in self.arg.task:  context = 'texture'
        elif 'ani' in self.arg.task:    context = 'animation'
        else:                           context = self.arg.task

        server = TacticServerStub(login='cgsup', password='dexter',
                                  server=dxConfig.getConf('TACTIC_IP'),
                                  project=prjCode)
        shot_exp = "@SOBJECT(sthpw/task"
        shot_exp += "['project_code','{SHOW}']".format(SHOW=prjCode)
        shot_exp += "['extra_code','{CODE}']".format(CODE=extraCode)
        shot_exp += "['context', '{TASK}']".format(TASK=context)
        shot_exp += ")"
        msg.debug('TACTIC query:', shot_exp)

        try:
            infos = server.eval(shot_exp)
            if infos:
                artist = infos[0]['assigned']
        except:
            print('### task info query ERROR:', shot_exp)
            pass

        return artist

    def getDags(self, level=10):
        camExclList = ['persp', 'top', 'front', 'side', 'left', 'right', 'back', 'bottom']
        dagList = []
        for asm in cmds.ls(assemblies=True):
            if str(asm) in camExclList:
                continue
            dagList.append(asm)

        dagWalked = []
        def dagWalk(node, nodeList, level, depth=1):
            if not 'Shape' in node:
                nodeList.append(node)
            if level <= depth:
                return
            children = cmds.listRelatives(node, children=True, fullPath=True)
            if children != None:
                for child in children:
                    dagWalk(child, nodeList, level, depth+1)

        for dag in dagList:
            dagWalk(dag, dagWalked, level)

        nodeDict = {}
        for dag in dagWalked:
            node = nodeDict
            dagTok = dag.split('|')
            for de in dagTok:
                if de:
                    node = node.setdefault(de, dict())

        return nodeDict

    def getDbRecord(self):
        dbRecord = {'show': self.arg.show,
                    'task': self.arg.task,
                    'asset': self.arg.asset,
                    'data_type': self.arg.task,
                    'time': datetime.datetime.now().isoformat(),
                    'artist': self.getTaskArtist(),
                    'files': {'maya_dev_file': [self.arg.scene],
                              'maya_pub_file': [self.arg.pubscene],
                              'masterUsd': [self.arg.master]
                             },
                    'argv': self.arg,
                    'hierarchy': self.getDags(),
                    'enabled':True
                    }

        if 'seq' in self.arg:
            dbRecord['sequence'] = self.arg.seq
        if 'shot' in self.arg:
            dbRecord['shot'] = self.arg.N.SHOTNAME

        if 'nslyr' in self.arg:
            dbRecord['nslyr'] = self.arg.nslyr

        if 'asset' not in self.arg:
            try:
                tmp = self.arg.N.MAYA.Decode(self.arg.desc)
                dbRecord['asset'] = tmp.asset
            except:
                pass

        if 'ver' not in self.arg:
            if self.arg.task in ['ani', 'agent']:
                dbRecord['version'] = utl.VerAsInt(self.arg.nsver)
            else:
                self.arg.F.MAYA.SetDecode(os.path.basename(self.arg.scene))
                dbRecord['version'] = utl.VerAsInt(self.arg.ver)
        else:
            dbRecord['version'] = utl.VerAsInt(self.arg.ver)

        if self.arg.has_attr('user'):
            dbRecord['user'] = self.arg.user
        else:
            dbRecord['user'] = ''

        if self.arg.has_attr('node'):
            dbRecord['nodes'] = [self.arg.node]
        elif self.arg.has_attr('nodes'):
            dbRecord['nodes'] = self.arg.nodes

        if self.arg.has_attr('abcFiles'):
            dbRecord['files']['abc_files'] = self.arg.abcFiles
        if self.arg.has_attr('texAttrUsd'):
            dbRecord['files']['texAttrUsd'] = [self.arg.texAttrUsd]

        return dbRecord

    def DoIt(self):
        if not self.arg.show:
            return var.SUCCESS
        dbRecord = self.getDbRecord()

        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[dbRecord['show']]

        condition = {'data_type': dbRecord['data_type'],
                     'version': dbRecord['version']}

        if 'shot' in dbRecord:
            condition['shot'] = dbRecord['shot']
        if 'asset' in dbRecord:
            condition['asset'] = dbRecord['asset']
        if 'nslyr' in dbRecord:
            condition['nslyr'] = dbRecord['nslyr']

        find = coll.find_one(condition)
        if find:    coll.update_one(condition, {'$set': dbRecord})
        else:       coll.insert_one(dbRecord)

        # print('#'*50)
        # pprint.pprint(dbRecord)
        # print('#'*50)

        return var.SUCCESS
