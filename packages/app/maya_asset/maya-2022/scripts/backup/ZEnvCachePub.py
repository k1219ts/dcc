################################################
#
# author        : Dexter RND daeseok.chae
# create        : 2017.06.05
# filename      : ZEnvCachePub.py
# last update   : 2017.06.08
#
################################################

import getpass
import datetime

import maya.mel as mel
import maya.cmds as cmds

# 2017.06.05 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
import pymongo
from pymongo import MongoClient

import dxConfig

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"

def getPubVersion(show, task, data_type, asset_name = "", shot = ""):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print show, task, data_type, asset_name
    record = {'show': show,
                               'task': task,
                               'data_type': data_type}
    if asset_name != "":
        record['asset_name'] = asset_name

    if shot != "":
        record['shot'] = shot

    recentDoc = coll.find_one(record,
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1

class ZEnvCachePub():
    def __init__(self):
        print "hello world!"
        self.outputPath = mel.eval("textField -q -tx ZEnvCacheGenWin_Path")
        self.nodeNames = mel.eval('ZTextScrollList_GetAllItems("ZEnvCacheGenList")')

    def doIt(self):
        print "DoIt!"
        self.ppRulebook = rulebook.Coder()
        self.ppRulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_publish.yaml")

        try:
            if self.outputPath.startswith('/netapp/dexter/show'):
                self.outputPath = self.outputPath.replace('/netapp/dexter/show', '/show')

            localOutputPath = self.outputPath.split('/cache')[0] + '/cache'

            productName = 'root'

            if 'shot' in localOutputPath:
                productName = 'shot'

            decodingRule = self.ppRulebook.asset.zenv_cache.decode(localOutputPath, product_name = productName)
            self.ppRulebook.flag['PROJECT'] = decodingRule['PROJECT']
            if productName == 'shot':
                self.ppRulebook.flag['SHOT'] = decodingRule['SHOT']
                self.ppRulebook.flag['SEQUENCE'] = decodingRule['SEQUENCE']
            else:
                self.ppRulebook.asset.flag['TYPE'] = decodingRule['TYPE']
                self.ppRulebook.asset.flag['ASSET'] = decodingRule['ASSET']
            # self.ppRulebook.flag['VER'] = decodingRule['VER']

            typeTaskCoder = self.ppRulebook.asset.zenv_cache

            files = {}

            files['root'] = [self.outputPath]

            task_publish = {}

            # task 1
            task_publish['ZEnvGroupNodes'] = self.nodeNames

            # # task 2
            # zenvGroupNodes = {}
            #
            # for node in self.nodeNames:
            #     zenvGroupNodes[node] = mel.eval('ZTextScrollList_GetAllItems("ZEnvCacheGenList")')
            #
            # task_publish['ZEnvGroupNodes'] = zenvGroupNodes


            record = {}

            record['show'] = str(self.ppRulebook.flag["PROJECT"])
            record['task'] = "asset"
            record['data_type'] = 'zenv_cache'
            record['files'] = files
            record['task_publish'] = task_publish
            record['time'] = datetime.datetime.now().isoformat()
            record['enabled'] = True
            record['artist'] = getpass.getuser()
            record['maya_version'] = 'maya2_2017'
            record['tags'] = tag_parser.run(self.outputPath)

            # record Shot
            if productName == 'shot':
                record['shot'] = str(self.ppRulebook.flag['SHOT'])
                record['version'] = getPubVersion(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type='zenv_cache',
                                                  shot = str(self.ppRulebook.flag['SHOT']))
            # record other
            else:
                record['version'] = getPubVersion(show=str(self.ppRulebook.flag["PROJECT"]),
                                                  task='asset',
                                                  data_type='zenv_cache',
                                                  asset_name=str(self.ppRulebook.asset.flag["ASSET"]))

                record['asset_type'] = str(self.ppRulebook.asset.flag["TYPE"])
                record['asset_name'] = str(self.ppRulebook.asset.flag["ASSET"])

            COLLNAME = str(self.ppRulebook.flag["PROJECT"])
            client = MongoClient(DBIP)
            database = client[DBNAME]
            dbColl = database[COLLNAME]

            dbColl.insert_one(record)
            print "success db write", record

        except Exception as e:
            COLLNAME = "puberror"
            dbName = "test"
            client = MongoClient(DBIP)
            database = client[dbName]
            dbColl = database[COLLNAME]

            record = {"user": getpass.getuser(),
                      "errorMsg": str(e),
                      "time": datetime.datetime.now().isoformat(),
                      "project": str(self.ppRulebook.flag["PROJECT"]),
                      "asset": str(self.ppRulebook.asset.flag["ASSET"]),
                      "type": "zenv_cache",
                      "outputpath": self.outputPath,
                      "maya_version": "maya2_2017"}

            dbColl.insert_one(record)