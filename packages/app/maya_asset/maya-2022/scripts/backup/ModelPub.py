#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#    2015.06.03 $1
#-------------------------------------------------------------------------------

import os, sys
import datetime
import getpass
import string
import subprocess

from pymel.all import *
import maya.cmds as cmds
from dxname import tag_parser
from pymongo import MongoClient
import pymongo

import dxExportMesh

import dxConfig

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"

def pub( filename, doMaya, doAbc, doTex, expGrp ):
    nodes = []
    if expGrp:
        nodes = expGrp.split(',')

    exportClass = dxExportMesh.ExportMesh( filename, nodes )
    exportClass.mesh_export( maya=doMaya, abc=doAbc, tx=doTex )

    if 'pub' in filename:
        recordDB(exportClass.outputFilePath, outputPath = filename)

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

def recordDB(outputFileInfo, outputPath):
    data_type = 'model'

    if "hair" in outputPath:
        data_type = 'zenn'

    showName = ""
    shotName = ""
    assetType = ""
    assetName = ""

    maya_version = 'maya2_'

    try:
        files = outputFileInfo

        splitText = string.split(outputPath, '/')

        showName = splitText[2]
        shotName = splitText[3]
        assetType = splitText[4]
        assetName = splitText[5]

        if '2017' in __file__:
            maya_version += '2017'
        else:
            maya_version += '2016.5'
        maya_version += '_tractor'

        ######### 내일 이 밑에서부터 수정 #########
        record = {}

        # record Basic
        record['show'] = showName
        record['task'] = "asset"
        record['data_type'] = data_type
        record['files'] = files
        record['task_publish'] = {}
        record['time'] = datetime.datetime.now().isoformat()
        record['enabled'] = True
        record['artist'] = getpass.getuser()
        record['maya_version'] = maya_version
        record['tags'] = tag_parser.run(outputPath)

        # record Shot
        if shotName == 'shot':
            record['shot'] = shotName
            record['version'] = getPubVersion(show=showName,
                                              task='asset',
                                              data_type=data_type,
                                              shot=shotName)
        # record other
        else:
            record['version'] = getPubVersion(show=showName,
                                              task='asset',
                                              data_type=data_type,
                                              asset_name=assetName)

            record['asset_type'] = assetType
            record['asset_name'] = assetName

        COLLNAME = showName
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
                  "project": showName,
                  "asset": assetName,
                  "type": data_type,
                  "outputpath": outputPath,
                  "maya_version": maya_version}

        dbColl.insert_one(record)

if __name__ == '__main__':
    print 'open file : %s' % sys.argv[1]
    #print sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    if len(sys.argv) > 6:
        expGrp = sys.argv[6]
    else:
        expGrp = ''

    cmds.file( sys.argv[1], force=True, open=True )
    pub( sys.argv[2],
         int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
         expGrp )

    print '#result: model publish by batchscript'
    os._exit(0)
