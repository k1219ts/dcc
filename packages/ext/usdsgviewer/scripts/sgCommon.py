# -*- coding: utf-8 -*-
import os
import datetime as dt
import dxConfig
import requests
import pprint

from pxr import Sdf

from PySide2 import QtWidgets, QtCore

from pymongo import MongoClient
DB_IP = dxConfig.getConf('DB_IP')
client = MongoClient(DB_IP)

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'


def getShowList():
    showList = {}
    params = {'api_key': API_KEY, 'status': 'in_progres'}
    infos = requests.get("http://%s/dexter/search/project.php" % dxConfig.getConf('TACTIC_IP'),
                         params=params).json()

    skipProject = ['test', 'testshot', 'china']
    for i in infos:
        if i['code'] in skipProject:
            continue
        showList[i['name']] = i
    showList['pipe'] = {'name': u'pipe', 'title': u'신규파이프라인 (pipe)', 'code': ''}
    showList['cdh1'] = {'name': u'cdh1', 'title': u'신규파이프라인 (cdh1)', 'code': ''}
    return showList


def getShotList(show):
    try:
        path = '/show/%s/_3d/shot/' % show
        seqList = os.listdir(path)
        shotList = []
        for seq in seqList:
            if '.' not in seq and '_' not in seq:
                shots = os.listdir(os.path.join(path, seq))
                shotList = shotList + shots

        model = QtCore.QStringListModel()
        model.setStringList(sorted(shotList))
        completer = QtWidgets.QCompleter()
        completer.setModel(model)
        completer.setMaxVisibleItems(30)
    except:
        return False
    return completer


def simpleTime(time):
    tmp = dt.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')
    return tmp.strftime('%Y-%m-%d %H:%M')


def getRigInfoDB(show, assetName):
    g_DB = client['PIPE_PUB']
    coll = g_DB[show]

    rigInfo = []
    for data in coll.find({'asset_name': assetName, 'task': 'rig'}).limit(1).sort([('$natural', -1)]):
        data['time'] = simpleTime(data['time'])
        rigInfo.append(data)

    return rigInfo


def getRigPubTime(show, assetName):
    rigList = []
    rigLastVer = None
    rigTime = None

    rigPath = '/show/{show}/_3d/asset/{asset}/rig'.format(show=show, asset=assetName)

    if os.path.exists(rigPath):
        for f in sorted(os.listdir(rigPath), reverse=True):
            if os.path.isdir(os.path.join(rigPath, f)) and \
               '_low' not in f and \
               f not in ['scenes', 'preview']:
                rigList.append(f)

        rigLastVer = rigList[0]
        rigLastVerPath = os.path.join(rigPath, rigLastVer)
        rigTime = dt.datetime.fromtimestamp(os.path.getmtime(rigLastVerPath))

    return rigLastVer, rigTime


def getCacheOutTime(path):
    pubTime = dt.datetime.fromtimestamp(os.path.getctime(path))
    return pubTime


def getCustomLayerData(path):
    customData = []
    layer = Sdf.Layer.FindOrOpen(path)
    customData = layer.customLayerData
    del layer

    return customData


def getAniCustomLayerData(prim, nsLayer):
    customData = None
    for i in prim.GetPrimStack():
        if '%s_ani.usd' % nsLayer in i.layer.realPath:
            path = i.layer.realPath
            layer = Sdf.Layer.FindOrOpen(path)
            customData = layer.customLayerData
            del layer
    return customData


def getAniCacheOutTime(prim, nsLayer):
    pubTime = None
    for i in prim.GetPrimStack():
        if '%s.usd' % nsLayer in i.layer.realPath and '/ani' in i.layer.realPath:
            pubTime = dt.datetime.fromtimestamp(os.path.getctime(i.layer.realPath))
    return pubTime
