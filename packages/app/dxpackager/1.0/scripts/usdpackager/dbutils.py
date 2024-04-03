#coding:utf-8
import os, datetime, requests, getpass, json, re

import pymongo
from pymongo import MongoClient

import dxConfig

DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'VENDOR_PACKAGE'

IFPSTF = {
    #/stuff/{PROJ}/stuff/ftp/_vendor/{VENDOR}/from_dexter/{DATENUM}
    'PROJ': 2,
    'VENDOR': 6,
    # 'DATENUM': 8
}

def updatePackage(srcFile, pkgFile, pkgInfo={}):
    pkgFileTok = pkgFile.split('/')

    dbRecord = {
        'vendor': pkgFileTok[IFPSTF['VENDOR']],
        'show': pkgFileTok[IFPSTF['PROJ']],
        'time': datetime.datetime.now().isoformat(),
        'user': getpass.getuser(),
        'pkgdir':pkgFile.split('/_2d')[0].split('/_3d')[0],
        'nslyr': 'none',
        'sublyr': 'none',
        'pkgType': 'usd'
    }

    condition = {
        'show': dbRecord['show'],
        'pkgdir': dbRecord['pkgdir'],
        'nslyr': 'none',
        'sublyr': 'none'
    }

    if '/_3d/asset/' in srcFile:
        assetTok = srcFile.split('/_3d/asset/')[-1].split('/')
        if len(assetTok) > 1:
            dbRecord['asset'] = assetTok[0]
            condition['asset'] = assetTok[0]
            
        if len(assetTok) > 2:
            dbRecord['task'] = assetTok[1]
            condition['task'] = assetTok[1]
            if  bool(re.match(r'v[0-9]{3}', assetTok[2])):
                dbRecord['version'] = assetTok[2]
                condition['version'] = assetTok[2]
            else:
                if len(assetTok) > 3 and bool(re.match(r'v[0-9]{3}', assetTok[3])):
                    dbRecord['sublyr'] = assetTok[2]
                    dbRecord['version'] = assetTok[3]
                    condition['sublyr'] = assetTok[2]
                    condition['version'] = assetTok[3]

    elif '/_3d/shot/' in srcFile:
        shotTok = srcFile.split('/_3d/shot/')[-1].split('/')
        if len(shotTok) > 2:
            dbRecord['seq'] = shotTok[0]
            dbRecord['shot'] = shotTok[1]
            condition['seq'] = shotTok[0]
            condition['shot'] = shotTok[1]

        if len(shotTok) > 3:
            dbRecord['task'] = shotTok[2]
            condition['task'] = shotTok[2]
            if  bool(re.match(r'v[0-9]{3}', shotTok[3])):
                dbRecord['version'] = shotTok[3]
                condition['version'] = shotTok[3]
            else:
                if len(shotTok) > 4 and bool(re.match(r'v[0-9]{3}', shotTok[4])):
                    dbRecord['nslyr'] = shotTok[3]
                    dbRecord['version'] = shotTok[4]
                    condition['nslyr'] = shotTok[3]
                    condition['version'] = shotTok[4]

    elif '/_2d/shot/' in srcFile:
        shotTok = srcFile.split('/_2d/shot/')[-1].split('/')
        if len(shotTok) > 2:
            dbRecord['seq'] = shotTok[0]
            dbRecord['shot'] = shotTok[1]
            condition['seq'] = shotTok[0]
            condition['shot'] = shotTok[1]

        if len(shotTok) > 3:
            dbRecord['task'] = shotTok[2]
            condition['task'] = shotTok[2]
            if  bool(re.match(r'v[0-9]{3}', shotTok[3])):
                dbRecord['version'] = shotTok[3]
                condition['version'] = shotTok[3]
            else:
                if len(shotTok) > 4 and bool(re.match(r'v[0-9]{3}', shotTok[4])):
                    dbRecord['sublyr'] = shotTok[3]
                    dbRecord['version'] = shotTok[4]
                    condition['sublyr'] = shotTok[3]
                    condition['version'] = shotTok[4]

    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[dbRecord['vendor']]
 
    dbFile = srcFile
    for pi in pkgInfo:
        if pi == 'file':
            dbFile = pkgInfo[pi]
        else:
            dbRecord[pi] = pkgInfo[pi]

    condition['pkgType'] = dbRecord['pkgType']

    find = coll.find_one(condition)
    if find:
        coll.update_one(condition, {'$push': { 'files': dbFile }})
    else:
        dbRecord['files'] = [ dbFile ]
        coll.insert_one(dbRecord)