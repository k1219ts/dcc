#coding:utf-8
__author__ = "daeseok.chae @ Dexter Studio"
__date__ = "2019.11.14"
__comment__ = '''
    DB Setup
'''

# base module
import getpass
import datetime
import sys
import os

# sys.path.append("/backstage/libs/python_lib")

# using Mongo DB
from pymongo import MongoClient
import dxConfig
gDBIP = dxConfig.getConf("DB_IP")
client = MongoClient(gDBIP)
gDB = client["ASSETLIB"]

def test():
    print "ADD DB: ON"


def AddItem(filepath):

    itemDict = {'category':'unknown',
                'subCategory': 'unknown',
                'tag': [],
                'comment' : '',
                'reply':[{'user':getpass.getuser(), 'comment':'add item', 'time':datetime.datetime.now().isoformat()}],
                'name' : '',
                'files': {}
    }

    # find files insert 'files' paths.
    if not '/assetlib/3D/asset' in filepath:
        return "this location isn't assetlib"

    assetNameIndexKey = 'asset'
    if "element" in filepath:
        itemDict['tag'].append('element')
        assetNameIndexKey = 'element'

    splitFilePath = filepath.split('/')
    assetIndex = splitFilePath.index(assetNameIndexKey)
    assetName = splitFilePath[assetIndex + 1]

    dirpath = "/".join(splitFilePath[:assetIndex + 2])
    filepath = os.path.join(dirpath, "%s.usd" % assetName)
    previewFile = os.path.join(dirpath, "preview.jpg")

    itemDict['files']['usdfile'] = filepath
    itemDict['files']['preview'] = previewFile

    # if not os.path.exists(previewFile):
    customPreviewFile = previewFile.replace('.jpg', '.####.jpg')
    command = "/backstage/bin/DCC rez-env usdtoolkit-19.11 -- usdrecorder -w 320 -ht 240 --purposes render --renderer Prman --outputImagePath {OUTPUTFILE} {USDFILE}".format(USDFILE=filepath, OUTPUTFILE=customPreviewFile)
    fullCmd = 'echo dexter2019 | su render -c "%s"' % command
    ret = os.system('echo dexter2019 | su render -c "%s"' % command)
    if ret == 0:
        # print "# Success make preview"
        renameCmd = 'echo dexter2019 | su render -c "mv %s %s"' % (customPreviewFile.replace('.####.', '.0000.'), previewFile)
        ret = os.system(renameCmd)
        if ret == 0:
            print "# Success Make Preview" + command
        else:
            return "# Failed Rename :" + renameCmd
    else:
        return "# Failed " + fullCmd

    # set asset name
    itemDict['name'] = assetName
    if not gDB.item.find_one({'name':assetName}):
        gDB.item.insert_one(itemDict)


def AddTag(assetName, tagName):
    item = gDB.item.find_one({'name': assetName})
    # if ',' in tagName:
    list = tagName.split(',')
    for i in list:
        if not i in item['tag']:
            item['tag'].append(i)
    gDB.item.update_one({'_id': item['_id']}, {"$set": item})
