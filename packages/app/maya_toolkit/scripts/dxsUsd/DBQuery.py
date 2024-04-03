#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter CGSupervisor'
__date__ = '2020.07.02'
__comment__ = 'DB File'
##########################################

# IP config
import dxConfig
DB_IP = dxConfig.getConf('DB_IP')

# Mongo DB
import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)
g_DB = client['USD_PUBLISH']

import getpass
import datetime
import os
import sys

def getShow(showDir):
    splitShowDir = showDir.split("/")
    if "show" in splitShowDir:
        showIndex = splitShowDir.index("show")
        return splitShowDir[showIndex + 1]
    else:
        return "unknown"

def getDirSize(dirPath):
    dirSize = 0
    for dirpath, dirnames, filenames in os.walk(dirPath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                dirSize += os.path.getsize(fp)
    # for filename in os.listdir(dirPath):
    #     dirSize += os.path.getsize(os.path.join(dirPath, filename))
    return dirSize

def assetQueryDBObjId(showDir, assetName, version, type, **kwargs):
    print showDir, assetName, version, type, kwargs
    showName = getShow(showDir)
    coll = g_DB[showName]
    checkDict = {"name": assetName, "version": version, "type": type}
    if kwargs.has_key("elementName"):
        checkDict["elementName"] = kwargs["elementName"]
    if kwargs.has_key("elementType"):
        checkDict["elementType"] = kwargs["elementType"]

    dataDict = coll.find_one(checkDict)  # .sort('time', pymongo.DESCENDING).limit(1)

    return dataDict["_id"]

def assetInsertDB(showDir, assetName, version, type, outDirs = [], **kwargs):
    print showDir, assetName, version, type, outDirs, kwargs
    try:
        showName = getShow(showDir)
        coll = g_DB[showName]
        checkDict = {"name" : assetName, "version": version, "type":type}
        if kwargs.has_key("elementName"):
            checkDict["elementName"] = kwargs["elementName"]
        if kwargs.has_key("elementType"):
            checkDict["elementType"] = kwargs["elementType"]

        dataDict = coll.find_one(checkDict) #.sort('time', pymongo.DESCENDING).limit(1)

        if dataDict is None:
            # first insert
            dataDict = {"show":showName,
                        "task":"asset",
                        "type": type,
                        "name":assetName,
                        "version":version,
                        "last_time":datetime.datetime.now().isoformat(),
                        "overCount":0,
                        "outDirs" : outDirs
                        }

            log = {'user': getpass.getuser(),
                   'time': datetime.datetime.now().isoformat()}

            if kwargs.has_key('REZ_USED_RESOLVE'):
                log['REZ_USED_RESOLVE'] = kwargs['REZ_USED_RESOLVE']

            comment = ''
            if kwargs.has_key('comment'):
                comment = kwargs['comment']
            log['comment'] = comment

            dataDict['logs'] = [log]

            totalSize = 0
            for outDir in outDirs:
                totalSize += getDirSize(outDir)
            dataDict["memory"] = totalSize

            dataDict.update(kwargs)
            coll.insert_one(dataDict)
        else:
            # overwrite data
            dataDict.update(kwargs)
            dataDict["overCount"] += 1
            dataDict["last_time"] = datetime.datetime.now().isoformat()

            log = {'user':getpass.getuser(),
                   'time':datetime.datetime.now().isoformat()}

            if kwargs.has_key('REZ_USED_RESOLVE'):
                log['REZ_USED_RESOLVE'] = kwargs['REZ_USED_RESOLVE']

            comment = ''
            if kwargs.has_key('comment'):
                comment = kwargs['comment']
            log['comment'] = comment

            if not dataDict.has_key('logs'):
                dataDict['logs'] = [log]
            else:
                dataDict['logs'].append(log)

            totalSize = 0
            for outDir in outDirs:
                totalSize += getDirSize(outDir)
            dataDict["memory"] = totalSize
            dataDict["outDirs"] = outDirs

            dataDict.update(kwargs)
            coll.update_one({"_id":dataDict["_id"]}, {"$set":dataDict})
    except:
        coll = g_DB["error"]
        dataDict = {"show": showDir,
                    "name": assetName,
                    "version": version,
                    "type" : type,
                    "outDirs": outDirs}
        dataDict.update(kwargs)
        coll.insert_one(dataDict)

def shotInsertDB(showDir, shot, user, name, version, type, outDir, leftParameter = {}):
    print showDir, shot, user, name, version, type, outDir, leftParameter
    try:
        showName = showDir
        seq = shot.split("_")[0]
        coll = g_DB[showName]
        checkDict = {"seq": seq,"shot": shot, "name" : user, "version": version, "type":type}

        dataDict = coll.find_one(checkDict) #.sort('time', pymongo.DESCENDING).limit(1)

        if dataDict is None:
            # first insert
            dataDict = {"show":showName,
                        "seq": seq,
                        "shot": shot,
                        "task":"shot",
                        "type": type,
                        "name":name,
                        "version":version,
                        "last_time":datetime.datetime.now().isoformat(),
                        "overCount":0,
                        "outDirs" : outDir
                        }

            log = {'user': user,
                   'time': datetime.datetime.now().isoformat()}

            if leftParameter.has_key('REZ_USED_RESOLVE'):
                log['REZ_USED_RESOLVE'] = leftParameter.pop('REZ_USED_RESOLVE')

            comment = ''
            if leftParameter.has_key('comment'):
                comment = leftParameter.pop('comment')
            log['comment'] = comment

            dataDict['logs'] = [log]

            totalSize = getDirSize(outDir)
            dataDict["memory"] = totalSize

            dataDict['leftParm'] = leftParameter
            return coll.insert_one(dataDict)
        else:
            # overwrite data
            dataDict["overCount"] += 1
            dataDict["last_time"] = datetime.datetime.now().isoformat()

            log = {'user': name,
                   'time': datetime.datetime.now().isoformat()}

            if leftParameter.has_key('REZ_USED_RESOLVE'):
                log['REZ_USED_RESOLVE'] = leftParameter.pop('REZ_USED_RESOLVE')

            comment = ''
            if leftParameter.has_key('comment'):
                comment = leftParameter.pop('comment')
            log['comment'] = comment

            if not dataDict.has_key('logs'):
                dataDict['logs'] = [log]
            else:
                dataDict['logs'].append(log)

            totalSize = getDirSize(outDir)
            dataDict["memory"] = totalSize
            dataDict["outDirs"] = outDir

            dataDict['leftParm'] = leftParameter
            return coll.update_one({"_id":dataDict["_id"]}, {"$set":dataDict})
    except:
        coll = g_DB["error"]
        dataDict = {"show": showDir,
                    "shot": shot,
                    "task": "shot",
                    "type": type,
                    "outDirs": outDir}
        dataDict.update(leftParameter)
        coll.insert_one(dataDict)

#---------------------------------------------------------------------------
#
#   MAIN
#
#---------------------------------------------------------------------------
if __name__ == '__main__':
    import batchCommon

    optparser = batchCommon.shotDBOptParserSetup()
    opts, args = optparser.parse_args(sys.argv)

    leftParameter = {}
    print args
    for index in range(1, len(args[1:]), 2):
        leftParameter[args[index]] = args[index + 1]

    print opts.showDir, opts.shot, opts.name, opts.version, opts.type, opts.outDir, leftParameter
    shotInsertDB(opts.showDir, opts.shot, opts.user, opts.name, opts.version, opts.type, opts.outDir, leftParameter)
