#coding:utf-8
import dxConfig
from pymongo import MongoClient
from Define import *
import opentimelineio as otio

DB_IP = dxConfig.getConf('DB_IP')

# MongoDB
DatabaseName = "Editorial"

client = MongoClient(DB_IP)
db = client[DatabaseName]


def getData(coll, shotName, clipName, startTC, endTC, plateType='', fps=24.0):
    queryDict = {
        Column2.CLIP_NAME.name.lower(): clipName,
        "$or": [
            {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
                       Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
            {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
                       Column2.TC_OUT.name.lower(): {"$gte":startTC, "$lte": endTC}}]},
            {"$and": [{Column2.TC_IN.name.lower(): {"$gte": startTC, "$lte": endTC},
                       Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
        ]
    }
    if shotName:
        queryDict[Column2.SHOT_NAME.name.lower()] = shotName

    if plateType:
        queryDict[Column2.TYPE.name.lower()] = plateType

    fps = float(fps)

    startTime = otio.opentime.RationalTime.from_timecode(startTC, fps)
    endTime = otio.opentime.RationalTime.from_timecode(endTC, fps)
    durationTime = otio.opentime.duration_from_start_end_time(startTime, endTime)
    TCRange = otio.opentime.TimeRange(startTime, durationTime)

    queryItem = coll.find(queryDict)

    if queryItem.count() == 1:
        return queryItem[0]

    for item in coll.find(queryDict):
        dbStartTime = otio.opentime.RationalTime.from_timecode(item['tc_in'], fps)
        dbEndTime = otio.opentime.RationalTime.from_timecode(item['tc_out'], fps)
        dbDurationTime = otio.opentime.duration_from_start_end_time(dbStartTime, dbEndTime)
        dbTCRange = otio.opentime.TimeRange(dbStartTime, dbDurationTime)

        if dbTCRange.contains(TCRange) and dbTCRange.overlaps(TCRange):
            return item

def getShotList(coll, seqName):
    queryDict = {
        Column2.SHOT_NAME.name.lower(): {"$regex":"%s_*" % seqName}
    }
    return coll.find(queryDict)

def editList(coll, xmlFile):
    queryDict = {
        Column2.XML_NAME.name.lower(): xmlFile
    }
    return list(coll.find(queryDict))

def getEditFileList(coll):
    return coll.distinct(Column2.XML_NAME.name.lower())

def getPlateList(coll, seqName, typeInFix):
    itemList = coll.find({})
    plateList = []
    for item in itemList:
        if seqName in item['shot_name'] and typeInFix in item['type']:
            plateList.append(item)

    print plateList
    return plateList

def getPlateListInFolders():
    showName = 'prat2'
    seqName = 'PS05'

    seqPath = os.path.join('/show', showName, '_2d', 'shot', seqName)

    folderPath = []
    for shotName in os.listdir(seqPath):
        shotNameDir = os.path.join(seqPath, shotName, 'plates')
        # print os.listdir(os.path.join(seqPath, shotName))
        for plateType in os.listdir(shotNameDir):
            if 'org' in plateType:
                folderPath.append(os.path.join(os.path.join(shotNameDir, plateType), sorted(os.listdir(os.path.join(shotNameDir, plateType)))[-1]))

    return folderPath

if __name__ == '__main__':
    import os
    coll = db['prat2']
    plateList = getPlateList(coll, "PS05", "org")

    excludeList = []
    for shotItem in plateList:
        shotName = shotItem['shot_name']
        typeName = shotItem['type']
        version = shotItem['version']

        try:
            print os.listdir(os.path.join('/show', 'prat2', '_2d', 'shot', shotName.split('_')[0], shotName, 'plates', typeName))
        except:
            print excludeList.append(os.path.join('/show', 'prat2', '_2d', 'shot', shotName.split('_')[0], shotName, 'plates', typeName))

    print sorted(getPlateListInFolders())
    print excludeList
    # startTC = '14:00:10:21'
    # endTC = '14:00:11:20'
    #
    # startTime = otio.opentime.RationalTime.from_timecode(startTC, 24.0)
    # endTime = otio.opentime.RationalTime.from_timecode(endTC, 24.0)
    # durationTime = otio.opentime.duration_from_start_end_time(startTime, endTime)
    # TCRange = otio.opentime.TimeRange(startTime, durationTime)
    # print startTime.to_timecode()
    # print endTime.to_timecode()
    #
    # queryDict = {
    #     Column2.CLIP_NAME.name.lower(): 'C213C003_210109_C46Q',
    #     "$or": [
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": startTC, "$lte": endTC}}]},
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$gte": startTC, "$lte": endTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
    #     ]
    # }
    # for item in coll.find(queryDict):
    #     dbStartTime = otio.opentime.RationalTime.from_timecode(item['tc_in'], 24.0)
    #     dbEndTime = otio.opentime.RationalTime.from_timecode(item['tc_out'], 24.0)
    #     dbDurationTime = otio.opentime.duration_from_start_end_time(dbStartTime, dbEndTime)
    #     dbTCRange = otio.opentime.TimeRange(dbStartTime, dbDurationTime)
    #
    #     print dbStartTime.to_timecode()
    #     print dbEndTime.to_timecode()
    #
    #     print dbTCRange.contains(TCRange)
    #     print dbTCRange.overlaps(TCRange)
    #
    # print
    #
    # startTC = '14:00:11:16'
    # endTC = '14:00:12:12'
    #
    # startTime = otio.opentime.RationalTime.from_timecode(startTC, 24.0)
    # endTime = otio.opentime.RationalTime.from_timecode(endTC, 24.0)
    # durationTime = otio.opentime.duration_from_start_end_time(startTime, endTime)
    # TCRange = otio.opentime.TimeRange(startTime, durationTime)
    # print startTime.to_timecode()
    # print endTime.to_timecode()
    #
    # queryDict = {
    #     Column2.CLIP_NAME.name.lower(): 'C213C003_210109_C46Q',
    #     "$or": [
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$lte": startTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": startTC, "$lte": endTC}}]},
    #         {"$and": [{Column2.TC_IN.name.lower(): {"$gte": startTC, "$lte": endTC},
    #                    Column2.TC_OUT.name.lower(): {"$gte": endTC}}]},
    #     ]
    # }
    # # print coll.find(queryDict).count()
    # for item in coll.find(queryDict):
    #     dbStartTime = otio.opentime.RationalTime.from_timecode(item['tc_in'], 24.0)
    #     dbEndTime = otio.opentime.RationalTime.from_timecode(item['tc_out'], 24.0)
    #     dbDurationTime = otio.opentime.duration_from_start_end_time(dbStartTime, dbEndTime)
    #     dbTCRange = otio.opentime.TimeRange(dbStartTime, dbDurationTime)
    #
    #     print dbStartTime.to_timecode()
    #     print dbEndTime.to_timecode()
    #
    #     print dbTCRange.contains(TCRange)
    #     print dbTCRange.overlaps(TCRange)
