#coding:utf-8
import xlrd2
import os
import sys
from Define import Column2
import datetime
import DBConfig

def compareDict(src, dst):
    excludeKeyList = ['history_version', 'time']
    # fist key check
    # if sorted(src.keys()) != sorted(dst.keys()):
    #     return False

    for key in src.keys():
        if key in excludeKeyList:
            continue

        if not src.has_key(key) or not dst.has_key(key):
            return False

        # if key == 'mov_cut_fps':
        #     if src['mov_cut_fps'] != dst[key]:
        #         print key, src[key], dst[key]
        #         return False

        if src[key] and dst[key] and src[key] != dst[key]:
            print key, src[key], dst[key]
            return False
    return True

def main(xlsFileName):
    if not os.path.exists(xlsFileName):
        assert False, "Not found XLS File"

    showName = xlsFileName.split('/')[3]
    CollectionName = showName.lower()
    coll = DBConfig.db[CollectionName]

    updateDBXLS = xlrd2.open_workbook(xlsFileName)
    sheet = updateDBXLS.sheet_by_name('scan_list')

    for row in range(1, sheet.nrows):
        rowData = sheet.row_values(row)

        item = {}
        for column in Column2:
            try:
                if column.name == Column2.ISSUE.name:
                    item[column.name.lower()] = rowData[column.value].replace('\n', '&')
                elif column.name == Column2.EDIT_ISSUE.name:
                    item[column.name.lower()] = rowData[column.value].replace('\n', '&')
                else:
                    item[column.name.lower()] = rowData[column.value]
            except IndexError as e:
                pass

        item['time'] = datetime.datetime.now().isoformat()
        # TODO deprecate: Column2.V1 = [retime, scale]
        try:
            item.pop(Column2.RETIME.name.lower())
            item.pop(Column2.SCALE.name.lower())
        except AttributeError as e:
            pass

        # already data Search
        shotName = rowData[Column2.SHOT_NAME.value]
        # shotName = ''
        clipName = rowData[Column2.CLIP_NAME.value]
        startTC = rowData[Column2.TC_IN.value]
        endTC = rowData[Column2.TC_OUT.value]
        # type = rowData[Column2.]
        if not shotName or not clipName:
            continue

        findItem = DBConfig.getData(coll, shotName, clipName, startTC, endTC)

        if findItem:
            previousHistory = None
            if findItem.has_key('history'):
                previousHistory = findItem.pop('history')
            historyItem = findItem

            if historyItem.has_key('lastModified'):
                historyItem.pop('lastModified')
            id = historyItem.pop('_id')

            if compareDict(historyItem, item):
                continue
            try:
                item['history'] = {'v%03d' % (int(findItem['history_version'][1:])): historyItem}
            except:
                item['history'] = {'v001': historyItem}
            if previousHistory:
                for version in previousHistory.keys():
                    item['history'][version] = previousHistory[version]

            try:
                item['history_version'] = 'v%03d' % (int(findItem['history_version'][1:]) + 1)
            except:
                item['history_version'] = 'v001'
            print "update item"
            coll.update_one({'_id': id}, {'$set': item, '$currentDate': {'lastModified': True}}, )
        else:
            item['history_version'] = 'v001'
            print "insert item"
            coll.insert_one(item)



if __name__ == "__main__":
    main(sys.argv[-1])