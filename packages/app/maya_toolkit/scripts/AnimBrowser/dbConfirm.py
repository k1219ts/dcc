from pymongo import MongoClient

from bson import ObjectId

dbName = "inventory"
collName = "anim_item"
# collName = "anim_tags"
# dbPlugin = MongoDB(dbName, collName)

import dxConfig

DBIP = dxConfig.getConf("DB_IP")
client = MongoClient(DBIP)
database = client[dbName]

def deleteItem(_objectID):
    coll = database[collName]
    objectID = ObjectId(_objectID)

    coll.delete_one({'_id':objectID})

def findEnableFalseItem():
    coll = database[collName]
    return coll.find({"enabled":False})

def findItem():
    coll = database[collName]
    return coll.find({})

def updateItem(key, value, findQuery = {}):
    coll = database[collName]

    updateItem = {"$set":{key:value}}
    findingItem = coll.find(findQuery)
    for item in findingItem:
        print item['_id']
        coll.update({'_id':ObjectId(item['_id'])}, updateItem)

updateItem(key = "ishik", value = 1, findQuery={'ishik':True})
# deleteForItem = findEnableFalseItem()
# for i in deleteForItem:
#     deleteItem(i['_id'])