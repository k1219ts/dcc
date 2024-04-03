from pymongo import MongoClient

from bson import ObjectId
from bson import Code

import dxConfig

DBIP = dxConfig.getConf("DB_IP")
client = MongoClient(DBIP)
database = client["inventory"]

def getRelativeInfo(findObjId):
    coll = database["anim_relate"]

    objID = ObjectId(findObjId)

    # return coll.find().clone()
    return coll.find_one({'_id' : objID})

def getExistRelativeInfo(findItemId):
    coll = database["anim_relate"]

    objID = ObjectId(findItemId)

    # return coll.find([{'$or' : {{'$match' : {"0._id" : objID}},
    #                                 {'$match': {"1._id": objID}},
    #                                 {'$match': {"2._id": objID}}}
    #                         }])
    queryItem = coll.find({})

    for item in queryItem:
        print "#" * 10
        print item
        if item.has_key("0"):
            for data in item["0"]:
                if data['_id'] == objID:
                    return item['_id']

        if item.has_key("1"):
            for data in item["1"]:
                if data['_id'] == objID:
                    return item['_id']

        if item.has_key("2"):
            for data in item["2"]:
                if data['_id'] == objID:
                    return item['_id']

    return None

def getItemForObjID(findObjId):
    coll = database["anim_item"]

    objID = ObjectId(findObjId)

    return coll.find_one({'_id' : objID})

    # returnValue = []
    # for i in coll.find({'_id' : objID}):
    #     returnValue.append(i)
    #
    # return i

def appendTierItem(updateObjId, selfTier, selfObjId, parentObjId):
    coll = database["anim_relate"]

    objID = ObjectId(updateObjId)

    findItem = coll.find_one({"_id":objID})

    updateQuery = {"$set" : {}}
    if not findItem.has_key(str(selfTier)):
        updateQuery["$set"][str(selfTier)] = [{"_id": ObjectId(selfObjId), "parent": ObjectId(parentObjId)}]
    else:
        updateQuery["$set"][str(selfTier)] = findItem[str(selfTier)] + [{"_id":ObjectId(selfObjId), "parent" : ObjectId(parentObjId)}]
    coll.update_one({"_id":objID}, updateQuery)

def addRelativeInfo(parentId, childTier, childId):
    coll = database["anim_relate"]

    query = {}
    query["0"] = [{'_id' : ObjectId(parentId)}]
    query[str(childTier)] = [{'_id' : ObjectId(childId), "parent":ObjectId(parentId)}]

    return coll.insert_one(query)

def updateHashTag(findObjId, hashTag):
    coll = database["anim_item"]

    objID = ObjectId(findObjId)

    coll.find_one_and_update({'_id' : objID}, {"$set":{"hashTag":hashTag}})

    return coll.find_one({'_id':objID})

def updateTier(objectID, tier1, tier2, tier3):
    coll = database["anim_item"]

    objID = ObjectId(objectID)

    return coll.find_one_and_update({'_id':objID}, {"$set":{"tag1tier":tier1, 'tag2tier':tier2, 'tag3tier':tier3}})

def renameTier(changedTierNum, changedTierName, category = "", tierList = []):
    coll = database["anim_tags"]

    # query = {"$set":{"tag%dtier" % changedTierNum : changedTierName}}
    query = {}
    query['category'] = category
    for index, tier in enumerate(tierList):
        if tier == "":
            break
        query['tag%dtier' % (index + 1)] = tierList[index]

    print "#" * 20, "tags", "#" * 20
    for item in coll.find(query):
        print item

    print "#" * 20, "items", "#" * 20
    coll = database["anim_item"]
    for item in coll.find(query):
        print item

    print {"$set":{"tag%dtier" % changedTierNum : changedTierName}}
    # coll.find_and_update(query, {"$set":{"tag%dtier" % changedTierNum : changedTierName}})

    # return coll.find_and_update({'_id':objID}, {"$set":{"tag1tier":tier1, 'tag2tier':tier2, 'tag3tier':tier3}})