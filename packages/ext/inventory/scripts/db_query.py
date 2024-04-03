import operator
import pymongo
from pymongo import MongoClient
import getpass
import dxConfig

DB_IP = dxConfig.getConf('DB_IP')
DB_NAME = 'inventory'
CONFIGCOLL = 'user_config'
COLL = 'assets'
CORECOLL = 'core_elements'

def getTagList():
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    tags = coll.find().distinct('tags')

    return tags

def getDistinct(key):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return sorted(coll.distinct(key))

def getFindDistinct(searchTerm, key):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    dist = sorted(coll.find(searchTerm).distinct(key))
    if 'movie' in dist:
        dist[dist.index('movie')], dist[0] = dist[0], dist[dist.index('movie')]
    return dist
    #return sorted(coll.find(searchTerm).distinct(key))

def searchByTerm(term):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    #print term
    return coll.find(term).sort('time',pymongo.DESCENDING)
    #return coll.find(term).sort('time', pymongo.ASCENDING)


def searchByTag(tag, term=None):
    # TAG TEXT INDEX
    #print tag, repr(tag)
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    baseTerm = {"$text": {"$search":tag,
                          "$caseSensitive": False}
                }
    if term:
        baseTerm.update(term)
        result = coll.find(baseTerm)
    else:
        result = coll.find(baseTerm)
    return result

def getCoreElement():
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[CORECOLL]
    return coll.find()

def updateUserConfig(user, configData):
    # configData - > {'play':'click',...}

    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[CONFIGCOLL]
    return coll.find_one_and_update({'user':user},
                                    {'$set':configData},
                                    upsert=True
                                    )

def getUserConfig():
    user = getpass.getuser()
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[CONFIGCOLL]
    return coll.find_one({'user':user})

def getRecentUpdated():
    user = getpass.getuser()
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    items = {}
    for assetType in getDistinct("type"):
        items[assetType] = []
        newItems = coll.find({'type':assetType, 'enabled':True}).sort("time",-1).limit(5)
        for item in newItems:
            items[assetType].append(item)
    return items

def getRandom(limit):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.aggregate([{'$sample': {'size': limit}}])

def getTagCount():
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]

    result = coll.aggregate([
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ])

    dic = {}
    for i in result:
        dic[i['_id']] = i['count']
    return dic

def updateBookmark(user, ids):
    """
    :param ids: [ObjectId, ObjectId, ...]
    :return:
    """
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[CONFIGCOLL]
    return coll.find_one_and_update({'user':user},
                                    {'$set':{'bookmark':ids}},
                                    upsert=True
                                    )

def getByIds(ids):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.find({'_id': {"$in":ids}})

def insertOne(doc):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.insert_one(doc)

def updateProject(id, project):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.find_one_and_update({'_id':id},
                                    {'$set':{'project':project}}
                                    )

def getInventoryConfig():
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[CONFIGCOLL]

    return coll.find_one({"config" : "inventory_config"})

def updateName(id, name):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.find_one_and_update({'_id':id},
                                    {'$set':{'name':name}}
                                    )

def updateTags(id, tags):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    coll.find_one_and_update({'_id':id}, {'$set':{'tags':tags}} )

    return coll.find_one({'_id':id})

def deleteItem(id):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    return coll.find_one_and_update({'_id': id},
                                    {'$set': {'enabled': False}}
                                    )


def getStat(action='click', limit=20):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db['stats']
    assetColl = db['assets']
    docList = []

    ids = []
    for i in coll.find({}).sort(action, pymongo.DESCENDING).limit(limit):
        ids.append(i['itemId'])

    return ids, list(assetColl.find({'_id': {"$in": ids}}))



def team_click_count(startTime, endTime):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db['logs']
    assetColl = db['assets']

    teamDic = {}
    optionTerm = {'time': {'$gte': startTime,
                            '$lte': endTime}}

    for i in coll.find(optionTerm):
        teamRef = i['user'].split('-')[0]
        if teamDic.has_key(teamRef):
            pass
        else:
            teamDic[teamRef] = {'click':{},
                                'doubleclick':{}}


        teamRoot = teamDic[teamRef]

        if teamRoot[i['action']].has_key(i['itemId']):
            teamRoot[i['action']][i['itemId']] += 1
        else:
            teamRoot[i['action']][i['itemId']] = 1

    searchIds = []
    for team in teamDic:
        for i in sorted(teamDic[team]['click'].items(), reverse=True, key=operator.itemgetter(1))[:5]:
            searchIds.append(i[0])
        for i in sorted(teamDic[team]['doubleclick'].items(), reverse=True, key=operator.itemgetter(1))[:5]:
            searchIds.append(i[0])
    return teamDic, list(assetColl.find({'_id': {"$in": searchIds}}))

def convertObjIDtoDoc(ids, limit=5):
    # case list
    # ids    [objid, objid]

    # case dict
    # ids    {objid : 10,
    #         objid : 9,}

    client = MongoClient(DB_IP)
    db = client['inventory']
    assetColl = db['assets']

    returnResult = []

    if type(ids) == dict:
        orderedId = [i[0] for i in sorted(ids.items(), reverse=True, key=operator.itemgetter(1))][:limit]
        searchResult = list(assetColl.find({'_id': {"$in": orderedId}
                                            }
                                           ))

        for i in orderedId:
            for doc in searchResult:
                if i == doc['_id']:
                    returnResult.append(doc)
                    break


    elif type(ids) == list:
        searchResult = list(assetColl.find({'_id': {"$in": ids[-limit:]}
                                            }
                                           ))

        for i in ids:
            for doc in searchResult:
                if i == doc['_id']:
                    returnResult.append(doc)
                    break

    return returnResult



def getPersonalClickStat(name,limit=5):
    client = MongoClient(DB_IP)
    db = client['inventory']
    coll = db['logs']
    assetColl = db['assets']

    ud = {'click'  : {},
          'doubleclick' : {}
          }
    clickList = []
    doubleClickList = []

    for i in coll.find({'user':name}):
        if ud[i['action']].has_key(i['itemId']):
            ud[i['action']][i['itemId']] += 1
        else:
            ud[i['action']][i['itemId']] = 1

        if i['action'] == 'click':
            if not(i['itemId'] in clickList):
                clickList.append(i['itemId'])
        elif i['action'] == 'doubleclick':
            if not (i['itemId'] in doubleClickList):
                doubleClickList.append(i['itemId'])


    clickDoc = convertObjIDtoDoc(ud['click'], limit=limit)
    doubleClickDoc = convertObjIDtoDoc(ud['doubleclick'], limit=limit)

    clickLogDoc = convertObjIDtoDoc(clickList, limit=limit)
    doubleClickLogDoc = convertObjIDtoDoc(doubleClickList, limit=limit)

    return clickDoc, doubleClickDoc, clickLogDoc, doubleClickLogDoc

def getClickHistory(limit =100):
    returnResult = []

    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db['logs']

    infos = coll.find({"action":"click"}).sort("time", -1).limit(limit)
    for record in infos:
        assetColl = db['assets']
        doc = assetColl.find_one({'_id': record['itemId']})
        doc['user'] = record['user']
        returnResult.append(doc)

    return returnResult