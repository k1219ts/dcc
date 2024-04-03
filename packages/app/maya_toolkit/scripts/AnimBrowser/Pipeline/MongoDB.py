'''
'    @author    : daeseok.chae
'    @date      : 2017.02.10
'    @brief     : Dexter MongoDB Base Setting Class
'''
import getpass
import datetime as datetime

import os

from pymongo import MongoClient

from bson import ObjectId

# dbName = "inventory"
# collName = "assets"
# dbPlugin = MongoDB(dbName, collName)

import dxConfig

class MongoDB():
    def __init__(self, dbName = "", collName = ""):
        DBIP = dxConfig.getConf("DB_IP")
        client = MongoClient(DBIP)
        database = client[dbName]
        self.coll = database[collName]

    def setCategory(self, str = ""):
        self.category = str

    def setTag1TierName(self, str = ""):
        self.tag1Tier = str

    def setTag2TierName(self, str =""):
        self.tag2Tier = str

    def setTag3TierName(self, str =""):
        self.tag3Tier = str

    def setFiles(self, dic = {}):
        self.files = dic

    def setFileNum(self, num = 0):
        self.fileNumber = num

    def setIsHIK(self, isHIK = 2):
        self.isHIK = isHIK

    def setHashTag(self, tag = []):
        self.hashTag = tag

    def updateRecord(self):
        self.dbRecord = {"tag1tier": self.tag1Tier,
                    "tag2tier": self.tag2Tier,
                    "tag3tier": self.tag3Tier,
                    "fileNum": self.fileNumber,
                    "files": self.files,
                    "ishik": self.isHIK,
                    "enabled": False,
                    "user": getpass.getuser(),
                    "time": datetime.datetime.now().isoformat(),
                    "category": self.category,
                    "hashTag" : self.hashTag
        }

        return self.dbRecord

    def existDocument(self, checkDB):
        if not self.coll == None:
            return self.coll.find(checkDB).count()

    def getTagData(self, category = ""):
        if category == "":
            return self.coll.find({})
        else:
            return self.coll.find({'category' : category})

    def getTag2Tier(self, tier1Name):
        return self.coll.distinct("tag2tier")

    def getHashTags(self, category):
        query = {}
        if not category == "":
            query = {"category":category}
        return self.coll.find(query).distinct("hashTag")

    def getTier(self):
        tierList = []
        for tier1 in self.coll.distinct("tag_tier1"):
            tierList.append(tier1)
        for tier2 in self.coll.distinct("tag_tier2"):
            tierList.append(tier2)
        for tier3 in self.coll.distinct("tag_tier3"):
            tierList.append(tier3)

        tierList = list(set(tierList))
        return tierList

    def getContentInfo(self, category, tag1Tier, tag2Tier, tag3Tier):
        dbItem = {"category":category,
                  "tag1tier": str(tag1Tier),
                  "tag2tier": str(tag2Tier),
                  "tag3tier": str(tag3Tier),
                  "enabled": True
                  }

        reWork = []

        for i in self.coll.find(dbItem):
            reWork.append(i)

        return reWork

    def removeDocument(self, category, tag1Tier, tag2Tier, tag3Tier, fileNum):
        item = self.coll.find_one({"tag1tier":tag1Tier, "tag2tier":tag2Tier, "tag3tier":tag3Tier, "fileNum":fileNum, "category":category})
        objectID = ObjectId(item['_id'])

        directoryPath = os.path.dirname(item['files']['preview'])

        cmd = 'echo "dexter" | su render -c "rm -rf %s"' % directoryPath
        os.system(cmd)
        self.coll.delete_one({'_id':objectID})


    def insertDocument(self):
        self.resultID = self.coll.insert_one(self.dbRecord).inserted_id

    def insertTagDocument(self, dbRecord):
        if not self.existDocument(dbRecord):
            self.coll.insert(dbRecord)
            return True
        else:
            return False

    def removeTagDocument(self, dbRecord):
        self.coll.remove(dbRecord)

    def findForTag(self, tagList, category = {}):
        hashTagList = []
        for tag in tagList:
            if tag.startswith('#'):
                hashTagList.append(tag[1:])

        query = {}
        if len(category) > 1:
            query = {"$or":[]}
        for team in category:
            teamQuery = {"category":team, "enabled":True, "hashTag" : {"$all":hashTagList}}
            for tag in tagList:
                if not tag.startswith("#"):
                    if not teamQuery.has_key("$and"):
                        teamQuery["$and"] = []
                    teamQuery["$and"].append({"$or":[{'tag1tier' : tag}, {'tag2tier' : tag}, {'tag3tier' : tag}]})
            if query.has_key("$or"):
                query['$or'].append(teamQuery)
            else:
                query = teamQuery

        print query
        findItem = self.coll.find(query)

        reWork = list()
        for i in findItem:
            reWork.append(i)

        return reWork
