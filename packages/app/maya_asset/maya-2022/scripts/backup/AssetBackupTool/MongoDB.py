'''
'    @author    : daeseok.chae
'    @date      : 2017.02.10
'    @brief     : Dexter MongoDB Base Setting Class
'''
import getpass
import datetime as datetime

from pymongo import MongoClient

import dxConfig

class MongoDB():
    def __init__(self, dbName = "", collName = ""):
        DBIP = dxConfig.getConf("DB_IP")
        client = MongoClient(DBIP)
        database = client[dbName]
        self.coll = database[collName]
        self.files = {}
    
    def setName(self, str = ""):
        self.name = str
        
    def setProject(self, str = ""):
        self.project = str

    def setCategory(self, str=""):
        self.category = str
        
    def setDesc(self, str = ""):
        self.description = str
        
    def setTags(self, list = []):
        self.tags = list
        
    def setFiles(self, dic = {}):
        self.files = dic
        
    def getRecord(self):
        self.dbRecord = {"name": self.name,
                    "project": self.project,
                    "category": self.category,
                    "description": self.description,
                    "tags": self.tags,
                    "files": self.files,
                    "enabled": False,
                    "user": getpass.getuser(),
                    "time": datetime.datetime.now().isoformat(),
                    "type": "ASSET_SRC"
        }
        
        print "dbRecord :", self.dbRecord
        return self.dbRecord

    def updateDocument(self, objectID, files):
        self.coll.update({'_id':objectID}, {'$set': {'files':files}}, upsert=True)
    
    def existDocument(self, checkDB):
        print "checkDB :", checkDB
        if not self.coll == None:
            if self.coll.find(checkDB).limit(1).count():
                return True
            else:
                return False
            
    def insertDocument(self):
        self.resultID = self.coll.insert_one(self.dbRecord).inserted_id
        print "resultID :", self.resultID

