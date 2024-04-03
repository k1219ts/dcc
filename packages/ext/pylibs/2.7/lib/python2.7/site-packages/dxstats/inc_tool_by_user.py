#encoding=utf-8
#!/usr/bin/env python
from pymongo import MongoClient
import datetime
import dxConfig

DB_IP = dxConfig.getConf('DB_IP')
DB_NAME = 'stats'
COLL = 'tool_stats'
TCOLL = 'time_stats'

def run(toolName, user):
    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]

    coll.update_one({'user': user},
                    {'$inc':{toolName:1}},
                    upsert=True)

    coll = db[TCOLL]
    coll.insert_one({'user': user, 'tool':toolName,
                     'time':datetime.datetime.now().isoformat()})
