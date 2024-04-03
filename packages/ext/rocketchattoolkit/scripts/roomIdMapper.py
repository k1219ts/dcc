#coding:utf-8
from rocketchat.api import RocketChatAPI

roomNameMapper = {
    "emd": "비상선언_공지방",
    "slc": "사일런스_공지",
    "prat2": "해적2 공지",
    "ncx": "NCX 공지방",
    "ncl": "",
    "cdh": "외계인_공지방",
    "cdh1": "외계인_공지방",
    "wdl": '원더랜드_공지방',
    "tmn": '더문_공지방',
    'csp': 'CGSUP'
}

hostSetup = {
    'domain': 'http://10.10.10.232:61015',
    'username': 'daeseok.chae',
    'password': 'dexter123'
}

DB_IP = '10.10.10.232:27017'
# Mongo DB
DatabaseName = 'rocketchat' # TODO: need input
CollectionName = 'rocketchat_message' # TODO: need input

from pymongo import MongoClient

client = MongoClient(DB_IP)
g_DB = client[DatabaseName]
coll = g_DB[CollectionName]
roomColl = g_DB['rocketchat_room']

for showName in roomNameMapper.keys():
    try:
        print showName, roomColl.find_one({'fname':roomNameMapper[showName]})['_id']
    except:
        pass


hostSetup = {
    'domain':'https://chat.dexterstudios.com', # 10.10.10.232:61015
    'username': 'VelozBot',
    'password': 'dexterbot123'
}

api = RocketChatAPI(settings=hostSetup)
api.send_message('test 입니다', '8sKxmJd39Xv4nrTDa')
#
#
#
# for room in api.get_private_rooms():
#     print roomColl.find_one({'name':room['name']})['fname'], room['id']