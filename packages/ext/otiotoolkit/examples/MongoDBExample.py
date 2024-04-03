#coding:utf-8
##########################################
__author__  = 'daeseok.chae in Dexter CGSupervisor'
__date__ = '2020.02.14'
__comment__ = 'Python MongoDB Example'
##########################################

# IP config
import sys
sys.path.append("/backstage/libs/python_lib")

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')

# Mongo DB
DatabaseName = 'ExampleDB' # TODO: need input
CollectionName = 'ExampleCol' # TODO: need input


from pymongo import MongoClient
client = MongoClient(DB_IP)
g_DB = client[DatabaseName]
coll = g_DB[CollectionName]

def addItemByOne(item=dict):
    '''
    dictionary data
    :param item:
    :return:
    '''
    coll.insert_one(item)

def addItem(item=list):
    '''
    dict item in list
    :param item:
    :return:
    '''
    coll.insert_many(item)

def getItem(queryDict=dict):
    '''
    get item by multiple
    :param queryDict: {FindKey:Value}
    :return:
    '''
    findItem = coll.find(queryDict)
    return findItem

def getItemByOne(queryDict=dict):
    '''
    get item by one
    :param queryDict: {FindKey:Value}
    :return:
    '''
    findItem = coll.find_one(queryDict)
    return findItem

#---------------------------------------------------------------------------
#
#   MAIN
#
#---------------------------------------------------------------------------
if __name__ == '__main__':
    addItemByOne({'Key':'Value'})

    addItem([{'Key': 'Value', 'Key2':['Value1', 'Value2']}, {'Key': 'Value'}])

    print getItemByOne({'Key':'Value'})

    print '*' * 80

    for item in getItem({'Key': 'Value'}):
        print item

