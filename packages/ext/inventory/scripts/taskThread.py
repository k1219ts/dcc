# # -*- coding: utf-8 -*-

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import pymongo
from pymongo import MongoClient

import socket
import getpass
import dxConfig
import datetime
from bson import ObjectId

DB_IP = dxConfig.getConf('DB_IP')
DB_NAME = 'inventory'


class StatUpdateThread(QtCore.QThread):
    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)

        self.queryTerm = None
        self.updateTerm = None
        self.result = None
        self.action = '' # click, doubleclick ??

        self.finished.connect(self.deleteLater)


    def run(self):
        if getpass.getuser() == 'taehyung.lee':
            return
        else:
            client = MongoClient(DB_IP)
            db = client[DB_NAME]
            coll = db['stats']
            self.result = coll.find_one_and_update(self.queryTerm,
                                                   self.updateTerm,
                                                   upsert=True)

            coll = db['logs']
            coll.insert_one({'itemId' : self.queryTerm['itemId'],
                             'time':datetime.datetime.now().isoformat(),
                             'user': socket.gethostname().replace('.', '_'),
                             'action' : self.action
                             })

    def __del__(self):
        self.wait()


# class StatUpdateThread(QtCore.QRunnable):
#     def __init__(self):
#         QtCore.QRunnable.__init__(self)
#
#         self.queryTerm = None
#         self.updateTerm = None
#         self.result = None
#         self.action = '' # click, doubleclick ??
#
#
#     def run(self):
#         client = MongoClient(DB_IP)
#         db = client[DB_NAME]
#         coll = db['stats']
#         self.result = coll.find_one_and_update(self.queryTerm,
#                                                self.updateTerm,
#                                                upsert=True)
#
#         coll = db['logs']
#         coll.insert_one({'itemId' : self.queryTerm['itemId'],
#                          'time':datetime.datetime.now().isoformat(),
#                          'user': socket.gethostname().replace('.', '_'),
#                          'action' : self.action
#                          })
#         return
