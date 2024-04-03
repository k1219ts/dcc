# -*- coding: utf-8 -*-
import getpass
import os
import sys


if sys.platform == 'darwin':
    sys.path.insert(0, '/netapp/backstage/pub/lib/python_lib')

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore
import viewer
import items

import pymongo
from pymongo import MongoClient
import getpass
import dxConfig

DB_IP = dxConfig.getConf('DB_IP')
DB_NAME = 'inventory'
COLL = 'logs'

class InventoryMonitor(viewer.BaseList):
    def __init__(self, parent):
        super(InventoryMonitor, self).__init__(parent)

        self.resize(1600,900)

        self.getRecentAction(action='doubleclick', limit=100)

    def getRecentAction(self, action='click', limit=100):
        client = MongoClient(DB_IP)
        db = client[DB_NAME]
        coll = db[COLL]
        infos = coll.find().sort("time", -1).limit(limit)
        for record in infos:
            assetColl = db['assets']
            doc = assetColl.find_one({'_id':record['itemId']})
            imgItem = viewer.makeItems(self, doc)

            if type(imgItem) == items.ThumbnailItem:
                self.itemWidget(imgItem).titleLabel.setText(record['user'])
            else:
                imgItem.setText(record['user'])


    def keyPressEvent(self, event):
        # TEST DEV FOR CATEGORY EDIT
        if event.key() == QtCore.Qt.Key_F5:
            print "refresh category"
            self.clear()
            self.getRecentAction(action='click', limit=100)




def main():
    print "MAIN!!"
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(":appIcon.png"))
    app.setDoubleClickInterval(200)
    thInst = InventoryMonitor(None)

    thInst.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()