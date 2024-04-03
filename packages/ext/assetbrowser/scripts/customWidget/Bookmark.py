#coding=utf-8
from PySide2 import QtWidgets, QtGui, QtCore
from core import Database
from Item import ItemListWidget

from pymongo import MongoClient
import dxConfig
import getpass
gDBIP = dxConfig.getConf("DB_IP")
client = MongoClient(gDBIP)
gDB = client["ASSETLIB"]
userName = getpass.getuser()

class BookmarkListWidget(ItemListWidget):
    def __init__(self, parent=None):
        ItemListWidget.__init__(self)

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setIconSize(QtCore.QSize(50, 50))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setWindowFlags(QtCore.Qt.Window)
        self.deleteKey = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete),self)
        self.deleteKey.activated.connect(self.deleteItem)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDragDropOverwriteMode(False)

        usrBookmarkList = Database.GetBookmarkList(userName)
        if usrBookmarkList:
            for i in usrBookmarkList:
                if i['category'] == 'Texture':
                    item = Database.gDB.source.find_one({'name': i['name']})
                else:
                    item = Database.gDB.item.find_one({'name': i['name']})
                itemName = item['name']
                # print(itemName)
                icon_path = item['files']['preview']
                itm = QtWidgets.QListWidgetItem(itemName)
                itm.setIcon(QtGui.QIcon(icon_path))
                itm.setSizeHint(QtCore.QSize(70, 70))
                self.addItem(itm)

    def get_selectedItem(self):
        itemDB= []
        selectedItems = self.selectedItems()
        for item in selectedItems:
            itemName = item.text()
            for i in Database.gDB.user_config.find({'user': userName}):
                for item in i['bookmark']:
                    if item['category'] == 'Texture' and item['name'] == itemName:
                        for i in Database.gDB.source.find({'name': itemName}):
                            itemDB.append(i)
                    else:
                        for i in Database.gDB.item.find({'name': itemName}):
                            itemDB.append(i)
        self.item = itemDB
        for i in itemDB:
            item = i
            self.thumbnailPath = item['files']['preview']
            reply = item['reply'][0]
            self.user = reply['user']
            self.time = reply['time'].split('T')[0]
            self.checkCategory = item['category']

            if self.checkCategory == 'Texture':
                self.assetName = item['name']
                self.filePath = item['files']['filePath']
            else:
                self.assetName = item['name']
                self.filePath = item['files']['usdfile']


    def deleteItem(self):
        selItem=[]
        deletedNameList =[]

        for sel in self.selectedItems():
            itemName = sel.text()
            bookmarkItem = Database.gDB.user_config.find_one({'user': userName})

            for i in bookmarkItem['bookmark']:
                if i['category'] == 'Texture' and i['name'] == itemName:
                    item= Database.gDB.source.find_one({'name': itemName})
                    deletedNameList.append(item['name'])
                    selItem.append(itemName)
                    self.takeItem(self.row(sel))

                elif i['name'] == itemName:
                    print(i['name'])
                    item = Database.gDB.item.find_one({"name": itemName})
                    deletedNameList.append(item['name'])
                    selItem.append(itemName)
                    self.takeItem(self.row(sel))
        Database.UpdateBookmarkList(userName, deletedNameList)


    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        source_items = event.source().selectedItems()
        if source_items[0].listWidget() == self:
            return

        for i in source_items:
            source_name = i.text()
            bookmarkWidgetList =[]
            getBookmarkItems =Database.gDB.user_config.find_one({'user':userName})
            for db in getBookmarkItems['bookmark']:
                bookmarkWidgetList.append(db['name'])

            if source_name in bookmarkWidgetList:
                pass
            else:
                current = self.category.currentItem()
                category = current.parent().text(0)
                if category == 'Texture':
                    item = Database.gDB.source.find_one({"name": source_name})
                else:
                    item = Database.gDB.item.find_one({"name": source_name})

                itemName = item["name"]
                getCategory = item['category']
                icon_path = item['files']['preview']
                itm = QtWidgets.QListWidgetItem(itemName)
                itm.setIcon(QtGui.QIcon(icon_path))
                itm.setSizeHint(QtCore.QSize(70, 70))
                self.addItem(itm)
                dbData=item["_id"]
                Database.AddBookmarkItem(userName, dbData,getCategory,itemName)
                event.accept()
