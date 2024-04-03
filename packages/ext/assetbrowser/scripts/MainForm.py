#coding=utf-8
# from PyQt4 import QtGui
# from PyQt4 import QtCore
# import sys

from PySide2 import QtWidgets, QtGui, QtCore
import os
from MainFormUI import Ui_Form
from core import Database
RESOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Resources")
ICON_PATH = os.path.join(RESOURCE_PATH, "icon")
import getpass
import sys
#sys.path.append('/backstage/apps/Maya/versions/2018/global/linux/scripts/xbUtils')


class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setWindowTitle("USD ASSETLIB BROWSER")

        self.ui.itemTreeWidget.setDragEnabled(True)
        self.ui.categoryWidget.clicked.connect(self.category_cliked)
        self.ui.searchEdit.returnPressed.connect(self.search_returnPressed)
        num= "Asset Total:"+ str(Database.gDB.item.find().count()) +',    Texture Total: ' + str(Database.gDB.source.find().count())
        self.print_itemNumber(num)


    def print_itemNumber(self,num):
        self.ui.statusLabel.clear()
        self.ui.statusLabel.setText(num)


    def search_returnPressed(self):
        self.ui.itemTreeWidget.clear()
        inputText = self.ui.searchEdit.text()
        inputText = inputText.lower()

        itemList = []
        SCcursor = Database.gDB.source.find({'name': {'$exists': True}})
        cursor = Database.gDB.item.find({'name': {'$exists': True}})
        cursorList = [SCcursor,cursor]
        for i in cursorList:
            for itr in i:
                if inputText in itr['name']:
                    if not itr in itemList:
                        itemList.append(itr)

                elif inputText.lower() in itr['name'].lower():
                    if not itr in itemList:
                        itemList.append(itr)

                elif itr['tag']:
                    for tag in itr['tag']:
                        if inputText in tag.lower():
                            if not itr in itemList:
                                itemList.append(itr)
                else:
                    pass

        for item in itemList:
            itemName = item['name']
            icon_path = item['files']['preview']
            itm = QtWidgets.QListWidgetItem(itemName)
            itm.setIcon(QtGui.QIcon(icon_path))
            itm.setSizeHint(QtCore.QSize(200, 200))
            self.ui.itemTreeWidget.addItem(itm)

        num = inputText + " : " + str(len(itemList))
        self.print_itemNumber(num)

    def category_cliked(self):
        # clear itemList
        if self.ui.itemTreeWidget.count() >= 1:
            self.ui.itemTreeWidget.clear()

        find_item = self.ui.categoryWidget.currentItem()

        if find_item is None:
            pass

        elif find_item.parent(): #sub_category
            current =self.ui.categoryWidget.currentItem()
            sub = current.text(0)
            main = current.parent().text(0)

            for item in Database.GetItems(main, sub):
                itemName = item['name']
                self.icon_path = item['files']['preview']
                self.itm = QtWidgets.QListWidgetItem(itemName)
                self.itm.setIcon(QtGui.QIcon(self.icon_path))
                self.itm.setSizeHint(QtCore.QSize(200, 200))
                self.ui.itemTreeWidget.addItem(self.itm)

            if current.parent().parent():
                # if main == 'Default' and sub == "Updated Asset":
                #     cursor = Database.gDB.item.find({'name': {'$exists': True}})
                #     for itr in cursor:
                #         if itr['category'] == 'default' or itr['subCategory'] == 'unknown':
                #             itemName = itr['name']
                #             icon_path = itr['files']['preview']
                #             itm = QtWidgets.QListWidgetItem(itemName)
                #             itm.setIcon(QtGui.QIcon(icon_path))
                #             itm.setSizeHint(QtCore.QSize(200, 200))
                #             self.ui.itemTreeWidget.addItem(itm)

                if main == "Texture":
                    for item in Database.GetSCItems(main, sub):
                        # print(item)
                        itemName = item['name']
                        self.icon_path = item['files']['preview']
                        self.itm = QtWidgets.QListWidgetItem(itemName)
                        self.itm.setIcon(QtGui.QIcon(self.icon_path))
                        self.itm.setSizeHint(QtCore.QSize(200, 200))
                        self.ui.itemTreeWidget.addItem(self.itm)

            else:
                if main == 'Default' and sub == "Updated Asset":
                    cursor = Database.gDB.item.find({'name': {'$exists': True}})

                    for itr in cursor:
                        if itr['category'] == 'default' or itr['subCategory'] == 'unknown':
                            itemName = itr['name']
                            icon_path = itr['files']['preview']
                            itm = QtWidgets.QListWidgetItem(itemName)
                            itm.setIcon(QtGui.QIcon(icon_path))
                            itm.setSizeHint(QtCore.QSize(200, 200))
                            self.ui.itemTreeWidget.addItem(itm)

                if main == "Texture":
                    for item in Database.GetSCItems(main, sub):
                        # print(item)
                        itemName = item['name']
                        self.icon_path = item['files']['preview']
                        self.itm = QtWidgets.QListWidgetItem(itemName)
                        self.itm.setIcon(QtGui.QIcon(self.icon_path))
                        self.itm.setSizeHint(QtCore.QSize(200, 200))
                        self.ui.itemTreeWidget.addItem(self.itm)


        # elif find_item.parent().parent(): #child_category
        #     current =self.ui.categoryWidget.currentItem()
        #     sub = current.text(0)
        #     main = current.parent().text(0)





        elif find_item.parent() is None: #Main category
            pass
