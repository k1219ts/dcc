# -*- coding: utf-8 -*-
import getpass
from pymodule.Qt import QtCore

from core import Database
from models.asset import Asset
from models.texture import Texture
from controllers.itemController import ItemController

from libs.hashableQStandardItem import HashableQStandardItem
from libs.standardItemModel import StandardItemModel

userName = getpass.getuser()

class BookmarkController(ItemController):
    def __init__(self, parent=None):
        ItemController.__init__(self, parent)

        self.item_model.size_hint = QtCore.QSize(70, 70)
        self.item_model.icon_width = 60
        self.item_model.icon_height = 35

    def bookmarks(self):
        return Database.GetBookmarkList(userName)

    def reload_list(self, bookmarks=None):
        self.clear()

        if bookmarks is None:
            bookmarks = self.bookmarks()

        temp_bookmarks = []
        missing_bookmark = False
        for bookmark in bookmarks:
            try:
                item = Database.GetItem(bookmark["category"], bookmark["ID"])
            except:
                missing_bookmark = True
                continue

            self.add_item(item)
            temp_bookmarks.append(bookmark)

        if missing_bookmark:
            Database.UpdateBookmarkList(userName, temp_bookmarks)

    def delete_items(self, qitem_selection_ranges):
        bookmarks = self.bookmarks()

        for selection_range in qitem_selection_ranges:
            for model_index in selection_range.indexes():
                document = self.find_document(model_index)

                for row, bookmark in enumerate(bookmarks):
                    if bookmark["ID"] == document.object_id:
                        del bookmarks[row]
                        break

        user = Database.UpdateBookmarkList(userName, bookmarks)
        self.reload_list(user["bookmark"])

    def add_items(self, source):
        selection_ranges = source.selectionModel().selection()

        # TODO: 드롭할 때마다 매번 체크하기 때문에 추후 정리 필요
        Database.AddUserInBookmark(userName)

        bookmarkWidgetList = []
        for bookmark in self.bookmarks():
            bookmarkWidgetList.append(bookmark["ID"])

        for selection_range in selection_ranges:
            for model_index in selection_range.indexes():
                document = source.find_document(model_index)
                if document.object_id in bookmarkWidgetList:
                    pass
                else:
                    item = Database.GetItem(document.category, document.object_id)
                    self.add_item(item)

                    dbData = item["_id"]
                    itemName = item["name"]
                    getCategory = item["category"]
                    Database.AddBookmarkItem(userName, dbData, getCategory, itemName)
