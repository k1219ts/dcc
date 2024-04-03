# -*- coding: utf-8 -*-
import getpass
from pymodule.Qt import QtCore

from core import Database
from models.asset import Asset
from models.texture import Texture

from libs.hashableQStandardItem import HashableQStandardItem
from libs.standardItemModel import StandardItemModel

userName = getpass.getuser()

class ItemController(object):
    def __init__(self, parent=None):

        self.parent = parent
        self._documents = {}

        self.init()

    def init(self):
        self.item_model = StandardItemModel()
        self.item_proxy_model = QtCore.QSortFilterProxyModel()
        self.item_proxy_model.setSourceModel(self.item_model)

        self.parent.setModel(self.item_proxy_model)

    def clear(self):
        self._documents = {}
        self.item_model.clear()

    def add_item(self, item):
        status = ""
        if "status" in item:
            status = item["status"]

        itm = HashableQStandardItem(item["name"])
        itm.setData(item["files"]["preview"])
        itm.status = status

        self.item_model.appendRow(itm)
        self.add_document(itm, item)

    def find_item(self, model_index):
        if not model_index.isValid():
            return

        item_model_index = self.item_proxy_model.mapToSource(model_index)
        item = self.item_model.item(item_model_index.row())

        return item

    def find_document(self, model_index):
        item = self.find_item(model_index)
        return self.document(item)

    def add_document(self, list_widget_item, document):
        if document["category"] == "Texture":
            self._documents[list_widget_item] = Texture(document)
        else:
            self._documents[list_widget_item] = Asset(document)

    # get_document
    def document(self, list_widget_item):
        if list_widget_item in self._documents:
            return self._documents[list_widget_item]

        return None

    def delete_items(self, qitem_selection_ranges):
        for selection_range in qitem_selection_ranges:
            for model_index in selection_range.indexes():
                document = self.find_document(model_index)
                Database.AddDeleteItem(document.name, userName)
                Database.DeleteDocument(document.object_id, document.category)

            self.item_proxy_model.removeRows(
                selection_range.top(), selection_range.height())

    def deprecated_edit_tags(self, qitem_selection_ranges, tag_name):
        for selection_range in qitem_selection_ranges:
            for model_index in selection_range.indexes():
                document = self.find_document(model_index)
                try:
                    Database.AddTag(document.object_id, tag_name, document.category)
                except:
                    continue

                tags = tag_name.split(",")
                document.tag = tags
