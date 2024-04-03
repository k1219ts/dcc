# -*- coding: utf-8 -*-
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

from controllers.bookmarkController import BookmarkController
from customWidget.Item import ItemView

class BookmarkView(ItemView):
    def __init__(self, parent=None):
        ItemView.__init__(self)

        self.item_controller = BookmarkController(self)

        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setDragEnabled(False)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setIconSize(QtCore.QSize(70, 70))
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setViewMode(QtWidgets.QListView.IconMode)
        self.setWindowFlags(QtCore.Qt.Window)

    def reload_list(self):
        self.item_controller.reload_list()

    def deleteItem(self):
        selection_ranges = self.selectionModel().selection()
        self.item_controller.delete_items(selection_ranges)
        # TODO : File Summary 내용 삭제

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        try:
            self.item_controller.add_items(event.source())
        except:
            event.ignore()
            return

        event.accept()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Delete:
            self.deleteItem()
