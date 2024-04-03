from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

from libs.hashableQStandardItem import HashableQStandardItem
from libs.utils import resize_pixmap

class StandardItemModel(QtGui.QStandardItemModel):
    def __init__(self, *args, **kwargs):
        QtGui.QStandardItemModel.__init__(self, *args, **kwargs)
        self.setItemPrototype(HashableQStandardItem())

        self._size_hint = QtCore.QSize(170, 170)
        self._icon_width = 160
        self._icon_height = 120

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DecorationRole:
            it = self.itemFromIndex(index)
            value = it.data(QtCore.Qt.DecorationRole)
            if value is None:
                path = it.data(HashableQStandardItem.PathRole)
                pixmap = resize_pixmap(path, it.status, self._icon_width, self._icon_height)
                value = QtGui.QIcon(pixmap)
                it.setData(value, QtCore.Qt.DecorationRole)
            return value
        elif role == QtCore.Qt.SizeHintRole:
            return self.size_hint
        else:
            return QtGui.QStandardItemModel.data(self, index, role)


    @property
    def size_hint(self):
        return self._size_hint

    @size_hint.setter
    def size_hint(self, value):
        self._size_hint = value

    def set_size_hint(self, size_hint):
        self._size_hint = size_hint

    @property
    def icon_width(self):
        return self._icon_width

    @icon_width.setter
    def icon_width(self, value):
        self._icon_width = value

    @property
    def icon_height(self):
        return self._icon_height

    @icon_height.setter
    def icon_height(self, value):
        self._icon_height = value
