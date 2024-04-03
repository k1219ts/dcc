# -*- coding: utf-8 -*-
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

class HashableQStandardItem(QtGui.QStandardItem):

    PathRole = QtCore.Qt.UserRole + 1

    def __init__(self, *args):
        super(HashableQStandardItem, self).__init__(*args)
        self.path = ""
        self._status = ""

    def __hash__(self):
        return hash(id(self))

    # def __init__(self, *args, **kwargs):
    #     QStandardItem.__init__(self, *args, **kwargs)

    def data(self, role=QtCore.Qt.UserRole + 1):
        if role == HashableQStandardItem.PathRole:
            return self.path
        return QtGui.QStandardItem.data(self, role)

    def setData(self, value, role=QtCore.Qt.UserRole + 1):
        if role == HashableQStandardItem.PathRole:
            self.path = value
        else:
            QtGui.QStandardItem.setData(self, value, role)
            # self.emitDataChanged()

    def type(self):
        return QtCore.Qt.UserType

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
