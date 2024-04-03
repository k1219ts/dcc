# -*- coding: utf-8 -*-
from pymodule.Qt import QtWidgets

class HashableQListWidgetItem(QtWidgets.QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))
