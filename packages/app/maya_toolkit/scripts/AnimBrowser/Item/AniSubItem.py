import os
from PySide2 import QtWidgets

class AniSubItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parentText, itemName, parentWidget = None):
        QtWidgets.QTreeWidgetItem.__init__(self, parentWidget)

        self.itemFullPath = os.path.join(parentText, itemName) + "/"
        self.itemName = itemName
        self.setText(0, itemName)
