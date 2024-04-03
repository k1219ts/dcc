from PySide2 import QtWidgets

class DirItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, groupName = "", parentWidget = None):
        QtWidgets.QTreeWidgetItem.__init__(self, parentWidget)

        self.groupName = groupName
        self.setText(0, groupName)
