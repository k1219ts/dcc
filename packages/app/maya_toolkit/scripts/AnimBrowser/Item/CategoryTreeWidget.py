from pymodule.Qt import QtWidgets

class CategoryTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent):
        QtWidgets.QTreeWidget.__init__(self, parent)

        self.setDragEnabled(True)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        print "dragEnterEvent", event.source()
        event.accept()

    def dragMoveEvent(self, event):
        print "dragMoveEvent", event.source()
        event.accept()

    def dropEvent(self, event):
        print "dropEvent", event.source()
        event.accept()