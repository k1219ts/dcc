from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

from .ContentItem import ContentItem
from AnimBrowser.Pipeline.MongoDB import MongoDB

DBNAME = "inventory"
COLLNAME = "anim_item"

class ContentTab(QtWidgets.QListWidget):
    def __init__(self, animationInfoList = list(), itemFullPath = ""):
        QtWidgets.QListWidget.__init__(self)

        self.animationInfoList = animationInfoList
        self.itemFullPath = itemFullPath
        
        self.setViewMode(QtWidgets.QListView.IconMode)
        
        self.iconSize = QtCore.QSize(282, 232)

        self.setGridSize(self.iconSize)
        self.setSpacing(5)
        # self.setIconSize(self.iconSize)
        self.setResizeMode(QtWidgets.QListView.Adjust)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragOnly)

        self.beforeItem = None

        self.maxLoadCount = len(self.animationInfoList) / 20
        self.currentLoadCount = -1
        self.isMoreLoading = True

    def loadContent(self):
        if self.isMoreLoading == False:
            return

        dialog = QtWidgets.QProgressDialog("data loading... [0 / %d]" % len(self.animationInfoList), "Cancel", 0, len(self.animationInfoList), self)
        dialog.show()
        for index, aniInfo in enumerate(self.animationInfoList):
            QtWidgets.QApplication.processEvents()
            dialog.setValue(index)
            dialog.setLabelText("data loading... [%s / %s]" % (index, len(self.animationInfoList)))
            ContentItem(self, contentInfo = aniInfo)

        dialog.close()

        self.isMoreLoading = False

    def refreshUI(self):
        if self.itemFullPath == "":
            return

        tierSplitItem = self.itemFullPath.split('/')
        self.dbPlugin = MongoDB(DBNAME, COLLNAME)
        self.animationInfoList = self.dbPlugin.getContentInfo(tierSplitItem[0],
                                                                 tierSplitItem[1],
                                                                 tierSplitItem[2],
                                                                 tierSplitItem[3])

        self.clear()

        self.loadContent()

    def setIconScale(self, scaleValue):
        self.setGridSize(self.iconSize * scaleValue)

        for index in range(self.count()):
            self.item(index).setContentSize(scaleValue)

    def dragEnterEvent(self, event):
        event.accept()
                        
