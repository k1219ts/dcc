import sgCommon

from PySide2 import QtWidgets, QtGui, QtCore
import pprint


class MetaTableModel(QtCore.QAbstractTableModel):
    def __init__(self, datain, colorValue, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.dataArray = datain
        self.colorValue = colorValue

    def rowCount(self, parent):
        if self.dataArray:
            return len(self.dataArray)
        return None

    def columnCount(self, parent):
        if self.dataArray:
            return len(self.dataArray[0])
        return None

    def data(self, index, role):
        value = self.dataArray[index.row()][index.column()]
        if index.isValid():
            if role == QtCore.Qt.BackgroundRole:
                if str(value) in self.colorValue:
                    return QtGui.QColor(QtGui.QColor(255, 0, 0, 255))

            if role != QtCore.Qt.DisplayRole:
                return None
            return value


class variantWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, main, parent, variantSetName, value, selected, color=''):
        super(variantWidgetItem, self).__init__(parent)

        self.parent = parent
        self.main = main
        self.variants = main.variants
        self.baseWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.baseWidget)

        # Column 0 -- label
        self.variantName = QtWidgets.QLabel(variantSetName)
        self.variantName.setMaximumSize(QtCore.QSize(110, 25))
        self.layout.addWidget(self.variantName)

        # Column 1 -- comboBox
        self.variantSelection = QtWidgets.QComboBox()
        self.layout.addWidget(self.variantSelection)
        self.variantSelection.addItems(value)
        currentIdx = self.variantSelection.findText(selected)
        self.variantSelection.setCurrentIndex(currentIdx)
        self.parent.setItemWidget(self, 0, self.baseWidget)
        self.variantSelection.activated.connect(self.selectVariant)

        # init color
        self.setColor()

    def selectVariant(self, index):
        self.main.ui.variant_treeWidget.clear()
        self.main.ui.metadata_treeWidget.clear()

        currentItem = self.main.ui.sceneGraph_treeWidget.currentItem()
        nsLayer = currentItem.text(0)
        primPath = currentItem.text(1)

        self.main.setVariantItem(primPath, nsLayer, self.variantName.text(), self.variantSelection.itemText(index))
        self.updateSceneGraph()

        pprint.pprint(self.main.variantSpace)
        self.setColor()

        if not self.main.variant4View.has_key(primPath):
            self.main.variant4View[primPath] = {}
        self.main.variant4View[primPath][self.variantName.text()] = self.variantSelection.itemText(index)

    def updateSceneGraph(self):
        currentItem = self.main.ui.sceneGraph_treeWidget.currentItem()
        primPath = currentItem.text(1)
        currentItem.takeChildren()

        prim = self.main.stage.GetPrimAtPath(primPath)
        self.main.getAllPrims(prim, currentItem, primPath)

    def setColor(self):
        currentItem = self.main.ui.sceneGraph_treeWidget.currentItem()
        primPath = currentItem.text(1)

        varName = self.variantName.text()
        varVer = self.variantSelection.currentText()
        if self.main.variantSpace[primPath].has_key(varName):
            saveVer = self.main.variantSpace[primPath][varName]

            # print 'run setColor', varName, varVer, saveVer

            if varVer != saveVer:
                self.variantName.setStyleSheet('Color: yellow')
            else:
                self.variantName.setStyleSheet('Color: white')

class metadataWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent):
        super(metadataWidgetItem, self).__init__(parent)

        self.parent = parent
        self.baseWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.baseWidget)

        # Column 0 -- tableView
        self.metaTableView = QtWidgets.QTableView()
        self.layout.addWidget(self.metaTableView)
        self.metaTableView.clearSpans()
        self.parent.treeWidget().setItemWidget(self, 0, self.baseWidget)

        self.metaTableView.horizontalHeader().setVisible(False)
        self.metaTableView.verticalHeader().setVisible(False)

    def setWidgetSize(self):
        header = self.metaTableView.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        self.metaTableView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.metaTableView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.metaTableView.setFixedSize(695, self.metaTableView.verticalHeader().length())
