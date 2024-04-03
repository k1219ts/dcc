from Katana import UI4
from Katana import NodegraphAPI

from PyQt5 import QtCore, QtGui, QtWidgets

from VariantMenuUI import Ui_Form
import SetVariants

import UI4.Util.Caches

class customTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, mainWidget, parent, variantName, variantValueList):
        QtWidgets.QTreeWidgetItem.__init__(self, parent)

        itemFont = QtGui.QFont()
        itemFont.setPointSize(20)
        self.setText(0, variantName)
        self.setFont(0, itemFont)
        self.setTextAlignment(0, QtCore.Qt.AlignRight)

        self.variantName = variantName
        self.isClose = True

        self.mainWidget = mainWidget

        filterModel = QtCore.QSortFilterProxyModel()
        self.variantInfoEdit = QtWidgets.QComboBox()
        variantCompleter = QtWidgets.QCompleter(variantValueList)
        variantCompleter.setModel(filterModel)
        # variantCompleter.setCompletionMode( QtWidgets.QCompleter.UnfilteredPopupCompletion )
        variantCompleter.setCaseSensitivity( QtCore.Qt.CaseInsensitive )

        shaderItem = QtGui.QStandardItemModel()
        for index, variantValue in enumerate(variantValueList):
            item = QtGui.QStandardItem(variantValue)
            item.setFont(itemFont)
            shaderItem.setItem(index, item)
        filterModel.setSourceModel(shaderItem)


        self.variantInfoEdit.addItems(variantValueList)
        self.variantInfoEdit.setCompleter(variantCompleter)
        self.variantInfoEdit.currentIndexChanged.connect(self.setVariant)
        self.variantInfoEdit.setEditable(True)

        itemDelegate = QtWidgets.QStyledItemDelegate()
        self.variantInfoEdit.setItemDelegate(itemDelegate)

        comboStyle = '''
            QComboBox { }
        '''
        self.variantInfoEdit.setStyleSheet(comboStyle)

        lineEdit = QtWidgets.QLineEdit()
        lineEdit.setFont(itemFont)
        lineEdit.setCompleter(variantCompleter)
        self.variantInfoEdit.setLineEdit(lineEdit)
        parent.setItemWidget(self, 1, self.variantInfoEdit)

        varGroup = NodegraphAPI.GetRootNode().getParameter('variables')
        variableParam = varGroup.getChild(self.variantName)
        value = variableParam.getChild('value').getValue(0)
        self.variantInfoEdit.setCurrentIndex(self.variantInfoEdit.findText(value))
        lineEdit.setText(value)

    def setVariant(self, text):
        #Todo: variant select, after event
        varGroup = NodegraphAPI.GetRootNode().getParameter('variables')
        variableParam = varGroup.getChild(self.variantName)

        variableParam.getChild('value').setValue(str(self.variantInfoEdit.currentText()), 0)

        self.mainWidget.closeWidget()

class VariantMenuMain(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setWindowFlags(QtCore.Qt.Popup)
        nodeGraphTab = UI4.App.Tabs.FindTopTab("Node Graph")
        offset = nodeGraphTab.mapToGlobal(QtCore.QPoint(0, 0))

        self.move(nodeGraphTab.frameGeometry().center() - self.frameGeometry().center() + offset)
        self.setWindowOpacity(0.9)

        # variantNameList = ["assetVariant"]
        self.reloadUI()

        self.ui.pushButton.clicked.connect(self.refreshClicked)

        self.show()

    def closeWidget(self):
        if self.isClose:
            self.close()

    def reloadUI(self):
        self.isClose = False
        while self.ui.treeWidget.topLevelItemCount() > 0:
            self.ui.treeWidget.topLevelItem(0).isClose = False
            self.ui.treeWidget.takeTopLevelItem(0)
            treeitem = self.ui.treeWidget.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()

        variantGroup = NodegraphAPI.GetRootNode().getParameter("variables")
        for variant in variantGroup.getChildren():
            variantName = variant.getName()
            variantValueList = []
            if variant.getChild("options"):
                for variable in variant.getChild("options").getChildren():
                    variantValueList.append(variable.getValue(0))
                customTreeWidgetItem(self, self.ui.treeWidget, variantName, variantValueList)
            # self.ui.treeWidget.addTopLevelItem()

        self.ui.treeWidget.setMinimumHeight(
            self.ui.treeWidget.topLevelItemCount() * 30 + (self.ui.treeWidget.topLevelItemCount() * 5) + 10)
        self.ui.treeWidget.setMaximumHeight(
            self.ui.treeWidget.topLevelItemCount() * 30 + (self.ui.treeWidget.topLevelItemCount() * 5) + 10)
        self.isClose = True

    def refreshClicked(self):
        # First, Flush Cache
        # Katana.CacheManager.flush()
        UI4.Util.Caches.FlushCaches()

        variantSetNameList = list()
        variantGroup = NodegraphAPI.GetRootNode().getParameter('variables')
        for variant in variantGroup.getChildren():
            variantName = variant.getName()
            variantSetNameList.append(variantName)

        variantNodes = NodegraphAPI.GetAllNodesByType('PxrUsdInVariantSelect')
        variantNodes+= NodegraphAPI.GetAllNodesByType('UsdInVariantSelect')
        for node in variantNodes:
        # for node in NodegraphAPI.GetAllNodesByType('PxrUsdInVariantSelect'):
            selectParam = node.getParameter('args.variantSelection.value')
            if selectParam.isExpression():
                expression = selectParam.getExpression()
                variantSetName = expression.split('.')[-2]
                if variantSetName in variantSetNameList:
                    SetVariants.doIt(node, create=False)

        self.reloadUI()
