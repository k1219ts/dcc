import string
import time

from Katana import QtCore, QtGui, QtWidgets, UI4, QT4Widgets, QT4FormWidgets
from Katana import NodegraphAPI, Utils
from Katana import UniqueName, FormMaster

import ScriptActions as SA

class UsdVersionResolverEditor(QtWidgets.QWidget):
    def __init__(self, parent, node):
        node.upgrade()

        self.__node   = node
        self.__frozen = True

        QtWidgets.QWidget.__init__(self, parent)
        QtWidgets.QVBoxLayout(self)

        self.toolbarLayout = QtWidgets.QHBoxLayout()
        self.layout().addItem(self.toolbarLayout)

        self.addButton = UI4.Widgets.ToolbarButton(
            'Reset', self,
            UI4.Util.IconManager.GetPixmap('Icons/reset16.png'),
            UI4.Util.IconManager.GetPixmap('Icons/resetHilite16.png')
        )
        self.addButton.clicked.connect(self.addButtonClicked)
        self.toolbarLayout.addWidget(self.addButton)
        self.toolbarLayout.addStretch()

        # version tree widget
        self.treeStretchBox = UI4.Widgets.StretchBox(self, allowHorizontal=False, allowVertical=True)
        self.layout().addWidget(self.treeStretchBox)

        self.treeWidget = QT4Widgets.SortableTreeWidget(self.treeStretchBox)
        self.treeWidget.setHeaderLabels(['Name', '', '', '', ''])
        self.treeWidget.setSelectionMode(QtWidgets.QTreeWidget.SingleSelection)
        self.treeWidget.setAllColumnsShowFocus(True)
        self.treeWidget.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        self.treeStretchBox.layout().addWidget(self.treeWidget)

        self.layout().addStretch()

        # Default Member Variable
        self.shotName   = None
        self.vctrlGroup = None
        self.updateCurrentState()
        self.updateTreeContents()


    def updateCurrentState(self):
        self.shotName = SA.GetShotName()
        if self.shotName:
            self.vctrlGroup = SA.GetRefNode(self.__node, '%s_VersionCtrlGroup' % self.shotName)


    def showEvent(self, event):
        QtWidgets.QWidget.showEvent(self, event)
        self.updateCurrentState()
        self.updateTreeContents()
        if self.__frozen:
            self.__frozen = False
            self._thaw()

    def hideEvent(self, event):
        QtWidgets.QWidget.hideEvent(self, event)
        if not self.__frozen:
            self.__frozen = True
            self._freeze()

    def _thaw(self):
        self.__setupEventHandlers(True)

    def _freeze(self):
        self.__setupEventHandlers(False)

    def __setupEventHandlers(self, enabled):
        Utils.EventModule.RegisterCollapsedHandler(self.__updateCB, 'parameter_setValue', enabled=enabled)

    def __updateCB(self, args):
        eventNode  = args[0][2]['node']
        eventParam = args[0][2]['param'].getFullName()
        if eventParam.find('rootNode.variables') > -1:
            print '[INFO UsdVersionResolver]: Update widgets'
            self.updateCurrentState()
            self.updateTreeContents()

    def addButtonClicked(self):
        self.shotName = SA.GetShotName()
        if not self.shotName:
            return
        refName = '%s_VersionCtrlGroup' % self.shotName
        self.vctrlGroup = SA.GetRefNode(self.__node, refName)
        if self.vctrlGroup:
            self.vctrlGroup.delete()
        self.vctrlGroup = SA.CreateVersionCtrlGroup(self.__node).doIt()
        self.updateTreeContents()


    def updateTreeContents(self):
        self.treeWidget.clear()
        if not self.vctrlGroup:
            return

        SG = SA.GetScenegraph(self.__node)
        SG.doIt()
        widgetNameMap = dict()
        for loc, vers in SG.info:
            relPath  = loc.split('/World/')[-1]
            splitPath= relPath.split('/')
            for i in range(len(splitPath)):
                name  = string.join(splitPath[:i+1], '/')
                parent= string.join(splitPath[:i], '/')
                if parent:
                    parentWidget = widgetNameMap[parent]
                else:
                    parentWidget = self.treeWidget
                if not widgetNameMap.has_key(name):
                    item = VersionItemWidget(parentWidget, name, SG)
                    widgetNameMap[name] = item
                    item.setExpanded(True)



class VersionItemWidget(QT4Widgets.SortableTreeWidgetItem):
    def __init__(self, parent, relPath, SG):
        QT4Widgets.SortableTreeWidgetItem.__init__(self, parent)
        self.parent = parent
        self.relPath= relPath
        self.SG = SG

        self.setText(0, relPath.split('/')[-1])
        self.vctrlGroup = SA.GetVersionGroup(SG.node)

        self.verWidgets = list()
        self.addCtrlWidget()


    def addCtrlWidget(self):
        if not self.SG.variantMap.has_key(self.relPath):
            return
        variants = self.SG.variantMap[self.relPath]
        if self.relPath.find('Rig/') > -1:
            variants = ['aniVer', 'simVer', 'groomVer']
        self._dependAction = False

        # create widget
        self.verWidgets = [None] * len(variants)
        for i in range(len(variants)):
            name = variants[i]

            versionWidget = QtWidgets.QWidget()
            layoutWidget  = QtWidgets.QHBoxLayout(versionWidget)
            labelWidget   = QtWidgets.QLabel(name.replace('Ver', ''))
            labelWidget.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            layoutWidget.addWidget(labelWidget)

            self.verWidgets[i] = QtWidgets.QComboBox()
            self.verWidgets[i].setFixedWidth(60)
            objName = self.relPath.split('/')[-1] + '_' + name
            self.verWidgets[i].setObjectName(objName)
            self.verWidgets[i].currentIndexChanged.connect(lambda _, x=i: self.ctrlProc(x))
            layoutWidget.addWidget(self.verWidgets[i])

            self.treeWidget().setItemWidget(self, i+1, versionWidget)

        for i in range(len(variants)):
            name = variants[i]
            objName = str(self.verWidgets[i].objectName())
            pxrVarNode = SA.GetRefNode(self.vctrlGroup, objName)
            if pxrVarNode:
                valueData = SA.GetVariantValues(pxrVarNode, self.SG.getLocation(self.relPath), name)
                if valueData:
                    items, value = valueData
                    self.setWidgetItem(self.verWidgets[i], items, value)
        self._dependAction = True

    def ctrlProc(self, index):
        sender = self.verWidgets[index]
        value  = str(sender.currentText())
        self.SetNodeVersion(sender, value)
        if self._dependAction:
            self.SetDependency(index)

    def setWidgetItem(self, widget, items, value):
        widget.addItems(items)
        if value in items:
            setIndex = items.index(value)
        else:
            setIndex = len(items) - 1
        widget.setCurrentIndex(setIndex)


    def SetNodeVersion(self, widget, value):
        objName = str(widget.objectName())
        pxrVarNode = SA.GetRefNode(self.vctrlGroup, objName)
        pxrVarNode.getParameter('args.variantSelection.value').setValue(value, 0)
        pxrVarNode.getParameter('args.variantSelection.enable').setValue(1, 0)

    def SetDependency(self, index):
        if len(self.verWidgets) == 1:
            return
        self._dependAction = False
        currentObjname = str(self.verWidgets[index].objectName())
        currentPxrVarNode = SA.GetRefNode(self.vctrlGroup, currentObjname)
        for i in range(index+1, len(self.verWidgets)):
            objName = str(self.verWidgets[i].objectName())
            pxrVarNode = SA.GetRefNode(self.vctrlGroup, objName)
            if pxrVarNode:
                name = objName.split('_')[-1]
                valueData = SA.GetVariantValues(pxrVarNode, self.SG.getLocation(self.relPath), name)
                self.verWidgets[i].clear()
                if valueData:
                    items, value = valueData
                    self.setWidgetItem(self.verWidgets[i], items, value)
        self._dependAction = True
