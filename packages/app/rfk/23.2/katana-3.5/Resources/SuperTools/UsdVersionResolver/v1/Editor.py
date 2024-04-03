import time

from Katana import QtCore, QtGui, UI4, QT4Widgets, QT4FormWidgets
from Katana import NodegraphAPI, Utils
from Katana import UniqueName, FormMaster

import ScriptActions as SA

class UsdVersionResolverEditor(QtGui.QWidget):
    def __init__(self, parent, node):
        node.upgrade()

        self.__node = node
        self.__frozen = True

        QtGui.QWidget.__init__(self, parent)
        QtGui.QVBoxLayout(self)

        self.__toolbarLayout = QtGui.QHBoxLayout()
        self.layout().addItem(self.__toolbarLayout)

        self.__addButton = UI4.Widgets.ToolbarButton(
            'Reset', self,
            UI4.Util.IconManager.GetPixmap('Icons/reset16.png'),
            UI4.Util.IconManager.GetPixmap('Icons/resetHilite16.png')
        )
        self.connect(self.__addButton, QtCore.SIGNAL('clicked()'), self.__addButtonClicked)
        self.__toolbarLayout.addWidget(self.__addButton)
        self.__toolbarLayout.addStretch()

        # tree widget
        self.__treeStretchBox = UI4.Widgets.StretchBox(self, allowHorizontal=False, allowVertical=True)
        self.layout().addWidget(self.__treeStretchBox)

        self.__treeWidget = QT4Widgets.SortableTreeWidget(self.__treeStretchBox)
        self.__treeWidget.setHeaderLabels(['Name', 'Version', 'Version', 'Version', ''])
        self.__treeWidget.setSelectionMode(QtGui.QTreeWidget.SingleSelection)
        self.__treeWidget.setAllColumnsShowFocus(True)
        self.__treeWidget.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)

        self.__treeStretchBox.layout().addWidget(self.__treeWidget)

        self.layout().addStretch()

        # Default Variable
        self.shotName = None; self.vctrlGroup = None
        self.updateCurrentState()
        self.__updateTreeContents()


    def updateCurrentState(self):
        self.shotName = SA.GetShotName()
        if self.shotName:
            self.vctrlGroup = SA.GetRefNode(self.__node, '%s_VersionCtrlGroup' % self.shotName)

    def showEvent(self, event):
        QtGui.QWidget.showEvent(self, event)
        self.updateCurrentState()
        self.__updateTreeContents()
        if self.__frozen:
            self.__frozen = False
            self._thaw()

    def hideEvent(self, event):
        QtGui.QWidget.hideEvent(self, event)
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
        eventNode = args[0][2]['node']
        eventParam= args[0][2]['param'].getFullName()
        if eventParam.find('rootNode.variables') > -1:
            print '[INFO UsdVersionResolver]: Update widgets'
            self.updateCurrentState()
            self.__updateTreeContents()


    def __updateTreeContents(self):
        self.__treeWidget.clear()
        if not self.vctrlGroup:
            return

        treeInfo = SA.GetVersionTree(self.__node)
        treeInfo.doIt()
        main, layerMap = treeInfo.computeTreeItem()
        for m in main:
            mainItem = VersionItemWidget(self.__treeWidget, m, treeInfo)
            if layerMap[m]:
                for l in layerMap[m]:
                    lyrItem = VersionItemWidget(mainItem, m + '/' + l, treeInfo)
            self.__treeWidget.setItemExpanded(mainItem, True)


    def __addButtonClicked(self):
        self.shotName = SA.GetShotName()
        if not self.shotName:
            return

        refName = '%s_VersionCtrlGroup' % self.shotName
        self.vctrlGroup = SA.GetRefNode(self.__node, refName)
        if self.vctrlGroup:
            self.vctrlGroup.delete()
        self.vctrlGroup = SA.CreateVersionCtrlGroup(self.__node)
        self.__updateTreeContents()



class VersionItemWidget(QT4Widgets.SortableTreeWidgetItem):
    def __init__(self, parent, relpath, treeInfo):
        QT4Widgets.SortableTreeWidgetItem.__init__(self, parent)
        self.parent = parent
        self.setText(0, relpath.split('/')[-1])

        self.relpath  = relpath
        self.treeInfo = treeInfo
        self.vctrlGroup = SA.GetVersionGroup(treeInfo._node)
        self.m_widgets  = list()

        self.addCtrlWidget()

    def addCtrlWidget(self):
        if not self.treeInfo.variantMap.has_key(self.relpath):
            return

        variants = self.treeInfo.variantMap[self.relpath]
        if self.relpath.find('rig/') > -1:
            variants = ['aniVersion', 'simVersion', 'zennVersion']

        self._dependAction = False

        # create widget
        self.m_widgets = [None] * len(variants)
        for i in range(len(variants)):
            name = variants[i]

            versionWidget= QtGui.QWidget()
            layoutWidget = QtGui.QHBoxLayout(versionWidget)

            labelWidget = QtGui.QLabel('  ' + name.replace('Version', ''))
            layoutWidget.addWidget(labelWidget)

            self.m_widgets[i] = QtGui.QComboBox()
            self.m_widgets[i].setFixedWidth(60)
            objname = self.relpath.split('/')[-1] + '_' + name
            self.m_widgets[i].setObjectName(objname)
            self.m_widgets[i].currentIndexChanged.connect(lambda _, x=i: self.ctrlProc(x))
            layoutWidget.addWidget(self.m_widgets[i])

            self.treeWidget().setItemWidget(self, i+1, versionWidget)

        for i in range(len(variants)):
            name = variants[i]
            objname = str(self.m_widgets[i].objectName())
            pxrVarNode = SA.GetRefNode(self.vctrlGroup, objname)
            if pxrVarNode:
                valueData = self.treeInfo.variantValues(self.relpath, name, pxrVarNode)
                if valueData:
                    items, value = valueData
                    self.SetWidgetItem(self.m_widgets[i], items, value)

        self._dependAction = True


    def ctrlProc(self, index):
        sender = self.m_widgets[index]
        value  = str(sender.currentText())
        self.SetNodeVersion(sender, value)
        if self._dependAction:
            self.SetDependency(index)


    def SetNodeVersion(self, widget, value):
        objname = str(widget.objectName())
        pxrVarNode = SA.GetRefNode(self.vctrlGroup, objname)
        pxrVarNode.getParameter('args.variantSelection.value').setValue(value, 0)
        pxrVarNode.getParameter('args.variantSelection.enable').setValue(1, 0)

    def SetWidgetItem(self, widget, items, value):
        widget.addItems(items)
        if value in items:
            setIndex = items.index(value)
        else:
            setIndex = len(items) - 1
        widget.setCurrentIndex(setIndex)


    def SetDependency(self, index):
        if len(self.m_widgets) == 1:
            return
        self._dependAction = False
        currentObjname = str(self.m_widgets[index].objectName())
        currentPxrVarNode = SA.GetRefNode(self.vctrlGroup, currentObjname)
        for i in range(index+1, len(self.m_widgets)):
            objname = str(self.m_widgets[i].objectName())
            pxrVarNode = SA.GetRefNode(self.vctrlGroup, objname)
            if pxrVarNode:
                varname = objname.split('_')[-1]
                valueData = self.treeInfo.variantValues(self.relpath, varname, currentPxrVarNode)
                # self.m_widgets[i].setCurrentIndex(-1)
                self.m_widgets[i].clear()
                if valueData:
                    items, value = valueData
                    self.SetWidgetItem(self.m_widgets[i], items, value)
        self._dependAction = True
