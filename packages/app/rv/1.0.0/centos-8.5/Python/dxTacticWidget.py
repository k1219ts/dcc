# -*- coding: utf-8 -*-
import os
import pprint

from PySide2 import QtCore, QtWidgets
from rv import commands, qtutils


API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

class setCustomWidget:
    def __init__(self, parent=None):
        # snapShot
        parent.snapTree = RecordTree(parent.tab_snapShot)
        parent.snapTree.setObjectName("snapTree")
        parent.gridLayout_snap.addWidget(parent.snapTree, 1, 0, 1, 1)
        parent.snapTree.setSortingEnabled(True)
        parent.snapTree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        snapTree = ['name', 'ver', 'process', 'context', 'status', 'type', 'artist', 'time']
        parent.snapTree.setColumnCount(len(snapTree))
        for idx, header in enumerate(snapTree):
            parent.snapTree.headerItem().setText(idx, header)

        # breakdown
        parent.brkdownTree = RecordTree(parent.tab_snapShot)
        parent.brkdownTree.setObjectName("reviewTree")
        parent.gridLayout_snap.addWidget(parent.brkdownTree, 2, 0, 1, 1)
        parent.brkdownTree.setSortingEnabled(True)
        parent.brkdownTree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        parent.brkdownTree.setSizePolicy(sizePolicy)
        parent.brkdownTree.resize(parent.brkdownTree.width(), 100)

        brkdown = ['name', 'ver', 'process', 'context', 'type', 'artist', 'time']
        parent.brkdownTree.setColumnCount(len(brkdown))
        for idx, header in enumerate(brkdown):
            parent.brkdownTree.headerItem().setText(idx, header)

        # Review Supervisor
        parent.reSupTree = RecordTree(parent.tab_reviewSup)
        parent.reSupTree.setObjectName("reSupTree")
        parent.gridLayout_reSup.addWidget(parent.reSupTree, 1, 0, 1, 1)
        parent.reSupTree.setSortingEnabled(True)
        parent.reSupTree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        # Milestone
        parent.mileTree = RecordTree(parent.tab_milestone)
        parent.mileTree.setObjectName("mileTree")
        parent.gridLayout_mile.addWidget(parent.mileTree, 1, 0, 1, 1)
        parent.mileTree.setSortingEnabled(True)
        parent.mileTree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        # Feedback Topic
        parent.feedTree = RecordTree(parent.tab_feedbackTopic)
        parent.feedTree.setObjectName("feedTree")
        parent.gridLayout_feed.addWidget(parent.feedTree, 1, 0, 1, 1)
        parent.feedTree.setSortingEnabled(True)
        parent.feedTree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)

        headers = ['name', 'ver', 'process', 'context', 'status', 'artist', 'time']
        Qtrees = [parent.reSupTree, parent.mileTree, parent.feedTree]
        for tree in Qtrees:
            tree.setColumnCount(len(headers))
            for idx, header in enumerate(headers):
                tree.headerItem().setText(idx, header)


class RecordTree(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        super(RecordTree, self).__init__(parent)
        self.widgets = qtutils.sessionWindow()

        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)

        self.itemDoubleClicked.connect(self.recordDoubleClicked)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.menuContext)

    def menuContext(self, point):
        menu = QtWidgets.QMenu()
        menu.addAction('Play Selection', self.playSelection)
        menu.addAction('Play All', self.playAll)
        menu.addSeparator()
        menu.addAction('Clear Selection', self.clearSelection)
        menu.exec_(self.mapToGlobal(point))

    def playSelection(self):
        playItems = []
        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)

            if self.isItemSelected(item):
                try:
                    itemPath = item.tacticData['path']
                except:
                    itemPath = item.tacticData['tactic_path']
                if itemPath:
                    playItems.append(itemPath)

        commands.clearSession()
        commands.addSources(playItems, '', True)

    def playAll(self):
        playItems = []
        for index in range(self.topLevelItemCount()):
            item = self.topLevelItem(index)
            try:
                itemPath = item.tacticData['path']
            except:
                itemPath = item.tacticData['tactic_path']
            if itemPath:
                playItems.append(itemPath)

        commands.clearSession()
        commands.addSources(playItems, '', True)

    def clearSelection(self):
        self.selectionModel().clearSelection()

    def recordDoubleClicked(self, item):
        if item:
            try:
                itemPath = item.tacticData['path']
            except:
                itemPath = item.tacticData['tactic_path']
            playItems = []

            if itemPath:
                if os.path.isdir(itemPath):
                    for fd in os.listdir(itemPath):
                        if fd.startswith('.'):
                            continue
                        playItems.append(os.path.join(itemPath, fd))
                else:
                    playItems.append(itemPath)
                commands.clearSession()
                commands.addSources(playItems, '', True)
            else:
                self.MessagePopup('업로드된 snapshot이 없습니다.')

    def dragEnterEvent(self, event):
        urls = []

        for item in self.selectedItems():
            try:
                itemPath = item.tacticData['path']
            except:
                itemPath = item.tacticData['tactic_path']
            urls.append(QtCore.QUrl('file://' + itemPath))

        event.mimeData().setUrls(urls)
        event.accept()

    def dropEvent(self, event):
        if event.source() == self:
            return

    def dragMoveEvent(self, event):
        event.accept()

    def dragLeaveEvent(self, event):
        event.accept()

    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self.widgets, 'dxTacticReview', msg,
                                          QtWidgets.QMessageBox.Ok)

class MovItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(MovItem, self).__init__(parent)
        self.tacticData = {}


class ReviewItem(MovItem):
    def __init__(self, parent=None, add=False,):
        super(ReviewItem, self).__init__(parent)
        self.hasSub=False

        if add:
            self.addCheck = QtWidgets.QCheckBox()
            self.addCheck.setStyleSheet("""
            QCheckBox:checked {
            color: rgb(0,255,0);
            }
            """)

            for i in range(self.treeWidget().columnCount()):
                self.setForeground(i, QtGui.QBrush(QtGui.QColor('#606060')))

            self.treeWidget().setItemWidget( self, 0, self.addCheck)
            self.addCheck.setChecked(False)
            self.addCheck.stateChanged.connect(self.checkBoxChanged)
        else:
            self.setText(0, 'R')

    def checkBoxChanged(self, state):
        self.setSelected(state)


class ShotLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(ShotLineEdit, self).__init__(parent)

        # SHOT COMPLETER
        self.shotCompleter = CustomQCompleter([])
        self.setCompleter(self.shotCompleter)

    def setShotList(self, shots):
        self.shotCompleter.setStringList(shots)

    def text_changed(self, text):
        if text:
            cShot = self.shotCompleter.getStringList()
            suggest_list = []
            for i in cShot:
                if text in i:
                    suggest_list.append(i)
            self.shotCompleter.updateModel(suggest_list)
        else:
            self.shotCompleter.resetToAllshot()


class CustomQCompleter(QtWidgets.QCompleter):
    def __init__(self, stringList, parent=None):
        super(CustomQCompleter, self).__init__(parent)
        # ALL POSSIBLE ITEM LIST
        self.stringList = stringList

        self.shotModel = QtCore.QStringListModel(self.stringList)
        self.setModel(self.shotModel)

        # STRINGLIST -> SHOTMODEL ( QSTRINGLISTMODEL )-> SETMODEL

    def resetToAllshot(self):
        self.shotModel.reset()
        self.shotModel.setStringList(self.stringList)

    def setStringList(self, stringList):
        self.stringList = stringList
        self.updateModel(stringList)

    def getStringList(self):
        return self.stringList

    def getModelList(self):
        return self.model().stringList()

    def updateModel(self, updateStringList):
        self.shotModel.reset()
        self.shotModel.setStringList(updateStringList)
