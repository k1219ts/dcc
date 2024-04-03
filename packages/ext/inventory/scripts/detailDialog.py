# -*- coding: utf-8 -*-
#from PyQt4 import QtCore, QtGui
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import subprocess, sys

import db_query

from tagModifyDialog import TagModifyDialog

class DetailDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, item=None):
        super(DetailDialog, self).__init__(parent)
        self.resize(1200,900)
        self.setWindowTitle('Detail Dialog')
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.detailTree = DetailTree(self, item)
        self.gridLayout.addWidget(self.detailTree)
        self.setStyleSheet("""
        DetailDialog{color: rgb(200,200,200); background: rgb(48,48,48); border-width: 1px;}
        """)
        self.detailTree.expandAll()
        self.detailTree.resizeColumnToContents(0)



class DetailTree(QtWidgets.QTreeWidget):
    def __init__(self, parent=None, item = None):
        super(DetailTree, self).__init__(parent)
        #self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        self.setStyleSheet("""
        QTreeView {color: #CCCCCC; font: 14px; width: 100px; background-color: #323232;
        border: 1px solid #3d3d3d;}
        QTreeView::item:selected { background-color: #FF8D1D; color: #000000;}
        QTreeView::item:hover { border: 1px solid #6BB6FF;}
        """)
        self.header().setStyleSheet("""
        QHeaderView:section { color: #FFA91D; border: 1px solid #505050;
        background-color: #323232; height: 28px; font: bold; }
        """)

        self.setColumnCount(2)
        self.headerItem().setText(0, 'Key')
        self.headerItem().setText(1, 'Value')

        self.itemDoubleClicked.connect(self.detailDoubleClicked)
        self.item = item
        try:
            self.setDetailData(item.getItemData())
        except:
            self.setDetailData(item) # item = {.......}

    def detailDoubleClicked(self, item, col):
        if unicode(item.text(0)).startswith('/'):
            if sys.platform == 'darwin':
                subprocess.call(["open", "-R", unicode(item.text(0))])
            else:
                subprocess.Popen(['nautilus', unicode(item.text(0))])

        elif unicode(item.text(1)).startswith('/'):
            if sys.platform == 'darwin':
                subprocess.call(["open", "-R", unicode(item.text(1))])
            else:
                subprocess.Popen(['nautilus', unicode(item.text(1))])

        elif item.parent() is not None:
            if item.parent().text(0) == "tags":
                self.tagModifyDialog(item.parent())
                #self.tagModifyDialog(item)

    def tagModifyDialog(self, item):
        tagList = []
        # for i in item.text(1).split('\n'):
        #     tagList.append(i)
        for index in range(item.childCount()):
            tagList.append(item.child(index).text(1))

        dialog = TagModifyDialog(self, tagList)
        if dialog.exec_() == 1: # OK
            tagList = dialog.tagEdit.toPlainText().split('\n')
            infos = db_query.updateTags(self.item.getItemData()['_id'], tagList)
            self.item.setItemData(infos)

            self.clearDetailData()
            self.setDetailData(infos)
            self.expandAll()
            self.resizeColumnToContents(0)

    def clearDetailData(self):
        while self.topLevelItemCount() > 0:
            self.takeTopLevelItem(0)
            treeitem = self.topLevelItem(0)
            if treeitem:
                if treeitem.childCount() > 0:
                    treeitem.takeChildren()

    def setDetailData(self, data, root=None):
        for i in sorted(data.keys()):
            if root:
                item = QtWidgets.QTreeWidgetItem(root)
            else:
                item = QtWidgets.QTreeWidgetItem(self)

            item.setText(0, i)
            value = data[i]
            if type(value) == list:
                #valueText = ''
                for subv in value:
                    #valueText += subv + '\n'
                    subitem = QtWidgets.QTreeWidgetItem(item)
                    subitem.setText(1, subv)
                # subitem = QtWidgets.QTreeWidgetItem(item)
                # subitem.setText(1, valueText[:-1])
            elif type(value) == unicode:
                item.setText(1, value)

            elif type(value) == dict:
                self.setDetailData(value, item)

            else:
                item.setText(1, unicode(value))

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            return
        else:
            orgPath = '\n'.join([unicode(i.text(1)) for i in self.selectedItems()])
            urls = [QtCore.QUrl(u'file://'+unicode(i.text(1))) for i in self.selectedItems()]

            event.mimeData().setText(orgPath)
            event.mimeData().setUrls(urls)
        event.accept()


    def dragMoveEvent(self, event):
        event.accept()


    def dropEvent(self, event):
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    import pymongo
    from pymongo import MongoClient
    import dxConfig

    DB_IP = dxConfig.getConf('DB_IP')
    DB_NAME = 'inventory'
    COLL = 'assets'

    client = MongoClient(DB_IP)
    db = client[DB_NAME]
    coll = db[COLL]
    doc = coll.find_one()

    ce = DetailDialog(None, doc) #
    ce.show()
    sys.exit(app.exec_())