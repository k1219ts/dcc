# -*- coding: utf-8 -*-
import sys
import os
import shutil

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

from ui_download_dialog import Ui_Dialog

class DownloadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, items=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.ui.treeWidget.setColumnCount(4)
        self.ui.treeWidget.headerItem().setText(0, 'file name')
        self.ui.treeWidget.headerItem().setText(1, 'size')
        self.ui.treeWidget.headerItem().setText(2, 'progress')
        self.ui.treeWidget.headerItem().setText(3, 'status')


        homePath = QtCore.QDir.homePath()
        today = QtCore.QDate.currentDate().toString('yyyyMMdd')

        self.ui.locationLineEdit.setText(os.path.join(unicode(homePath), unicode(today)))

        self.connectSetting()

        self.items = items

        for item in self.items:
            dItem = DownloadItem(self.ui.treeWidget)
            dItem.setItemData(item)
            dItem.setText(0, os.path.basename(item['files']['mov']))
            dItem.setText(1, sizeof_fmt(os.stat(item['files']['mov']).st_size))
            dItem.setText(3, 'Waiting')

    def startDownload(self):
        for i in range(self.ui.treeWidget.topLevelItemCount()):
            item = self.ui.treeWidget.topLevelItem(i)
            src = item.getItemData()['files']['mov']
            dstDir = unicode(self.ui.locationLineEdit.text())
            if not(os.path.exists(dstDir)):
                os.makedirs(dstDir)
            dst = os.path.join(dstDir, os.path.basename(src))
            if not(os.path.exists(src)):
                continue
            # copyFileObj(open(fromFile, 'rb'), open(toFile, 'wb'), self.report)
            self.do_copy(open(src, 'rb'),
                         open(dst, 'wb'),
                         item,
                         self.report)

            item.setText(3, 'Done')


    def connectSetting(self):
        self.ui.closeButton.clicked.connect(self.done)
        self.ui.selectDirButton.clicked.connect(self.selectDir)
        self.ui.downloadButton.clicked.connect(self.startDownload)

    def selectDir(self):
        basePath = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                              "Select search base directory",
                                                              QtCore.QDir.homePath())
        if basePath:
            self.ui.locationLineEdit.setText(basePath)

    def do_copy(self,fsrc,fdst, item, callback, length=16*1024):
        copied = 0
        while True:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            callback(item, copied)

    def report(self, item, copySize):
        currentPercent = (float(copySize) / float(item.fileSize)) * 100
        item.progress.setValue(currentPercent)
        QtWidgets.QApplication.processEvents()


class DownloadItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(DownloadItem, self).__init__(parent)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.treeWidget().setItemWidget(self, 2, self.progress)
        self.itemData = {}
        self.fileSize = 0

    def setItemData(self, data):
        self.itemData = data
        self.fileSize = os.stat(data['files']['mov']).st_size

    def getItemData(self):
        return self.itemData



def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return '%.1f%s%s' % (num, 'P', suffix)


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
    doc = list(coll.find({"type":"VFX_REF"}).limit(10))

    ce = DownloadDialog(None, doc) #
    ce.show()
    sys.exit(app.exec_())