# -*- coding: utf-8 -*-
import sys, os, datetime, json

from PySide2 import QtWidgets, QtCore

import nuke
from ui_CAT import Ui_Form


class CATs(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CATs, self).__init__()
        #QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle('Comp Asset Tool')
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.widgetFont = self.font()
        self.widgetFont.setPointSize(12)
        self.setFont(self.widgetFont)

        self.isFileDic = {True: QtWidgets.QFileIconProvider().icon(QtWidgets.QFileIconProvider.File),
                          False: QtWidgets.QFileIconProvider().icon(QtWidgets.QFileIconProvider.Folder)
                          }

        self.ui.browserTree.setColumnCount(2)
        self.ui.browserTree.headerItem().setText(0, 'Nuke Asset Node')
        self.ui.browserTree.headerItem().setText(1, 'Last Update')

        self.ui.browserTree.header().resizeSection(0, 300)
        self.ui.browserTree.header().resizeSection(1, 80)


        self.ui.prjLabel.setText("Project")
        self.ui.prjLabel.setAlignment(QtCore.Qt.AlignRight)
        self.ui.seqLabel.setText("Sequence")
        self.ui.seqLabel.setAlignment(QtCore.Qt.AlignRight)

        prjSeqFile = '/stdrepo/CMP/TD_hslth/Global_File/prj_seq.json'

        prjSeq = json.load(open(prjSeqFile, 'r'))
        self.prjDic = prjSeq['prj_code']

        #self.ui.shotsearchButton.clicked.connect(self.nkNodeListup)
        self.ui.pasteButton.clicked.connect(self.nkPaste)
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.prjComboBox.currentIndexChanged.connect(self.setSeq)
        self.ui.seqComboBox.currentIndexChanged.connect(self.nkNodeListup)

        for i in sorted(self.prjDic.keys(), reverse=False):
            if self.prjDic[i] in prjSeq['valid_prj']:
                self.ui.prjComboBox.addItem(i, self.prjDic[i])

    def setSeq(self, index):
        self.ui.seqComboBox.clear()
        currentPrj = self.ui.prjComboBox.currentText()
        prjPath = '/show/%s/asset/comp/' % (self.prjDic[currentPrj])

        if os.path.isdir(prjPath):
            seqList = os.listdir(prjPath)
            for i in seqList:
                self.ui.seqComboBox.addItem(i)
        else:
            pass

    def nkNodeListup(self):
        self.ui.browserTree.clear()
        currentPrj = self.ui.prjComboBox.currentText()
        currentSeq = self.ui.seqComboBox.currentText()
        fullPath = '/show/%s/asset/comp/%s' %(self.prjDic[currentPrj], currentSeq)
        try:
            fullList = os.listdir(fullPath)
            for i in fullList:
                item = QtWidgets.QTreeWidgetItem(self.ui.browserTree)
                absPath = os.path.join(fullPath, i)
                item.setText(0, i)
                item.setIcon(0, self.isFileDic[os.path.isfile(absPath)])
                mtime = datetime.datetime.fromtimestamp(os.stat(absPath)[8])
                item.setText(1, str(mtime))

        except:
            item = QtWidgets.QTreeWidgetItem(self.ui.browserTree)
            noSuch = "No Such Node list"
            item.setText(0, noSuch)


    def nkPaste(self):
        prjName = self.prjDic[self.ui.prjComboBox.currentText()]
        seqName = self.ui.seqComboBox.currentText()
        items = self.ui.browserTree.currentItem()
        nkName = items.text(0)
        nuke.nodePaste('/show/%s/asset/comp/%s/%s' %(prjName, seqName, nkName))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ce = CATs()
    ce.show()
    sys.exit(app.exec_())
