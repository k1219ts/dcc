# -*- coding: utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, os
import getpass, time
import Qt
from Qt import QtWidgets
from Qt import QtGui
from Qt import QtCore
from LayInvenMov_ui import Ui_Form
from LayMongoMov import LayInventorydb

if "Side" in Qt.__binding__:
    import maya.OpenMayaUI as mui
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken
    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    def main():
        mainVar = InvenMovAdd(getMayaWindow())
elif "PyQt" in Qt.__binding__:
    def main():
        app = QtWidgets.QApplication(sys.argv)
        mainVar = InvenMovAdd(None)
        sys.exit(app.exec_())
else:
    print "No Qt binding available"

if __name__ == "__main__":
    main()

class InvenMovAdd(QtWidgets.QWidget):
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.openpath = '/dexter/Cache_DATA/LAY/003_Reference/01_mov'
        self.ui.file_txt.setText(self.openpath)
        self.connection()
        self.show()

    def connection(self):
        self.ui.open_btn.clicked.connect(self.openDir)
        self.ui.send_btn.clicked.connect(self.sendDb)
        self.ui.minus_btn.clicked.connect(self.minusList)

    def openDir(self):
        # image file open window
        files = []
        files = QtWidgets.QFileDialog.getOpenFileNames(self, "Movie Files Open",
                                                       self.openpath, "Movie files (*.mov *.avi *.mp4 *.flv *.mkv *.gif *.m4v *.divx)")
        for file in files[0]:
            fullpath = QtCore.QFileInfo(file).filePath()
            fullpath = fullpath.split(self.openpath)[-1]
            try:
                item = self.ui.file_list.findItems(fullpath, QtCore.Qt.MatchContains)[0]
            except:
                item = ''
            if not item:
                self.ui.file_list.addItem(fullpath)
        self.totalList()

    def sendDb(self):
        # db send and spool send
        org = []
        total = self.totalList()
        for i in range(0, total):
            source =  self.openpath + unicode(self.ui.file_list.item(i).text())
            org.append(source)
        # print org
        if org:
            Laydb = LayInventorydb()
            Laydb.dbImport(org)
            # time.sleep(5)
            self.ui.file_list.clear()
            self.totalList()

    def totalList(self):
        total = self.ui.file_list.count()
        self.ui.user_txt.setText(str(total))
        return total

    def minusList(self):
        # list -> image list minus
        index = self.ui.file_list.selectedItems()
        for i in index:
            delitem = self.ui.file_list.row(i)
            self.ui.file_list.takeItem(delitem)
        self.totalList()
