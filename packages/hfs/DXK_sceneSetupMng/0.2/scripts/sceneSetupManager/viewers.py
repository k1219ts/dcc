# -*- coding: utf-8 -*-
import os

# QT
import pymodule.Qt as Qt
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore

# CHN download
import dxConfig
SITE = dxConfig.getHouse()
if SITE == 'CHN':
    import re
    import pexpect
    from StringIO import StringIO

import items

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

class DataBaseViewer(QtWidgets.QWidget):
    def __init__(self, parent=None, dict={} ):
        QtWidgets.QWidget.__init__(self)
        self.dataviewer = QtWidgets.QTreeWidget()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.dataviewer)
        styles = """
                QTreeWidget { background: rgb(80, 80, 80); color: white; }
                QTreeWidget::item:selected{background: rgb(40, 170, 162);}
                QTreeWidget::item { padding: 5 10 5 10 px; margin: 0px; border: 0 px}
                QLineEdit { background: rgb(80, 80, 80); color: white; border-color: white; }
                """
        font = Qt.QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(10)
        self.dataviewer.setFont(font)
        self.setStyleSheet(styles)
        self.setLayout(layout)
        self.resize(1200, 500)
        self.setWindowTitle("MongoDB Viewer")
        self.dataviewer.headerItem().setText(0, 'key')
        self.dataviewer.headerItem().setText(1, 'data')
        self.dataviewer.header().resizeSection(0, 100)
        self.dataviewer.setRootIsDecorated(False)

        if not dict['data_type'] in ['geoCache','zenn']:
            for i in dict['files'].keys():
                item = QtWidgets.QTreeWidgetItem(self.dataviewer)
                b = QtGui.QBrush(QtGui.QColor(100, 200, 200, 255))
                item.setForeground(0, b)
                item.setText( 0, i )
                if len(dict['files'][i]) < 2:
                    item.setText( 1, str(dict['files'][i]) )
                else:
                    if type(dict['files'][i]) in [ unicode, str ]:
                        item.setText(1, str(dict['files'][i]))
                    else:
                        for j in range(len(dict['files'][i])):
                            childItem = QtWidgets.QTreeWidgetItem(item)
                            childItem.setText(1, str(dict['files'][i][j]) )
                            item.addChild(childItem)

                item.sortChildren(1, QtCore.Qt.AscendingOrder)
                item.setExpanded(True)

            infoItem = QtWidgets.QTreeWidgetItem(self.dataviewer)
            b = QtGui.QBrush(QtGui.QColor(100, 200, 200, 255))
            infoItem.setForeground(0, b)
            infoItem.setText(0, 'Info')
            for key in dict.keys():
                if key != 'files':
                    childItem = QtWidgets.QTreeWidgetItem(infoItem)
                    childItem.setText(0, key)
                    childItem.setText(1, str(dict[key]))
                    infoItem.addChild(childItem)

            infoItem.sortChildren(1, QtCore.Qt.AscendingOrder)
            infoItem.setExpanded(True)

        if dict['data_type'] in ['geoCache','zenn']:
            self.dataviewer.headerItem().setText(0, 'asset')
            self.dataviewer.headerItem().setText(1, 'key')
            self.dataviewer.headerItem().setText(2, 'data')
            self.dataviewer.header().resizeSection(0, 200)
            self.dataviewer.header().resizeSection(1, 100)
            for asset in dict['files']['assets'].keys():
                item = QtWidgets.QTreeWidgetItem(self.dataviewer)
                b = QtGui.QBrush(	QtGui.QColor ( 100, 200, 200, 255 ))
                item.setForeground(0, b)
                item.setText( 0, asset )
                for key in dict['files']['assets'][asset].keys():
                    childItem = QtWidgets.QTreeWidgetItem(item)
                    childItem.setText( 1, str(key) )
                    childItem.setText( 2, str(dict['files']['assets'][asset][key]) )
                    item.addChild(childItem)


                item.sortChildren(1, QtCore.Qt.AscendingOrder)
                item.setExpanded(True)

            infoItem = QtWidgets.QTreeWidgetItem(self.dataviewer)
            b = QtGui.QBrush(QtGui.QColor(100, 200, 200, 255))
            infoItem.setForeground(0, b)
            infoItem.setText(0, 'Info')
            for key in dict.keys():
                if key != 'files':
                    childItem = QtWidgets.QTreeWidgetItem(infoItem)
                    childItem.setText(1, key)
                    childItem.setText(2, str(dict[key]) )
                    infoItem.addChild(childItem)

            infoItem.sortChildren(1, QtCore.Qt.AscendingOrder)
            infoItem.setExpanded(True)


class ImportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        # ui
        label = QtWidgets.QLabel("Do you want to import new cache \n or to update an existing cache?\n")
        self.save_btn = QtWidgets.QPushButton("Import")
        self.open_btn = QtWidgets.QPushButton("Update")
        self.close_btn = QtWidgets.QPushButton("Close")
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.save_btn)
        layout2.addWidget(self.open_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addLayout(layout2, 3, 0)
        self.setLayout(layout)
        self.setWindowTitle("Scene Setup Manager")
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)

        # connection
        self.close_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.save)
        self.open_btn.clicked.connect(self.open)

        self.show()
        self.exec_()

    def save(self):
        self.result = 'import'
        self.close()

    def open(self):
        self.result = 'update'
        self.close()



class PikaClass(QtCore.QThread):
    messageReceived = QtCore.Signal(str)
    threadEnd = QtCore.Signal()

    def __init__(self, parent=None, fileList=[]):
        QtCore.QThread.__init__(self, parent)
        print "init!!"
        self.fileList = fileList
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(False)
        self.timer.start(1000)
        self.child = None
        self.expectIO = StringIO()
        self.setTerminationEnabled(True)
        self.timer.timeout.connect(self.shout)

    def shout(self):
        if self.child:
            print 'log file : ', self.expectIO.getvalue()
            # self.emit(QtCore.SIGNAL("messageReceived"), (self.expectIO.getvalue()))
            # self.emit(self.messageReceived, (self.expectIO.getvalue()))
            self.messageReceived.emit(self.expectIO.getvalue())
            self.expectIO.seek(0)
            self.expectIO.truncate()

    def run(self):
        print 'start to copy!!'
        server = '11.0.0.12'
        for i in self.fileList:
            print '###down:', i
            targetFile = i
            if '*' in targetFile:
                targetFile = targetFile.replace('/*', '/')
            cpcmd = 'python %s/scp_test.py %s %s' % (CURRENTPATH, i, targetFile)
            # cpcmd = 'python /dexter/Cache_DATA/RND/jeongmin/sceneSetupManager/scp_test.py %s %s' % (i, i)
            cmd = "ssh root@%s '%s %s %s'" % (server, cpcmd, i, targetFile)

            ssh_newkey = 'Are you sure you want to continue connecting'
            self.child = pexpect.spawn(cmd, timeout=None)
            r = self.child.expect([ssh_newkey, "root@%s's password:" % server])

            print 'r is ', r

            if r == 0:
                self.child.sendline('yes')
                rc = self.child.expect("root@%s's password:" % server)
                self.child.sendline('D$DXchina!B123')
                # self.child.logfile = sys.stdout
                self.child.logfile = self.expectIO
                # sys.stdout = self.expectIO
                rc = self.child.expect(pexpect.EOF)

            elif r == 1:
                self.child.sendline('D$DXchina!B123')
                # self.child.logfile = sys.stdout
                self.child.logfile = self.expectIO
                # sys.stdout = self.expectIO
                rc = self.child.expect(pexpect.EOF)

            print '***', self.child.before
            if 'No such file or directory' in str(self.child.before):
                print 'NO FILE IN KOREA'
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle("Warning!")
                msg.setText("%s file not exists in korea" % targetFile)
                msg.exec_()
                self.child.close(force=True)
            self.child.close(force=True)
        self.threadEnd.emit()

    def __del__(self):
        print "delete!!!!"
        self.wait()
        self.quit()

class ExistDialog(QtWidgets.QWidget):
    def __init__(self, parent=None, fileDict=[]):
        QtWidgets.QWidget.__init__(self)
        # ui
        styles = """
                QTreeWidget { background: rgb(80, 80, 80); color: white; }
                QTreeWidget::item { font:11px; padding: 5 0 5 10 px; margin: 0px; border: 0 px}
                QTreeWidget::item:selected{background: rgb(150, 124, 185);}
                """
        self.setStyleSheet(styles)
        brush = QtGui.QBrush()
        brush.setColor(QtGui.QColor(253, 100, 107, 255))
        label = QtWidgets.QLabel('File Not Exist')
        label.setStyleSheet('''QLabel{color:rgb(255,100,100)}''')
        self.download_treeWidget = QtWidgets.QTreeWidget()
        self.download_treeWidget.setRootIsDecorated(False)
        self.download_treeWidget.headerItem().setText(0, 'import')
        self.download_treeWidget.headerItem().setText(1, 'type')
        self.download_treeWidget.headerItem().setText(2, 'name')
        self.close_btn = QtWidgets.QPushButton("Close")
        self.download_btn = QtWidgets.QPushButton("Download")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressView = QtWidgets.QTextEdit()
        self.progressView.setEnabled(False)
        self.checkAll_pushButton = QtWidgets.QPushButton("All")
        self.checkAll_pushButton.setStyleSheet('''QLabel{color:rgb(255,100,100)}''')
        layout2 = QtWidgets.QHBoxLayout()
        layout2.addWidget(self.download_btn)
        layout2.addWidget(self.close_btn)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label)
        layout.addWidget(self.checkAll_pushButton)
        layout.addWidget(self.download_treeWidget)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.progressView)
        layout.addLayout(layout2, 5, 0)
        self.setLayout(layout)
        self.setWindowTitle("Massage")
        font = Qt.QtGui.QFont()
        font.setPointSize(10)
        label.setFont(font)
        self.download_treeWidget.header().resizeSection(0, 30)
        self.resize(800,400)
        for i in sorted(fileDict):
            for k in fileDict[i]:
                downloadItem = items.Download_CheckableItem( self.download_treeWidget, i, k )

        self.download_btn.clicked.connect(self.downLoad)
        self.close_btn.clicked.connect(self.killProcess)
        self.checkAll_pushButton.clicked.connect(self.checkAll)
        self.show()

    def checkAll(self):
        for i in range(self.download_treeWidget.topLevelItemCount()):
            item = self.download_treeWidget.topLevelItem(i)
            item.check.setChecked(0)

    def downLoad( self ):
        fileList = []
        for i in range(self.download_treeWidget.topLevelItemCount()):
            item = self.download_treeWidget.topLevelItem(i)
            if item.check.isChecked():
                print item.text(2)
                self.filepath = item.text(2)
                fileList.append(self.filepath)
                if not os.path.exists(os.path.dirname(self.filepath)):
                    os.makedirs(os.path.dirname(self.filepath))

        self.process = PikaClass(fileList=fileList)
        self.process.messageReceived.connect(self.progress)
        self.process.threadEnd.connect(self.killProcess)
        self.process.start()
        # self.connect(self.process, QtCore.SIGNAL("messageReceived"), self.progress)
        # self.connect(self.process, QtCore.Signal("threadEnd"), self.killProcess)
        # self.process.messageReceived.connect(self.progress)
        self.process.messageReceived.connect(self.progress)

    def progress(self, log):
        print log
        print '***progress'
        fileName = re.search('\r([\w\W]*) *', str(log))
        kb = re.search(' ([0-9]*)[K|M][B]', str(log))
        percent = re.search(' ([0-9]*)%', str(log))
        kbpersec = re.search('([0-9]*.[0-9])[K|M]B/s', str(log))
        estime = re.search('([0-9]{2}:[0-9]{2}) ETA', str(log))
        self.progressView.setText('download file: ' + fileName.group())
        if kb:
            self.progressView.append('downloaded size: ' + kb.group())
        if percent:
            print 'percent: ', percent.group(1)
            self.progressBar.setValue(int(percent.group(1)))
        if kbpersec:
            self.progressView.append('download speed: ' + kbpersec.group())
        if estime:
            self.progressView.append('estimate time: ' + estime.group())
        if 'No such file or directory' in str(log):
            print '*** NO FILE IN KOREA'
        print "log from main widget", log
        print repr(log)

    def killProcess(self):
        try:
            if self.process != None:
                print 'a'
                self.process.terminate()
                print 'b'
                self.process.wait()
                print 'c'
                self.process.exit(0)
                print 'd'
                self.process.quit()
                print 'e'
                # self.process.__del__()
                print 'f'
                # del self.process
        except Exception as e:
            print "Except ?", e.message

        self.close()
