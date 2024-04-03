# coding:utf-8
# ----------------------------------------------------------------------
# @ author : daeseok.chae in Dexter Studio CGSupervisor
# @ date   : 2020.10.22
#
# ----------------------------------------------------------------------
from PyQt4 import QtGui
import os
import re

currentDir = os.path.dirname(__file__)

class FileItem(QtGui.QTreeWidgetItem):
    def __init__(self, parent, parentPath, fileInfo):
        QtGui.QTreeWidgetItem.__init__(self, parent)
        filenameIndex = -1
        filename = fileInfo[filenameIndex]
        self.isDirectory = False
        self.setIcon(0, QtGui.QIcon('%s/file.png' % currentDir))
        if fileInfo[0][0] == 'd': # directory
            self.setIcon(0, QtGui.QIcon('%s/directory.png' % currentDir))
            self.setTextColor(0, QtGui.QColor("#035aeb"))
            self.isDirectory = True
        elif fileInfo[0][0] == 'l': # symbol link
            self.setTextColor(0, QtGui.QColor("#00ffff"))
            filenameIndex -= 2
            filename = fileInfo[filenameIndex]
            self.isDirectory = True
        r = re.compile("\033\[[0-9;]+m")
        filename = r.sub("", filename)
        self.pwd = os.path.join(parentPath, filename)
        if fileInfo[0][0] == 'l':
            self.pwd = r.sub("", fileInfo[-1])
        self.setText(0, filename)
        if ":" in fileInfo[filenameIndex - 1]:
            # month day time
            datetime = "%s/%s %s" % (fileInfo[filenameIndex - 3], fileInfo[filenameIndex - 2], fileInfo[filenameIndex - 1])
        else:
            # month day year
            datetime = "%s/%s/%s" % (fileInfo[filenameIndex - 1], fileInfo[filenameIndex - 3], fileInfo[filenameIndex - 2])
        self.setText(3, datetime)
        self.setText(1, fileInfo[-5])

    def getPwd(self):
        return self.pwd