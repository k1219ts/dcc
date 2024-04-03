# encoding:utf-8
# !/usr/bin/env python

import os
from PySide2 import QtCore, QtGui, QtWidgets, QtCompat

chList = list()
chName = list()
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "ui", "acvConv.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))

class DragDropTest(QtWidgets.QListWidget):
    def __init__(self, parent):
        QtWidgets.QListWidget.__init__(self, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
            lst = list()
            for i in event.mimeData().urls():
                lst.append(i.toLocalFile().split(os.sep)[-1])
                chList.append(str(i.toLocalFile()))
                chName.append(str(i.toLocalFile().split(os.sep)[-1]))
            self.addItems(lst)
        else:
            event.ignore()

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()

    def connectSignal(self):
        self.actList = DragDropTest(self.wd)
        self.actList.setGeometry(QtCore.QRect(1, 1, 228, 266))
        self.convBtn.clicked.connect(self.getList)

    def getList(self):
        id = self.actList.count()
        for i in range(int(id)):
            self.convStart(str(chList[i]))

    def convStart(self, fPath):
        dirPath = os.sep.join(fPath.split(os.sep)[:-1]) + "/"
        reb = fPath.split(os.sep)[-1].split(".")
        newFileName = reb[0] + "." + reb[1]
        keyList = {"2016R2": "2017ff04", "MiarmyProForMaya20165": "MiarmyProForMaya2017", "Maya 2016": "Maya 2017"}
        with open(fPath, "r") as f:
            filedata = f.read()
            if "2016 Extension 2 SP1" in filedata:
                filedata = filedata.replace("2016 Extension 2 SP1", "2017")
            else:
                if "2016 Extension 2" in filedata:
                    filedata = filedata.replace("2016 Extension 2", "2017")
                else:
                    pass
            for i in keyList.keys():
                if i in filedata:
                    filedata = filedata.replace(i, keyList[i])
                else:
                    pass
        if not os.path.exists(dirPath + "Converted_v2017/"):
            os.mkdir(dirPath + "Converted_v2017/")
        else:
            pass
        with open(dirPath + "Converted_v2017/" + newFileName, "w") as s:
            s.write(filedata)
        del filedata
        self.actList.clear()

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()