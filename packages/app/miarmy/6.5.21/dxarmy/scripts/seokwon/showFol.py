#!/usr/bin/python
# encoding:utf-8

import os, sys
from PyQt4 import QtCore, QtGui

class keyPress(QtGui.QListWidget):
    def __init__(self, parent):
        QtGui.QListWidget.__init__(self, parent)

    def getR(self, dir):
        rangeNum = self.count()
        cur = self.currentRow()
        if cur == min(range(0, rangeNum)) and dir == "Up":
            return -1
        elif cur == max(range(0, rangeNum)) and dir == "Down":
            return -1
        else:
            return cur

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self._ent_item()
        elif event.key() == QtCore.Qt.Key_Down:
            if self.getR("Down") != -1:
                self.setCurrentRow(self.getR("Down") + 1)
        elif event.key() == QtCore.Qt.Key_Up:
            if self.getR("Up") != -1:
                self.setCurrentRow(self.getR("Up") - 1)

    def _ent_item(self):
        selItem = self.selectedItems()
        if not selItem:
            return
        for j in selItem:
            os.system("nautilus -w %s" % j.text())
        myWindow.close()

class getInputs(QtGui.QWidget):
    def __init__(self,parent=None):
        super(getInputs, self).__init__(parent)
        self.resize(250,50)
        self.move(840, 500)
        self.setWindowTitle("Search Shot")
        self.le = QtGui.QLineEdit(self)
        self.le.setGeometry(QtCore.QRect(0,0,250,50))
        self.le.returnPressed.connect(self.doit)

    def doit(self):
        getInput = str(self.le.text())
        res = doit(getInput)
        self.newMain(res)

    def newMain(self, getList):
        height = len(getList) * 31
        if height < 720:
            pass
        else:
            height = 720
        self.resize(450, height)
        self.le.close()
        self.listd = keyPress(self)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.listd.setFont(font)
        self.listd.setGeometry(QtCore.QRect(1, 1, 448, height - 2))
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.listd.addItems(getList)
        self.listd.show()
        self.listd.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.listd.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.listd.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.listd.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.listd.setCurrentRow(0)
        self.listd.setFocus()

def doit(getWords):
    getList = list()
    if len(getWords.split(" ")) == 3:
        prj = getWords.split(" ")[0]
        sqs = getWords.split(" ")[1]
        if prj in os.listdir("/show/"):
            if "shot" in os.listdir("/show/" + prj + "/"):
                if sqs in os.listdir("/show/" + prj + "/shot/"):
                    if (sqs + "_" + getWords.split(" ")[2]) in os.listdir("/show/" + prj + "/shot/" + sqs):
                        opnPath = "/show/" + prj + "/shot/" + sqs + "/" + sqs + "_" + getWords.split(" ")[2]
                        os.system("nautilus -w %s" % opnPath)
                    else:
                        for n in os.listdir("/show/" + prj + "/shot/" + sqs):
                            if str(n).count(getWords.split(" ")[2]) == 1:
                                getList.append("/show/" + prj + "/shot/" + sqs + "/" + str(n) + "/")
    elif len(getWords.split(" ")) == 2:
        if getWords.split(" ")[0] in os.listdir("/show/"):
            prj = getWords.split(" ")[0]
            if "shot" in os.listdir("/show/" + prj + "/"):
                if getWords.split(" ")[1] in os.listdir("/show/" + prj + "/shot/"):  # "god2 POS"
                    getList = [("/show/" + prj + "/shot/" + getWords.split(" ")[1] + "/" + str(i) + "/") for i in os.listdir("/show/" + prj + "/shot/" + getWords.split(" ")[1] + "/")]
                else:  # "god2 0030"
                    for j in os.listdir("/show/" + prj + "/shot/"):
                        try:
                            for k in os.listdir("/show/" + prj + "/shot/" + str(j) + "/"):
                                if str(k).count(getWords.split(" ")[1]) == 1:
                                    getList.append("/show/" + prj + "/shot/" + str(j) + "/" + str(k) + "/")
                        except:
                            pass
        else:  # "POS 0030"
            for q in os.listdir("/show/"):
                if "shot" in os.listdir("/show/" + str(q) + "/"):
                    for w in [dr for dr in os.listdir("/show/" + str(q) + "/shot/") if os.path.isdir("/show/" + str(q) + "/shot/" + dr)]:
                        if str(w).count(getWords.split(" ")[0]) == 1:
                            for c in os.listdir("/show/" + str(q) + "/shot/" + str(w) + "/"):
                                if str(c).count(getWords.split(" ")[1]) == 1:
                                    getList.append("/show/" + str(q) + "/shot/" + str(w) + "/" + str(c) + "/")
    elif len(getWords.split(" ")) == 1:
        if getWords.isdigit() == True:   # 0030
            for b in os.listdir("/show/"):
                if "shot" in os.listdir("/show/" + str(b) + "/"):
                    for v in os.listdir("/show/" + str(b) + "/shot/"):
                        try:
                            for x in os.listdir("/show/" + str(b) + "/shot/" + str(v) + "/"):
                                if str(x).count(getWords) == 1:
                                    getList.append("/show/" + str(b) + "/shot/" + str(v) + "/" + str(x) + "/")
                        except:
                            pass
        elif getWords in os.listdir("/show/"):  # god2
            if "shot" in os.listdir("/show/" + getWords + "/"):
                opnPath = "/show/" + getWords + "/shot/"
            else:
                opnPath = "/show/" + getWords +"/"
            os.system("nautilus -w %s" % opnPath)
        else:   # POS
            for h in os.listdir("/show/"):
                if "shot" in os.listdir("/show/" + str(h) + "/"):
                    for l in os.listdir("/show/" + str(h) + "/shot/"):
                        if str(l).count(getWords) == 1:
                            getList.append("/show/" + str(h) + "/shot/" + str(l) + "/")
    return getList

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    app = QtGui.QApplication(sys.argv)
    myWindow = getInputs()
    myWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
