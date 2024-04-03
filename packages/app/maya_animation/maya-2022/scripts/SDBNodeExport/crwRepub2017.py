# encoding:utf-8
# !/usr/bin/env python

import os
import maya.cmds as cmds
import json
from PySide2 import QtCore, QtGui, QtWidgets, load_ui
import dxUI

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "crwRepub.ui")

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        self.connectSignal()

    def connectSignal(self):
        self.moBtn.clicked.connect(self.getValues)

    def pathSet(self):
        fPat = str(os.path.dirname(cmds.file(q=1, sn=1)))
        setPath = str(os.sep.join(fPat.split(os.sep)[0:-1]))
        if not os.path.exists(setPath + "/data"):
            os.mkdir(setPath + "/data/")
        else:
            pass

    def getValues(self):
        sel = str(cmds.ls(sl=True)[0])
        self.pathSet()
        if sel.count("matchmove_SDBNode") != 1:
            cmds.warning("Select matchmove SDBNode please.")
        else:
            att = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ", "scaleX", "scaleY", "scaleZ"]
            exp = list()
            for i in att:
                exp.append(str(cmds.getAttr(sel + "." + i)))
            fileName = str(os.path.basename(cmds.file(q=1, sn=1))).split(".")[0].replace("ani", "crw") + ".json"
            fileDir = str(os.sep.join(cmds.file(q=True, sn=True).split(os.sep)[:-2]) + "/data/")
            exPath = fileDir + fileName
            self.jsonMake(exp, exPath)
            self.jsonPath.setText(exPath)
        if os.path.isfile(exPath) == True:
            QtWidgets.QMessageBox.information(self, "Completion", "Json file checked.")
        else:
            cmds.warning("Not Exist Json File.")

    def jsonMake(self, data, filepath):
        if data:
            f = open(filepath, 'w')
            json.dump(data, f, indent=4, sort_keys=True)
            f.close()

def main():
    global myWindow
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()