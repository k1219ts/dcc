# encoding:utf-8

import os
import maya.cmds as cmds
import subprocess
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "offsetKeyframeRestore.ui")

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()

    def connectSignal(self):
        self.picBtn.clicked.connect(self.pick)
        self.assBtn.clicked.connect(self.assign)
        self.manBtn.clicked.connect(self.openMan)

    def openMan(self):
        exp = "/usr/bin/evince"
        fileName = "/netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/offsetRestoring/ofsRst.pdf"
        subprocess.Popen([exp, fileName])
        #os.system("evince /netapp/backstage/pub/apps/maya2/versions/2017/team/animation/linux/scripts/offsetRestoring/ofsRst.pdf")

    def getMinV(self, con):
        getList = dict()
        for i in cmds.ls("*:" + con):
            minVal = int(cmds.playbackOptions(q=True, max=True))
            for j in cmds.listAttr(str(i), k=True):
                if cmds.keyframe(str(i) + "." + str(j), q=True):
                    getMinVal = min(cmds.keyframe(str(i) + "." + str(j), q=True))
                    if getMinVal < minVal:
                        minVal = getMinVal
            getList[str(i) + "." + str(j)] = minVal
        return getList

    def chkWd(self, nsChar):
        if self.mChk.isChecked():
            pass
        else:
            cmds.selectKey(nsChar.split(":")[0] + ":move_CON", rm=True)
        if self.dChk.isChecked():
            pass
        else:
            cmds.selectKey(nsChar.split(":")[0] + ":direction_CON", rm=True)
        if self.pChk.isChecked():
            pass
        else:
            cmds.selectKey(nsChar.split(":")[0] + ":place_CON", rm=True)

    def pick(self):
        sel = str(cmds.ls(sl=True)[0])
        self.stCon = sel.split(":")[-1]
        self.charList = self.getMinV(self.stCon)

    def assign(self):
        sel = cmds.ls(sl=True)
        newCharList = self.getMinV(self.stCon)
        if sel:
            new = [str(i).split(":")[0] for i in sel]
            for c in range(len(newCharList)):
                key = self.charList.keys()[c]
                if key in newCharList.keys():
                    if key.split(":")[0] in new:
                        subValue = self.charList[key] - newCharList[key]
                        cmds.selectKey(key.split(":")[0] + ":*_CON")
                        self.chkWd(key)
                        cmds.keyframe(e=True, iub=True, r=True, o="over", tc=subValue)
                    else:
                        pass
                else:
                    cmds.warning("Start keyframe of " + self.charList.keys()[c].split(":")[0] + "is changed.")
        else:
            for c in range(len(newCharList)):
                key = self.charList.keys()[c]
                if key in newCharList.keys():
                    subValue = self.charList[key] - newCharList[key]
                    cmds.selectKey(key.split(":")[0] + ":*_CON")
                    self.chkWd(key)
                    cmds.keyframe(e=True, iub=True, r=True, o="over", tc=subValue)
                else:
                    cmds.warning("Start keyframe of " + self.charList.keys()[c].split(":")[0] + "is changed.")

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