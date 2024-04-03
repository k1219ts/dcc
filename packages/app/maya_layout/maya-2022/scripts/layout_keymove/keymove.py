# -*- coding : utf-8 -*-
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import os, sys, subprocess
import Qt
from Qt import QtWidgets, QtGui, QtCore
from keymoveui import Ui_Form
import maya.cmds as cmds

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#######  PyQt4 or PySide2 May biding code  #########
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
        mainVar = KeyMove(getMayaWindow())
        mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
        mainVar.show()

if __name__ == "__main__":
    main()
########################################################

class KeyMove(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.Setting()
        self.Connection()

    def Setting(self):
        # help icon > help document link
        icon = Qt.QtGui.QIcon()
        helpimg = "resource/help.png"
        icon.addPixmap(Qt.QtGui.QPixmap(os.path.join(CURRENT_DIR, helpimg)),
                                        Qt.QtGui.QIcon.Normal, Qt.QtGui.QIcon.Off)
        self.ui.helpbtn.setIcon(icon)
        self.ui.helpbtn.setObjectName("helpbtn")
        self.ui.helpbtn.setCursor(QtCore.Qt.WhatsThisCursor)
        self.ui.helpbtn.setToolTip("help")

    def Connection(self):
        self.ui.ok_btn.clicked.connect(self.keyInput)
        self.ui.selbtn.clicked.connect(self.keyFind)
        self.ui.helpbtn.clicked.connect(self.Help)

    def keyFind(self):## animcurve find keyframenode
        ani = cmds.ls(selection=True)
        self.anim = cmds.listAnimatable(ani)
        self.anim.sort()
        self.keyTime(self.anim)

    def keyTime(self, anim = None):## node time value query
        self.lists = []
        self.ui.list_txt.clear()
        for i in anim:
            keyani = cmds.keyframe(i, query=True, keyframeCount=True)
            if (keyani > 0):
                times = cmds.keyframe(i, query=True, index = (0, keyani), timeChange=True)
                if not times == 'None':
                    list = "%s : %s"%(i, times)
                    self.lists.append(list)
        print self.lists
        self.keyPrint()

    def keyPrint(self):## keyframe time list print
        chname = ''
        for i in self.lists:
            if not i.startswith(chname):
                self.ui.list_txt.addItem("#"*100+"\n")
            self.ui.list_txt.setStyleSheet("color: rgb(255, 255, 255)")
            self.ui.list_txt.addItem(i)
            chname = i.split('.')[0]

    def keyInput(self):## start, end range change vlaue : relative add or minus change
        self.startkey = self.ui.start_txt.text()
        self.endkey = self.ui.end_txt.text()
        self.chkey = self.ui.change_txt.text()
        if self.startkey.isdigit() and self.endkey.isdigit():
            shift = float(self.chkey) - float(self.endkey)
            if shift < 0:
                self.KeyScale()
                self.KeyShift(shift)
            else:
                self.KeyShift(shift)
                self.KeyScale()
            self.keyTime(self.anim)

    def KeyShift(self, shift=None):## keyframe not range shift
        # print "ok", self.startkey, self.endkey, self.chkey
        movestart = float(self.endkey) + 1.0
        moveend = cmds.playbackOptions(q=True, maxTime = True)
        print type(movestart), moveend
        for i in self.anim:
            cmds.keyframe(i, edit=True, time=(movestart, moveend),
                          relative=True, timeChange=shift)

    def KeyScale(self):## keyframe range scale move
        for i in self.anim:
            cmds.scaleKey(i, time=(self.startkey, self.endkey),
                          newStartTime=self.startkey, newEndTime=self.chkey)

    def Help(self):
        pdf = "/usr/bin/evince" #pdf viewer
        #odp = "/usr/bin/libreoffice"  # odp viewer
        helpdoc = "resource/keymove_help.pdf"
        fileName = os.path.join(CURRENT_DIR, helpdoc)
        subprocess.Popen([pdf, fileName])
