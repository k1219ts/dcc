#encoding:utf-8
import maya.cmds as cmds
import maya.mel as mel
import os
import string
import xml.etree.ElementTree as xml
from cStringIO import StringIO

# Qt Module
from PySide2 import QtGui, QtCore
import shiboken
import pysideuic


def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    from_class = parsed.find('class').text
    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' % from_class]
        base_class = eval('QtGui.%s' % widget_class)
    return form_class, base_class

uiFile = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/ui/animExport.ui'

form, base = loadUiType(uiFile)

class Window(base, form):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        '''A custom window with a demo set of ui widgets'''
        self.setupUi(self)
        self.connectSignal()   #선언
        self.num = 1
        animPlug = '/usr/autodesk/maya2017/bin/plug-ins/animImportExport.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            cmds.loadPlugin(animPlug)
            cmds.pluginInfo(animPlug, edit=True, autoload=True)

    def connectSignal(self):
        self.pushButton.clicked.connect(self.tarRoot)  # 버튼 등록 및 실행
        self.pushButton_2.clicked.connect(self.expAm)
        self.pushButton_3.clicked.connect(self.nextFile)
        self.timeSetBtn.clicked.connect(self.timeSet)
        self.nextBtn.clicked.connect(self.nextanFile)
        
    def tarRoot(self):

        self.tarJnt = cmds.ls(sl=True)
        self.label_2.setText(self.tarJnt[0])

    def nextFile(self):
        cmds.file(f=True, new=True)
        self.animDir = cmds.fileDialog2(fileMode=1, caption="Import FBX File")
        cmds.file(self.animDir[0], i=True)
        self.animDir = str(self.animDir[0])

        if self.animDir.index("omaya/"):
            self.indNumb = self.animDir.index("omaya/")
            self.offNum = self.indNumb + 6
            self.getname = self.animDir[self.offNum:]
            self.dirPath = self.animDir[:self.offNum]
            self.animPath = self.dirPath + "../05_Anim/"
            if not os.path.exists(self.animPath):
                os.mkdir(self.animPath)
            self.amName = self.getname.replace(".fbx", ".anim")
            if self.amName[0:5] == "maya_":
                self.amName = self.amName.replace("maya_", "")
            self.animFile = self.animPath + self.amName
            self.outneb = string.zfill(self.num, 2)
            self.outTex = str(self.outneb) + " : /05_Anim/" + self.amName + "\n"
        else:
            cmds.warning("Check your Tomaya folder name.")

    def expAm(self):
        if len(self.tarJnt) == 0:
            cmds.warning("Select Root Joint")
        else:
            cmds.select(self.tarJnt, hierarchy=True)
            self.tarHjnt = cmds.ls(sl=True)
            cmds.select(self.tarJnt, d=True)
            self.tarRjnt = cmds.ls(sl=True)
            cmds.select(self.tarHjnt)
            for a in range(len(self.tarHjnt)):
                self.tarHjnt[a] = str(self.tarHjnt[a])
            for b in range(len(self.tarRjnt)):
                self.tarRjnt[b] = str(self.tarRjnt[b])
            self.tm = cmds.keyframe(self.tarJnt, q=True, lastSelected=True, timeChange=True)
            cmds.playbackOptions(minTime=0, maxTime=self.tm[0])
            cmds.currentTime(0)
            self.etAt = [".sx", ".sy", ".sz", ".v", ".radi", ".liw"]
            self.trAt = [".tx", ".ty", ".tz"]
            self.rtAt = [".rx", ".ry", ".rz"]
            self.alAt = self.trAt + self.rtAt + self.etAt
            mel.eval("channelBoxCommand -break; ")
            for i in range(len(self.tarHjnt)):
                for j in range(len(self.etAt)):
                    self.temA = self.tarHjnt[i] + self.etAt[j]
                    mel.eval("CBdeleteConnection %s;" % self.temA)
                    cmds.setAttr(self.temA, lock=True)
            for k in range(len(self.tarRjnt)):
                for l in range(len(self.trAt)):
                    self.temB = self.tarRjnt[k] + self.trAt[l]
                    mel.eval("CBdeleteConnection %s;" % self.temB)
                    cmds.setAttr(self.temB, lock=True)
            for m in range(len(self.tarHjnt)):
                for n in range(len(self.rtAt)):
                    self.temC = self.tarHjnt[m] + self.rtAt[n]
                    cmds.setAttr(self.temC, 0)
            cmds.BakeSimulation(self.tarHjnt, hi="below", sb=1.0)
            cmds.currentTime(0)
            for o in range(len(self.tarHjnt)):
                for p in range(len(self.alAt)):
                    self.temD = self.tarHjnt[o] + self.alAt[p]
                    mel.eval('CBunlockAttr %s;' % self.temD)
            cmds.selectKey(keyframe=True)
            cmds.filterCurve(f="euler")
            cmds.delete(self.tarHjnt, sc=True, uac=False, hi="below", cp=False, s=False)
            self.ofs = self.lineEdit.text()
            self.ofsin = int(self.ofs)
            self.offS = str(int(self.ofs) - 1)
            if self.ofsin > 1:
                cmds.cutKey(self.tarHjnt, o="keys", t=(1,self.offS), cl=True)
                self.offsp = int(self.offS) + 1
                self.offsm = "-" + self.offS
                cmds.keyTangent(self.tarHjnt, time=(0,self.offsp), itt="auto", ott="auto")
                cmds.keyframe(self.tarHjnt, e=True, iub=True, r=True, o="over", t=("%d:" % self.offsp, ), tc=self.offsm)
                cmds.selectKey(self.tarHjnt, clear=True)
                cmds.selectKey(self.tarHjnt, keyframe=True)
                cmds.file(self.animFile, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)
                self.enT = int(self.tm[0]) - int(self.offS)
                cmds.playbackOptions(minTime=0, aet=self.enT)
                if os.path.isfile(self.animFile):
                    self.plainTextEdit.insertPlainText(self.outTex)
                self.num += 1
            elif self.ofsin == 1:
                cmds.selectKey(self.tarHjnt, clear=True)
                cmds.selectKey(self.tarHjnt, keyframe=True)
                cmds.file(self.animFile, force=True, options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1", typ="animExport", eas=True)
                if os.path.isfile(self.animFile):
                    self.plainTextEdit.insertPlainText(self.outTex)
                self.num += 1
            else:
                cmds.warning("Input start frame with positive integer number")

    def timeSet(self):

        self.tlm = cmds.keyframe(self.tarJnt, q=True, lastSelected=True, timeChange=True)
        cmds.playbackOptions(minTime=0, maxTime=self.tlm[0])
        cmds.currentTime(0)
        cmds.select(self.tarJnt)
        if not cmds.toggleAxis(q=True, o=True) == 1:
            cmds.toggleAxis(o=True)

    def nextanFile(self):

        cmds.select(self.tarJnt, hierarchy=True)
        cmds.selectKey(keyframe=True)
        cmds.currentTime(0)
        cmds.delete(all=True, c=True)
        cmds.select(cl=True)
        cmds.select(self.tarJnt)
        self.animDir = cmds.fileDialog2(fileMode=1, caption="Import Anim File")
        cmds.file(self.animDir[0], i=True)
        self.showAn = str(self.animDir[0].split(os.sep)[-1]).split(".")[0]
        self.lineEditB.setText(self.showAn)
'''
        if self.animDir.index("Anim/"):
            self.aniNum = self.animDir.index("Anim/")
            self.offNum = self.aniNum + 5
            self.aniN = self.animDir[self.offNum:]
            self.showAn = self.aniN.replace(".anim", "")
            self.animPlainText.setPlainText(self.showAn)
        else:
            self.animPlainText.setPlainText("Your Anim File is not in 05_Anim folder")
'''
def main():
    global myWindow
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()
