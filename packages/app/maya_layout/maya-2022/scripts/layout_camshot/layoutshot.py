# -*- coding: utf-8 -*-
import os
import re
from maya import cmds, mel
from PySide2 import QtWidgets, QtGui, QtCore
import layshotui
reload(layshotui)
CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

def main():
    mainVar = Shotset(getMayaWindow())

if __name__ == "__main__":
    main()

class Shotset(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = layshotui.Ui_shotform()
        self.ui.setupUi(self)
        self.ui.shotmain.sortByColumn(2, QtCore.Qt.AscendingOrder)
        #self.ui.shotmain.resizeColumnToContents(0)
        self.ui.shotmain.setColumnWidth(0 ,250)
        self.ui.shotmain.setColumnWidth(1 ,200)
        self.ui.shotmain.setColumnWidth(2 ,250)
        self.ui.shotmain.setColumnWidth(3 ,150)
        self.ui.shotmain.setColumnWidth(4 ,150)
        self.ui.shotmain.setColumnWidth(5 ,80)
        self.ui.shotmain.setColumnWidth(6 ,80)
        self.mayascene()
        self.shotList()
        self.titleset()
        self.connection()
        self.show()

    def mayascene(self):
        mayascene = cmds.file(q=True, sn=True)
        cammb = '/'.join(mayascene.split('/')[0:4])
        self.previzcam = '%s/camera/layoutcam.mb' % cammb
        if not os.path.exists(self.previzcam):
            self.previzcam = os.path.join(CURRENTPATH, 'layoutcam.mb')

    def connection(self):
        self.ui.endbtn.clicked.connect(self.close)  ### close window
        self.ui.resetbtn.clicked.connect(self.resetList)       ### reset item
        self.ui.shotmain.itemChanged.connect(self.changeShot)  #### item change
        self.ui.shotmain.itemClicked.connect(self.changeShot)  #### item change
        self.ui.addshotbtn.clicked.connect(self.addShot)       #### item add
        self.ui.minusshotbtn.clicked.connect(self.minusShot)   #### item minus

    def shotList(self):
    ## shot list get
        shotob = cmds.ls(type = 'shot')
        self.shots = {}

        for i in shotob:
        ## shot attribute get
            camera = cmds.shot(i, q = True, cc = True)
            start = int(cmds.getAttr(i + '.startFrame'))
            end = int(cmds.getAttr(i + '.endFrame'))
            shotname = cmds.getAttr(i + '.shotName')
            track = cmds.getAttr(i + '.track')
            scale = cmds.getAttr(i + '.scale')
            scalefl = "%0.3f"%scale

            shots = {'shotob' : i, 'camera' : camera, 'shotname' : shotname, 'start' : start,
                      'end' : end, 'track' : track, 'scale' : scalefl}
            self.shotPrint(shots)

    def shotPrint(self, shots):
    ## shot list print
        #item = QtWidgets.QTreeWidgetItem(self.ui.shotmain)
        item = Treeitem(self.ui.shotmain, shots)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)

        for i in range(0, 7):
            item.setTextAlignment(i, QtCore.Qt.AlignCenter)
        self.ui.shotmain.setCurrentItem(item)
        self.ui.shotmain.sortByColumn(2, QtCore.Qt.AscendingOrder)

    def addShot(self):
    ## shot item add
        index = self.ui.shotmain.selectedItems()
        for i in index:
            addflag = 0
            seletshot = self.ui.shotmain.indexOfTopLevelItem(i)
            ## selected item row
            if not ((self.shotnum-1) == seletshot):
                imsi = self.ui.shotmain.topLevelItem(seletshot+1)
                pre = i.text(2).strip('C')
                after = imsi.text(2).strip('C')
                if int(after) - 10 > int(pre):
                    addflag = 1
                else:
                    addflag = 0
            else:
                pre = i.text(2).strip('C')
                addflag = 1

            if addflag == 1:
                center = pre
                innum = int(center) / 10
                inname = str(10 * (innum + 1)).zfill(4)
                cam = unicode('C' + inname + '_00')
                camshot = 'C' + inname
                stime = int(i.text(4)) + 1
                etime = stime + 99
                scale = "%0.3f" % 1

                cameralist = cameraList()
                if not camshot in cameralist:
                ## camera not exits create camera
                    mel.eval('file -import -type "mayaBinary"  -ignoreVersion -rpr "PrevisCam" '
                             '-options "v=0;" "%s";' % self.previzcam)
                    cameralist = cameraList()
                    for i in cameralist:
                        if i.endswith('camera'):
                            cmds.rename(i, camshot)
                            cmds.parent(camshot, 'cam')

                cmds.shot(cam, cc=camshot, sn=camshot, st=stime, et=etime, sst=stime, set=etime, track=1)
                shots = {'shotob': cam, 'camera': camshot, 'shotname': camshot, 'start': stime,
                         'end': etime, 'track': 1, 'scale': scale}
                print shots
                self.shotPrint(shots)
                self.titleset()
        # self.ui.shotmain.sortByColumn(2, QtCore.Qt.AscendingOrder)

    def minusShot(self):
    ## shot item delete
        index = self.ui.shotmain.selectedItems()
        if self.shotnum != 0:
            for i in index:
                indextext = i.shotname.text()
                ## selected item column text
                seletshot = self.ui.shotmain.indexOfTopLevelItem(i)
                ## selected item row
                self.ui.shotmain.takeTopLevelItem(seletshot)
                ## selected item delete
                cmds.delete(indextext)
                self.titleset()
            self.ui.shotmain.setCurrentItem(self.ui.shotmain.topLevelItem(seletshot - 1))
            ## current item cursor move

    def changeShot(self, item, column):
    ## item change ---> shot attribute change
        self.shotob = item.shotname.text()
        item.cameracombo.activated[str].connect(self.comboCh)
        shots = {'shotname': item.text(2), 'start': item.text(3), 'end': item.text(4),
                 'track': item.text(5), 'scale' : item.text(6)}

        if item.text(3).isdigit() and item.text(4).isdigit() and item.text(5).isdigit():
            if re.match("C[0-9][0-9][0-9][0-9]", shots['shotname'])is None or len(shots['shotname']) != 5:
             ## shotname rematch
                print "shotname not match"
            else:
                ccamera = item.cameracombo.currentText()
                shotname = item.text(2)
                cmds.shot(self.shotob, e = True, cc = ccamera, sn = shotname,
                          st = shots['start'], et = shots['end'],  sst = shots['start'],
                          set = shots['end'], trk = int(shots['track']))
                self.shotRename(shotname, ccamera)

    def comboCh(self, camera):
    ## item combo change
        cmds.shot(self.shotob, e=True, cc=unicode(camera))

    def titleset(self):
    ## shot counter
        self.shotnum = self.ui.shotmain.topLevelItemCount()
        self.ui.shotnum.setText(str(self.shotnum))

    def resetList(self):
    ## reset item
        self.ui.shotmain.clear()
        self.shotList()
        self.ui.shotmain.sortByColumn(2, QtCore.Qt.AscendingOrder)

    def shotRename(self, shotname = None, cameraname = None):
    ## shotName changed shotObjectName change
        shots = self.shotob.split('_')
        if not shots[0] == shotname:
            newshotob = '%s_%s'%(shotname, shots[-1])
            cmds.rename(self.shotob, newshotob)
            cmds.rename(cameraname, shotname)
            self.shotob = newshotob
            cmds.setAttr(self.shotob + '.shotName', shotname, type = 'string')
            self.resetList()

class Treeitem(QtWidgets.QTreeWidgetItem):
## Camera Shot SequencerTree item setting
    def __init__(self, parent, shotinfo):
        super(Treeitem,self).__init__(parent)
        font = QtGui.QFont()
        font.setPointSize(11)

        self.cameracombo = QtWidgets.QComboBox()
        self.cameracombo.setFont(font)
        self.cameracombo.addItems(cameraList())
        self.cameracombo.setCurrentText(shotinfo['camera'])

        self.shotname = QtWidgets.QLabel()
        self.shotname.setAlignment(QtCore.Qt.AlignCenter)
        self.shotname.setFont(font)
        self.shotname.setText(shotinfo['shotob'])

        self.scale = QtWidgets.QLabel()
        self.scale.setAlignment(QtCore.Qt.AlignCenter)
        self.scale.setFont(font)
        self.scale.setText(shotinfo['scale'])

        self.treeWidget().setItemWidget(self, 0, self.shotname)
        self.treeWidget().setItemWidget(self, 1, self.cameracombo)
        self.treeWidget().setItemWidget(self, 6, self.scale)

        shotname = shotinfo['shotname']
        if shotname.find('_') > 0:
            shotname = shotname.split('_')[0]
            cmds.setAttr(shotinfo['shotob'] + '.shotName', shotname, type='string')

        self.setText(2, shotname)
        self.setText(3, str(shotinfo['start']))
        self.setText(4, str(shotinfo['end']))
        self.setText(5, str(shotinfo['track']))

def cameraList():
## current scene camera list
    startcam = [u'front', u'persp', u'side', u'top']
    noncam = cmds.listCameras(p=True)
    cameralist = list(set(noncam) - set(startcam))
    cameralist.sort()
    return cameralist
