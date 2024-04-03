# -*- coding: utf-8 -*-
####################################################
#          coding by RND youkyoung.kim             #
####################################################
import os, sys, re
from maya import cmds, mel
import Qt
from Qt import QtWidgets
from Qt import QtCore
from shotCreate_ui import ShotCreate_Form

CURRENTPATH = os.path.dirname(os.path.abspath(__file__))

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
        window =  ShotCreate(getMayaWindow())
        window.move(QtWidgets.QDesktopWidget().availableGeometry().center() - window.frameGeometry().center())
        window.show()

if __name__ == "__main__":
    main()

class ShotCreate(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = ShotCreate_Form()
        self.ui.setupUi(self)
        self.windowSetting()
        self.shotCheck()
        self.connection()

    def windowSetting(self):
        self.ui.shot_spin.setValue(0)
        self.ui.shot_tree.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.ui.shot_tree.setColumnWidth(0 ,50)
        self.ui.shot_tree.setColumnWidth(1 ,300)
        self.ui.shot_tree.setColumnWidth(2 ,200)
        self.ui.shot_tree.setColumnWidth(3 ,100)
        self.ui.shot_tree.setColumnWidth(4 ,100)
        self.ui.shot_tree.setColumnWidth(5 ,100)
        self.ui.shot_tree.setColumnWidth(6 ,100)
        self.ui.shot_tree.setColumnWidth(7 ,100)
        self.ui.shot_tree.setColumnWidth(8 ,50)
        self.ui.fbx_btn.setStyleSheet(
            "QPushButton#fbx_btn {background-color: black; border: 1px solid #999;}"
            "QPushButton#fbx_btn:hover {background-color: #aaa; border: 1px solid #999; color: black;}"
        )
        self.ui.shot_btn.setStyleSheet(
            "QPushButton#shot_btn {background-color: #336699; border: 1px solid #999;}"
            "QPushButton#shot_btn:hover {background-color: darkred; border: 1px solid #999;}"
        )

    def connection(self):
        self.ui.shot_btn.clicked.connect(self.openSequencerWindow)
        self.ui.fbx_btn.clicked.connect(self.openFbxWindow)
        self.ui.create_btn.clicked.connect(self.createSpin)
        self.ui.add_btn.clicked.connect(self.createAdd)
        self.ui.minus_btn.clicked.connect(self.deleteShot)
        self.ui.ok_btn.clicked.connect(self.shotCheck)
        self.ui.cancel_btn.clicked.connect(self.close)
        self.ui.shot_tree.itemSelectionChanged.connect(self.selectShot)
        self.ui.shot_tree.itemDoubleClicked.connect(self.editShot)

    def openSequencerWindow(self):
        # seqencer window open
        mel.eval("SequenceEditor")

    def openFbxWindow(self):
        # unreal fbx converter
        sys.path.append("C:\Users\dexter\PycharmProjects")
        from layout_unreal import unrealFBX
        unrealFBX.main()

    def shotCheck(self):
        self.ui.shot_tree.clear()
        shots = sorted(cmds.ls(type='shot'))
        if shots:
            self.ui.create_btn.setEnabled(False)
            for i, j in enumerate(shots):
                self.shotTreeSet(i, j)
        else:
            self.ui.create_btn.setEnabled(True)

    def shotTreeSet(self, num=None, shot=None):
        shotattr = self.shotGetAttr(shot)
        item = Treeitem(self.ui.shot_tree)
        item.setText(0, str(num))
        item.setText(1, str(shot))
        for i in range(2, 9):
            item.setText(i, str(shotattr[i-2]))

    def createSpin(self):
        total = int(self.ui.shot_spin.value())
        for i in range(total):
            self.createShots(i+1)

    def createShots(self, num=None):
        if num:
            shotimsi = str(10 * (num)).zfill(4)
            shotname = unicode('C' + shotimsi)
            start = (1000 * num) + 1.0
            end = (1000 * num) + 1000.0
            camera = self.cameraCheck(shotname)
            shotname = '%s_00' %shotname
            cmds.shot(shotname, sn=shotname, cc=camera, st=start, et=end, sst=start, set= end, track=1)
            self.shotCheck()

    def cameraCheck(self, cameraname=None):
        cameralist = cameraList()
        if cameraname in cameralist:
            camera = cameraname
        else:
            camera = 'persp'
        return camera

    def createAdd(self):
        nextframe = 1000
        shot = sorted(cmds.ls(type='shot'))[-1]
        start = int(cmds.getAttr(shot + '.startFrame')) + nextframe
        end = int(cmds.getAttr(shot + '.endFrame')) + nextframe
        sstart = int(cmds.getAttr(shot + '.sequenceStartFrame')) + nextframe
        send = int(cmds.getAttr(shot + '.sequenceEndFrame')) + nextframe

        imsiname = int(shot.lstrip('C').split('_')[0])+10
        shotname = unicode('C' + str(imsiname).zfill(4))
        camera = self.cameraCheck(shotname)
        shotname = '%s_00' %shotname
        cmds.shot(shotname, sn=shotname, cc=camera, st=start, et=end, sst=sstart, set=send, track=1)
        self.shotCheck()

    def deleteShot(self):
        index = self.ui.shot_tree.selectedItems()
        for i in index:
            indextext = i.text(1)
            seletshot = self.ui.shot_tree.indexOfTopLevelItem(i)
            self.ui.shot_tree.takeTopLevelItem(seletshot)
            cmds.delete(indextext)

    def selectShot(self):
        index = self.ui.shot_tree.selectedItems()
        for i in index:
            indextext = i.text(1)
            cmds.select(indextext)

    def shotGetAttr(self, shot=None):
        shotattr = []
        currentcam = str(cmds.shot(shot, q = True, cc = True))
        timestart = int(cmds.shot(shot, q = True, st = True))
        timeend = int(cmds.shot(shot, q = True, et = True))
        sequencestart = int(cmds.shot(shot, q = True, sst = True))
        sequenceend = int(cmds.shot(shot, q=True, set=True))
        scale = cmds.shot(shot, q=True, s=True)
        track = int(cmds.shot(shot, q=True, track=True))
        scale = '%0.3f' %scale

        shotattr.append(currentcam)
        shotattr.append(timestart)
        shotattr.append(timeend)
        shotattr.append(sequencestart)
        shotattr.append(sequenceend)
        shotattr.append(scale)
        shotattr.append(track)
        return shotattr

    def editShot(self, item=None, col=None):
        shotname = item.text(1)
        camera = item.text(2)
        scale = item.text(7)
        shotwindow = ShotEdit(QtWidgets.QDialog)
        shotwindow.shotName(shotname, camera, scale)
        shotwindow.show()
        result = shotwindow.exec_()
        if not result:
            self.shotCheck()

#------------------------------------------------------------
class Treeitem(QtWidgets.QTreeWidgetItem):
    # treewidget item
    def __init__(self, parent):
        super(Treeitem, self).__init__(parent)
        font = Qt.QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(75)

        for i in range(9):
            self.setFont(i, font)
            self.setTextAlignment(i, QtCore.Qt.AlignLeft)

#------------------------------------------------------------
def cameraList():
    # current scene camera list
    startcam = [u'front', u'persp', u'side', u'top']
    noncam = cmds.listCameras(p=True)
    cameralist = list(set(noncam) - set(startcam))
    cameralist.sort()
    return cameralist

#------------------------------------------------------------
class ShotEdit(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self)
        self.setWindowTitle("Shot Edit")
        self.resize(450, 200)
        self.gridLayout = QtWidgets.QGridLayout()
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(self.spacerItem)
        self.cancel_btn = QtWidgets.QPushButton()
        self.cancel_btn.setMinimumSize(QtCore.QSize(70, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.cancel_btn.setFont(font)
        self.horizontalLayout_4.addWidget(self.cancel_btn)
        self.ok_btn = QtWidgets.QPushButton()
        self.ok_btn.setMinimumSize(QtCore.QSize(70, 30))
        self.ok_btn.setFont(font)
        self.horizontalLayout_4.addWidget(self.ok_btn)
        self.gridLayout.addLayout(self.horizontalLayout_4, 2, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.shot_label = QtWidgets.QLabel()
        self.shot_label.setMinimumSize(QtCore.QSize(100, 30))
        self.shot_label.setMaximumSize(QtCore.QSize(100, 16777215))
        self.shot_label.setFont(font)
        self.horizontalLayout.addWidget(self.shot_label)
        self.shot_txt = QtWidgets.QLabel()
        self.shot_txt.setMinimumSize(QtCore.QSize(140, 30))
        self.shot_txt.setMaximumSize(QtCore.QSize(140, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.shot_txt.setFont(font)
        self.horizontalLayout.addWidget(self.shot_txt)
        self.chshot_txt = QtWidgets.QLineEdit()
        self.chshot_txt.setMinimumSize(QtCore.QSize(120, 30))
        self.horizontalLayout.addWidget(self.chshot_txt)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.scale_label = QtWidgets.QLabel()
        self.scale_label.setMinimumSize(QtCore.QSize(100, 30))
        self.scale_label.setMaximumSize(QtCore.QSize(100, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.scale_label.setFont(font)
        self.horizontalLayout_2.addWidget(self.scale_label)
        self.scale_txt = QtWidgets.QLabel()
        self.scale_txt.setMinimumSize(QtCore.QSize(140, 30))
        self.scale_txt.setMaximumSize(QtCore.QSize(140, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.scale_txt.setFont(font)
        self.horizontalLayout_2.addWidget(self.scale_txt)
        self.chscale_txt = QtWidgets.QLineEdit()
        self.chscale_txt.setMinimumSize(QtCore.QSize(120, 30))
        self.horizontalLayout_2.addWidget(self.chscale_txt)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setMinimumSize(QtCore.QSize(100, 30))
        self.camera_label.setMaximumSize(QtCore.QSize(100, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(50)
        self.camera_label.setFont(font)
        self.horizontalLayout_3.addWidget(self.camera_label)
        self.camera_txt = QtWidgets.QLabel()
        self.camera_txt.setMinimumSize(QtCore.QSize(140, 30))
        self.camera_txt.setMaximumSize(QtCore.QSize(140, 16777215))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.camera_txt.setFont(font)
        self.horizontalLayout_3.addWidget(self.camera_txt)
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.setMinimumSize(QtCore.QSize(120, 30))
        self.camera_combo.setMaximumSize(QtCore.QSize(16777215, 30))
        self.horizontalLayout_3.addWidget(self.camera_combo)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)
        self.setLayout(self.gridLayout)

        self.cancel_btn.setText("Cancel")
        self.ok_btn.setText("Ok")
        self.shot_label.setText("Shot Name :")
        self.scale_label.setText("Scale :")
        self.camera_label.setText("Camera :")
        self.connection()

    def connection(self):
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.changeShot)

    def shotName(self, shotname=None, cameraname=None, scale=None):
        self.shotname = shotname
        self.cameraname = cameraname
        self.scale = scale
        self.shot_txt.setText(shotname)
        self.scale_txt.setText(scale)
        self.camera_txt.setText(cameraname)
        self.chshot_txt.setText(shotname)
        self.chscale_txt.setText(scale)
        cameralist = cameraList()
        self.camera_combo.addItems(cameralist)
        self.camera_combo.setCurrentText(cameraname)

    def changeShot(self):
        chshot = self.chshot_txt.text()
        chscale = self.chscale_txt.text()
        chcamera = self.camera_combo.currentText()
        if not self.scale == chscale:
            cmds.setAttr(self.shotname + '.scale', float(chscale))

        if not self.cameraname == chcamera:
            cmds.shot(self.shotname, e=True, cc=unicode(chcamera))

        if not chshot == self.shotname:
            cmds.rename(self.shotname, chshot)
            ckshot = cmds.getAttr(chshot + '.shotName')
            print ckshot
            if not chshot == ckshot:
                cmds.setAttr(chshot + '.shotName', chshot, type='string')
        self.close()
