# Original code(MEL) by Rob Bredow (at) 185vfx.com

# development by kwantae Kim(kkt0525@gmail.com) at DEXTER DIGITAL
# Last Updated 2016.11.02


# Update list
# development 2015.12.02
# Multi camera support, cameraShape error fix 2016.05.17
# locator List 2016.06.08
# load setting 2016.06.13
# minor bug fix 2016.07.21
# maya 2016.5 fix 2016.11.02
# maya 2017 fix 2017.08.01


import os

# from PyQt4 import QtCore
# from PyQt4 import QtGui
# from PyQt4 import uic
import ui_DD_tacker_widget

reload(ui_DD_tacker_widget)
from PySide2 import QtCore, QtGui, QtUiTools, QtWidgets

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as mui
from shiboken2 import wrapInstance

# UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "DD_tacker.ui")
window_obj = "DD_tacker v0.3.0"


def mayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    if ptr is not None:
        return wrapInstance(long(ptr), QtWidgets.QWidget)


def run():
    # global dialog
    # try:
    #    dialog.close()
    # except:
    dialog = Qtacker()
    dialog.show()


class Qtacker(QtWidgets.QDialog):
    def __init__(self, parent=mayaWindow()):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = ui_DD_tacker_widget.ui_DD_tacker_Dialog()
        self.ui.setupUi(self)
        self.setObjectName(window_obj)

        self.init_GUI()
        self.connect_slot()

    def init_GUI(self):
        self.proc_load()
        self.reload_cam()
        self.load_setting()

    def connect_slot(self):
        QtCore.QObject.connect(self.ui.btnReload, QtCore.SIGNAL("clicked()"), self.reload_cam)
        QtCore.QObject.connect(self.ui.btnEnable, QtCore.SIGNAL("clicked()"), self.enable_center3D)
        QtCore.QObject.connect(self.ui.btnDisable, QtCore.SIGNAL("clicked()"), self.clear_SelectedPanZoom)

    def reload_cam(self):
        self.ui.listCam.clear()
        self.ui.listCam.addItem("Select Camera")

        self.ui.all_cam_list = cmds.ls(cameras=True)
        self.ui.cam_list = []

        for c in self.ui.all_cam_list:
            self.ui.cam_list.append(cmds.listRelatives(c, type="transform", parent=True)[0])

        for c in cmds.listCameras(p=False, o=True):
            try:
                self.ui.cam_list.remove(c)
            except:
                print "Removing Othographic Cameras Failed."

        self.ui.listCam.addItems(self.ui.cam_list)

    def load_setting(self):
        cam_list = cmds.ls(cameras=True)
        read = []

        # print cam_list

        shape = cmds.ls(lf=1)
        for cName in cam_list:
            for chkNode in shape:
                nType = cmds.nodeType(chkNode)
                findShape = chkNode.count('panv_' + cName)
                if (nType == 'expression' and findShape > 0):
                    # print nType, chkNode, cnt
                    read = chkNode.split('_' + cName + '_')
                    # print cName, read[1]
                    self.ui.listLoc.addItem("[ " + cName + " ], " + read[1])

    def proc_load(self):
        proc = 'proc matrix screenSpaceGetMatrix(string $attr){\n'
        proc += '  float $v[]=`getAttr $attr`;\n'
        proc += '  matrix $mat[4][4]=<<$v[0], $v[1], $v[2], $v[3];\n'
        proc += '             $v[4], $v[5], $v[6], $v[7];\n'
        proc += '             $v[8], $v[9], $v[10], $v[11];\n'
        proc += '             $v[12], $v[13], $v[14], $v[15]>>;\n'
        proc += ' return $mat;\n'
        proc += '}\n\n'
        proc += 'proc vector screenSpaceVecMult(vector $v, matrix $m){\n'
        proc += '  matrix $v1[1][4]=<<$v.x, $v.y, $v.z, 1>>;\n'
        proc += '  matrix $v2[1][4]=$v1*$m;\n'
        proc += '  return <<$v2[0][0], $v2[0][1],  $v2[0][2]>>;\n'
        proc += '}\n\n'
        # proc += 'float $ptPosWs[];'
        # proc += 'vector $ptVecWs;'
        # proc += 'matrix $cam_mat[4][4];'
        # proc += 'vector $ptVecCs;'
        # proc += 'float $hfv;'
        # proc += 'float $ptx;'
        # proc += 'float $panh;'
        # proc += 'float $vfv;'
        # proc += 'float $pty;'
        # proc += 'float $panv;'
        # proc += '}\n\n'

        # execute proc
        print proc
        mel.eval(proc)

    def clear_allPanZoom(self):
        cnt = 0
        shape = cmds.ls(lf=1)

        for chkNode in shape:
            nType = cmds.nodeType(chkNode)
            if (chkNode[:3] == "pan"):
                cmds.delete(shape[cnt])
            cnt += 1

        self.ui.enbLoc.setText("select Camera and one Locator")

    def clear_SelectedPanZoom(self):

        chkList = self.ui.listLoc.selectedItems()
        if not chkList:
            self.ui.enbLoc.setText('please select Disable Locator!')
            return

        cnt = 0
        readItm = []
        selectedItm = []

        for read in self.ui.listLoc.selectedItems():
            readItm.append(str(read.text()))

        for c in readItm:
            tempItm = c.split(',')
            print tempItm
            selectedItm.append(tempItm[0])

        print selectedItm

        selectedItm[0] = selectedItm[0].replace(" ", "")
        selectedItm[0] = selectedItm[0].replace("[", "")
        selectedItm[0] = selectedItm[0].replace("]", "")

        print selectedItm[0]

        shape = cmds.ls(lf=1)
        for chkNode in shape:
            nType = cmds.nodeType(chkNode)
            findShape = chkNode.count(selectedItm[0])
            if (nType == 'expression' and findShape > 0):
                cmds.delete(shape[cnt])
                print nType, chkNode, cnt
            cnt += 1

        for itm in self.ui.listLoc.selectedItems():
            self.ui.listLoc.takeItem(self.ui.listLoc.row(itm))

        self.ui.enbLoc.setText(selectedItm[0] + ', Tacker Disabled')

    def enable_center3D(self):
        melH = ""
        melV = ""
        sCam = cmds.listRelatives(str(self.ui.listCam.currentText()), allDescendents=True, type="camera")[0]
        sLoc = ""
        chfh = ""
        chfv = ""
        cnt = 0

        node = cmds.ls(sl=1)
        shape = cmds.ls(sl=1, dag=1, lf=1)

        print sCam, sLoc

        for chkNode in shape:
            nType = cmds.nodeType(chkNode)
            if (nType == "locator"):
                sLoc = node[cnt]
            cnt += 1

        if sLoc == "":
            self.ui.enbLoc.setText('Locator not selected')
            return 0

        chfh = str(cmds.getAttr(sCam + ".horizontalFilmAperture"))
        chfv = str(cmds.getAttr(sCam + ".verticalFilmAperture"))

        # get the world space position of the point into a vector
        melOut = 'float $ptPosWs[] = `xform -q -ws -t ' + sLoc + '`;\n'
        melOut += 'vector $ptVecWs = <<$ptPosWs[0],$ptPosWs[1],$ptPosWs[2]>>;\n\n'

        # Grab the worldInverseMatrix from cam
        melOut += 'matrix $cam_mat[4][4] = screenSpaceGetMatrix("' + sCam + '.worldInverseMatrix");\n\n'

        # Multiply the point by that matrix
        melOut += 'vector $ptVecCs = screenSpaceVecMult($ptVecWs,$cam_mat);\n\n'

        # Adjust the point's position for the camera perspective
        melOut += 'float $hfv = `camera -q -hfv ' + sCam + '`;\n'
        melOut += 'float $ptx = (($ptVecCs.x/(-$ptVecCs.z))/tand($hfv/2))/2.0+.5;\n'
        melOut += 'float $panh = ' + chfh + '*($ptx-0.5);\n\n'

        melOut += 'float $vfv = `camera -q -vfv ' + sCam + '`;\n'
        melOut += 'float $pty = (($ptVecCs.y/(-$ptVecCs.z))/tand($vfv/2))/2.0+.5;\n'
        melOut += 'float $panv = ' + chfv + '*((1-(1-$pty))-0.5);\n\n'

        # set attribute
        melH += melOut + sCam + '.horizontalPan = $panh;\n'
        melV += melOut + sCam + '.verticalPan = $panv;\n'

        cmds.setAttr(sCam + ".panZoomEnabled", 1)
        cmds.expression(o=sCam, s=melH, n="panh_" + sCam + "_" + sLoc)
        cmds.expression(o=sCam, s=melV, n="panv_" + sCam + "_" + sLoc)

        self.ui.listLoc.addItem("[ " + sCam + " ], " + sLoc)
        self.ui.enbLoc.setText(sLoc + ', Tacker Enabled')