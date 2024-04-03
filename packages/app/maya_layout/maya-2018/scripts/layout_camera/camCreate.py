# -*- coding: utf-8 -*-
####################################################
#          coding by RND youkyoung.kim             #
####################################################
import os, sys, re
from maya import cmds
import Qt
from Qt import QtWidgets
from Qt import QtCore
from camCreate_ui import CameraCreate_Form
import camJsonRead
import camJsonWrite

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
        window = CameraCreate(getMayaWindow())
        window.move(QtWidgets.QDesktopWidget().availableGeometry().center() - window.frameGeometry().center())
        window.show()

if __name__ == "__main__":
    main()

class CameraCreate(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = CameraCreate_Form()
        self.ui.setupUi(self)
        self.windowSetting()
        self.cameraCheck()
        self.connection()

    def connection(self):
        self.ui.ok_btn.clicked.connect(self.cameraCheck)
        self.ui.cancel_btn.clicked.connect(self.close)
        self.ui.create_btn.clicked.connect(self.createSpin)
        self.ui.list_tree.itemSelectionChanged.connect(self.selectCam)
        self.ui.list_tree.itemDoubleClicked.connect(self.editCamera)
        self.ui.plus_btn.clicked.connect(self.createAdd)
        self.ui.minus_btn.clicked.connect(self.deleteCam)
        self.ui.jsonwrite_btn.clicked.connect(camJsonWrite.main)

    def windowSetting(self):
        # default value set and treewidget set
        self.groupname = 'cam'
        self.cameramodel = ''
        self.ui.cam_spin.setValue(0)
        self.ui.list_tree.sortByColumn(0, QtCore.Qt.AscendingOrder)
        self.ui.list_tree.setColumnWidth(0 ,50)
        self.ui.list_tree.setColumnWidth(1, 150)
        self.ui.list_tree.setColumnWidth(2, 150)
        self.ui.jsonwrite_btn.setStyleSheet(
            "QPushButton#jsonwrite_btn {background-color: #336699; border: 1px solid #999;}"
            "QPushButton#jsonwrite_btn:hover {background-color: darkred; border: 1px solid #999;}"
        )

    def cameraCheck(self):
        # camera json file read and setting
        show = camJsonRead.mayaScene()
        # show = 'wgf'
        self.ui.list_tree.clear()
        if show:
            cameralist = camJsonRead.cameraList()
            self.ui.show_txt.setText(show)
            try:
                readjson = camJsonRead.CameraRead()
                self.cameramodel = readjson.cameraModel()
            except:
                pass

            if self.cameramodel:
                self.ui.model_txt.setText(self.cameramodel)

            if cameralist:
                self.ui.create_btn.setEnabled(False)
                # camera exist check >> true = create btn hidden
                for i, j in enumerate(cameralist):
                    self.camTreeSet(i, j)
        else:
            self.ui.create_btn.setEnabled(False)
            self.ui.plus_btn.setEnabled(False)
            self.ui.minus_btn.setEnabled(False)

    def buttonSetting(self, flag=None):
        self.ui.create_btn.setEnabled(flag)
        self.ui.plus_btn.setEnabled(flag)
        self.ui.minus_btn.setEnabled(flag)
#
    def camTreeSet(self, num=None, camname=None):
        # camera list treewidget print
        try:
            camshape = cmds.ls(camname, dag=True)[1]
            focal = cmds.getAttr(camshape + '.focalLength')

            item = Treeitem(self.ui.list_tree)
            item.setText(0, str(num))# number
            item.setText(1, str(camname)) #camera name
            item.setText(2, camshape)
            item.setText(3, str(focal))
        except:
            pass

    def groupMake(self):
        # camera group node create
        if not cmds.ls(self.groupname):
            cmds.group(em=True, n=self.groupname)

    def createSpin(self):
        # cameras create
        total = int(self.ui.cam_spin.value())
        if total:
            self.groupMake()
            for i in range(total):
                self.createCam(i+1)
            self.cameraCheck()

    def createAdd(self):
        # one camera create
        self.createCam()
        self.cameraCheck()

    def createCam(self, num=None):
        # create camera rename and group
        camname = cmds.camera()[0]
        if num:
            chname = str(10 * (num)).zfill(4)
            changename = unicode('C' + chname)
        else:
            changename = unicode('C0000')
        cmds.parent(camname, self.groupname)
        cmds.rename(camname, changename)

    def deleteCam(self):
        # select camera delete
        index = self.ui.list_tree.selectedItems()
        for i in index:
            indextext = i.text(1)
            seletcam = self.ui.list_tree.indexOfTopLevelItem(i)
            self.ui.list_tree.takeTopLevelItem(seletcam)
            cmds.delete(indextext)

    def editCamera(self, item=None, col=None):
        # camera name, focal length value edit dialog
        camname = item.text(1)
        focal = item.text(3)
        editwindow = CameraEdit(QtWidgets.QDialog)
        editwindow.cameraName(camname, focal)
        editwindow.show()
        result = editwindow.exec_()
        if not result:
            self.cameraCheck()

    def selectCam(self):
        # item click >> select change
        index = self.ui.list_tree.selectedItems()
        for i in index:
            indextext = i.text(1)
            cmds.select(indextext)

#------------------------------------------------------------
class Treeitem(QtWidgets.QTreeWidgetItem):
    # treewidget item
    def __init__(self, parent):
        super(Treeitem, self).__init__(parent)
        font = Qt.QtGui.QFont()
        font.setPointSize(11)

        for i in range(4):
            self.setFont(i, font)
            self.setTextAlignment(i, QtCore.Qt.AlignCenter)

#---------------------------------------------------------------
class CameraEdit(QtWidgets.QDialog):
    # camera edit dialog
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self)
        self.focal_length = ['', '10.0', '14.0', '17.5', '25.0', '32.0', '35.0', '50.0', '65.0']
        self.setWindowTitle("Camera Edit")
        self.resize(312, 150)
        gridLayout = QtWidgets.QGridLayout()
        verticalLayout = QtWidgets.QVBoxLayout()
        horizontalLayout = QtWidgets.QHBoxLayout()
        self.currentcam_txt = QtWidgets.QLabel()
        self.currentcam_txt.setMinimumSize(QtCore.QSize(120, 30))
        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.currentcam_txt.setFont(font)
        horizontalLayout.addWidget(self.currentcam_txt)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout.addItem(spacerItem)
        self.currentfocal_txt = QtWidgets.QLabel()
        self.currentfocal_txt.setMinimumSize(QtCore.QSize(120, 30))

        self.currentfocal_txt.setFont(font)
        horizontalLayout.addWidget(self.currentfocal_txt)
        verticalLayout.addLayout(horizontalLayout)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)

        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)
        verticalLayout.addWidget(line)
        horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.chcam_txt = QtWidgets.QLineEdit()
        self.chcam_txt.setMinimumSize(QtCore.QSize(120, 30))
        self.chcam_txt.setFont(font)
        horizontalLayout_2.addWidget(self.chcam_txt)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_2.addItem(spacerItem1)
        self.chfocal_combo = QtWidgets.QComboBox()
        self.chfocal_combo.setMinimumSize(QtCore.QSize(120, 30))
        self.chfocal_combo.setFont(font)

        horizontalLayout_2.addWidget(self.chfocal_combo)
        verticalLayout.addLayout(horizontalLayout_2)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        verticalLayout.addItem(spacerItem2)
        horizontalLayout_3 = QtWidgets.QHBoxLayout()

        font = Qt.QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        horizontalLayout_3.addItem(spacerItem3)
        cancel_btn = QtWidgets.QPushButton()
        cancel_btn.setMinimumSize(QtCore.QSize(70, 30))
        cancel_btn.setFont(font)

        horizontalLayout_3.addWidget(cancel_btn)
        ok_btn = QtWidgets.QPushButton()
        ok_btn.setMinimumSize(QtCore.QSize(70, 30))
        ok_btn.setFont(font)

        horizontalLayout_3.addWidget(ok_btn)
        verticalLayout.addLayout(horizontalLayout_3)
        gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)
        cancel_btn.setText("Cancel")
        ok_btn.setText("Ok")

        self.chfocal_combo.setEditable(True) # combobox text editable
        self.setLayout(gridLayout)

        self.chfocal_combo.addItems(sorted(self.focal_length))
        cancel_btn.clicked.connect(self.reject)
        ok_btn.clicked.connect(self.changeCam)

    def cameraName(self, camname=None, focal=None):
        # current cameraname and focal length get
        self.oldcam = camname
        self.currentcam_txt.setText(camname)
        self.currentfocal_txt.setText(focal)
        self.chcam_txt.setText(camname)
        try:
            index = self.focal_length.index(str(focal))
        except:
            self.chfocal_combo.setItemText(0, focal)
            index = 0
        self.chfocal_combo.setCurrentIndex(index)

    def changeCam(self):
        # cameraname and focal length changed >> rename and focal value set
        chcam = self.chcam_txt.text()
        focal = float(self.chfocal_combo.currentText())

        if re.match("C[0-9][0-9][0-9][0-9]", chcam) is None or len(chcam) != 5:
            ck = 0
            camJsonRead.messageBox('Camera Name : C0000 format not match !!')
        else:
            ck = 1
            cmds.rename(self.oldcam, chcam)

        if chcam != self.oldcam:
            camshape = cmds.listRelatives(chcam, shapes=True)[0]
        elif chcam == self.oldcam:
            camshape = cmds.listRelatives(self.oldcam, shapes=True)[0]
        else:
            print "camera name not match"

        if ck:
            cmds.setAttr(camshape + '.focalLength', focal)
        self.close()
