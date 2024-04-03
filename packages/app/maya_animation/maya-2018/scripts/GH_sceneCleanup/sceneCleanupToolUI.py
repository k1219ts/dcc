# encoding=utf-8
# !/usr/bin/env python

import os
from shiboken2 import wrapInstance
from PySide2 import QtCore, QtGui, QtWidgets

import maya.OpenMayaUI as OpenMayaUI
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
import maya.cmds as cmds

import GH_sceneCleanup.modules as modules
reload(modules)

def maya_main_window():
    main_window_ptr = OpenMayaUI.MQtUtil.mainWindow()
    return wrapInstance(long(main_window_ptr), QtWidgets.QWidget)

def maya_api_version():
    return int(cmds.about(api=True))

_win = None

def showUI():
    global bariquantWindow
    bariquantWindow = bariquantWindow(parent = maya_main_window())
    bariquantWindow.run()
    return bariquantWindow

MAYAVERSION = os.getenv("MAYA_VER")

CHECK_LABEL_DICT = {"BakeConst"    : "Bake, Delete Constraint",
                  "DelDispLYR"   : "Delete Display Layers",
                  "DelUnusedNode": "Delete Unused Node",
                  "MergeAnimLYR" : "Merge Anim Layers",
                  "DelImgBar"    : "Delete ImagePlane Bar",
                  "DelUnknNode"  : "Delete Unknown Node",
                  "DelUnknPlgn"  : "Delete Unknown Plugins",
                  "CleanupSeq"   : "Delete Unused Sequencer",
                  "CheckNames"   : "Check Duplicated Names",}


class bariquantWindow(MayaQWidgetDockableMixin, QtWidgets.QMainWindow):
    MAYA2017 = 201700

    def __init__(self, parent=None):
        super(bariquantWindow, self).__init__(parent)
        self.setWindowTitle('BARIQUANT')
        self.resize(500, 300)

        self.cleanupProcess = QtCore.QProcess

        main_widget = QtWidgets.QWidget(self)
        main_vbox = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.textBrwsr = QtWidgets.QTextBrowser()
        self.textBrwsr.setLineWrapMode( QtWidgets.QTextBrowser.NoWrap )
        self.textBrwsr.setMinimumHeight( 200 )
        main_vbox.addWidget( self.textBrwsr )

        hbox = QtWidgets.QHBoxLayout()
        spacer_i = QtWidgets.QSpacerItem(QtWidgets.QSizePolicy.Expanding,
                                         QtWidgets.QSizePolicy.Expanding )
        hbox.addItem( spacer_i )
        clearTxtBTN = QtWidgets.QPushButton()
        clearTxtBTN.setFixedSize( 100, 20 )
        clearTxtBTN.setSizePolicy( QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding )
        clearTxtBTN.setText( 'clear log' )
        clearTxtBTN.clicked.connect( self.clearTextbrowser )
        hbox.addWidget( clearTxtBTN )
        hbox.setAlignment(QtCore.Qt.AlignRight)
        main_vbox.addLayout(hbox)

        check_groupBox = QtWidgets.QGroupBox()
        check_groupBox.setTitle("CLEAN UP SCENE")
        vbox = QtWidgets.QVBoxLayout(check_groupBox)
        vbox.setContentsMargins(10, 10, 10, 0)
        vbox.setSpacing(6)
        main_vbox.addWidget(check_groupBox)

        self.checkboxDic = {}

        for i in CHECK_LABEL_DICT:
            self.checkboxDic[i] = self.createCheckBox(i, CHECK_LABEL_DICT[i])
            vbox.addWidget(self.checkboxDic[i])

        DoCleanupBTN = QtWidgets.QPushButton()
        DoCleanupBTN.setText("Clean Up")
        DoCleanupBTN.setMinimumHeight(50)
        DoCleanupBTN.clicked.connect(self.cleanup)
        main_vbox.addWidget(DoCleanupBTN)

        self.setupUI()

    def dockCloseEventTriggered(self):
        self.deleteInstances()

    def deleteControl(self, control):
        if cmds.workspaceControl(control, q=True, exists=True):
            cmds.workspaceControl(control, e=True, close=True)
            cmds.deleteUI(control, control=True)

    def deleteInstances(self):
        if maya_api_version() < bariquantWindow.MAYA2017:
            for obj in maya_main_window().children():
                if str(type(obj)) == "<class '{}.MyDockingWindow'>".format(os.path.splitext(
                                                        os.path.basename(__file__)[0] )):
                    if obj.__class__.__name__ == "bariquantWindow":
                        obj.setParent(None)
                        obj.deleteLater()

    def createCheckBox(self, objectName, label):
        checkbox = QtWidgets.QCheckBox()
        checkbox.setObjectName(objectName)
        checkbox.setText(label)
        return checkbox

    def setupUI(self):
        self.checkboxDic["DelUnknNode"].setChecked(True)
        self.checkboxDic["DelUnknPlgn"].setChecked(True)
        self.checkboxDic["DelUnusedNode"].setChecked(True)
        self.checkboxDic["CheckNames"].setChecked(True)

    def clearTextbrowser(self):
        self.textBrwsr.clear()

    def cleanup(self):
        if self.checkboxDic["CheckNames"].isChecked():
            modules.checkDuplicateNames(self)
        if self.checkboxDic["BakeConst"].isChecked():
            modules.delConstrains(self)
        if self.checkboxDic["DelDispLYR"].isChecked():
            modules.delDispLYR(self)
        if self.checkboxDic["DelUnusedNode"].isChecked():
            modules.deleteUnusedNodeCmd(self)
        if self.checkboxDic["MergeAnimLYR"].isChecked():
            modules.AnimLYRmerge(self)
        if self.checkboxDic["DelImgBar"].isChecked():
            modules.DeleteBarPlane(self)
        if self.checkboxDic["DelUnknNode"].isChecked():
            modules.deleteUnknownNode(self)
        if MAYAVERSION == '2016.5' and self.checkboxDic["DelUnknPlgn"].isChecked():
            modules.cleanUnknownPlugins(self)
        if MAYAVERSION == '2017' and self.checkboxDic["DelUnknPlgn"].isChecked():
            modules.cleanUnknownPlugins(self)
        if self.checkboxDic["CleanupSeq"].isChecked():
            modules.cleanupSequencer(self)

        QtWidgets.QMessageBox.information(self, unicode("알림", 'utf-8'), unicode("성공적으로 완료되었습니다.", 'utf-8'))

    def run(self):
        self.setObjectName('bariquantMainWindow')
        workSpaceControlName = self.objectName() + 'WorkspaceControl'
        self.deleteControl(workSpaceControlName)
        self.show(dockable=True)
        #cmds.workspaceControl(workSpaceControlName, e=True, ttc=["AttributeEditor", -1], wp="preferred", mw=420)
        #self.raise_()
        #self.setDockableParameters(width=420)
"""
def show():
    global _win
    if _win:
        _win.parent().close()
        _win.parent().deleteLater()
    _win = bariquantWindow()
    _win.show(dockable=True)
    _win.parent().setAcceptDrops(True)
"""
