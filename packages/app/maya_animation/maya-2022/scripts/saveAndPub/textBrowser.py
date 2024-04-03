# encoding=utf-8
# !/usr/bin/env python

import os
from PySide2 import QtGui, QtCore
import maya.cmds as cmds
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = savePubBrowser()
    _win.show(dockable=True)
    _win.setAcceptDrops(True)

MAYAVERSION = os.getenv("MAYA_VER")

class savePubBrowser(MayaQWidgetDockableMixin, QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(savePubBrowser, self).__init__(parent)
        self.setWindowTitle('info window')
        self.setObjectName('savePubBrowser')
        self.resize(500, 700)

        self.cleanupProcess = QtCore.QProcess

        main_widget = QtGui.QWidget(self)
        main_vbox = QtGui.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        check_groupBox = QtGui.QGroupBox()
        check_groupBox.setTitle("CLEAN UP SCENE")
        vbox = QtGui.QVBoxLayout(check_groupBox)
        vbox.setContentsMargins(10, 10, 10, 0)
        vbox.setSpacing(6)
        main_vbox.addWidget(check_groupBox)

        clearTxtBTN = QtGui.QPushButton()
        clearTxtBTN.setFixedSize(100, 30)
        clearTxtBTN.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        clearTxtBTN.setText('clear log')
        clearTxtBTN.clicked.connect(self.clearTextbrowser)
        hbox = QtGui.QVBoxLayout()
        #spacer_i = QtGui.QSpacerItem(300, 0, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        hbox.addWidget(clearTxtBTN)
        #hbox.setAlignment(QtCore.Qt.AlignRight)

        main_vbox.addWidget(clearTxtBTN)

        self.textBrwsr = QtGui.QTextBrowser()
        self.textBrwsr.setLineWrapMode(QtGui.QTextBrowser.NoWrap)
        self.textBrwsr.setMinimumHeight(200)
        main_vbox.addWidget(self.textBrwsr)

        DoCleanupBTN = QtGui.QPushButton()
        DoCleanupBTN.setText("Clean Up")
        DoCleanupBTN.setMinimumHeight(50)
        DoCleanupBTN.clicked.connect(self.cleanup)
        main_vbox.addWidget(DoCleanupBTN)

        self.setupUI()

    def setupUI(self):
        self.checkboxDic["DelUnknNode"].setChecked(True)
        self.checkboxDic["DelUnknPlgn"].setChecked(True)
        self.checkboxDic["DelUnusedNode"].setChecked(True)
        self.checkboxDic["CheckNames"].setChecked(True)

    def clearTextbrowser(self):
        self.textBrwsr.clear()
