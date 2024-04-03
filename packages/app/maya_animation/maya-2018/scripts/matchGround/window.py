# -*- coding: utf-8 -*-
import os
import logging
import maya.cmds as cmds
import aniCommon
import mg_module; reload(mg_module)
from PySide2 import QtWidgets, QtCompat

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
UIFILE = os.sep.join([CURRENT_DIR, 'ui', 'mg_window.ui'])

CHARACTERS = ['Normal Spider', 'Wolf']

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        ui = QtCompat.loadUi(UIFILE, self)
        setup_ui(ui, self)

        self.setWindowTitle('Match Ground UI')
        self.characters_comboBox.addItems(CHARACTERS)
        self.minTime_lineEdit.setText(str(cmds.playbackOptions(q=True, min=True)))
        self.maxTime_lineEdit.setText(str(cmds.playbackOptions(q=True, max=True)))
        self.useRedirect_checkBox.setChecked(False)
        self.useRedirect_checkBox.setEnabled(False)
        self.bakeCharacter_comboBox.addItems(CHARACTERS)

        self.connectSignals()

    def connectSignals(self):
        self.addGround_Btn.clicked.connect(self.addGround)
        self.ok_Btn.clicked.connect(self.doIt)
        self.cancel_Btn.clicked.connect(self.close)
        self.doBake_Btn.clicked.connect(self.doBake)
        self.cancelBake_Btn.clicked.connect(self.close)
        self.test_button.clicked.connect(self.doItNew)

    def closeEvent(self, event):
        logger.debug(u'Window Closed')

    def getOptionsFromUi(self):
        options = dict()
        options['ground'] = str(self.groundObject_lineEdit.text())
        options['character'] = str(self.characters_comboBox.currentText())
        options['minTime'] = str(self.minTime_lineEdit.text())
        options['maxTime'] = str(self.maxTime_lineEdit.text())
        return options

    def addGround(self):
        sel = cmds.ls(sl=True)
        if sel:
            self.groundObject_lineEdit.setText(sel[0])

    def doIt(self):
        options = self.getOptionsFromUi()
        for opt in options:
            if not options[opt]:
                QtWidgets.QMessageBox.warning(self, 'waring!', 'No {0}'.format(opt))
                return
        selection = cmds.ls(sl=True)
        if not selection:
            QtWidgets.QMessageBox.warning(self, 'waring!', 'Select Rig Object')
            return
        logger.debug(u'Ground : {0}, Character : {1}'.format(options['ground'],
                                                             options['character']))
        logger.debug(u'Perform Match Ground')

        mg_module.matchGround(selection,
                              options['character'],
                              options['ground'],
                              options['minTime'],
                              options['maxTime'])

    def doItNew(self):
        options = self.getOptionsFromUi()
        for opt in options:
            if not options[opt]:
                QtWidgets.QMessageBox.warning(self, 'waring!', 'No {0}'.format(opt))
                return
        selection = cmds.ls(sl=True)
        if not selection:
            QtWidgets.QMessageBox.warning(self, 'waring!', 'Select Rig Object')
            return
        logger.debug(u'Ground : {0}, Character : {1}'.format(options['ground'],
                                                             options['character']))
        logger.debug(u'Perform Match Ground')

        mg_module.matchGroundNew(nodes=selection,
                                 character=options['character'],
                                 ground=options['ground'])



    def doBake(self):
        character = str(self.bakeCharacter_comboBox.currentText())
        nodes = cmds.ls(sl=True)
        minTime = cmds.playbackOptions(q=True, min=True)
        maxTime = cmds.playbackOptions(q=True, max=True)
        mg_module.bakeDeleteNodes(nodes=nodes,
                                  character=character,
                                  minTime=minTime,
                                  maxTime=maxTime)


def showUI():
    global _win
    try:
        _win.close()
    except:
        pass
    _win = MainWindow()
    _win.show()
    _win.resize(200, 100)
    return _win