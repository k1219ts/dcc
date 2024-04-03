#encoding:utf-8

import os
import logging
import maya.cmds as cmds

from PySide2 import QtWidgets, QtCore, QtUiTools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
UIFILE = os.sep.join([CURRENT_DIR, 'ui', 'moveKeyUi.ui'])

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class MoveKey(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MoveKey, self).__init__(parent)
        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)
        self.setWindowTitle('MoveMove')
        self.step_doubleSpinBox.setSingleStep(0.1)
        self.step_doubleSpinBox.setValue(1.0)

        self.connectSignals()

    def connectSignals(self):
        self.left_Btn.clicked.connect(lambda: self.move(positive=False))
        self.right_Btn.clicked.connect(lambda: self.move(positive=True))

    def move(self, positive):
        step = self.step_doubleSpinBox.value()
        if positive:
            cmds.keyframe(animation="keys",
                          option="over",
                          relative=True,
                          timeChange=(0 + step))
        else:
            cmds.keyframe(animation="keys",
                          option="over",
                          relative=True,
                          timeChange=(0 - step))

def showUI():
    global _win
    try:
        _win.close()
    except:
        pass
    _win = MoveKey()
    _win.show()
    _win.resize(200, 100)
    return _win