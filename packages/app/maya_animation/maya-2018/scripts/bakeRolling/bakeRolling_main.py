import sys

from PySide2 import QtGui, QtWidgets, QtCore

# ---- import dev modules
from bakeRolling_ui import Ui_bkr_win as winUi
import bakeRolling_functions as fn

'''
import sys
import importlib as im

workdir = '/works/tasks/pmc/drone'
modname = 'bakeRolling'

def reload_submodules(modname):
    mod = im.import_module(modname)
    reload(mod)
    if hasattr(mod, '__all__'):
        for m in mod.__all__:
            reload_submodules('{}.{}'.format(modname, m))


if not workdir in sys.path:
	sys.path.append(workdir)

reload_submodules(modname)
'''


global BRK_WINDOW



class win(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = winUi()
        self.ui.setupUi(self)

        self.move(parent.x()+100, parent.y()+200)
        #self.move(parent.x(), parent.y()-450)
        fn.functions(self.ui)

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

def main():
    mainWin = win(getMayaWindow())
    mainWin.show()

    return mainWin
try:
    BRK_WINDOW.close()
except:
    pass

BRK_WINDOW = main()
