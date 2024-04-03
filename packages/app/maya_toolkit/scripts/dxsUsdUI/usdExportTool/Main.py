# -*- coding: utf-8 -*-

import site
import os
import sys
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
import MainForm
import subprocess

ScriptRoot = os.path.dirname(os.path.abspath(__file__))

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    mainVar = MainForm.MainForm(None)
    mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
    mainVar.setWindowTitle('USD Export')
    iconpath = '%s/ui/pxr_usd.png' % ScriptRoot
    mainVar.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(iconpath)))
    mainVar.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
