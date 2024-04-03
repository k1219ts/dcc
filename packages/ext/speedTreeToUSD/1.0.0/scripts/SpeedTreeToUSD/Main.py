#coding:utf-8
from __future__ import print_function

import os, sys

from PySide2 import QtWidgets, QtCore, QtGui

import DXUSD.Utils as utl
from SpeedTreeToUSD.ui.Window import Win
import SpeedTreeToUSD.Vars as var


def Main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    win = Win(None)

    cenpos = QtWidgets.QDesktopWidget().availableGeometry().center()
    winpos = win.frameGeometry().center()
    win.move(cenpos - winpos)
    win.setWindowTitle('Speedtree to USD')

    icon = QtGui.QPixmap(utl.SJoin(var.ICON, 'speedtree_logo_small.png'))
    win.setWindowIcon(QtGui.QIcon(icon))

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    Main()
