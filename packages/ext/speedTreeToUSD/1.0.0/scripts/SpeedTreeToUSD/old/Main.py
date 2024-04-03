# -*- coding: utf-8 -*-

import site
import os
import sys

from PySide2 import QtWidgets, QtCore, QtGui

import SpeedTreeToUSD.MainForm as MainForm

ScriptRoot = os.path.dirname(os.path.abspath(__file__))


def Main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("plastique"))

    mainVar = MainForm.MainForm(None)
    mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
    mainVar.setWindowTitle('Speedtree to USD')

    iconpath = '%s/resources/ui/pxr_usd.png'%ScriptRoot
    mainVar.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(iconpath)))

    mainVar.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    Main()
