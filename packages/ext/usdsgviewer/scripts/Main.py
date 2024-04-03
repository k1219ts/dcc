# -*- coding: utf-8 -*-

import os
import sys

from PySide2 import QtWidgets, QtGui

import MainForm


def main():
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle(QtWidgets.QStyleFactory.create("plastique"))

    mainVar = MainForm.MainForm(None)
    mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
    mainVar.setWindowTitle('USD Scenegraph Viewer')

    iconpath = '%s/ui/pxr_usd.png' % os.path.dirname(os.path.abspath(__file__))
    mainVar.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(iconpath)))

    mainVar.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
