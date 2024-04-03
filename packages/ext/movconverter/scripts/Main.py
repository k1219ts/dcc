# -*- coding: utf-8 -*-

import os
import sys

from PySide2 import QtWidgets, QtGui

import MainForm


def main():
    app = QtWidgets.QApplication(sys.argv)

    mainVar = MainForm.MainForm(None)
    mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
    mainVar.setWindowTitle('movConverter')

    mainVar.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
