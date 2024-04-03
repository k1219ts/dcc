# encoding=utf-8
# !/usr/bin/env python

import os
from PySide2 import QtCore, QtWidgets, QtGui, load_ui

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = tacticSelectContextUI()
    _win.show()

class tacticSelectContextUI(QtWidgets.QMainWindow):
    def __init__(self, parent=None, task=list()):
        super(tacticSelectContextUI, self).__init__(parent)
        self.parent=parent
        self.setWindowTitle('Select Context')
        self.setObjectName('tacticSelectContextUI')
        self.resize(300, 200)

        main_widget = QtGui.QWidget(self)
        main_vbox = QtGui.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.contextComboBox = QtGui.QComboBox()
        self.contextComboBox.addItems(task)
        main_vbox.addWidget(self.contextComboBox)

        hbox = QtGui.QHBoxLayout()
        DoCheckInBTN = QtGui.QPushButton()
        DoCheckInBTN.setText("Check In")
        DoCheckInBTN.clicked.connect(self.checkin)
        cancelBTN = QtGui.QPushButton()
        cancelBTN.setText("Cancel")
        cancelBTN.clicked.connect(self.closeUI)

        hbox.addWidget(DoCheckInBTN)
        hbox.addWidget(cancelBTN)

        main_vbox.addLayout(hbox)

    def checkin(self):
        self.selectedTask = str(self.contextComboBox.currentText())
        self.close()

    def closeUI(self):
        self.close()