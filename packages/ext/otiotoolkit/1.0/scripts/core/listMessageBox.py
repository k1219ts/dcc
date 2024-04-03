#coding:utf-8

from PySide2 import QtWidgets, QtCore, QtGui

class QListMessageBox(QtWidgets.QDialog):
    def __init__(self, items):
        QtWidgets.QDialog.__init__(self)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.resize(800, 300)

        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        self.listWidget = QtWidgets.QListWidget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for item in items:
            self.listWidget.addItem(item)

        gridLayout.addWidget(self.listWidget, 0, 0, 1, 3)

        okBtn = QtWidgets.QPushButton()
        okBtn.setText("&OK")
        okBtn.clicked.connect(self.accept)
        gridLayout.addWidget(okBtn, 1, 1, 1, 1)

        cancelBtn = QtWidgets.QPushButton()
        cancelBtn.setText("&CANCEL")
        cancelBtn.clicked.connect(self.reject)
        gridLayout.addWidget(cancelBtn, 1, 2, 1, 1)
        self.setLayout(gridLayout)

    def accept(self):
        self.result = True
        # self.velozStatus = ""
        # if self.omitBtn.isChecked():
        #     self.velozStatus = "Omit"
        # elif self.holdBtn.isChecked():
        #     self.velozStatus = "Hold"
        # elif self.waitingBtn.isChecked():
        #     self.velozStatus = "Waiting"

        self.close()

    def reject(self):
        self.result = False
        self.close()

    def closeEvent(self, event):
        print "Event"
        event.accept()