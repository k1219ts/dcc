#coding:utf-8
from PySide2 import QtWidgets, QtGui

class PlateSpoolArgumentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, windowName='Plate Spool Options'):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        # self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setMinimumSize(800, 150)

        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        label = QtWidgets.QLabel()
        label.setText('Shot Name')
        gridLayout.addWidget(label, 0, 0, 1, 1)

        self.shotNameEdit= QtWidgets.QLineEdit()
        # self.shotNameEdit.setText('')
        self.shotNameEdit.setPlaceholderText('If you enter a shot name, only the shot is set. Separate the shot names with spaces.')
        gridLayout.addWidget(self.shotNameEdit, 0, 1, 1, 3)

        label2 = QtWidgets.QLabel()
        label2.setText('inOffset')
        gridLayout.addWidget(label2, 1, 0, 1, 1)

        self.inOffsetEdit = QtWidgets.QLineEdit()
        self.inOffsetEdit.setText('0')
        self.inOffsetEdit.setValidator(QtGui.QIntValidator(-100, 100))
        gridLayout.addWidget(self.inOffsetEdit, 1, 1, 1, 1)

        label2 = QtWidgets.QLabel()
        label2.setText('outOffset')
        gridLayout.addWidget(label2, 1, 2, 1, 1)

        self.outOffsetEdit = QtWidgets.QLineEdit()
        self.outOffsetEdit.setText('0')
        self.outOffsetEdit.setValidator(QtGui.QIntValidator(-100, 100))
        gridLayout.addWidget(self.outOffsetEdit, 1, 3, 1, 1)

        okBtn = QtWidgets.QPushButton()
        okBtn.setText("&OK")
        okBtn.clicked.connect(self.accept)
        gridLayout.addWidget(okBtn, 2, 0, 1, 2)

        cancelBtn = QtWidgets.QPushButton()
        cancelBtn.setText("&CANCEL")
        cancelBtn.clicked.connect(self.reject)
        gridLayout.addWidget(cancelBtn, 2, 2, 1, 2)

        self.setLayout(gridLayout)

    def accept(self):
        self.result = True
        self.close()

    def reject(self):
        self.result = False
        self.close()

    def closeEvent(self, event):
        print "Event"
        event.accept()