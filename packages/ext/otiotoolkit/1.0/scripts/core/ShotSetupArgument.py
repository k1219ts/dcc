#coding:cp949
from PySide2 import QtWidgets

class ShotSetupArgumentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, windowName='Shot Setup Options'):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        # self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setMinimumSize(800, 150)

        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        self.shotBurnCheckBox = QtWidgets.QCheckBox()
        self.shotBurnCheckBox.setText("ShotName BurnIn")
        self.shotBurnCheckBox.setChecked(True)
        gridLayout.addWidget(self.shotBurnCheckBox, 0, 0, 1, 1)
        self.effectBurnCheckBox = QtWidgets.QCheckBox()
        self.effectBurnCheckBox.setText("Effect BurnIn")
        self.effectBurnCheckBox.setChecked(True)
        gridLayout.addWidget(self.effectBurnCheckBox, 0, 1, 1, 1)
        self.velozSeqMov = QtWidgets.QCheckBox()
        self.velozSeqMov.setText("Upload Veloz Sequence Mov")
        self.velozSeqMov.setChecked(True)
        gridLayout.addWidget(self.velozSeqMov, 0, 2, 1, 1)
        self.editOrder = QtWidgets.QCheckBox()
        self.editOrder.setText("Veloz EditOrder Change")
        self.editOrder.setChecked(False)
        gridLayout.addWidget(self.editOrder, 0, 3, 1, 1)

        self.vfxDetailUpdate = QtWidgets.QCheckBox()
        self.vfxDetailUpdate.setChecked(True)
        self.vfxDetailUpdate.setText('VFX Detail Update')
        gridLayout.addWidget(self.vfxDetailUpdate, 0, 4, 1, 1)

        label = QtWidgets.QLabel()
        label.setText('Shot Name')
        gridLayout.addWidget(label, 1, 0, 1, 1)

        self.shotNameEdit = QtWidgets.QLineEdit()
        self.shotNameEdit.setPlaceholderText('If you enter a shot name, only the shot is set. Separate the shot names with spaces.')
        gridLayout.addWidget(self.shotNameEdit, 1, 1, 1, 5)

        label2 = QtWidgets.QLabel()
        label2.setText('FPS')
        gridLayout.addWidget(label2, 2, 0, 1, 1)

        self.editFPS = QtWidgets.QComboBox()
        self.editFPS.addItems(['23.98', '24.00', '29.98', '30.00'])
        gridLayout.addWidget(self.editFPS, 2, 1, 1, 1)

        self.velozUpdate = QtWidgets.QCheckBox()
        self.velozUpdate.setChecked(True)
        self.velozUpdate.setText("Veloz Update")
        gridLayout.addWidget(self.velozUpdate, 2, 2, 1, 1)

        self.velozDurationUpdate = QtWidgets.QCheckBox()
        self.velozDurationUpdate.setChecked(False)
        self.velozDurationUpdate.setText("Veloz Duration Update")
        gridLayout.addWidget(self.velozDurationUpdate, 2, 3, 1, 1)

        # Radio Button
        label3 = QtWidgets.QLabel()
        label3.setText('VelozStatus')
        gridLayout.addWidget(label3, 3, 0, 1, 1)
        self.omitBtn = QtWidgets.QRadioButton("Omit")
        self.omitBtn.setChecked(True)
        self.holdBtn = QtWidgets.QRadioButton("Hold")
        self.waitingBtn = QtWidgets.QRadioButton("Waiting")
        gridLayout.addWidget(self.omitBtn, 3, 1, 1, 1)
        gridLayout.addWidget(self.holdBtn, 3, 2, 1, 1)
        gridLayout.addWidget(self.waitingBtn, 3, 3, 1, 1)

        okBtn = QtWidgets.QPushButton()
        okBtn.setText("&OK")
        okBtn.clicked.connect(self.accept)
        gridLayout.addWidget(okBtn, 4, 0, 1, 3)

        cancelBtn = QtWidgets.QPushButton()
        cancelBtn.setText("&CANCEL")
        cancelBtn.clicked.connect(self.reject)
        gridLayout.addWidget(cancelBtn, 4, 3, 1, 3)

        self.setLayout(gridLayout)

    def accept(self):
        self.result = True
        self.velozStatus = ""
        if self.omitBtn.isChecked():
            self.velozStatus = "Omit"
        elif self.holdBtn.isChecked():
            self.velozStatus = "Hold"
        elif self.waitingBtn.isChecked():
            self.velozStatus = "Waiting"
        self.close()

    def reject(self):
        self.result = False
        self.close()

    def closeEvent(self, event):
        print "Event"
        event.accept()