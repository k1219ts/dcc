from PySide2 import QtWidgets, QtCore, QtGui
import os
from core import calculator

ScriptRoot = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))

class PlateSetupArgumentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, windowName='Plate Setup Options', xlsFilePath=''):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        # self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setMinimumSize(800, 150)

        self.xlsFilePath = xlsFilePath

        gridLayout = QtWidgets.QGridLayout(self)
        gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)

        label = QtWidgets.QLabel()
        label.setText('Scan Root Path')
        gridLayout.addWidget(label, 0, 0, 1, 1)

        self.scanRootEdit = QtWidgets.QLineEdit()
        self.scanRootEdit.setText('')
        gridLayout.addWidget(self.scanRootEdit, 0, 1, 1, 4)

        self.findScanRootDir = QtWidgets.QPushButton()
        self.findScanRootDir.setText('')
        self.findScanRootDir.setMinimumSize(QtCore.QSize(30, 30))
        self.findScanRootDir.setMaximumSize(QtCore.QSize(30, 30))
        self.findScanRootDir.setIconSize(QtCore.QSize(35, 35))
        self.findScanRootDir.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(ScriptRoot, 'resources', 'folder.png'))))
        gridLayout.addWidget(self.findScanRootDir, 0, 5, 1, 1)

        self.findScanRootDir.clicked.connect(self.openDirectory)

        label2 = QtWidgets.QLabel()
        label2.setText('Shot Name')
        gridLayout.addWidget(label2, 1, 0, 1, 1)

        self.shotNameEdit = QtWidgets.QLineEdit()
        self.shotNameEdit.setText('')
        gridLayout.addWidget(self.shotNameEdit, 1, 1, 1, 5)

        self.versionUp = QtWidgets.QCheckBox()
        self.versionUp.setChecked(False)
        self.versionUp.setText("Force Version Up")
        gridLayout.addWidget(self.versionUp, 2, 0, 1, 1)

        okBtn = QtWidgets.QPushButton()
        okBtn.setText("&OK")
        okBtn.clicked.connect(self.accept)
        gridLayout.addWidget(okBtn, 3, 0, 1, 3)

        cancelBtn = QtWidgets.QPushButton()
        cancelBtn.setText("&CANCEL")
        cancelBtn.clicked.connect(self.reject)
        gridLayout.addWidget(cancelBtn, 3, 3, 1, 3)

        self.setLayout(gridLayout)

    def openDirectory(self):
        dirText = ''
        if not os.path.exists(dirText) and self.xlsFilePath:
            showName = calculator.parseShowName(self.xlsFilePath)
            dirText = '/stuff/%s/scan' % showName.lower()
        else:
            dirText = os.getenv("HOME")

        dialog = FindDirectoryDialog(self, "find scan root directory", dirText)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        result = dialog.exec_()
        if result == 1:
            path = dialog.selectedFiles()[-1]
            self.scanRootEdit.setText(path)

    def accept(self):
        self.result = True
        self.close()

    def reject(self):
        self.result = False
        self.close()

    def closeEvent(self, event):
        print "Event"
        event.accept()


class FindDirectoryDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, windowName='', dirPath=''):
        QtWidgets.QFileDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setDirectory(dirPath)
        self.setMinimumSize(1200, 800)