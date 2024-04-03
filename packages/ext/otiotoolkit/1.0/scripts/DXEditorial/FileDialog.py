from PySide2 import QtWidgets

class FindFileDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, windowName='', dirPath='' ):
        QtWidgets.QFileDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setDirectory(dirPath)
        self.setMinimumSize(1200, 800)
