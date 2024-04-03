# encoding=utf-8

import os, sys
try:
    from PyQt4 import QtGui, QtCore
except:
    from Qt import QtGui, QtCore, QtWidgets, load_ui

import modules

_win = None

MOVIE_FORMATS = ['mkv', 'mov', 'avi', 'mp4']

def showUI():
    global _win
    if _win:
        _win.close()
        _win.deleteLater()
    _win = Movicle()
    _win.show()

class Movicle(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Movicle, self).__init__(parent)
        self.setWindowTitle('Movicle - Movie Converter')
        self.setAcceptDrops(True)

        main_widget = QtGui.QWidget(self)
        main_vbox = QtGui.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        label_main = QtGui.QLabel()
        label_main.setText('Drop Here')
        label_main.setFont(QtGui.QFont('SansSerif', 100, QtGui.QFont.Bold))
        label_main.setAlignment(QtCore.Qt.AlignHCenter)
        main_vbox.addWidget(label_main)

        self.infoBrowser = QtGui.QTextBrowser()
        self.infoBrowser.setLineWrapMode(QtGui.QTextBrowser.NoWrap)
        main_vbox.addWidget(self.infoBrowser)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.inputFileList = list()
        self.outputFileList = list()
        ls_filePath = event.mimeData().urls()
        info = 'Input : \n'
        info += '{inputFiles}\n'
        info += 'Output : \n'
        info += '{outputFiles}'
        inputString = str()
        outputString = str()

        for mf in MOVIE_FORMATS:
            for qfilePath in ls_filePath:
                filePath = str(qfilePath.toString())
                if filePath.endswith(mf):
                    inputString += '      {}\n'.format( filePath )
                    outPath = os.path.splitext(filePath)[0] + '_conv.mov'
                    outputString += '      {}\n'.format( outPath )
                    self.inputFileList.append(filePath)
                    self.outputFileList.append(outPath)
        info = info.format(inputFiles=inputString, outputFiles=outputString)
        self.infoBrowser.setText(info)
        self.convert()

    def convert(self):
        modules.convert(self.inputFileList, self.outputFileList)
        QtGui.QMessageBox.information(self, "Finished", "Conversion Finished")


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = Movicle()
    #win.resize(1500, 600)
    win.show()
    sys.exit(app.exec_())