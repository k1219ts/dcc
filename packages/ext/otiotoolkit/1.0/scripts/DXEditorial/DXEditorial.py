#coding:utf-8

import os
import sys

from DXEditorialUI import Ui_Form
from FileDialog import FindFileDialog
from adapters.FCP7_XMLParser import FCPXML7Parser
from Define import *

from PySide2 import QtWidgets, QtGui

ScriptRoot = os.path.dirname(os.path.abspath(__file__))

class DXEditorial(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setWindowTitle('DXEditorial Project')
        iconpath = '%s/ui/pxr_usd.png' % ScriptRoot
        self.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(iconpath)))

        self.xmlFile = ''
        self.ui.xmlEdit.textChanged.connect(self.xmlParse)

        self.ui.xmlBrowseBtn.clicked.connect(self.xmlBtnClicked)
        self.ui.scanBrowseBtn.clicked.connect(self.scanBtnClicked)

        self.ui.exportExcelBtn.clicked.connect(self.exportExcelFile)
        self.ui.importExcelBtn.clicked.connect(self.importExcelFile)
        self.ui.rvPlayerBtn.clicked.connect(self.rvPlayerFromOTIO)

    def xmlBtnClicked(self):
        filePath = self.selectFile(['*.xml'], self.ui.xmlEdit.text())
        if filePath:
            self.ui.xmlEdit.setText(filePath)
            # self.xmlParse()

    def scanBtnClicked(self):
        dirPath = self.selectDirectory(self.ui.scanEdit.text())
        if dirPath:
            self.ui.scanEdit.setText(dirPath)

    def xmlParse(self):
        print self.ui.xmlEdit.text()
        if self.xmlFile != self.ui.xmlEdit.text() and os.path.exists(self.ui.xmlEdit.text()):
            if os.path.splitext(self.ui.xmlEdit.text())[-1] == '.xml':
                self.xmlFile = self.ui.xmlEdit.text()
                self.parser = FCPXML7Parser(self.xmlFile)
                self.parser.doIt()

                self.ui.excelTable.setColumnCount(len(Column2))
                self.ui.excelTable.setRowCount(len(self.parser.excelMng.excelList))
                for row, data in enumerate(self.parser.excelMng.excelList):
                    for column in Column2:
                        item = QtWidgets.QTableWidgetItem(str(data[column.name]))
                        self.ui.excelTable.setItem(row, column.value, item)


    def selectFile(self, filter, dirText):
        if not os.path.exists(dirText):
            dirText = os.getenv("HOME")

        dialog = FindFileDialog(self, "select file", dirText)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(filter)

        result = dialog.exec_()
        if result == 1:
            path = dialog.selectedFiles()[-1]
        else:
            path = ''

        return path

    def selectDirectory(self, dirText):
        if not os.path.exists(dirText):
            dirText = os.getenv("HOME")

        dialog = FindFileDialog(self, "select directory", dirText)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        # dialog.setNameFilters(filter)

        result = dialog.exec_()
        if result == 1:
            path = dialog.selectedFiles()[-1]
        else:
            path = ''

        return path

    def exportExcelFile(self):
        print "exportExcelFile"
        self.parser.save()

    def importExcelFile(self):
        print "importExcelFile"

    def rvPlayerFromOTIO(self):
        print "rvPlayerFromOTIO"

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    # nautilusFile = os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
    # setFile = ''
    # if nautilusFile:
    #     setFile = nautilusFile.split('\n')[0]
    mainVar = DXEditorial()
    mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
    mainVar.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
