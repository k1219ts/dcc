# -*- coding: utf-8 -*-
import sys, os

from PySide2 import QtWidgets, QtCore

from ui_precomp import Ui_Dialog

class PrecompUI(QtWidgets.QDialog):
    def __init__(self, parent=None, basePath=None, filePath=None, version=None):

        QtWidgets.QDialog.__init__(self, parent=None)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.setWindowTitle("precomp path")

        self.widgetFont = self.font()
        self.widgetFont.setPointSize(10)
        self.setFont(self.widgetFont)

        self.ui.label_5.setStyleSheet("""
        QLabel{
        border: 1px solid black;
        border-radius: 3px;
        background: rgb(78,78,78);
        }
        """)

        self.ui.label_6.setStyleSheet("""
        QLabel{
        border: 1px solid black;
        border-radius: 3px;
        background: rgb(78,78,78);
        }
        """)


        self.basePath = basePath
        self.filePath = filePath
        self.origVersion = version

        self.ui.label_5.setText(basePath)
        self.ui.label_6.setText(filePath)

        self.isValidPath = False
        self.validColor = {True:'#547699', False:'#dc143c'}

        self.ui.lineEdit_3.textChanged.connect(self.nameChanged)
        self.ui.spinBox.valueChanged.connect(self.nameChanged)
        self.ui.buttonBox.accepted.connect(self.okClicked)
        self.ui.buttonBox.rejected.connect(self.cancelClicked)

        self.ui.spinBox.setMinimum(1)

        self.ui.lineEdit_3.setText('precomp')
        self.ui.spinBox.setValue(1)
        self.result = ''


    def okClicked(self):
        htmlParse = QtWidgets.QTextEdit()
        htmlParse.setHtml(self.ui.label_5.text() + self.ui.label_6.text())
        self.result = htmlParse.toPlainText()
        self.accept()
        print("ok")

    def cancelClicked(self):
        self.reject()
        print("cancel")

    def nameChanged(self):
        text = str(self.ui.lineEdit_3.text())
        version = 'v' + str(self.ui.spinBox.value()).zfill(3)

        self.isValidPath = not(os.path.exists(self.basePath + text + '/' + text + '_' + version))
        baseText = self.basePath + \
                   '<font size = "3" style="background-color: %s; color: black">%s</font>' % (self.validColor[self.isValidPath],text) + '/' + \
                   '<font size = "3" style="background-color: %s; color: black">%s_%s</font>' % (self.validColor[self.isValidPath],text, version) + '/'
        self.ui.label_5.setText(baseText)


        versionToSplit = self.filePath.split('precomp')[-1].replace(self.origVersion, version)
        filenameText = '<font size = "3" style="background-color: %s; color: black">%s</font>' % (self.validColor[self.isValidPath], text) + \
                        versionToSplit.split(version)[0] + \
                        '<font size = "3" style="background-color: %s; color: black">%s</font>' % (self.validColor[self.isValidPath], version) + \
                        versionToSplit.split(version)[-1]

        self.ui.label_6.setText(filenameText)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ce = PrecompUI(None, '/show/mkk/shot/FFT2/FFT2_0640/comp/src/precomp/', 'FFT2_0640_precomp_%V_v001.%04d.exr', 'v001')
    ce.show()
    sys.exit(app.exec_())
