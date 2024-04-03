# -*- coding: utf-8 -*-

try:
    from PySide.QtCore import *
    from PySide.QtGui import *
    import pysideuic
    import shiboken
    import xml.etree.ElementTree as xml
    from cStringIO import StringIO
except:
    import sip
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from PyQt4 import uic
import os, sys

VERSION = "v0.2"
currentpath = os.path.abspath(__file__)
UIROOT = os.path.join(os.path.dirname(currentpath), 'ui')
currentDir = os.path.dirname(currentpath)
UIFILE = os.path.join(UIROOT, "TrimMov.ui")
css = open(os.path.join(UIROOT, 'studioLibrary.css'), 'r').read()
rcss = css.replace("RESOURCE_DIRNAME", currentDir+"/res")
rcss = rcss.replace("BACKGROUND_COLOR", "rgb(40,40,40)")
rcss = rcss.replace("ACCENT_COLOR", "rgb(255,90,40)")

def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' %form_class]
        base_class = eval('%s' % widget_class)

    return form_class, base_class
try:
    formclass, baseclass = loadUiType(UIFILE)
except:
    formclass, baseclass = uic.loadUiType(UIFILE)

class trimMovWindow(formclass, baseclass):
    def __init__(self, parent=None):
        super(trimMovWindow, self).__init__(parent)
        self.setupUi(self)
        # self.setWindowIcon( QIcon(QPixmap( os.path.join(UIROOT, 'alembic_black.png') )) )
        self.setWindowTitle('Movie Trim %s' %VERSION)
        self.setStyleSheet(rcss)
        self.move(QPoint(1200 / 2, 200))

        # command binding
        self.MovBrowse_Btn.clicked.connect(self.browseMovieFile)
        self.OutDirBrowse_Btn.clicked.connect(self.outpathSelect)
        self.DoIt_Btn.clicked.connect(self.DoTrim)

        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def browseMovieFile(self):
        current = str(self.MovName_lineEdit.text().toUtf8())
        startPath = ''
        if os.path.exists(current):
            startPath = current

        fileName = str(QFileDialog.getOpenFileName(self, "Select Movie File", startPath, "mov (*.*)").toUtf8())

        if not fileName:
            return
        self.MovName_lineEdit.setText(unicode(fileName, 'utf-8'))

    def outpathSelect(self):
        current = str(self.OutName_lineEdit.text().toUtf8())
        startPath = ""
        if os.path.exists(current):
            startPath = current
        dirName = str(QFileDialog.getExistingDirectory(self, "Directory Select", startPath).toUtf8())
        if not dirName:
            return
        self.OutName_lineEdit.setText(unicode(dirName, 'utf-8'))

    def DoTrim(self):
        movPath = str(self.MovName_lineEdit.text().toUtf8())
        outDir = str(self.OutName_lineEdit.text().toUtf8())

        StartFrame = int(self.StartTime_lineEdit.text())
        EndFrame = int(self.EndTime_lineEdit.text())

        prefix = os.path.splitext(os.path.basename(movPath))[0]

        outPath = os.path.join(outDir, prefix)

        outPathTemp = outPath + "Temp"
        outFileName = outPath + "_%s_%s.mov" % (StartFrame, EndFrame)

        if not os.path.isdir(outPathTemp):
            os.makedirs(outPathTemp)

        cmd = '%s/pdpExport.sh "%s" "%s" %d %d' % (currentDir, movPath, outPathTemp, StartFrame, EndFrame)

        os.system(cmd)

        cmd = '%s/imgToMov.sh %d %d 24 "H.264 LT" "%s" "%s" "%s" "movie"' % (
            currentDir, StartFrame, EndFrame, outPathTemp, outFileName, prefix)
        os.system(cmd)

        QMessageBox.information(self, unicode("알림", 'utf-8'), unicode("성공", 'utf-8'))

def show():
    global app

    # Use a shared instance of QApplication
    import maya.OpenMayaUI as mui
    app = QApplication.instance()

    # Get a pointer to the maya main window
    ptr = mui.MQtUtil.mainWindow()
    # Use sip to wrap the pointer into a QObject
    try:
        win = shiboken.wrapInstance(long(ptr), QWidget)
    except:
        win = sip.wrapinstance(long(ptr), QObject)
    form = trimMovWindow(win)
    form.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = trimMovWindow()
    win.show()
    sys.exit(app.exec_())
