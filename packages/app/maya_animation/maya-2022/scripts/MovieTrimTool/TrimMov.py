# -*- coding: utf-8 -*-

try:
    from PySide.QtCore import *
    from PySide.QtGui import *
    import pysideuic
    import shiboken
    import xml.etree.ElementTree as xml
    from cStringIO import StringIO
    from PySide.phonon import Phonon
except:
    import sip
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    from PyQt4 import uic
    from PyQt4.phonon import Phonon

import os, sys
import subprocess

FFMPEG = "ffmpeg"

VERSION = "v2.0"
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
    formclass, baseclass = uic.loadUiType(UIFILE)
except:
    formclass, baseclass = loadUiType(UIFILE)

class trimMovWindow(formclass, baseclass):
    def __init__(self, parent=None):
        super(trimMovWindow, self).__init__(parent)
        self.setupUi(self)
        # self.setWindowIcon( QIcon(QPixmap( os.path.join(UIROOT, 'alembic_black.png') )) )
        self.setWindowTitle('Movie Trim %s' %VERSION)
        self.setStyleSheet(rcss)
        self.move(QPoint(1200 / 2, 200))

        #self.seekSlider = Phonon.SeekSlider()
        #self.horizontalLayout_7.insertWidget(1, self.seekSlider)
        #self.seekSlider.setIconVisible(0)

        # command binding
        self.MovBrowse_Btn.clicked.connect(self.browseMovieFile)
        self.OutDirBrowse_Btn.clicked.connect(self.outpathSelect)
        self.DoIt_Btn.clicked.connect(self.DoTrim)
        self.byFrame_radioButton.toggled.connect(self.timeUnitChange)
        self.timeSlider.sliderReleased.connect(self.slider_value_change)
        self.buttonMimes.clicked.connect(self.handleButtonMimes)
        self.timeSlider.hide()
        #self.seekSlider.hide()

        self.show()

    def handleButtonMimes(self):
        dialog = MimeDialog(self)
        dialog.exec_()

    def keyPressEvent(self, event):
        #if event.key() == Qt.Key_Escape:
            #self.close()
        if event.key() == Qt.Key_S:
            self.play_or_pause()

    def timeUnitChange(self):
        if self.byFrame_radioButton.isChecked():
            self.StartTime_lineEdit.clear()
            self.EndTime_lineEdit.clear()
        else:
            self.StartTime_lineEdit.setText("00:00:00")
            self.EndTime_lineEdit.setText("00:00:00")

    def browseMovieFile(self):
        #current = str(self.MovName_lineEdit.text().toUtf8())
        startPath = ''
        #if os.path.exists(current):
            #startPath = current

        fileNameStr = QFileDialog.getOpenFileName(self, "Select Movie File", startPath, "mov (*.*)")

        try:
            fileName = str(fileNameStr.toUtf8())
        except:
            fileName = fileNameStr[0]

        if not fileName:
            return

        try:
            self.fps = self.getFPS(fileName)
        except:
            self.fps = 24

        try:
            self.MovName_lineEdit.setText(unicode(fileName, 'utf-8'))
        except:
            self.MovName_lineEdit.setText(fileName)
        self.FPS_lineEdit.setText(str(self.fps))
        self.loadVideo(fileName)

    def getFPS(self, filename):
        _a, _b, _c = os.popen3('ffmpeg -i "%s"' % filename)
        _out = _c.read()
        _outLines = _out.splitlines()
        _outLinesList = _outLines[18].split(" ")
        fpsIndex = _outLinesList.index("fps,")
        fps = float(_outLinesList[fpsIndex - 1])
        #fpsIndex = _out.index("fps, ")
        #fps = float(_out[fpsIndex - 4:fpsIndex])
        print fps
        return fps

    def outpathSelect(self):
        current = str(self.OutName_lineEdit.text().toUtf8())
        startPath = ""
        if os.path.exists(current):
            startPath = current
        dirName = str(QFileDialog.getExistingDirectory(self, "Directory Select", startPath).toUtf8())
        if not dirName:
            return
        self.OutName_lineEdit.setText(unicode(dirName, 'utf-8'))

    def loadVideo(self, movPath):
        try:
            self.media_obj.clear()
        except:
            pass

        for count in reversed(range(self.VideoLayout.count())):
            widgetToRemove = self.VideoLayout.itemAt(count).widget()
            self.VideoLayout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)

        try:
            media_src = Phonon.MediaSource(unicode(movPath, 'utf-8'))
        except:
            media_src = Phonon.MediaSource(movPath)

        self.media_obj = Phonon.MediaObject(self)
        self.media_obj.setCurrentSource(media_src)

        self.video_widget = Phonon.VideoWidget(self)
        self.VideoLayout.addWidget(self.video_widget)

        Phonon.createPath(self.media_obj, self.video_widget)
        self.media_obj.tick.connect(self.time_change)
        self.media_obj.setTickInterval(60)
        #self.connect(self.media_obj, SIGNAL('tick(qint64)'), self.time_change)
        self.media_obj.totalTimeChanged.connect(self.total_time_change)
        self.media_obj.play()
        self.timeSlider.setEnabled(True)

        #self.VideoLayout.addWidget(self.videoPlayer)
        #self.videoPlayer.load(media_src)
        self.seekSlider.setMediaObject(self.media_obj)
        #self.videoPlayer.play()

    def play_or_pause(self):
        if Phonon.PlayingState == self.media_obj.state():
            self.media_obj.pause()
        elif Phonon.PausedState == self.media_obj.state():
            self.media_obj.play()

    def slider_value_change(self):
        value = self.timeSlider.value()
        print value
        self.media_obj.seek(value)

    def time_change(self, time):
        if not self.timeSlider.isSliderDown():
            self.timeSlider.setValue(time)

        displayTime = QTime((time / 3600000) % 60,
                            (time / 60000) % 60,
                            (time / 1000) % 60,
                            (time % 1000.0) / 10 )
                            #((time/1000.0) % 60.0 * 10) % 10)
        #timeString = displayTime.toString('hh:mm:ss.zz')
        try:
            timeStringPython = displayTime.toPyTime()
        except:
            timeStringPython = displayTime.toPython()

        timeString = '%02d:%02d:%02d.%02d' %(timeStringPython.hour,
                                       timeStringPython.minute,
                                       timeStringPython.second,
                                       timeStringPython.microsecond/1000 )

        #self.time_lcdNumber.display(timeString)
        self.time_lineEdit.setText(timeString)
        timeFrame = self.timeUnitConversiton(timeString, isFrame=False)
        #self.frame_lcdNumber.display(str(int(timeFrame)))
        self.frame_lineEdit.setText(str(int(timeFrame)))
        #print timeFrame

    def total_time_change(self, time):
        self.timeSlider.setRange(0, time)

    def timeUnitConversiton(self, time, isFrame=True):
        if isFrame:
            TimeS = (int(time) / self.fps) % 60
            TimeM = int((int(time) / self.fps) / 60)
            TimeH = int(TimeM / 60)

            return TimeH, TimeM, TimeS
        else:
            st = str(time).split(":")
            Time = ((int(st[0]) * 60 + int(st[1])) * 60 + float(st[2])) * self.fps

            return Time


    def DoTrim(self):
        movPath = str(self.MovName_lineEdit.text().toUtf8())
        outDir = str(self.OutName_lineEdit.text().toUtf8())
        #fps = float(self.FPS_lineEdit.text())

        StartFrame = self.StartTime_lineEdit.text()
        EndFrame = self.EndTime_lineEdit.text()

        prefix = os.path.splitext(os.path.basename(movPath))[0]

        outPath = os.path.join(outDir, prefix)

        if self.byFrame_radioButton.isChecked():
            outFileName = outPath + "_%d.mov" % (int(StartFrame))

            startTimeH, startTimeM, startTimeS = self.timeUnitConversiton(StartFrame)

            StartTime = "%02d:%02d:%02.02f" % (startTimeH, startTimeM, startTimeS)
            DurationFrame = int(EndFrame) - int(StartFrame)

        else:
            StartTime = str(StartFrame)

            StartTimeToFrame = self.timeUnitConversiton(StartTime, isFrame=False)
            EndFrameToFrame = self.timeUnitConversiton(EndFrame, isFrame=False)

            DurationFrame = EndFrameToFrame - StartTimeToFrame

            outFileName = outPath + "_%d.mov" % (StartTimeToFrame)

        DurationH, DurationM, DurationS = self.timeUnitConversiton(DurationFrame)
        Duration = "%02d:%02d:%02.02f" % (DurationH, DurationM, DurationS)

        #cmd = '%s/trimMov.sh %s %s "%s" "%s" %d' % (currentDir, StartTime, Duration, outFileName, movPath, fps)

        #cmd = [FFMPEG, '-i', movPath, '-ss', StartTime, '-t', Duration ]
        #cmd += ['-vcodec', 'libx264', '-acodec', 'libfaac', '-preset', 'slow', '-profile:v', 'baseline', '-b', '6000 k', '-tune', 'zerolatency' ]
        #cmd += ['-y', outFileName]

        print StartTime, Duration
        cmd = '{0} -i "{1}" -ss {2} -t {3}'.format(FFMPEG, movPath, StartTime, Duration)
        cmd += ' -vcodec libx264 -acodec copy -preset slow -profile:v baseline -b 6000k -tune zerolatency'
        cmd += ' -y "{0}"'.format(outFileName)
        print outFileName

        #subprocess.call(cmd)
        p = subprocess.Popen(cmd, shell=True)
        p.wait()

        QMessageBox.information(self, unicode("알림", 'utf-8'), unicode("성공", 'utf-8'))

class MimeDialog(QDialog):
    def __init__(self,parent):
        QDialog.__init__(self, parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle('MimeTypes')
        listbox = QListWidget(self)
        listbox.setSortingEnabled(True)
        backend = Phonon.BackendCapabilities
        listbox.addItems(backend.availableMimeTypes())
        layout = QVBoxLayout(self)
        layout.addWidget(listbox)
        self.resize(300, 500)

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
    try:
        form.close()
    except:
        pass

    form.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("TrimMovie Player")
    #pp.addLibraryPath("/usr/lib64/kde4/plugins/phonon_backend0")
    win = trimMovWindow()
    win.show()
    sys.exit(app.exec_())
