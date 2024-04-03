import sys
from PySide import QtGui, QtCore
from PySide.phonon import Phonon

def setStyleHelper(widget, style):
    widget.setStyle(style)
    widget.setPalette(style.standardPalette())
    for child in widget.children():
        if isinstance(child, QtGui.QWidget):
            setStyleHelper(shild, style)

def change_style(widget, style):
    style = QtGui.QStyleFactory.create(style)
    if style: setStyleHelper(widget,style)


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.mediaObject = Phonon.MediaObject(self)
        self.videoWidget = Phonon.VideoWidget(self)
        Phonon.createPath(self.mediaObject, self.videoWidget)
        self.audioOutput = Phonon.AudioOutput(Phonon.VideoCategory, self)
        Phonon.createPath(self.mediaObject, self.audioOutput)

        self.metaInformationResolver = Phonon.MediaObject(self)
        self.mediaObject.setTickInterval(1000)
        self.videoWidget.setScaleMode(0)

        self.connect(self.mediaObject, QtCore.SIGNAL('tick(qint64)'), self.tick)
        self.connect(self.mediaObject, QtCore.SIGNAL('stateChanged(Phonon::State, Phonon::State)'), self.stateChanged)
        self.connect(self.metaInformationResolver, QtCore.SIGNAL('stateChanged(Phonon::State, Phonon::State)'), self.metaStateChanged)

        self.connect(self.mediaObject, QtCore.SIGNAL('currentSourceChanged(Phonon::MediaSource)'), self.sourceChanged)

        self.setupActions()
        self.setupMenus()
        self.setupUi()
        self.timeLcd.display("00:00")

        self.video_id = self.videoWidget.winId()
        self.source = ''

    def caps(self):
        self.caps = Caps()
        self.caps.show()

    def screenshot(self):
        self.screenshot = Screenshot(self.video_id)
        self.screenshot.show()

    def muting(self):
        if self.audioOutput.isMuted():
            self.audioOutput.setMuted(0)
            self.muteAction.setIcon(QtGui.QIcon("mute_off.png"))
        else:
            self.audioOutput.setMuted(1)
            self.muteAction.setIcon(QtGui.QIcon("mute_on.png"))

    def sizeHint(self):
        return QtCore.QSize(600,450)

    def addFiles(self):
        files = QtGui.QFileDialog.getOpenFileNames(self,
                                                   self.tr("Select video files"),
                                                   QtGui.QDesktopServices.storageLocation(QtGui.QDesktopServices.MusicLocation))

        if files.isEmpty(): return
        for string in files: self.source = Phonon.MediaSource(string)
        if self.source: self.metaInformationResolver.setCurrentSource(self.source)

    def about(self):
        QtGui.QMessageBox.information(self, self.tr("About me"),
                                      self.tr("This simple video player example shows how to use Phonon"))

    def stateChanged(self, newState, oldState):
        if newState == Phonon.ErrorState:
            if self.mediaObject.errorType() == Phonon.FatalError:
                QtGui.QMessageBox.warning(self, self.tr("Fatal Error"), self.mediaObject.errorString())
            else:
                QtGui.QMessageBox.warning(self, self.tr("Error"), self.mediaObject.errorString())

        elif newState == Phonon.PlayingState:
            self.playAction.setEnabled(False)
            self.pauseAction.setEnabled(True)
            self.stopAction.setEnabled(True)

        elif newState == Phonon.StoppedState:
            self.stopAction.setEnabled(False)
            self.playAction.setEnabled(True)
            self.pauseAction.setEnabled(False)
            self.timeLcd.display("00:00")

        elif newState == Phonon.PausedState:
            self.pauseAction.setEnabled(False)
            self.stopAction.setEnabled(True)
            self.playAction.setEnabled(True)

    def tick(self, time):
        displayTime = QtCore.QTime(0, (time/60000) % 60, (time/1000) % 60)
        self.timeLcd.display(displayTime.toString('mm:ss'))

    def sourceChanged(self, source):
        self.timeLcd.display("00:00")

    def metaStateChanged(self, newState, oldState):
        if newState == Phonon.ErrorState:
            QtGui.QMessageBox.warning(self, self.tr("Errir opening files"),
                                      self.metaInformationResolver.errorString())

        self.mediaObject.setCurrentSource(self.metaInformationResolver.currentSource())

        source = self.metaInformationResolver.currentSource()

    def full(self):
        if not self.videoWidget.isFullScreen():
            self.videoWidget.enterFullScreen()
        else:
            self.videoWidget.exitFullScreen()

    def aspect_auto(self): self.videoWidget.setAspectRatio(0)
    def aspect_user(self): self.videoWidget.setAspectRatio(1)
    def aspect_43(self): self.videoWidget.setAspectRatio(2)
    def aspect_169(self): self.videoWidget.setAspectRatio(3)

    def scale_fit(self): self.videoWidget.setScaleMode(0)
    def scale_scale(self): self.videoWidget.setScaleMode(1)

    def change_stylecleanlooks(self): change_style(app, 'cleanlooks')
    def change_styleplastique(self): change_style(app, 'plastique')

    def setupActions(self):
        self.playAction = QtGui.QAction(QtGui.QIcon(""))
