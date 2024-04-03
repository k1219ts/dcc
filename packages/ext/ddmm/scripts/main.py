# -*- coding: utf-8 -*-

"""
**main.py**

**Platform:**
    Linux

**Description:**
    Main Module.

**Others:**

"""

import os
import sys
import commands
import time
from PIL import Image

# Qt Module
from PySide2 import QtWidgets, QtCore, QtGui
from ui.ui_playblast import Ui_DDMM

# STYLESHEET
import qdarkstyle

# custom module
from qsort import qsort1a

FFPLAY = "/opt/ffmpeg/bin/ffplay-dd"
ROOT = os.path.dirname(os.path.realpath(__file__))
CODEC = ['H.264 HQ', 'H.265 HQ', 'Apple ProResProxy']
DISPLAY = ['From Image']
FPS = ['24', '23.976', '29.976', '48', '30', '60']


# 모든 OS에서 한글이 깨지지 않게 처리
def hconv(text):
    return unicode(text, 'utf-8')

class DDMM_GUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DDMM_GUI, self).__init__(parent)
        self.ui = Ui_DDMM()
        self.ui.setupUi(self)

        darkStyleSheet = qdarkstyle.load_stylesheet()
        self.setStyleSheet(darkStyleSheet)

        # init
        self.rootDirectory = os.path.abspath(os.getcwd())

        # print sys.argv

        # if len(sys.argv) == 1:
        #     # 프로그램 실행 시 인수가 있는 경우 처리
        #     self.rootDirectory = os.path.abspath(os.getcwd())
        # elif len(sys.argv) == 2:
        #     # nautilus를 사용할 경우 인수가 2개 필요. 첫번째 인수는 경로, 두번째 인수는 파일로 인식
        #     self.rootDirectory = os.path.abspath(sys.argv[1]) #os.path.join(sys.argv[1], sys.argv[2])
        # else:
        #     self.rootDirectory = os.path.abspath(os.getcwd())

        for arg in sys.argv:
            if '.py' in arg:
                continue
            self.rootDirectory = os.path.abspath(arg)

        self.sourceRootDirectory = self.rootDirectory

        # print self.rootDirectory

        files = os.listdir(self.rootDirectory)
        self.sequenceDict = qsort1a(self.rootDirectory, files)

        # 시퀀스 폴더 바깥에 mov가 생성되도록 경로 수정
        self.movRootDirectory = os.sep.join( self.rootDirectory.split(os.sep)[:-1] )

        sequenceList = self.sequenceDict.keys()
        sequenceList.sort()
        temp = []
        # validate sequence list
        for c in sequenceList:
            sep = c.split(':')
            if len(sep) != 2: continue
            if sep[1].lower() == 'jpg': temp.append(c)
        self.ui.comboBox_sequence.addItems(temp)

        if len(temp) == 0:
            QtWidgets.QMessageBox.warning(self, hconv("사용안내"), hconv("시퀀스 파일이 있는 폴더에서 다시 실행하세요."))
            sys.exit()

        # set
        self.ui.label_logo.setPixmap(QtGui.QPixmap(ROOT+"/resources/dexter_studios_logo_dark.png"))
        self.ui.comboBox_codec.addItems(CODEC)
        self.ui.comboBox_displaySize.addItems(DISPLAY)
        self.ui.comboBox_codec.setCurrentIndex(0)
        self.ui.comboBox_fps.addItems(FPS)
        self.ui.comboBox_fps.setCurrentIndex(0)

        # update
        self.updateData()
        self.changeSequence(0)

        # create signal connection
        self.ui.toolButton_explorer.clicked.connect(self.explorerButton)
        self.ui.toolButton_browse.clicked.connect(self.browseButton)
        self.ui.toolButton_play.clicked.connect(self.playButton)
        self.ui.pushButton_cancel.clicked.connect(self.close)
        self.ui.pushButton_create.clicked.connect(self.videoProcess)
        self.ui.comboBox_displaySize.currentIndexChanged.connect(self.changeDisplaySize)
        self.ui.comboBox_sequence.currentIndexChanged.connect(self.changeSequence)
        self.ui.horizontalSlider_scale.sliderMoved.connect(self.slider_changedcale)
        self.ui.doubleSpinBox_scale.valueChanged.connect(self.spin_changeScale)
        self.ui.comboBox_fps.currentIndexChanged.connect(self.updateData)

    def closeEvent(self, e):
        try:
            self.win.close()
        except:
            pass

    def spin_changeScale(self, val):
        self.ui.horizontalSlider_scale.setValue( val * 10.0 )

    def slider_changedcale(self, val):
        self.ui.doubleSpinBox_scale.setValue(val / 10.0)

    def changeSequence(self, idx):
        fileNameAndExtension = str(self.ui.comboBox_sequence.itemText(idx))

        self.baseName, self.extension = fileNameAndExtension.split(':')
        self.sourceName = self.baseName # special key
        self.startFrame, self.endFrame, self.duration, self.size = self.sequenceDict[fileNameAndExtension]

        # get image size
        im = Image.open(os.path.join(self.sourceRootDirectory, self.baseName+'.'+self.startFrame+'.'+self.extension))
        self.width, self.height = im.size

        if self.width%2==1:
            self.width = self.width+1
        if self.height%2==1:
            self.height = self.height+1

        # 시퀀스 폴더 바깥에 mov가 생성되도록 경로 수정
        self.movieFileName = os.path.join(self.movRootDirectory, self.baseName+'.mov')

        # set text
        self.ui.label_frameRange.setText('%s ~ %s' % (int(self.startFrame), int(self.endFrame)))
        self.ui.label_duration.setText('/ %s frames' % self.duration)
        self.ui.label_size.setText('(%.2f MB)' % (self.size / 1024.0 / 1024.0))
        self.ui.lineEdit_output.setText(self.movieFileName)
        self.ui.lineEdit_width.setText(str(self.width))
        self.ui.lineEdit_height.setText(str(self.height))

    def changeDisplaySize(self, idx):
        if idx == 0: # from image
            self.ui.lineEdit_width.setEnabled(False)
            self.ui.lineEdit_height.setEnabled(False)
        else: # custom
            self.ui.lineEdit_width.setEnabled(True)
            self.ui.lineEdit_height.setEnabled(True)

    def updateData(self):
        self.fps = self.ui.comboBox_fps.currentText()
        self.scale = self.ui.doubleSpinBox_scale.value()

        # set text
        self.ui.label_fps.setText("/ %s fps" % self.fps)

    def explorerButton(self):
        os.system('/usr/bin/nautilus %s &' % self.movRootDirectory)

    def playButton(self):
        if os.path.isfile(self.movieFileName):
            os.system("%s %s &" % (FFPLAY, self.movieFileName))
        else:
            QtWidgets.QMessageBox.warning(self, hconv("안내"), hconv("재생할 파일이 존재하지 않습니다.\n 먼저 Create 버튼을 눌러 만드세요."))

    def browseButton(self):
        fileName = QtWidgets.QFileDialog.getSaveFileName(self, 'save filename', self.movieFileName)
        if fileName[0]:
            self.rootDirectory = os.path.dirname(str(fileName[0]))
            self.baseName = os.path.splitext(os.path.basename(str(fileName[0])))[0]
            self.movieFileName = str(fileName[0])
            # set text
            self.ui.lineEdit_output.setText(self.movieFileName)

    def videoProcess(self):
        self.updateData()

        if not self.ui.comboBox_fps.currentText() in FPS:
            QtWidgets.QMessageBox.question(self, hconv("실행안내"), hconv("fps를 임의로 입력 시\ntimecode가 정상적으로 출력되지 않을 수 있습니다."), QtWidgets.QMessageBox.Ok)

        msg = QtWidgets.QMessageBox.question(self, hconv("실행안내"), hconv("mov 파일을 만드시겠습니까?"), QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if msg != QtWidgets.QMessageBox.Ok:
            return

        codecName = str(self.ui.comboBox_codec.currentText())
        inputImageFile = '{NAME}.%04d.{EXT}'.format(NAME=self.sourceName, EXT=self.extension)
        outputImageSize = "%dx%d" % (self.width*self.scale, self.height*self.scale)

        # cmd  = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg-4.2.0', '--', 'ffmpeg_converter']
        cmd  = ['ffmpeg_converter']

        if 'H.265' in codecName:
            cmd += ['-c', 'h265']
        elif 'H.264' in codecName:
            cmd += ['-c', 'h264']
        elif 'ProRes' in codecName:
            cmd += ['-c', 'proresProxy']

        cmd += ['-r', str(self.fps)]
        cmd += ['-i', self.sourceRootDirectory]
        cmd += ['-o', self.movieFileName]
        cmd += ['-s',  outputImageSize]

        print 'cmds: ', ' '.join(cmd)
        os.system(' '.join(cmd))

        QtWidgets.QMessageBox.information(self, hconv("알림"), hconv("성공적으로 완료되었습니다."))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = DDMM_GUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
