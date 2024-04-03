import os

from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

class ContentItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent, contentInfo = {}):
        QtWidgets.QListWidgetItem.__init__(self, parent)
        self.contentInfo = contentInfo

        self.sizeHint = QtCore.QSize(282, 232)
        self.setSizeHint(self.sizeHint)

        self.contentUI = ContentWidget()
        try:
            if self.contentInfo['category'] == "Animation":
                self.contentUI.setContentName("{0}{1}".format(contentInfo["tag3tier"], contentInfo["fileNum"]))
            elif contentInfo['category'] == "Mocap":
                self.contentUI.setContentName(os.path.basename(contentInfo["files"]['anim']).split('.')[0])
            elif contentInfo['category'] == "Crowd":
                self.contentUI.setContentName(os.path.basename(contentInfo["files"]['ma']).split('.')[0])
        except:
            self.contentUI.setContentName("{0}{1}".format(contentInfo["tag3tier"], contentInfo["fileNum"]))
        self.contentUI.setThumbnail(contentInfo["files"]["preview"])

        dataTypeStr = ""
        if contentInfo['ishik'] == 0:
            dataTypeStr = 'STD'
        elif contentInfo['ishik'] == 1:
            dataTypeStr = 'HIK'
        elif contentInfo['ishik'] == 2:
            if contentInfo['category'] == "Mocap":
                dataTypeStr = "ANI"
            elif contentInfo['category'] == "Crowd":
                dataTypeStr = "ACT"
        else:
            dataTypeStr = ""

        if contentInfo['category'] == "Animation":
            self.contentUI.dataTypeIcon.setStyleSheet('background:rgba(255, 255, 255, 0); color:rgb(255, 0, 0)')
            self.contentUI.dataTypeIcon.setText("A" + dataTypeStr)
        elif contentInfo['category'] == "Mocap":
            self.contentUI.dataTypeIcon.setStyleSheet('background:rgba(255, 255, 255, 0); color:rgb(0, 255, 0)')
            self.contentUI.dataTypeIcon.setText("M" + dataTypeStr)
        elif contentInfo['category'] == "Crowd":
            self.contentUI.dataTypeIcon.setStyleSheet('background:rgba(255, 255, 255, 0); color:rgb(0, 0, 255)')
            self.contentUI.dataTypeIcon.setText("C" + dataTypeStr)
        else:
            self.contentUI.dataTypeIcon.setText("?" + dataTypeStr)

        if contentInfo.has_key('hashTag'):
            # self.contentUI.setToolTip(' '.join(contentInfo['hashTag']))
            toolTipInfo = ' '.join(contentInfo['hashTag'])
            print "# toolTip : ", toolTipInfo
            self.contentUI.previewLabel.setStatusTip(toolTipInfo)
            self.contentUI.previewLabel.setToolTip(toolTipInfo)

        parent.setItemWidget(self, self.contentUI)

    def isHIK(self):
        return self.contentInfo["ishik"]

    def getAnimFilePath(self):
        return self.contentInfo["files"]["anim"]
        
    def getBVHFilePath(self):
        return self.contentInfo["files"]["bvh"]        

    def getActionFilePath(self):
        return self.contentInfo['files']['ma']

    def setContentSize(self, scaleValue):
        self.contentUI.setContentSize(scaleValue)
        self.setSizeHint(self.sizeHint * scaleValue)
#
#         self.setMovie(movie)

class ContentWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        # self.setWindowFlags(QtCore.Qt.Dialog)
        self.setStyleSheet('''
                            QWidget{
                                background-color: black;
                            }
                           ''')

        self.setAttribute(QtCore.Qt.WA_AlwaysShowToolTips)
        print self.toolTipDuration()
        self.setToolTipDuration(1)

        self.widgetSize = QtCore.QSize(282, 237)
        self.resize(self.widgetSize)

        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setMargin(0)
        self.gridLayout.setSpacing(0)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)

        # second floor
        self.curFrameEdit = QtWidgets.QLineEdit(self)
        self.curFrameSize = QtCore.QSize(45, 16)
        self.curFrameEdit.setMaximumSize(self.curFrameSize)
        self.curFrameEdit.setMinimumSize(self.curFrameSize)
        self.curFrameEdit.setReadOnly(True)
        self.horizontalLayout.addWidget(self.curFrameEdit)

        self.maxFrameEdit = QtWidgets.QLineEdit(self)
        self.maxFrameSize = QtCore.QSize(45, 16)
        self.maxFrameEdit.setMaximumSize(self.maxFrameSize)
        self.maxFrameEdit.setMinimumSize(self.maxFrameSize)
        self.maxFrameEdit.setReadOnly(True)

        self.gifSlider = QtWidgets.QSlider(self)
        self.gifSlider.setOrientation(QtCore.Qt.Horizontal)
        self.gifSliderSize = QtCore.QSize(282 - (self.curFrameSize.width() + self.maxFrameSize.width()), 16)
        self.gifSlider.setMinimumSize(self.gifSliderSize)
        self.gifSlider.setMaximumSize(self.gifSliderSize)
        self.gifSlider.sliderMoved.connect(self.sliderMove)
        self.gifSlider.sliderPressed.connect(self.sliderPressed)

        self.horizontalLayout.addWidget(self.gifSlider)
        self.horizontalLayout.addWidget(self.maxFrameEdit)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        # third floor
        self.assetLabel = QtWidgets.QLabel(self)
        self.assetLabelSize = QtCore.QSize(282, 16)
        self.assetLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.assetLabel.setMinimumSize(self.assetLabelSize)
        self.assetLabel.setMaximumSize(self.assetLabelSize)
        self.gridLayout.addWidget(self.assetLabel, 2, 0, 1, 1)

        # first floor
        previewWidget = QtWidgets.QWidget(self)
        self.previewLabel = QtWidgets.QLabel(previewWidget)
        self.previewSize = QtCore.QSize(282, 200)
        self.previewLabel.setMinimumSize(self.previewSize)
        self.previewLabel.setMaximumSize(self.previewSize)

        self.dataTypeIcon = QtWidgets.QLabel(previewWidget)
        self.dataTypeIcon.setGeometry(10, 10, 80, 20)
        font = self.dataTypeIcon.font()
        font.setPixelSize(20)
        font.setBold(True)
        self.dataTypeIcon.setFont(font)

        self.gridLayout.addWidget(previewWidget, 0, 0, 1, 1)

    def setContentSize(self, scaleValue):
        self.resize(self.widgetSize * scaleValue)

        self.previewLabel.setMaximumSize(self.previewSize * scaleValue)
        self.previewLabel.setMinimumSize(self.previewSize * scaleValue)

        self.maxFrameEdit.setMaximumSize(self.maxFrameSize.width() * scaleValue, self.maxFrameSize.height() * scaleValue)
        self.maxFrameEdit.setMinimumSize(self.maxFrameSize.width() * scaleValue, self.maxFrameSize.height() * scaleValue)

        self.curFrameEdit.setMaximumSize(self.curFrameSize.width() * scaleValue, self.curFrameSize.height() * scaleValue)
        self.curFrameEdit.setMinimumSize(self.curFrameSize.width() * scaleValue, self.curFrameSize.height() * scaleValue)

        self.gifSlider.setMinimumSize(self.gifSliderSize.width() * scaleValue, self.gifSliderSize.height() * scaleValue)
        self.gifSlider.setMaximumSize(self.gifSliderSize.width() * scaleValue, self.gifSliderSize.height() * scaleValue)

        self.assetLabel.setMinimumSize(self.assetLabelSize.width() * scaleValue, self.assetLabelSize.height() * scaleValue)
        self.assetLabel.setMaximumSize(self.assetLabelSize.width() * scaleValue, self.assetLabelSize.height() * scaleValue)

        self.gif.setScaledSize(self.previewSize * scaleValue)
        self.gif.start()
        self.gif.stop()

        # self.setSizeHint(size)

    def setContentName(self, name):
        self.assetLabel.setText(name)

    def setThumbnail(self, thumbPath):
        self.previewLabel.setText("")
        self.thumbPath = thumbPath
        if os.path.isfile(thumbPath):
            self.gif = QtGui.QMovie(thumbPath)
            self.gif.frameChanged.connect(self.finishGif)
            self.previewLabel.setMovie(self.gif)
            self.gif.setScaledSize(self.previewSize)
            self.gif.setCacheMode(QtGui.QMovie.CacheAll)
            self.gif.start()
            self.gif.stop()

            self.maxFrameEdit.setText(str(self.gif.frameCount()))
            self.gifSlider.setMaximum(self.gif.frameCount())

    def finishGif(self, frameNumber):
        self.gifSlider.setValue(frameNumber)
        self.curFrameEdit.setText(str(frameNumber + 1))
        if self.gif.frameCount() - 1 == frameNumber:
            self.gif.stop()
            self.gif.start()

    def sliderPressed(self):
        self.gif.jumpToFrame(self.gifSlider.value())
        self.curFrameEdit.setText(str(self.gifSlider.value() + 1))
        self.gif.stop()

    def sliderMove(self, value):
        self.gif.jumpToFrame(value)
        self.curFrameEdit.setText(str(self.gifSlider.value() + 1))
        self.gif.stop()

    def enterEvent(self, event):
        self.gif.start()

    def leaveEvent(self, event):
        self.gif.stop()