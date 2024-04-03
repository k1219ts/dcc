# -*- coding: utf-8 -*-

import os, sys, subprocess
#from PyQt4 import QtCore,QtGui
from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore
from user_config import GlobalConfig
#-------------------------------------------------------------------------------
# ITEM FOR LEFT SIDE MENU
class CategoryItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super(CategoryItem, self).__init__(parent)
        self.id = 'Category'

        self.setSizeHint(0, QtCore.QSize(30, 32))
        self.itemFont = self.font(0)
        self.itemFont.setBold(True)
        self.itemFont.setPointSize(16)
        self.setFont(0, self.itemFont)

        self.db = ''
        self.coll = ''
        self.searchTerm = {'enabled':True}


class ShowItem(CategoryItem):
    def __init__(self, parent=None):
        super(ShowItem, self).__init__(parent)
        self.id = 'Show'
        self.setSizeHint(0, QtCore.QSize(30, 24))
        itemFont = QtGui.QFont()
        itemFont.setBold(True)
        self.setFont(0, itemFont)
#-------------------------------------------------------------------------------

# BASIC ITEM, CONTAINER FOR CONTAINERWIDGET
class ThumbnailItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None):
        super(ThumbnailItem, self).__init__(parent)
        # ACTUAL ITEM SIZE / NOT IMAGE
        self.setSizeHint(QtCore.QSize(210, 180))
        self.setTextAlignment(QtCore.Qt.AlignHCenter)
        self.itemdata = None

    def setItemData(self, data):
        self.itemdata = data

    def getItemData(self):
        return self.itemdata

    def getTag(self):
        return self.itemdata['tags']

    # SHOULD BE OVERRIDED
    def doubleClick(self, itemWidget):
        # print itemWidget.label.gmovie.state()
        # itemWidget.label.gmovie.setPaused(True)
        # print itemWidget.label.gmovie.state()

        # itemWidget.label.gmovie.stop() # NOT WORKING??
        prvPath = self.itemdata['files']['preview']
        if sys.platform == 'darwin':
            prvPath = u'file://' + prvPath
        #QtGui.QDesktopServices.openUrl(QtCore.QUrl(prvPath))
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(prvPath))
        except:
            subprocess.call(['xdg-open', prvPath])

    # SHOULD BE OVERRIDED
    def click(self, itemWidget):
        print 'thumbnail item click'
        movieLabel = itemWidget.label
        movieLabel.itemClicked()


# ITEM FOR ASSET TEAM WITHOUT PREVIEW GIF
# TODO: ALEMBIC PREVIEW WITH GLWIDGET
# TODO: ZELOS INTEGRATION
class AlembicItem(ThumbnailItem):
    def __init__(self, itemData=None):
        super(AlembicItem, self).__init__()
        if itemData:
            self.setItemData(itemData)
            image = QtGui.QPixmap(itemData['files']['thumbnail'])
            image = image.scaled(200,130,QtCore.Qt.IgnoreAspectRatio,
                                 QtCore.Qt.SmoothTransformation)

            self.setIcon(QtGui.QIcon(image))
            self.setText(' ' + itemData['name'])

    # def doubleClick(self, itemWidget=None):
    #     assetDir = os.path.dirname(self.getItemData()['files']['thumbnail'])
    #     QtGui.QDesktopServices.openUrl(QtCore.QUrl(assetDir))

    def click(self, itemWidget=None):
        print 'AlembicItem click'
        pass


class ImageItem(ThumbnailItem):
    def __init__(self, itemData=None):
        super(ImageItem, self).__init__()
        if itemData:
            self.setItemData(itemData)
            image = QtGui.QPixmap(itemData['files']['thumbnail'])
            image = image.scaled(200,130,QtCore.Qt.IgnoreAspectRatio,
                                 QtCore.Qt.SmoothTransformation)
            self.setIcon(QtGui.QIcon(image))
            self.setText(' '+itemData['name'])

    # def doubleClick(self, itemWidget):
    #     imagePath = self.getItemData()['files']['thumbnail']
    #     # prvPath = self.itemdata['files']['preview']
    #     # if sys.platform == 'darwin':
    #     #     prvPath = u'file://' + prvPath
    #     QtGui.QDesktopServices.openUrl(QtCore.QUrl(imagePath))

    def click(self, itemWidget=None):
        print 'ImageItem click'
        pass


class MovieLabel(QtWidgets.QLabel):
    def __init__(self, parent, gif, thumbnail):
        QtWidgets.QLabel.__init__(self, parent)
        self.setAlignment(QtCore.Qt.AlignCenter)

        self.gifPath = gif
        self.jpgPath = thumbnail
        self.movieFrameCount = 0

        self.jmovie = QtGui.QMovie(self.jpgPath)
        if gif:
            self.gmovie = QtGui.QMovie()

        #self.jmovie.setCacheMode(QtGui.QMovie.CacheAll)
        self.jmovie.setScaledSize(QtCore.QSize(200, 130))

        self.setMovie(self.jmovie)
        self.jmovie.start()

    def itemClicked(self):
        user_config = GlobalConfig.instance().config_dic
        self.gmovie.setSpeed(user_config['play_speed'] * 100)
        if user_config['play'] == 'click':
            # IF FIRST TIME TO MOUSE OVER
            if not (self.gmovie.fileName()):
                self.readGif()

            if self.gmovie.state() == QtGui.QMovie.Paused:
                self.gmovie.setPaused(False)

            elif self.gmovie.state() == QtGui.QMovie.Running:
                self.gmovie.setPaused(True)
            else:
                self.gmovie.start()
            self.toGif()

    def readGif(self):
        if self.gifPath:
            self.gmovie.setFileName(self.gifPath)

            self.movieFrameCount = self.gmovie.frameCount()
            self.gmovie.setCacheMode(QtGui.QMovie.CacheAll)
            self.gmovie.setScaledSize(QtCore.QSize(200, 130))
            self.gmovie.setSpeed(GlobalConfig.instance().config_dic['play_speed']*100)

    def toGif(self):
        if self.movieFrameCount:
            self.parent().horizontalSlider.setMaximum(self.movieFrameCount)
            self.setMovie(self.gmovie)
            #self.gmovie.start()

    def toJpg(self):
        self.gmovie.stop()
        self.setMovie(self.jmovie)
        self.jmovie.start()
        self.parent().horizontalSlider.setMaximum(0)

    def enterEvent(self, event):
        user_config = GlobalConfig.instance().config_dic
        if user_config['play'] == 'over':
            # IF FIRST TIME TO MOUSE OVER
            if not(self.gmovie.fileName()):
                self.readGif()

            if self.gmovie.state() == QtGui.QMovie.Paused:
                self.gmovie.setPaused(False)

            elif self.gmovie.state() == QtGui.QMovie.Running:
                self.gmovie.setPaused(True)
            else:
                self.gmovie.start()
            self.toGif()

    def leaveEvent(self, event):
        user_config = GlobalConfig.instance().config_dic
        if user_config['play'] == 'over':

            # SCROLL PAUSE CONTROL
            print "movie label leave event"
            self.gmovie.setPaused(True)


class ContainerWidget(QtWidgets.QWidget):
    def __init__(self, gif=None, thumbnail=None, parent=None):
        QtWidgets.QWidget.__init__(self, parent=None)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setMargin(0)

        self.label = MovieLabel(self, gif, thumbnail)
        self.verticalLayout.addWidget(self.label)
        if gif:
            self.horizontalSlider = NoWheelSlider(self)
            self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
            self.verticalLayout.addWidget(self.horizontalSlider)
            #self.horizontalSlider.setMaximum(self.label.movieFrameCount)
            self.horizontalSlider.setStyleSheet("""
            QSlider::groove:horizontal {height: 15px; margin: 2px 0; background: rgb(50,50,50);}
            QSlider::handle:horizontal { background: rgb(240,169,32); width: 20px; }
            """)
            self.horizontalSlider.sliderMoved.connect(self.moveFrame)
            self.label.gmovie.frameChanged.connect(self.showFrame)
            self.label.gmovie.stateChanged.connect(self.changeState)
            """
            self.connect(self.horizontalSlider,
                         QtCore.SIGNAL("sliderMoved(int)"),
                         self.moveFrame)
            # ------------------------------------------------------------------------------
            self.connect(self.label.gmovie,
                         QtCore.SIGNAL("frameChanged(int)"),
                         self.showFrame)
            # ------------------------------------------------------------------------------
            self.connect(self.label.gmovie,
                         QtCore.SIGNAL("stateChanged(QMovie::MovieState)"),
                         self.changeState)
            """

        self.titleLabel = QtWidgets.QLabel(self)
        self.titleLabel.setMaximumHeight(22)
        self.titleLabel.setStyleSheet("""
        color: rgb( 255,255,255);
        """)
        self.verticalLayout.addWidget(self.titleLabel)


    def showFrame(self, frame):
        self.horizontalSlider.setSliderPosition(frame)
        if self.label.movieFrameCount-1 == frame:
            # TO GET GIF FINISHED SIGNAL
            # QT QMOVIE CLASS DOES NOT EMIT SIGNAL FOR REPLAY OR FINISH
            self.label.gmovie.stop()

    def changeState(self, state):
        #print state
        user_config = GlobalConfig.instance().config_dic
        if state == 0:
            if user_config['auto_repeat']:
                self.label.gmovie.start()
            else:
                self.label.gmovie.stop()
        self.label.gmovie.setSpeed(user_config['play_speed']*100)


    def moveFrame(self, frame):
        # WHEN USER MOVE SLIDER MANUALLY
        self.label.gmovie.jumpToFrame(frame)

    def enterEvent(self, event):
        #print "container enter"
        user_config = GlobalConfig.instance().config_dic
        if user_config['play'] == 'over':
            self.label.toGif()
        # self.label.movie.start()

    def leaveEvent(self, event):
        #print "container leave"
        user_config = GlobalConfig.instance().config_dic
        if user_config['play'] == 'over':
            pass
            # THIS MAKE MOUSE OVER LEAVE EVENT TO STOP REPLAY WHEN ENTER AGAIN.
            #self.label.toJpg()


class NoWheelSlider(QtWidgets.QSlider):
    def __init__(self, org):
        QtWidgets.QSlider.__init__(self)


    def wheelEvent(self, QWheelEvent):
        self.parent().wheelEvent(QWheelEvent)
        #QWheelEvent.accept()


class Divider(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Divider, self).__init__(parent)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setMargin(0)

        self.label = QtWidgets.QLabel(self)
        self.label.setStyleSheet("""
        background-color: rgb(15,15,15);
        color: rgb(255,255,255)
        """)
        self.label.setAlignment(QtCore.Qt.AlignBottom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.label)
        self.line = QtWidgets.QFrame(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum,
                                           QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setLineWidth(5)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line.setStyleSheet("""
        color: rgb(240, 169, 32)
        """)

        #
        self.verticalLayout.addWidget(self.line)

    def setItemName(self, name):
        self.label.setText(name)


class TagItem(QtWidgets.QListWidgetItem):
    def __init__(self, parent=None, fontSize=12):
        super(TagItem, self).__init__(parent)
        self.setTextAlignment(QtCore.Qt.AlignCenter)

        minimum = 12
        maximum = 30

        itemFont = QtGui.QFont()
        if fontSize < minimum:
            itemFont.setPointSize(minimum)
        else:
            fontSize = (fontSize * 0.1) + 10
            if fontSize > maximum:
                itemFont.setPointSizeF(maximum)
            else:
                itemFont.setPointSizeF(fontSize)
                itemFont.setWeight(fontSize)
        self.setFont(itemFont)
