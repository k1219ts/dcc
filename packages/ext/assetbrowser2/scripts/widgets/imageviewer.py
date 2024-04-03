#-*- coding: utf-8 -*-
from __future__ import division
import os
import sys
from pymodule.Qt import QtCompat
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

import resources_rc
from widgets.ui_imageviewer import Ui_ImageViewer
from libs.utils import replace_path

class ImageViewer(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)

        # ui_path = os.path.dirname(os.path.abspath(__file__))
        # self.ui = QtCompat.loadUi(os.path.join(ui_path, "imageviewer.ui"), self)
        self.ui = Ui_ImageViewer()
        self.ui.setupUi(self)

        self.images = []
        self.current_index = None

        self.image_lbl = QtWidgets.QLabel()
        self.image_lbl.setBackgroundRole(QtGui.QPalette.Base)
        self.image_lbl.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_lbl.setScaledContents(True)

        self.ui.scroll_area.setWidget(self.image_lbl)
        self.ui.scroll_area.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.ui.left_lbl.clicked.connect(self.preview_image)
        self.ui.right_lbl.clicked.connect(self.next_image)

    def add_images_and_loadfile(self, images, current_index):
        self.images = images
        self.current_index = current_index
        self.loadFile(self.images[self.current_index])

    def preview_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.images)-1

        self.loadFile(self.images[self.current_index])

    def next_image(self):
        self.current_index += 1
        if self.current_index > len(self.images)-1:
            self.current_index = 0

        self.loadFile(self.images[self.current_index])

    def initializeImageFileDialog(self, dialog):
        self.first_dialog = True

        if self.first_dialog:
            self.first_dialog = False

    def open(self):
        pass

    def loadFile(self, fileName):
        fileName = replace_path(fileName)
        reader = QtGui.QImageReader(fileName)
        newImage = reader.read()
        if newImage.isNull():
            self.setWindowTitle("Image Viewer")
            print("Cannot load {}.".format(fileName))
        else:
            self.setWindowTitle("Image Viewer - {}".format(fileName))
            self.setImage(newImage)

    def setImage(self, newImage):
        image = newImage
        # if (image.colorSpace().isValid()):
        #     image.convertToColorSpace(QtGui.QColorSpace.SRgb)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.image_lbl.setPixmap(pixmap)

        self.scale_factor = 1.0
        self.ui.scroll_area.setVisible(True)

        self.ui.scroll_area.setWidgetResizable(False)

        scroll_area_width = self.width() - (64*2) - 25
        scroll_area_height = self.height() - 25

        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        is_scaled_mode = False
        if pixmap_width > pixmap_height:
            if scroll_area_width < pixmap_width:
                is_scaled_mode = True
                width = scroll_area_width
                height = int(scroll_area_width * (pixmap_height/pixmap_width))
                self.image_lbl.resize(width, height)
        else:
            if scroll_area_height < pixmap_height:
                is_scaled_mode = True
                width = int(scroll_area_height  * (pixmap_width/pixmap_height))
                height = scroll_area_height
                self.image_lbl.resize(width, height)

        if not is_scaled_mode:
            self.image_lbl.adjustSize()

    def normalSize(self):
        self.image_lbl.adjustSize()
        self.scale_factor = 1.0

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def scaleImage(self, factor):
        self.ui.scroll_area.setWidgetResizable(False)
        self.scale_factor *= factor
        self.image_lbl.resize(
            self.scale_factor * self.image_lbl.pixmap().size()
        )
        self.adjustSCrollBar(self.ui.scroll_area.horizontalScrollBar(), factor)
        self.adjustSCrollBar(self.ui.scroll_area.verticalScrollBar(), factor)

    def adjustSCrollBar(self, scrollBar, factor):
        val = int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep()/2))
        scrollBar.setValue(val)

    def adjustSize(self):
        self.ui.scroll_area.setWidgetResizable(True)
        self.normalSize()
        # self.image_lbl.adjustSize()

    def fitToWindow(self):
        self.ui.scroll_area.setWidgetResizable(False)
        self.image_lbl.adjustSize()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.preview_image()
        elif event.key() == QtCore.Qt.Key_Right:
            self.next_image()
        elif event.key() == QtCore.Qt.Key_Escape:
            pass
        # if event.key() == QtCore.Qt.Key_F:
        #     self.zoomIn()
        # elif event.key() == QtCore.Qt.Key_V:
        #     self.zoomOut()
        # elif event.key() == QtCore.Qt.Key_N:
        #     self.fitToWindow()
        # elif event.key() == QtCore.Qt.Key_M:
        #     self.adjustSize()
        else:
            QtWidgets.QDialog.keyPressEvent(self, event)

    def closeEvent(self, event):
        self.hide()

def main():
    images = [
        "/Users/rndvfx/synctest/tengyart-kSvpTrfhaiU-unsplash.jpg",
        "/Users/rndvfx/synctest/vadim-kaipov-f6jkAE1ZWuY-unsplash.jpg",
        "/Users/rndvfx/synctest/tim-hufner-nAMLTEerpWI-unsplash.jpg",
        "/Users/rndvfx/synctest/sam-moqadam-_kn16nC2vcw-unsplash.jpg",
        "/Users/rndvfx/synctest/girl-with-red-hat-r4A-lJTgXQg-unsplash.jpg"
    ]
    current_index = 0

    app = QtWidgets.QApplication(sys.argv)
    mainView = ImageViewer(None)
    mainView.add_images_and_loadfile(images, current_index)
    mainView.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
