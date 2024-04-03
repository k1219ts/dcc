# -*- coding: utf-8 -*-
import os

from pymodule.Qt import QtWidgets

from libs.utils import get_posix_file

class ImageQListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        QtWidgets.QListWidget.__init__(self)

    def dragMoveEvent(self, event):
        event.accept()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            for url in mime_data.urls():
                path = url.toLocalFile()
                if path.find(".file/id=") != -1:
                    path = get_posix_file(path).rstrip()
                filename, ext = os.path.splitext(path)
                if ext.lower() in [".png", ".jpg", ".jpeg"]:
                    self.addItem(path)

            event.accept()
        else:
            event.ignore()
