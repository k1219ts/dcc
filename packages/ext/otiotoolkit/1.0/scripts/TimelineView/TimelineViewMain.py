#!/usr/bin/env python
import os
import sys
# import opentimelineview as otioViewWidget
from opentimelineview import settings
import opentimelineio as otio

from PySide2 import QtWidgets, QtGui, QtCore
from Timeline import TimelineView, TimelineWidget

class Main(QtWidgets.QWidget):

    selection_changed = QtCore.Signal(otio.core.SerializableObject)
    navigationfilter_changed = QtCore.Signal(int)

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.resize(1500, 1200)
        self.setStyleSheet(settings.VIEW_STYLESHEET)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setMargin(0)

        file = '/prod_nas/__DD_PROD/PRAT2/edit/201130/editorial/haejeok_002_S010_201130.xml'
        movFile = file.replace('.xml', '.mov')
        fileContents = otio.adapters.read_from_file(file)
        #
        # self.timelineWidget = QtWidgets.QWidget(parent=self)
        # self.timelineWidget.setStyleSheet("background-color:black")
        # layout.addWidget(self.timelineWidget)

        if isinstance(fileContents, otio.schema.Timeline):
            self.timeline = fileContents
            self.add_stack(self.timeline, movFile)
            layout.addWidget(self.newStack)

        self.setLayout(layout)

        # if isinstance(fileContents, otio.schema.Timeline):
        #     self.timelineWidget.set_timeline(fileContents)
        #     # self.tracks_widget.setVisible(False)
        # elif isinstance(
        #         fileContents,
        #         otio.schema.SerializableCollection
        # ):
        #     for s in fileContents:
        #         TimelineWidgetItem(s, s.name, self.tracks_widget)
        #     # self.tracks_widget.setVisible(True)
        #     self.timelineWidget.set_timeline(None)

    def add_stack(self, stack, movFile):
        self.newStack = TimelineView(stack, movFile=movFile)
        self.newStack.selection_changed.connect(self.selection_changed)
        self.navigationfilter_changed.connect(self.newStack.navigationfilter_changed)
        self.newStack.frame_all()

    def center(self):
        frame = self.frameGeometry()
        desktop = QtWidgets.QApplication.desktop()
        screen = desktop.screenNumber(
            desktop.cursor().pos()
        )
        centerPoint = desktop.screenGeometry(screen).center()
        frame.moveCenter(centerPoint)
        self.move(frame.topLeft())

    def show(self):
        super(Main, self).show()
        self.newStack.frame_all()

        self.resize(self.size().width(), self.newStack.sceneRect().height())

def main():
    application = QtWidgets.QApplication(sys.argv)

    window = Main()
    # window.load(args.input)

    window.center()
    window.show()
    window.raise_()
    application.exec_()

if __name__ == '__main__':
    main()