import opentimelineio as otio
from PySide2 import QtWidgets, QtGui, QtCore
from Define import TIMELINEVIEWSIZE
from Items import *


class Track(QtWidgets.QGraphicsRectItem):
    def __init__(self, track, *args, **kwargs):
        super(Track, self).__init__(*args, **kwargs)
        self.track = track
        self.trackItems = []
        self.setBrush(QtGui.QBrush(QtGui.QColor(43, 52, 59, 255)))
        self._populate()

    def _populate(self):
        trackMap = self.track.range_of_all_children()
        for n, item in enumerate(self.track):
            timelineRange = trackMap[item]

            rect = QtCore.QRectF(
                0,
                0,
                otio.opentime.to_seconds(timelineRange.duration) * TIMELINEVIEWSIZE.TIME_MULTIPLIER,
                TIMELINEVIEWSIZE.TRACK_HEIGHT
            )

            if isinstance(item, otio.schema.Clip):
                newItem = ClipItem(item, timelineRange, rect)
            elif isinstance(item, otio.schema.Gap):
                newItem = GapItem(item, timelineRange, rect)
            else:
                print("Warning: could not add item {} to UI.".format(item))
                continue

            newItem.setParentItem(self)
            newItem.xValue = otio.opentime.to_seconds(timelineRange.start_time) * TIMELINEVIEWSIZE.TIME_MULTIPLIER
            newItem.setX(otio.opentime.to_seconds(timelineRange.start_time) * TIMELINEVIEWSIZE.TIME_MULTIPLIER)
            newItem.counteract_zoom()
            self.trackItems.append(newItem)


class TimeSlider(QtWidgets.QGraphicsRectItem):
    def __init__(self, *args, **kwargs):
        super(TimeSlider, self).__init__(*args, **kwargs)
        self.setBrush(QtGui.QBrush(QtGui.QColor(64, 78, 87, 255)))
        pen = QtGui.QPen()
        pen.setWidth(0)
        self.setPen(pen)
        self._ruler = None

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())

        super(TimeSlider, self).mousePressEvent(event)

    def add_ruler(self, ruler):
        self._ruler = ruler

    def counteract_zoom(self, zoomLevel=1.0):
        self.setX(zoomLevel * TIMELINEVIEWSIZE.TRACK_NAME_WIDGET_WIDTH)