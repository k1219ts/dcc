from PySide2 import QtGui, QtCore, QtWidgets
import collections
import math
import TrackWidget

RULER_SIZE = 10

class Ruler(QtWidgets.QGraphicsPolygonItem):
    timeSpace = collections.OrderedDict([("media_space", "Media Space"),
                                         ("trimmed_space", "Trimmed Space"),
                                         ("external_space", "External Space")])
    timeSpaceDefault = "external_space"

    def __init__(self, height, timeline, *args, **kwargs):
        poly = QtGui.QPolygonF()
        poly.append(QtCore.QPointF(0.5 * RULER_SIZE, - 0.5 * RULER_SIZE))
        poly.append(QtCore.QPointF(0.5 * RULER_SIZE, 0.5 * RULER_SIZE))
        poly.append(QtCore.QPointF(0, RULER_SIZE))
        poly.append(QtCore.QPointF(0, height))
        poly.append(QtCore.QPointF(0, RULER_SIZE))
        poly.append(QtCore.QPointF(-0.5 * RULER_SIZE, 0.5 * RULER_SIZE))
        poly.append(QtCore.QPointF(-0.5 * RULER_SIZE, -0.5 * RULER_SIZE))
        super(Ruler, self).__init__(poly, *args, **kwargs)

        self.timeline = timeline
        self.setBrush(QtGui.QBrush(QtGui.QColor(50, 255, 20, 255)))

        self.setAcceptHoverEvents(True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)

        self.labels = []
        self_time_space = self.timeSpaceDefault
        self._bounded_data = collections.namedtuple("bounded_data",
                                                    ["f",
                                                     "is_bounded",
                                                     "is_tail",
                                                     "is_head"])
        self.init()

    def init(self):
        # for trackItem in self.timeline.items():
        #     if isinstance(trackItem, TrackWidget.Track):
        #         frameNumberTail = FrameNumber("", position=-1)
        self.updateFrame()