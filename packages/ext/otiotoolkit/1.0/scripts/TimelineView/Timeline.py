import opentimelineio as otio
from PySide2 import QtWidgets, QtGui, QtCore
from Define import TIMELINEVIEWSIZE
import TrackWidget, Ruler
from collections import OrderedDict, namedtuple
from core.FlatternTrack import FlatternTrack

KEY_SYM = {
    QtCore.Qt.Key_Left: QtCore.Qt.Key_Right,
    QtCore.Qt.Key_Right: QtCore.Qt.Key_Left,
    QtCore.Qt.Key_Up: QtCore.Qt.Key_Down,
    QtCore.Qt.Key_Down: QtCore.Qt.Key_Up
}

def get_nav_menu_data():
    _nav_menu = namedtuple(
        "nav_menu",
        ["bitmask", "otioItem", "default", "exclusive"]
    )

    filter_dict = OrderedDict(
        [
            (
                "Clip",
                _nav_menu(0b00000001, TrackWidget.ClipItem, True, False)
            ),
            # (
            #     "Nested Clip",
            #     _nav_menu(0b00000010, track_widgets.NestedItem, True, False)
            # ),
            (
                "Gap",
                _nav_menu(0b00000100, TrackWidget.GapItem, True, False)
            ),
            # (
            #     "Transition",
            #     _nav_menu(0b00001000, track_widgets.TransitionItem, True, False)
            # ),
            # (
            #     "Only with Marker",
            #     _nav_menu(0b00010000, track_widgets.Marker, False, True)
            # ),
            # (
            #     "Only with Effect",
            #     _nav_menu(0b00100000, track_widgets.EffectItem, False, True)
            # ),
            # ("All", nav_menu(0b01000000, "None", False)) @TODO
        ]
    )
    return filter_dict


def get_filters(filter_dict, bitmask):
    filters = list()
    for item in filter_dict.itervalues():
        if bitmask & item.bitmask:
            filters.append(item)
    return filters

def group_filters(bitmask):
    inclusive_filters = list()
    exclusive_filters = list()
    filter_dict = get_nav_menu_data()
    for item in get_filters(filter_dict, bitmask):
        if item.exclusive:
            exclusive_filters.append(item)
        else:
            inclusive_filters.append(item)
    return inclusive_filters, exclusive_filters

class TimelineWidget(QtWidgets.QGraphicsScene):
    def __init__(self, timeline, *args, **kwargs):
        movFile = kwargs['movFile']
        kwargs.pop('movFile')
        super(TimelineWidget, self).__init__(*args, **kwargs)
        self.timeline = timeline
        self.stack = self.timeline.tracks
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(33, 33, 33)))

        self.editTimeline = FlatternTrack(self.timeline, movFile)

        self._adjust_scene_size()
        self._add_tracks()
        # self._add_time_slider()

        self.dataCache = self._cache_tracks()

    def _adjust_scene_size(self):
        sceneRange = self.stack.trimmed_range()

        startTime = otio.opentime.to_seconds(sceneRange.start_time)
        duration = otio.opentime.to_seconds(sceneRange.end_time_exclusive())

        height = (
                TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT +
                (len(self.timeline.video_tracks()) * TIMELINEVIEWSIZE.TRACK_HEIGHT)
        )

        self.setSceneRect(
            startTime * TIMELINEVIEWSIZE.TIME_MULTIPLIER,   # START POS X (LT)
            0,                                              # START POS Y (LT)
            duration * TIMELINEVIEWSIZE.TIME_MULTIPLIER,    # WIDTH (RB)
            height                                          # HEIGHT (RB)
        )
        # print startTime * TIMELINEVIEWSIZE.TIME_MULTIPLIER, duration * TIMELINEVIEWSIZE.TIME_MULTIPLIER

    def _add_track(self, track, yPos):
        sceneRect = self.sceneRect()
        rect = QtCore.QRectF(0, 0, sceneRect.width() * 10, TIMELINEVIEWSIZE.TRACK_HEIGHT)
        newTrack = TrackWidget.Track(track, rect)
        self.addItem(newTrack)
        newTrack.setPos(sceneRect.x(), yPos)

    def _add_tracks(self):
        videoTracksTop = TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT
        videoTracks = []
        otherTracks = []

        if isinstance(self.stack, otio.schema.Stack):
            videoTracks = [
                t for t in self.stack
                if t.kind == otio.schema.TrackKind.Video
            ]
            videoTracks.reverse()

            otherTracks = [
                t for t in self.stack
                if (
                        t.kind not in (
                    otio.schema.TrackKind.Video,
                    otio.schema.TrackKind.Audio
                )
                )
            ]
        else:
            if self.stack.kind == otio.schema.TrackKind.Video:
                videoTracks = [self.stack]
            else:
                otherTracks= [self.stack]

        if otherTracks:
            for t in otherTracks:
                print(
                    "Warning: track named '{}' has nonstandard track type:"
                    " '{}'".format(t.name, t.kind)
                )

            videoTracks.extend(otherTracks)

        for i, track in enumerate(videoTracks):
            self._add_track(track, videoTracksTop + i * TIMELINEVIEWSIZE.TRACK_HEIGHT)

    def _add_time_slider(self):
        sceneRect = self.sceneRect()
        sceneRect.setWidth(sceneRect.width() * 10)
        sceneRect.setHeight(TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT)
        self.timeSlider = TrackWidget.TimeSlider(sceneRect)
        self.addItem(self.timeSlider)
        self.timeSlider.setZValue(float("inf"))

    def get_next_item(self, item, key):
        otioItem = item.item
        nextItem = None
        if key in [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left]:
            head, tail = otioItem.parent().neighbors_of(otioItem)
            nextItem = head if key == QtCore.Qt.Key_Left else tail
        elif key in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            track = item.parentItem()
            if self._data_cache[track][key]:
                nextTrack = self._data_cache[track][key]
            else:
                return item

            atTime = otioItem.trimmed_range_in_parent().start_time

            atTime = min(atTime, self._data_cache[nextTrack]["end_time_inclusive"])
            nextItem = nextTrack.track.child_at_time(atTime)

        if nextItem:
            nextItem = self._data_cache['map_to_widget'][nextItem]

        return nextItem

    def _cache_tracks(self):
        '''
        Create a doubly linked list to navigate from track to track:
            track->get_next_up & track->get_next_up
        "map_to_wodget" : Create a map to retrieve the pyside widget from
        the otio item
        '''
        data_cache = dict()
        tracks = list()
        data_cache["map_to_widget"] = dict()
        for track_item in self.items():
            if not isinstance(track_item, TrackWidget.Track):
                continue
            tracks.append(track_item)
            track_range = track_item.track.available_range()
            data_cache[track_item] = {QtCore.Qt.Key_Up: None,
                                      QtCore.Qt.Key_Down: None,
                                      "end_time_inclusive":
                                      track_range.end_time_inclusive()
                                      }

            for item in track_item.childItems():
                data_cache["map_to_widget"][item.item] = item

        tracks.sort(key=lambda y: y.pos().y())
        index_last_track = len(tracks) - 1
        for i, track_item in enumerate(tracks):
            data_cache[track_item][QtCore.Qt.Key_Up] = \
                tracks[i - 1] if i > 0 else None
            data_cache[track_item][QtCore.Qt.Key_Down] = \
                tracks[i + 1] if i < index_last_track else None

        return data_cache



class TimelineView(QtWidgets.QGraphicsView):

    open_stack = QtCore.Signal(otio.schema.Stack)
    selection_changed = QtCore.Signal(otio.core.SerializableObject)

    def __init__(self, timeline, *args, **kwargs):
        movFile = kwargs['movFile']
        kwargs.pop('movFile')
        super(TimelineView, self).__init__(*args, **kwargs)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setScene(TimelineWidget(timeline, parent=self, movFile=movFile))
        self.setAlignment((QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop))
        self.setStyleSheet('border: 0px;')
        self.scene().selectionChanged.connect(self.parse_selection_change)
        self._navigation_filter = None
        self_last_item_cache = {'key': None, "item": None,
                                "previous_item": None}

    # def drawBackground(self, painter, rect):
    #     brush = QtGui.QBrush()
    #     brush.setStyle(QtCore.Qt.SolidPattern)
    #     # brush.setColor(QtGui.QColor(0, 0, 0))
    #     brush.setColor(QtGui.QColor(255, 255, 255))
    #     painter.fillRect(rect, brush)

    def parse_selection_change(self):
        selection = self.scene().selectedItems()
        if not selection:
            return

        for item in selection:
            # if isinstance(item, Ruler.Ruler):
            #     continue
            self.selection_changed.emit(item.item)
            break

    def mousePressEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        self.setDragMode(
            QtWidgets.QGraphicsView.ScrollHandDrag
            if modifiers == QtCore.Qt.AltModifier
            else QtWidgets.QGraphicsView.NoDrag
        )
        self.setInteractive(not modifiers == QtCore.Qt.AltModifier)

        super(TimelineView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super(TimelineView, self).mouseReleaseEvent(event)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)

    def wheelEvent(self, event):
        scaleBy = 1.0 + float(event.delta()) / 1000
        self.scale(scaleBy, 1)
        zoomLevel = 1.0 / self.matrix().m11()

        if TIMELINEVIEWSIZE.LIMIT_ZOOM_LEVEL <= zoomLevel:
            self.frame_all()
            return

        TIMELINEVIEWSIZE.CURRENT_ZOOM_LEVEL = zoomLevel
        print TIMELINEVIEWSIZE.LIMIT_ZOOM_LEVEL, zoomLevel, TIMELINEVIEWSIZE.CURRENT_ZOOM_LEVEL

        itemsToScale = [
            i for i in self.scene().items()
            if (isinstance(i, (TrackWidget.BaseItem, TrackWidget.TimeSlider)))
        ]

        for item in itemsToScale:
            item.counteract_zoom(zoomLevel)

    def _get_first_item(self):
        newXPos = 0
        newYPos = TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT

        newPosition = QtCore.QPointF(newXPos, newYPos)

        return self.scene().itemAt(newPosition, QtGui.QTransform())
    
    def _get_left_item(self, curSelectedItem):
        curItemXPos = curSelectedItem.pos().x()

        if curSelectedItem.parentItem():
            curTrackYPos = curSelectedItem.parentItem().pos().y()

            newXPos = curItemXPos - 1
            newYPos = curTrackYPos

            if newXPos < 0:
                newXPos = 0
        else:
            newXPos = curItemXPos
            newYPos = curSelectedItem.y()

        newPosition = QtCore.QPointF(newXPos, newYPos)

        return self.scene().itemAt(newPosition, QtGui.QTransform())

    def _get_right_item(self, curSelectedItem):
        curItemXPos = curSelectedItem.pos().x()

        if curSelectedItem.parentItem():
            curTrackYPos = curSelectedItem.parentItem().pos().y()

            newXPos = curItemXPos + curSelectedItem.rect().width()
            newYPos = curTrackYPos
        else:
            newXPos = curItemXPos
            newYPos = curSelectedItem.y()

        newPosition = QtCore.QPointF(newXPos, newYPos)

        return self.scene().itemAt(newPosition, QtGui.QTransform())

    def _get_up_item(self, curSelectedItem):
        curItemXPos = curSelectedItem.pos().x()

        if curSelectedItem.parentItem():
            curTrackYPos = curSelectedItem.parentItem().pos().y()

            newXPos = curItemXPos
            newYPos = curTrackYPos - TIMELINEVIEWSIZE.TRACK_HEIGHT

            newSelectedItem = self.scene().itemAt(
                QtCore.QPointF(
                    newXPos,
                    newYPos
                ),
                QtGui.QTransform()
            )

            if (not newSelectedItem or isinstance(newSelectedItem, otio.schema.Track)):
                newYPos = newYPos - TIMELINEVIEWSIZE.TRANSITION_HEIGHT
        else:
            newXPos = curItemXPos
            newYPos = curSelectedItem.y()

        newPosition = QtCore.QPointF(newXPos, newYPos)

        return self.scene().itemAt(newPosition, QtGui.QTransform())

    def _get_down_item(self, curSelectedItem):
        curItemXPos = curSelectedItem.pos().x()

        if curSelectedItem.parentItem():
            curTrackYPos = curSelectedItem.parentItem().pos().y()
            newXPos = curItemXPos
            newYPos = curTrackYPos + TIMELINEVIEWSIZE.TRACK_HEIGHT

            newSelectedItem = self.scene().itemAt(
                QtCore.QPointF(
                    newXPos,
                    newYPos
                ),
                QtGui.QTransform()
            )

            if (not newSelectedItem or isinstance(newSelectedItem, otio.schema.Track)):
                newYPos = newYPos + TIMELINEVIEWSIZE.TRANSITION_HEIGHT

            if newYPos < TIMELINEVIEWSIZE.TRACK_HEIGHT:
                newYPos = TIMELINEVIEWSIZE.TRACK_HEIGHT
        else:
            newXPos = curItemXPos
            newYPos = (TIMELINEVIEWSIZE.MARKER_SIZE + TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT + 1)
            newYPos = TIMELINEVIEWSIZE.TIME_SLIDER_HEIGHT

        newPosition = QtCore.QPointF(newXPos, newYPos)
        return self.scene().itemAt(newPosition, QtGui.QTransform())

    def _deselect_all_items(self):
        if self.scene().selectedItems():
            for selectedItem in self.scene().selectedItems():
                selectedItem.setSelected(False)

    def _select_new_item(self, newSelectedItem):
        if isinstance(newSelectedItem, QtWidgets.QGraphicsSimpleTextItem):
            newSelectedItem = newSelectedItem.parentItem()

        if (not isinstance(newSelectedItem, TrackWidget.Track) and newSelectedItem):
            self._deselect_all_items()
            newSelectedItem.setSelected(True)
            self.centerOn(newSelectedItem)

    def _get_new_item(self, event, curSelectedItem):
        key = event.key()
        modifier = event.modifiers()
        if not (key in (QtCore.Qt.Key_Left,
                        QtCore.Qt.Key_Right,
                        QtCore.Qt.Key_Up,
                        QtCore.Qt.Key_Down,
                        QtCore.Qt.Key_Return,
                        QtCore.Qt.Key_Enter
                        ) and not (modifier & QtCore.Qt.ControlModifier)):
            return None

        if key in [QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                   QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            if KEY_SYM[key] == self._last_item_cache['key'] and curSelectedItem == self._last_item_cache['item'] and not curSelectedItem == self._last_item_cache['previous_item']:
                newSelectedItem = self._last_item_cache['previous_item']
            else:
                filters = self.get_filters()
                while (not isinstance(curSelectedItem, TrackWidget.BaseItem) and curSelectedItem):
                    curSelectedItem = curSelectedItem.parentItem()
                if not curSelectedItem:
                    return None
                
                newSelectedItem = self.scene().get_next_item_filters(
                    curSelectedItem,
                    key,
                    filters
                )

            self._last_item_cache['item'] = newSelectedItem
            self._last_item_cache['previous_item'] = curSelectedItem
            self._last_item_cache['key'] = key
        elif key in [QtCore.Qt.Key_Return, QtCore.Qt.Key_Return]:
            if isinstance(curSelectedItem, TrackWidget.NestedItem):
                curSelectedItem.keyPressEvenet(event)
                newSelectedItem = None

        return newSelectedItem

    def keyPressEvent(self, event):
        super(TimelineView, self).keyPressEvent(event)
        self.setInteractive(True)

        # selections = [
        #     x for x in self.scene().selectedItems()
        #     if not isinstance(x, Ruler.Ruler)
        # ]

        newSelectedItem = self._get_first_item()

        self._select_new_item(newSelectedItem)

    def _snap(self, event, curSelectedItem):
        key = event.key()
        modifier = event.modifiers()

        if key in (QtCore.Qt.Key_Left,
                   QtCore.Qt.Key_Right) and (modifier & QtCore.Qt.ControlModifier):
            direction = 0
            if key == QtCore.Qt.Key_Left:
                direction = -1.0
            elif key == QtCore.Qt.Key_Right:
                direction = 1.0
            if direction:
                ruler = self.scene().get_ruler()
                ruler.snap(direction=direction,
                           scene_width=self.sceneRect().width())
                self.ensureVisible(ruler)

    def _keyPress_frame_all(self, event):
        key = event.key()
        modifier = event.modifiers()
        if key == QtCore.Qt.Key_F and (modifier & QtCore.Qt.ControlModifier):
            self.frame_all()

    def frame_all(self):
        zoomLevel = 1.0 / self.matrix().m11()
        scaleFactor = self.size().width() / self.sceneRect().width()
        self.scale(scaleFactor * zoomLevel, 1)
        zoomLevel = 1.0 / self.matrix().m11()
        TIMELINEVIEWSIZE.CURRENT_ZOOM_LEVEL = zoomLevel
        TIMELINEVIEWSIZE.LIMIT_ZOOM_LEVEL = zoomLevel
        itemsToScale = [
            i for i in self.scene().items()
            if (isinstance(i, (TrackWidget.BaseItem)))
        ]

        for item in itemsToScale:
            item.counteract_zoom(zoomLevel)

    def navigationfilter_changed(self, bitmask):
        nav_d = namedtuple("navigation_filter", ["inclusive", "exclusive"])
        incl_filter, excl_filter = group_filters(bitmask)
        self._navigation_filter = nav_d(incl_filter, excl_filter)
        self._last_item_cache = {"key": None, "item": None,
                                 "previous_item": None}

    def get_filters(self):
        return self._navigation_filter