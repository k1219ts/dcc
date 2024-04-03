from PySide2 import QtGui, QtCore, QtWidgets
import opentimelineio as otio
from PySide2.QtGui import QFontMetrics
from Define import TIMELINEVIEWSIZE

class EffectItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, item, rect, *args, **kwargs):
        super(EffectItem, self).__init__(rect, *args, **kwargs)
        self.item = item
        self.setFlags(QtWidgets.QGraphicsRectItem.ItemIsSelectable)
        self.init()
        self._set_tooltip()

    def init(self):
        rect = self.rect()
        rect.setY(TIMELINEVIEWSIZE.TRACK_HEIGHT - TIMELINEVIEWSIZE.EFFECT_HEIGHT)
        rect.setHeight(TIMELINEVIEWSIZE.EFFECT_HEIGHT)
        self.setRect(rect)

        dark = QtGui.QColor(0, 0, 0, 150)
        color = QtGui.QColor(255, 255, 255, 200)
        gradient = QtGui.QLinearGradient(
            QtCore.QPointF(0, self.boundingRect().top()),
            QtCore.QPointF(0, self.boundingRect().bottom())
        )
        gradient.setColorAt(0.2, QtCore.Qt.transparent)
        gradient.setColorAt(0.45, color)
        gradient.setColorAt(0.7, QtCore.Qt.transparent)
        gradient.setColorAt(1.0, dark)
        self.setBrush(QtGui.QBrush(gradient))

        pen = self.pen()
        pen.setColor(QtGui.QColor(0, 0, 0, 80))
        pen.setWidth(0)
        self.setPen(pen)

    def _set_tooltip(self):
        tooltips = list()
        for effect in self.item:
            name = effect.name if effect.name else ""
            effect_name = effect.effect_name if effect.effect_name else ""
            tooltips.append("{} {}".format(name, effect_name))
        self.setToolTip("\n".join(tooltips))

    def paint(self, *args, **kwargs):
        newArgs = [args[0], QtWidgets.QStyleOptionGraphicsItem()] + list(args[2:])
        super(EffectItem, self).paint(*newArgs, **kwargs)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemSelectedHasChanged:
            pen = self.pen()
            pen.setColor(
                QtGui.QColor(0, 255, 0, 255) if self.isSelected()
                else QtGui.QColor(0, 0, 0, 80)
            )
            self.setPen(pen)
            self.setZValue(
                self.zValue() + 1 if self.isSelected() else self.zValue() - 1
            )

        return super(EffectItem, self).itemChange(change, value)

class BaseItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, item, timeline_range, *args, **kwargs):
        super(BaseItem, self).__init__(*args, **kwargs)
        self.item = item
        self.timeline_range = timeline_range

        self._otio_sub_items = list()

        self.setFlags(QtWidgets.QGraphicsRectItem.ItemIsSelectable)
        self.setBrush(QtGui.QBrush(QtGui.QColor(180, 180, 180, 255)))

        pen = QtGui.QPen()
        pen.setWidth(0)
        pen.setCosmetic(True)
        self.setPen(pen)

        self.sourceInLabel = QtWidgets.QGraphicsSimpleTextItem(self)
        self.sourceOutLabel = QtWidgets.QGraphicsSimpleTextItem(self)
        self.sourceNameLabel = QtWidgets.QGraphicsSimpleTextItem(self)

        # self._add_markers()
        self._add_effects()
        self._set_labels()
        self._set_tooltip()

        self.xValue = 0.0
        self.currentXOffset = TIMELINEVIEWSIZE.TRACK_NAME_WIDGET_WIDTH

    # def _add_markers(self):
    #     trimmed_range = self.item.trimmed_range()
    #
    #     for m in self.item.markers:
    #         marked_time = m.marked_range.start_time
    #         if not trimmed_range.overlaps(marked_time):
    #             continue
    #
    #         # @TODO: set the marker color if its set from the OTIO object
    #         marker = Marker(m, None)
    #         marker.setY(0.5 * MARKER_SIZE)
    #         marker.setX(
    #             (
    #                     otio.opentime.to_seconds(m.marked_range.start_time) -
    #                     otio.opentime.to_seconds(trimmed_range.start_time)
    #             ) * TIME_MULTIPLIER
    #         )
    #         marker.setParentItem(self)
    #         self._add_otio_sub_item(marker)

    def _add_effects(self):
        if not hasattr(self.item, "effects"):
            return
        if not self.item.effects:
            return
        effect = EffectItem(self.item.effects, self.rect())
        effect.setParentItem(self)
        self._add_otio_sub_item(effect)

    def _add_otio_sub_item(self, item):
        self._otio_sub_items.append(item)

    def get_otio_sub_items(self):
        return self._otio_sub_items

    def _position_labels(self):
        self.sourceNameLabel.setY((TIMELINEVIEWSIZE.TRACK_HEIGHT - self.sourceNameLabel.boundingRect().height()) / 2.0)

    def _set_labels(self):
        self.sourceNameLabel.setText('PLACEHOLDER')
        self._position_labels()

    def _set_tooltip(self):
        self.setToolTip(self.item.name)

    def counteract_zoom(self, zoomLevel=1.0):
        self.setX(self.xValue + self.currentXOffset * zoomLevel)
        for label in (
                self.sourceNameLabel,
                self.sourceInLabel,
                self.sourceOutLabel
        ):
            label.setTransform(QtGui.QTransform.fromScale(zoomLevel, 1.0))

        self_rect = self.boundingRect()
        name_width = self.sourceNameLabel.boundingRect().width() * zoomLevel
        in_width = self.sourceInLabel.boundingRect().width() * zoomLevel
        out_width = self.sourceOutLabel.boundingRect().width() * zoomLevel

        frames_space = in_width + out_width + 3 * TIMELINEVIEWSIZE.LABEL_MARGIN * zoomLevel

        if frames_space > self_rect.width():
            self.sourceInLabel.setVisible(False)
            self.sourceOutLabel.setVisible(False)
        else:
            self.sourceInLabel.setVisible(True)
            self.sourceOutLabel.setVisible(True)

            self.sourceInLabel.setX(TIMELINEVIEWSIZE.LABEL_MARGIN * zoomLevel)

            self.sourceOutLabel.setX(
                self_rect.width() - TIMELINEVIEWSIZE.LABEL_MARGIN * zoomLevel - out_width
            )

        total_width = (name_width + frames_space + TIMELINEVIEWSIZE.LABEL_MARGIN * zoomLevel)
        if total_width > self_rect.width():
            self.sourceNameLabel.setVisible(False)
        else:
            self.sourceNameLabel.setVisible(True)
            self.sourceNameLabel.setX(0.5 * (self_rect.width() - name_width))

class GapItem(BaseItem):
    def __init__(self, *args, **kwargs):
        super(GapItem, self).__init__(*args, **kwargs)
        self.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100, 255)))
        self.sourceNameLabel.setText("GAP")


class ClipItem(BaseItem):
    def __init__(self, *args, **kwargs):
        super(ClipItem, self).__init__(*args, **kwargs)
        self.setBrush(QtGui.QBrush(QtGui.QColor(168, 197, 255, 255)))
        self.sourceNameLabel.setText(self.item.name)

