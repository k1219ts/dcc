import logging
import os
import tempfile
import capture

from .vendor.Qt import QtCore, QtWidgets, QtGui

import maya.cmds as mc
from . import lib
from . import widgets
reload(widgets)

log = logging.getLogger(__name__)


class Separator(QtWidgets.QFrame):
    """A horizontal line separator looking like a Maya separator"""

    def __init__(self):
        super(Separator, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class SeparatorHeader(QtWidgets.QWidget):
    """A label with a separator line to the right side of it."""

    def __init__(self, header=None, parent=None):
        super(SeparatorHeader, self).__init__(parent=parent)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.setLayout(layout)

        if header is None:
            header = ""

        label = QtWidgets.QLabel(header)
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(8)

        # We disable the label so it becomes Maya's grayed out darker
        # color without overriding the stylesheet so we can rely as much
        # on the styling of Maya as possible.
        label.setEnabled(False)

        label.setFont(font)
        label.setContentsMargins(0, 0, 0, 0)
        label.setSizePolicy(QtWidgets.QSizePolicy.Maximum,
                            QtWidgets.QSizePolicy.Maximum)

        separator = Separator()
        separator.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(label)
        layout.addWidget(separator)

        self.label = label


class ClickLabel(QtWidgets.QLabel):
    """A QLabel that emits a clicked signal when clicked upon."""
    clicked = QtCore.Signal()

    def mouseReleaseEvent(self, event):
        self.clicked.emit()
        return super(ClickLabel, self).mouseReleaseEvent(event)


class PreviewWidget(QtWidgets.QWidget):
    """The playblast image preview widget.

    Upon refresh it will retrieve the options through the function set as
    `options_getter` and make a call to `capture.capture()` for a single
    frame (playblasted) snapshot. The result is displayed as image.

    """

    def __init__(self, options_getter, parent=None):
        super(PreviewWidget, self).__init__(parent=parent)

        self.options_getter = options_getter

        layout = QtWidgets.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        preview = ClickLabel()
        layout.addWidget(preview)

        preview.clicked.connect(self.refresh)

        self.layout = layout
        self.preview = preview

    def refresh(self):
        frame = mc.currentTime(q=1)

        # When playblasting outside of an undo queue it seems that undoing
        # actually triggers a reset to frame 0. As such we sneak in the current
        # time into the undo queue to enforce correct undoing.
        mc.currentTime(frame, update=True)

        with lib.no_undo():
            options = self.options_getter()

            tempdir = tempfile.mkdtemp()

            # override some settings
            options = options.copy()
            options['complete_filename'] = os.path.join(tempdir, "temp.jpg")
            options['width'] = 1280 / 4
            options['height'] = 720 / 4
            options['viewer'] = False
            options['frame'] = frame
            options['off_screen'] = True
            options['format'] = "image"
            options['compression'] = "jpg"
            options['sound'] = None

            fname = capture.capture(**options)

            if not fname:
                log.warning("Preview failed")
                return

            image = QtGui.QPixmap(fname)
            self.preview.setPixmap(image)
            os.remove(fname)


class App(QtWidgets.QWidget):
    """The main capture window.

    This hosts a Preview widget along with a list of OptionPlugin widgets that
    each set specific options for capture. The combination of these options
    is used to perform the resulting capture.
    """

    # Signals
    options_changed = QtCore.Signal(dict)
    playblast_start = QtCore.Signal(dict)  # playblast about to start
    playblast_finished = QtCore.Signal(dict)  # playblast finished
    viewer_start = QtCore.Signal(dict)  # viewer about to start

    WINDOW_OBJECT = "CaptureGUI"
    WINDOW_TITLE = "Capture"

    def __init__(self, parent=None):
        super(App, self).__init__(parent=parent)

        self.setObjectName(self.WINDOW_OBJECT)
        self.setWindowTitle(self.WINDOW_TITLE)

        # Set dialog window flags so the widget can be correctly parented
        # to Maya main window
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Dialog)

        # Makes Maya perform magic which makes the window stay
        # on top in OS X and Linux. As an added bonus, it'll
        # make Maya remember the window position
        self.setProperty("saveWindowPref", True)

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        # Don't allow resizing to force minimum size
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        # Preview
        self.preview = PreviewWidget(self.get_options)
        self.layout.addWidget(self.preview)

        self.option_widgets = list()
        for plugin in [widgets.TimeWidget,
                       widgets.CameraWidget,
                       widgets.ScaleWidget,
                       widgets.CodecWidget,
                       widgets.OptionsWidget]:
            self.add_options_widget(plugin)

        # Buttons
        self.buttonsLayout = QtWidgets.QHBoxLayout()
        self.applyButton = QtWidgets.QPushButton("Capture")
        self.applyCloseButton = QtWidgets.QPushButton("Capture and Close")
        self.buttonsLayout.addWidget(self.applyButton)
        self.buttonsLayout.addWidget(self.applyCloseButton)
        self.applyButton.clicked.connect(self.apply)
        self.applyCloseButton.clicked.connect(self.apply_and_close)
        self.layout.addLayout(self.buttonsLayout)

        # Slots
        self.options_changed.connect(self.preview.refresh)

        self.preview.refresh()

    def add_options_widget(self, plugin):
        """Add and options widget plug-in to the App"""
        widget = plugin()

        if not widget.hidden:
            header = SeparatorHeader(plugin.label)
            self.layout.addWidget(header)
            self.layout.addWidget(widget)

        widget.options_changed.connect(self.on_widget_settings_changed)

        self.option_widgets.append(widget)

    def get_options(self):

        panel = lib.get_active_editor()

        # Get settings from widgets
        options = dict()
        for widget in self.option_widgets:
            options.update(widget.get_options(panel))

        return options

    def apply(self, *args):

        filename = lib._browse(None)

        # Return if playblast was cancelled
        if filename is None:
            return

        options = self.get_options()

        self.playblast_start.emit(options)

        # Perform capture
        options['filename'] = filename
        options['filename'] = lib._capture(options)

        self.playblast_finished.emit(options)
        filename = options['filename']  # get filename after callbacks

        # Show viewer
        if options['viewer']:
            if filename and os.path.exists(filename):
                self.viewer_start.emit(options)
                lib.open_file(filename)
            else:
                raise RuntimeError("Can't open playblast because file "
                                   "doesn't exist: {0}".format(filename))

        return filename

    def apply_and_close(self, *args):
        """Perform a capture and close only when there is a resulting path"""

        filename = self.apply()
        if filename:
            self.close()

    def on_widget_settings_changed(self):
        options = self.get_options()
        self.options_changed.emit(options)
