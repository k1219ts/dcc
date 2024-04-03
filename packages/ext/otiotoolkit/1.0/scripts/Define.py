from enum import Enum, unique
import dxConfig

# @unique
# class Column2_org(Enum):
#     # Excel Column Define
#     EDIT_ORDER = 0
#     SHOT_NAME = 1
#     CLIP_NAME = 2
#     TC_IN = 3
#     TC_OUT = 4
#     TYPE = 5
#     ISSUE = 6
#     MOV_CUT_IN = 7
#     XML_NAME = 8
#     SCAN_FPS = 9
#     SHOT_DURATION = 10
#     MOV_CUT_DURATION = 11
#     FRAME_IN = 12
#     FRAME_OUT = 13
#     VERSION = 14
#     ORIGINAL_ROOT_FOLDER = 15
#     ORIGINAL_ROOT_PATH = 16
#     RESOLUTION = 17
#     SCAN_DURATION = 18

# @unique
# class Column2(Enum):
#     # Excel Column Define
#     EDIT_ORDER = 0
#     SHOT_NAME = 1
#     CLIP_NAME = 2
#     TC_IN = 3
#     TC_OUT = 4
#     TYPE = 5
#     ISSUE = 6
#     MOV_CUT_IN = 7
#     XML_NAME = 8
#     SCAN_FPS = 9
#     CLIP_DURATION = 10
#     # MOV_CUT_DURATION = 11
#     FRAME_IN = 11
#     FRAME_OUT = 12
#     VERSION = 13
#     ORIGINAL_ROOT_FOLDER = 14
#     ORIGINAL_ROOT_PATH = 15
#     RESOLUTION = 16
#     SCAN_DURATION = 17

@unique
class Column2(Enum):
    # Excel Column Define
    EDIT_ORDER = 0
    SHOT_NAME = 1
    CLIP_NAME = 2
    TC_IN = 3
    TC_OUT = 4
    TYPE = 5
    ISSUE = 6
    EDIT_ISSUE = 7
    MOV_CUT_IN = 8
    XML_NAME = 9
    SCAN_FPS = 10
    CLIP_DURATION = 11
    FRAME_IN = 12
    FRAME_OUT = 13
    VERSION = 14
    ORIGINAL_ROOT_FOLDER = 15
    ORIGINAL_ROOT_PATH = 16
    RESOLUTION = 17
    SCAN_DURATION = 18

@unique
class RescanColumn(Enum):
    # Excel Column Define
    SHOT_NAME = 0
    CLIP_NAME = 1
    RECEIVED_DATE = 2
    RECEIVED_FOLDER = 3
    REQUEST_TC_IN = 4
    REQUEST_TC_OUT = 5
    REQUEST_ISSUE = 6
    REQUEST_DATE = 7
    NOTE = 8
    ORIGINAL_TC_IN = 9
    ORIGINAL_TC_OUT = 10
    ORIGINAL_DURATION = 11
    MOV_CUT_IN = 12

class Colors:
    ERROR   = '\033[91m'
    BOLD    = '\033[1m'
    WARNING = '\033[93m'
    ENDC    = '\033[0m'

class FORMAT:
    DISSOLVE = "Has Dissolve\n"
    RETIME = "Retime {RETIME}%\n"
    SOURCE = "Has Source\n"
    SPEEDRAMPTC = "SpeedRamp {RETIME}% TCInfo : {TC_IN} - {TC_OUT}\n"
    SCALE = "Scale {SCALE}%\n"
    ROTATION = "Rotation {ROTATE}\n"
    CENTER = "Center {CENTER}\n"
    ANCHOR_POINT = "Anchor {ANCHOR}\n"

    PREVIEWOKMSG = '{PROJECT} - {FILE} XLSX SETUP OK'
    ROLLOKMSG = '{PROJECT} - {ROLENAME} OTIO SETUP OK'
    EDITOKMSG = '{PROJECT} - {EDITDATE} edit setup OK'
    PLATEOKMSG = '{PROJECT} - Plate Setup OK'

    EDIT_CHANGED_TITLE_NOTE = '{EDITDATE} edit change\n'
    EDIT_COMPARE_NOTE = '{EDIT_POS} {EDIT_DUR} {EDIT_COMPARE}\n'
    EDIT_DURATION_CHANGE_NOTE = '{BEFORE_FRAME_IN_OUT} ({BEFORE_DURATION}) -> {NEW_FRAME_IN_OUT} ({NEW_DURATION})\n'
    EDIT_RETIME_CHANGED_MSG = 'retime changed {BEFORE_RETIME}% -> {AFTER_RETIME}%\n'

    # EDIT_CHANGED_MSG = '{EDITDATE} edit change\n{EDIT_POS} {EDIT_DUR} {EDIT_COMPARE}\n{BEFORE_FRAME_IN_OUT} ({BEFORE_DURATION}) -> {NEW_FRAME_IN_OUT} ({NEW_DURATION})'

class STRING:
    HORIZONTAL = "horiz"
    VERTICAL = "vert"
    RETIME_IN = 'speedkfin'
    RETIME_OUT = 'speedkfout'
    EDIT_COMPARE_ADD = 'add'
    EDIT_COMPARE_DELETE = 'delete'
    EDIT_COMPARE_CHANGED_IN = 'changed-in'

class CLIPTYPE:
    CLIP = "CLIP"
    SHOTNAME = "SHOTNAME"
    PREVIZ = "PREVIZ"
    NONE = "NONE"

class DEFAULT:
    RETIME = "0"
    SCALE = "100"
    ROTATION = "0"
    # CENTER = str({STRING.HORIZONTAL: u"0", STRING.VERTICAL: u"0"})
    CENTER = str({'horiz': u'0', 'vert': u'0'})
    ANCHOR_POINT = str({STRING.HORIZONTAL: u"0", STRING.VERTICAL: u"0"})
    FALSE = "FALSE"
    TRUE = "TRUE"

roomIdMapper = {
    'ncx': '4i7rawhKXhno9kq7n',
    'emd': 'owMcx5B4sj2sEjnqL',
    'prat2': 'n9c59z5EJCqhMbznJ',
    'wdl': 'x5Cj3PmxjxMmizve5',
    'cdh1': 'qu7dmncvmkK224Zki',
    'cdh': 'qu7dmncvmkK224Zki',
    'tmn': 'gqtK8M69EHwrdrgJv',
    'slc': 'LPWwfnkcpEmRwq7gZ',
    'csp': '8sKxmJd39Xv4nrTDa'
}

class TACTIC:
    IP = dxConfig.getConf("TACTIC_IP")
    API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
    LOGIN = 'cgsup'
    PASSWORD = 'dexter'

class TRACTOR:
    IP = dxConfig.getConf("TRACTOR_CACHE_IP")
    PORT = 80
    SERVICE_KEY = "Editorial"
    MAX_ACTIVE = 10
    PROJECT = "export"
    TIER = "cache"
    TAGS = ""
    PRIORITY = 100

class TIMELINEVIEWSIZE:
    TIME_SLIDER_HEIGHT = 0
    MEDIA_TYPE_SEPARATOR_HEIGHT = 5
    TRACK_HEIGHT = 45
    TRANSITION_HEIGHT = 10
    TIME_MULTIPLIER = 25
    LABEL_MARGIN = 5
    MARKER_SIZE = 10
    EFFECT_HEIGHT = (1.0 / 3.0) * TRACK_HEIGHT
    HIGHLIGHT_WIDTH = 5
    TRACK_NAME_WIDGET_WIDTH = 0.0
    SHORT_NAME_LENGTH = 7
    CURRENT_ZOOM_LEVEL = 1.0
    LIMIT_ZOOM_LEVEL = 0.0
