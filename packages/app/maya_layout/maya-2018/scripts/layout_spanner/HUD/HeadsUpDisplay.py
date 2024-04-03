# -*- coding:utf-8 -*-

import maya.cmds as cmds
import maya.mel as mm
import os
import sys
import getpass
import glob

import xml.etree.ElementTree as xml
from cStringIO import StringIO

# Qt Module
import pysideuic
from PySide2 import QtCore, QtGui
# import shiboken

# ======================================================================================================================= #

# ======================================================================================================================= #

UIFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DDHUD_UI.ui")
VERSION = "v.2.0"
windowObject = "HUD"
dockMode = False

currentpath = os.path.abspath( __file__ )
UIROOT = os.path.dirname(currentpath)
#rcss = open( os.path.join(UIROOT, 'darkorange.css'), 'r' ).read()
css = open(os.path.join(UIROOT, 'resources/css/studioLibrary.css'), 'r').read()
rcss = css.replace("RESOURCE_DIRNAME", UIROOT)
rcss = rcss.replace("BACKGROUND_COLOR", "rgb(40,40,40)")
rcss = rcss.replace("ACCENT_COLOR", "rgb(255,90,40)")
rcss = rcss.replace("ACCENT_COLOR_R", "85")
rcss = rcss.replace("ACCENT_COLOR_G", "85")
rcss = rcss.replace("ACCENT_COLOR_B", "255")

STATUS = [ "Blocking", "Detail", "Facial", "Detail Facial", "Final" ]

def loadUiType( uiFile ):
    parsed = xml.parse( uiFile )
    widget_class = parsed.find('widget').get('class')
    from_class = parsed.find('class').text
    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi( f, o, indent=0 )
        pyc = compile( o.getvalue(), '<string>', 'exec' )
        exec pyc in frame

        form_class = frame['Ui_%s' % from_class]
        base_class = eval('QtGui.%s' % widget_class )
    return form_class, base_class

form, base = loadUiType( UIFILE )

def expressionRemover():
    if cmds.objExists('frameCounterUpdate*') == 1:
        allExpr = cmds.ls("frameCounterUpdate*", type = "expression")
        for curExpr in allExpr:
            cmds.delete(curExpr)

class DDani_HeadsUpDisplayMain(form, base):
    def __init__(self, parent = None):
        super(DDani_HeadsUpDisplayMain, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Animation HUD %s" % VERSION)
        self.setStyleSheet( rcss )
        #self.setObjectName(windowObject)
        self.initGUI()
        self.connectSlot()

    def initGUI(self):
        self.DDHUD_ScenenameField.setText( cmds.file(q=1, namespace=1) )
        self.DDHUD_nameField.setText( getpass.getuser() )
        self.DDHUD_StatuscomboBox.addItems( STATUS )
        self.DDHUD_refreshBtn.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'refreshBtn.png'))))


    def connectSlot(self):
        self.DDHUD_refreshBtn.clicked.connect(self.initGUI)
        self.DDHUD_CreateButton.clicked.connect(self.DoIt)
        self.DDHUD_RemoveButton.clicked.connect(self.mg_removeHUD)

    def DoIt(self):
        artistName = str( self.DDHUD_nameField.text() )
        sceneName = str( self.DDHUD_ScenenameField.text() )
        status = str(self.DDHUD_StatuscomboBox.currentText())
        progress = str(self.DDHUD_ProgresscomboBox.currentText())

        self.mg_CreateHUD(artistName, sceneName, status, progress)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.mg_removeHUD()
            self.close()

    def closeEvent(self, event):
        self.mg_removeHUD()


# ======================================================================================================================= #

    def mg_CreateHUD(self, artistName, sceneName, status, progress ):
        mayaCmdsStr = "import maya.cmds as cmds"
        cmds.displayColor('headsUpDisplayLabels', 22)
        cmds.displayColor('headsUpDisplayValues', 16)
        self.offAllHud()
        self.mg_removeHUD()
        expressionRemover()

        cmds.headsUpDisplay('artistName', l="Animator     ",allowOverlap = 1,
                            b = 2,
                            s = 5,
                            lfs = "large",
                            bs = "small",
                            dataFontSize = "large",
                            command=("'%s'" % artistName) )

        cmds.headsUpDisplay('sceneName', l="Scene Info   ",allowOverlap = 1,
                            event="SceneOpened",
                            b = 4,
                            s = 2,
                            lfs = "large",
                            bs = "small",
                            dataFontSize = "large",
                            command = ("'%s'" % sceneName) )

        cmds.headsUpDisplay('dateName', l = "Date | Time  ",allowOverlap = 1,
                            #event="idle",
                            nodeChanges = "attributeChange",
                            dataFontSize = "large",
                            command = ('%s;cmds.date(format="DD / MM / YYYY   |   hh:mm ")' %mayaCmdsStr),
                            b = 11,
                            s = 0,
                            lfs = "large",
                            bs = "small")

        cmds.headsUpDisplay('absframeCounter', l= "Duration         ",allowOverlap = 1,
                            b = 1,
                            s = 6,
                            lfs = "large",
                            bs = "small",
                            dataFontSize = "large",
                            command = ("%s;endTime_ = cmds.playbackOptions(q=1, max=1);StartTime_ = cmds.playbackOptions(q=1, min=1);AbsTime_ = int(endTime_ - StartTime_ + 1);AbsTime_" %mayaCmdsStr),
                            event = "timeChanged")

        cmds.headsUpDisplay('frameCounter', l= "Frame         ",allowOverlap = 1,
                            b = 7,
                            s = 4,
                            lfs = "large",
                            bs = "small",
                            nodeChanges = "instanceChange",
                            dataFontSize = "large",
                            preset = "currentFrame")

        cmds.headsUpDisplay('status', l="Status        ",allowOverlap = 1,
                            b = 5,
                            s = 7,
                            lfs = "large",
                            bs = "small",
                            dataFontSize = "large",
                            command = ("'%s  %s %s'" % (status, progress, "%")))

        cmds.headsUpDisplay('camName', l="Camera        ", allowOverlap = 1,
                            s=8,
                            b=1,
                            lfs = "large",
                            bs = "small",
                            preset = "cameraNames")

       # add Moon =========================================

        if cmds.optionVar(q='focalLengthVisibility') == 0:
           mm.eval("ToggleFocalLength;")
        if cmds.optionVar(q="symmetryVisibility"):
            mm.eval("ToggleSymmetryDisplay;")
        if cmds.optionVar(q="viewportRendererVisibility"):
            mm.eval("ToggleViewportRenderer;")
        if cmds.optionVar(q="capsLockVisibility"):
            mm.eval("ToggleCapsLockDisplay;")


# ======================================================================================================================= #

    def mg_removeHUD(self, *args):
        expressionRemover()

        if cmds.ls('HUIdelNode*') != []:
            cmds.delete( cmds.ls('HUIdelNode*') )

        if cmds.headsUpDisplay('versionName', exists=1):cmds.headsUpDisplay('versionName', rem=1)

        if cmds.headsUpDisplay('artistName', exists=1):cmds.headsUpDisplay('artistName', rem=1)

        if cmds.headsUpDisplay('ownerName', exists=1):cmds.headsUpDisplay('ownerName', rem=1)

        if cmds.headsUpDisplay('sceneName', exists=1):cmds.headsUpDisplay('sceneName', rem=1)

        if cmds.headsUpDisplay('dateName', exists=1):cmds.headsUpDisplay('dateName', rem=1)

        if cmds.headsUpDisplay('status', exists=1):cmds.headsUpDisplay('status', rem=1)

        if cmds.headsUpDisplay('frameCounter', exists=1):cmds.headsUpDisplay('frameCounter', rem=1)

        if cmds.headsUpDisplay('absframeCounter', exists=1):cmds.headsUpDisplay('absframeCounter', rem=1)

        if cmds.headsUpDisplay('camName', exists=1):cmds.headsUpDisplay('camName', rem=1)

        # add Moon =========================================
        if cmds.optionVar(q='focalLengthVisibility') == 1:
            mm.eval("ToggleFocalLength;")

# ======================================================================================================================= #

    def offAllHud(self, *args):
        """
        buf_ = cmds.headsUpDisplay(lh=1)
        if buf_ != None:
            for hudList in buf_: cmds.headsUpDisplay(hudList, rem=1)
        """
        if cmds.optionVar(q='selectDetailsVisibility') == 1:
            mm.eval("ToggleSelectDetails;")

        if cmds.optionVar(q='objectDetailsVisibility') == 1:
            mm.eval("ToggleObjectDetails;")

        if cmds.optionVar(q='polyCountVisibility') == 1:
            mm.eval("TogglePolyCount;")

        if cmds.optionVar(q='subdDetailsVisibility') == 1:
            mm.eval("ToggleSubdDetails;")

        if cmds.optionVar(q='animationDetailsVisibility') == 1:
            mm.eval("ToggleAnimationDetails;")

        if cmds.optionVar(q='fbikDetailsVisibility') == 1:
            mm.eval("ToggleFbikDetails;")

        if cmds.optionVar(q='frameRateVisibility') == 1:
            mm.eval("ToggleFrameRate;")

        if cmds.optionVar(q='currentFrameVisibility') == 1:
            mm.eval("ToggleCurrentFrame;")

        if cmds.optionVar(q='sceneTimecodeVisibility') == 1:
            mm.eval("ToggleSceneTimecode;")

        if cmds.optionVar(q='currentContainerVisibility') == 1:
            mm.eval("ToggleCurrentContainerHud;")

        if cmds.optionVar(q='cameraNamesVisibility') == 1:
            mm.eval("ToggleCameraNames;")

        # edit Moon ===========================================
        if cmds.optionVar(q='focalLengthVisibility') == 0:
            mm.eval("ToggleFocalLength;")
        # =====================================================

        if cmds.optionVar(q='viewAxisVisibility') == 1:
            mm.eval("ToggleViewAxis;")

        if cmds.toggleAxis(q=1,o=1) == 1:
            mm.eval("ToggleOriginAxis;")

        if cmds.viewManip(q=1,v=1) == 1:
            mm.eval("ToggleViewCube;")

# ======================================================================================================================= #

def DDani_HUD():
    global DDani_hud
    try:
        DDani_hud.close()
    except:
        pass

    DDani_hud = DDani_HeadsUpDisplayMain()

    if sys.platform != "darwin":
        fontPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), "OpenSans-Regular.ttf")
        fontId = QtGui.QFontDatabase.addApplicationFont(fontPath)
        if fontId is not -1:
            family = QtGui.QFontDatabase.applicationFontFamilies(fontId)
            font = QtGui.QFont(family[0])
            font.setPointSize(9)
            DDani_hud.setFont(font)

    if dockMode:
        cmds.dockControl(label=windowObject, area="right", content = DDani_hud(), allowedArea=["left", "right"])
    else:
        DDani_hud.show()
        DDani_hud.resize(290, 120)
