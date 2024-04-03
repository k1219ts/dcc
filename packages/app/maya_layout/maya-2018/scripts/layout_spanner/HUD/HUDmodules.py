# -*- coding:utf-8 -*-

import maya.cmds as cmds
import maya.mel as mm

def expressionRemover():
    if cmds.objExists('frameCounterUpdate*') == 1:
        allExpr = cmds.ls("frameCounterUpdate*", type = "expression")
        for curExpr in allExpr:
            cmds.delete(curExpr)

def mg_CreateHUD(artistName, sceneName, status, progress):
    mayaCmdsStr = "import maya.cmds as cmds"
    cmds.displayColor('headsUpDisplayLabels', 22)
    cmds.displayColor('headsUpDisplayValues', 16)
    offAllHud()
    mg_removeHUD()
    expressionRemover()

    cmds.headsUpDisplay('artistName', l="Animator     ", allowOverlap=1,
                        b=16,
                        s=0,
                        lfs="small",
                        bs="small",
                        dataFontSize="small",
                        command=("'%s'" % artistName))

    cmds.headsUpDisplay('sceneName', l="Scene Info   ", allowOverlap=1,
                        event="SceneOpened",
                        b=15,
                        s=0,
                        lfs="small",
                        bs="small",
                        dataFontSize="small",
                        command=("'%s'" % sceneName))

    cmds.headsUpDisplay('dateName', l="Date | Time  ", allowOverlap=1,
                        # event="idle",
                        nodeChanges="attributeChange",
                        dataFontSize="small",
                        command=('%s;cmds.date(format=" YYYY / MM / DD   |   hh:mm ")' % mayaCmdsStr),
                        b=14,
                        s=0,
                        lfs="small",
                        bs="small")

    cmds.headsUpDisplay('absframeCounter', l="Duration         ", allowOverlap=1,
                        b=3,
                        s=5,
                        lfs="small",
                        bs="small",
                        dataFontSize="small",
                        command=("%s;endTime_ = cmds.playbackOptions(q=1, max=1);StartTime_ = cmds.playbackOptions(q=1, min=1);AbsTime_ = int(endTime_ - StartTime_ + 1);AbsTime_" % mayaCmdsStr),
                        event="timeChanged")

    cmds.headsUpDisplay('frameCounter', l="Frame         ", allowOverlap=1,
                        b=7,
                        s=4,
                        lfs="small",
                        bs="small",
                        nodeChanges="instanceChange",
                        dataFontSize="small",
                        preset="currentFrame")

    cmds.headsUpDisplay('status', l="Status        ", allowOverlap=1,
                        b=2,
                        s=6,
                        lfs="small",
                        bs="small",
                        dataFontSize="small",
                        command=("'%s  %s %s'" % (status, progress, "%")))

    cmds.headsUpDisplay('camName', l="Camera        ", allowOverlap=1,
                        s=8,
                        b=2,
                        lfs="small",
                        bs="small",
                        preset="cameraNames")

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


def mg_removeHUD():
    expressionRemover()

    if cmds.ls('HUIdelNode*') != []:
        cmds.delete(cmds.ls('HUIdelNode*'))

    if cmds.headsUpDisplay('versionName', exists=1): cmds.headsUpDisplay('versionName', rem=1)

    if cmds.headsUpDisplay('artistName', exists=1): cmds.headsUpDisplay('artistName', rem=1)

    if cmds.headsUpDisplay('ownerName', exists=1): cmds.headsUpDisplay('ownerName', rem=1)

    if cmds.headsUpDisplay('sceneName', exists=1): cmds.headsUpDisplay('sceneName', rem=1)

    if cmds.headsUpDisplay('dateName', exists=1): cmds.headsUpDisplay('dateName', rem=1)

    if cmds.headsUpDisplay('status', exists=1): cmds.headsUpDisplay('status', rem=1)

    if cmds.headsUpDisplay('frameCounter', exists=1): cmds.headsUpDisplay('frameCounter', rem=1)

    if cmds.headsUpDisplay('absframeCounter', exists=1): cmds.headsUpDisplay('absframeCounter', rem=1)

    if cmds.headsUpDisplay('camName', exists=1): cmds.headsUpDisplay('camName', rem=1)

    # add Moon =========================================
    if cmds.optionVar(q='focalLengthVisibility') == 1:
        mm.eval("ToggleFocalLength;")

        # ======================================================================================================================= #


def offAllHud():
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

    if cmds.toggleAxis(q=1, o=1) == 1:
        mm.eval("ToggleOriginAxis;")

    if cmds.viewManip(q=1, v=1) == 1:
        mm.eval("ToggleViewCube;")

    if cmds.optionVar(q='evaluationVisibility'):
        mm.eval("ToggleEvaluationManagerVisibility;")
