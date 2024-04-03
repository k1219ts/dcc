from PySide2 import QtWidgets, QtCore, QtGui

import nuke
import os
import json
import getpass
import time
import nukeToolbar

import Revealfolder
import renderHistory

# import rabbitmq_sender

import stamp
import hot_key
from rendering import sendTractor
import knobChange
# import plateArrange_v02

import compWrite
import branchOut
import frameAppendClip
import linuxBatch_global
import linuxBatch
import randomCurve

import autoBackdrop
import CardtoTrack
import PasteToSelected
import animatedSnap3D
import CAT

import nuke_callbacks
import cameraSearch

import collectFiles
from setup import projectSetting

import ta_def
from setup import compShotSetup

from tactic_client_lib import TacticServerStub

import cam_presets
cam_presets.nodePresetCamera()

import cryptomatte_utilities
cryptomatte_utilities.setup_cryptomatte_ui()


def renderHistoryShow():
    renderHistoryWidget = renderHistory.RenderHistory(QtWidgets.QApplication.activeWindow())
    renderHistoryWidget.setWindowFlags(QtCore.Qt.Dialog)
    renderHistoryWidget.show()

def invenPubShow():
    pubWidget = nuke_inven_pub.PubWidget(QtWidgets.QApplication.activeWindow())
    pubWidget.setWindowFlags(QtCore.Qt.Dialog)
    pubWidget.show()

def camera_search():
    # reload(cameraSearch)
    global mmvSearchDialog
    mmvSearchDialog = cameraSearch.CameraSearch(QtWidgets.QApplication.activeWindow())
    mmvSearchDialog.setWindowFlags(QtCore.Qt.Dialog)
    mmvSearchDialog.show()

def abcCenterPivot():
    for i in nuke.selectedNodes():
        if i.Class() == 'ReadGeo2':
            abcPath = i['file'].value()
            metaPath = abcPath[:-3] + 'json'
            if os.path.exists(metaPath):
                metaInfo = json.loads(open(metaPath, 'r').read())
                i['pivot'].setValue(metaInfo['translate'])


nuke.menu("Nuke").addCommand("DexterDigital/0. Comp Asset Tool", "compAssetTool()", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/1. Matchmove Search (DATABASE)", "camera_search()", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/2. Render History", "renderHistoryShow()", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/3. Revealfolder", "Revealfolder.Revealfolder()", "F6", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/4. sendTractor", "sendTractor.sendtoTractor()", "Ctrl+F12", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/5. Jpeg Write", "compWrite.compWrite('jpg/sRGB')", "F11", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/6. Exr Write", "compWrite.compWrite('exr')", "Ctrl+F11", icon='ddcmp.png')



if nuke.NUKE_VERSION_MAJOR >= 9:
    from W_hotbox import W_hotbox, W_hotboxManager
    import nuke_inven_pub
    nuke.menu("Nuke").addCommand("DexterDigital/6. Nuke Inventory Pub", "invenPubShow()", icon='ddcmp.png')


nuke.menu("Nuke").addCommand("DexterDigital/7. Collect Files", "collectFiles.collectFiles()", icon='ddcmp.png')
nuke.menu("Nuke").addCommand("DexterDigital/8. abc center pivot", "abcCenterPivot()",icon='ddcmp.png')

def create_read_custom(defaulttype="Read"):
    print("create_read_custom")

    # Get the selected node, and the path on it's 'file' knob if it
    # has one, or failing that, it's 'proxy' node, if it has that.
    sel_node = None
    default_dir = None
    try:
        sel_node = nuke.selectedNode()
    except:
        pass
    if ( sel_node is not None ) and ( sel_node != '' ):
        if 'file' in sel_node.knobs():
            default_dir = sel_node['file'].value()
        if (default_dir == None or default_dir == '') and 'proxy' in sel_node.knobs():
            default_dir = sel_node['proxy'].value()

    # Revert default_dir to None if it's empty so that the file browser
    # will do it's default unset behaviour rather than open on an empty path.
    if default_dir == '': default_dir = None

    # Raise the file browser and get path(s) for Read node(s).

    files = nuke.getClipname( "Read File(s)", default=default_dir, multiple=True )
    if files != None:
        maxFiles = nuke.numvalue("preferences.maxPanels")
        n = len(files)
        for f in files:
            isAbc = False
            stripped = nuke.stripFrameRange(f)
            nodeType = defaulttype
            if nukescripts.isAudioFilename( stripped ):
                nodeType = "AudioRead"
            if nukescripts.isGeoFilename( stripped ):
                nodeType = "ReadGeo2"
            if '.abc' in stripped:
                isAbc = True
            if nukescripts.isDeepFilename( stripped ):
                nodeType = "DeepRead"

            # only specify inpanel for the last n nodes. Old panels are kicked out using
            # a deferred delete, so reading large numbers of files can internally build
            # large numbers of active widgets before the deferred deletes occur.
            useInPanel = True
            if (maxFiles != 0 and n > maxFiles):
                useInPanel = False
            n = n-1

            if isAbc:
                nuke.createScenefileBrowser( f, "" )
            else:
                try:
                    if '%V' in f:
                        index = f.find('%0')
                        if index > 0:
                            padding = f[index:index + 4]

                            if (padding[-1] == 'd') and (padding[2].isdigit()):
                                numPadding = '#' * int(padding[2])
                                origFile = f.replace(padding, '#' * int(padding[2])).replace('%V', 'left')

                                range = ''
                                for seqFile in nuke.getFileNameList(os.path.dirname(f)):
                                    if nuke.stripFrameRange(seqFile) == os.path.basename(origFile):
                                        range = seqFile.split(' ')[-1]
                                f+= ' ' + range
                                nuke.createNode( nodeType, "file {"+f+"}", inpanel = useInPanel)

                    else:
                        nuke.createNode( nodeType, "file {"+f+"}", inpanel = useInPanel)

                except RuntimeError as err:
                    nuke.message(err.args[0])

nukescripts.create_read = create_read_custom

import hslthRenderDialog
nukescripts.showRenderDialog = hslthRenderDialog.showRenderDialog

# PDPLAYER FLIPBOOK SETTING
nukescripts.setFlipbookDefaultOption("flipbook", "Pdplayer 64")

nuke.menu('Nodes').menu('Image').addCommand('Write-ImagePlane',"w = nuke.createNode('Write')\nw.setName('Write_ImagePlane1')", icon='Write.png')


# custom callback acript
nuke.addOnScriptSave(nuke_callbacks.saveCallBack)
nuke.addOnScriptLoad(nuke_callbacks.loadCallBack)
nukescripts.addDropDataCallback(nuke_callbacks.dropCallback)

#import dropper
#nukescripts.dropData = dropper.dropData

def textForGrid():
    tn = nuke.createNode('Text')
    tn['knobChanged'].setValue("""
if nuke.thisKnob().name() == 'inputChange':
    import datetime
    dateList = nuke.thisNode().input(0).metadata()['input/ctime'].split(' ')[0].split('-')
    cdate = datetime.date(int(dateList[0]), int(dateList[1]), int(dateList[2]))
    today = datetime.date.today()
    deltaDate = (today - cdate).days
    print(deltaDate)
    if deltaDate == 0:
        nuke.thisNode()['opacity'].setValue(1)
    elif deltaDate <7:
        nuke.thisNode()['opacity'].setValue(0.5)
    else:
        nuke.thisNode()['opacity'].setValue(0.1)
""")

import feedTopic_Nuke
def feedbackTopic():
    # reload(feedTopic_Nuke)
    feedWidget = feedTopic_Nuke.FeedTopic_Nuke(QtWidgets.QApplication.activeWindow())
    feedWidget.setWindowFlags(QtCore.Qt.Dialog)
    feedWidget.show()

import layer_dialog
def layer_write():
    # reload(layer_dialog)
    layerDialog = layer_dialog.LayerDialog(QtWidgets.QApplication.activeWindow())
    layerDialog.setWindowFlags(QtCore.Qt.Dialog)
    layerDialog.show()

def QCGEN():
    # reload(qc_gen)
    qc_gen.qc_gen()

def Snap_To_QC():
    # reload(dd_upload_browser)
    dd_up = dd_upload_browser.DD_Upload(QtWidgets.QApplication.activeWindow())
    dd_up.setWindowFlags(QtCore.Qt.Dialog)
    dd_up.show()

def compAssetTool():
    CATWidget = CAT.CATs(QtWidgets.QApplication.activeWindow())
    CATWidget.setWindowFlags(QtCore.Qt.Dialog)
    CATWidget.show()

def hotKey():
    hot_window = hot_key.Hotkey(QtWidgets.QApplication.activeWindow())
    hot_window.show()

def compShotSetupFunction():
    # reload(compShotSetup)
    widget = compShotSetup.CompShotSetup(QtWidgets.QApplication.activeWindow())
    widget.setWindowFlags(QtCore.Qt.Dialog)
    widget.show()

def compPubWrite():
    fullPath = nuke.value('root.name')
    if fullPath.startswith('/netapp/dexter'):
        fullPath = fullPath.replace('/netapp/dexter', '')
    elif fullPath.startswith('/mach/'):
        fullPath = fullPath.replace('/mach', '')
    steps = fullPath.split(os.path.sep)
    project = steps[2]
    format = 'exr'  # default format 'exr'

    # show _config
    if 'DXCONFIGPATH' in os.environ:
        configFile = os.path.join(os.environ['DXCONFIGPATH'], 'Project.config')

        if os.path.isfile(configFile):
            with open(configFile, 'r') as f:
                configData = json.load(f)
            format = configData['delivery']['format']
            print('show format:', format)

    compWrite.compWrite(format)

def reloadAll():
    for i in nuke.selectedNodes():
        try:
            i['reload'].execute()
        except:
            pass

menubar = nuke.menu("Nuke");
m = menubar.addMenu("File")
m.addCommand("Open Recent Comp/@recent_file7", "nuke.scriptOpen(nuke.recentFile(7))", "#+7")
m.addCommand("Open Recent Comp/@recent_file8", "nuke.scriptOpen(nuke.recentFile(8))", "#+8")
m.addCommand("Open Recent Comp/@recent_file9", "nuke.scriptOpen(nuke.recentFile(9))", "#+9")
m.addCommand("Open Recent Comp/@recent_file10", "nuke.scriptOpen(nuke.recentFile(10))")
m.addCommand("Open Recent Comp/@recent_file11", "nuke.scriptOpen(nuke.recentFile(11))")
m.addCommand("Open Recent Comp/@recent_file12", "nuke.scriptOpen(nuke.recentFile(12))")
m.addCommand("Open Recent Comp/@recent_file13", "nuke.scriptOpen(nuke.recentFile(13))")
m.addCommand("Open Recent Comp/@recent_file14", "nuke.scriptOpen(nuke.recentFile(14))")
m.addCommand("Open Recent Comp/@recent_file15", "nuke.scriptOpen(nuke.recentFile(15))")
m.addCommand("Open Recent Comp/@recent_file16", "nuke.scriptOpen(nuke.recentFile(16))")
m.addCommand("Open Recent Comp/@recent_file17", "nuke.scriptOpen(nuke.recentFile(17))")
m.addCommand("Open Recent Comp/@recent_file18", "nuke.scriptOpen(nuke.recentFile(18))")
m.addCommand("Open Recent Comp/@recent_file19", "nuke.scriptOpen(nuke.recentFile(19))")
m.addCommand("Open Recent Comp/@recent_file20", "nuke.scriptOpen(nuke.recentFile(20))")
m.addCommand("Open Recent Comp/@recent_file21", "nuke.scriptOpen(nuke.recentFile(21))")
m.addCommand("Open Recent Comp/@recent_file22", "nuke.scriptOpen(nuke.recentFile(22))")
m.addCommand("Open Recent Comp/@recent_file23", "nuke.scriptOpen(nuke.recentFile(23))")
m.addCommand("Open Recent Comp/@recent_file24", "nuke.scriptOpen(nuke.recentFile(24))")
m.addCommand("Open Recent Comp/@recent_file25", "nuke.scriptOpen(nuke.recentFile(25))")

# add menu items to existing Nodes toolbar
nodeToolBar = nuke.menu('Nodes')
m = nodeToolBar.addMenu("X_Tools", icon="X_Tools.png")
m.addCommand("X_Distort", "nuke.createNode(\"X_Distort\")", icon="X_Distort.png")
m.addCommand("X_Denoise", "nuke.createNode(\"X_Denoise\")", icon="X_Denoise.png")
m.addCommand("X_Tesla", "nuke.createNode(\"X_Tesla\")", icon="X_Tesla.png")

DexterMenu_CMP = nodeToolBar.addMenu('Dexter CMP', icon='ddcmp.png')

#  ETC GIZMO
DexterMenu_CMP.addCommand('ETC/QualityCheck/QC_HSVSat', 'nuke.loadToolset("' + os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/QC_HSVSat.nk")',icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/QualityCheck/QC_PlayAB', 'nuke.loadToolset("' + os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/QC_PlayAB.nk")',icon='ddcmp.png')
# DexterMenu_CMP.addCommand('ETC/QualityCheck/QC_Netflix', 'nuke.loadToolset("' + os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/QC_Netflix.nk")',icon='ddcmp.png')
import QC_Netflix
DexterMenu_CMP.addCommand('ETC/QualityCheck/QC_Netflix','QC_Netflix.doit()')
DexterMenu_CMP.addCommand('ETC/Auto Backdrop', 'autoBackdrop.autoBackdrop()', 'alt+b', icon='Backdrop.png')
DexterMenu_CMP.addCommand("ETC/b_Erode", "nuke.createNode(\"b_erode\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/blur_Erode", "nuke.createNode(\"blur_Erode\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/Bokeh_Blur", "nuke.createNode(\"Bokeh_Blur\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/CamQuake", "nuke.createNode(\"CamQuake\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/ChromaSmear", "nuke.createNode(\"chromaSmear\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/colorsmear", "nuke.createNode(\"colorsmear\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/Converting", "nuke.createNode(\"Converting\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/CardToTrack", "CardtoTrack.corn3D()", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/DespillMadness2", "nuke.createNode(\"DespillMadness2\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/DespillMadnessv3", "nuke.createNode(\"DespillMadnessv3\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/EdgeFromAlpha", "nuke.createNode(\"EdgeFromAlpha\")", icon='ddcmp.png')
# DexterMenu_CMP.addCommand('ETC/hanui_ocula', 'nuke.createNode(\"hanui_ocula\")',icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/HIT", "nuke.createNode('Hit_V4')", icon='ddcmp.png' )
DexterMenu_CMP.addCommand("ETC/L_AlphaClean", "nuke.createNode(\"L_AlphaClean_v03\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/switchMatte", "nuke.createNode(\"switchMatte\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand("ETC/PointPositionMask", "nuke.createNode(\"PointPositionMask\")", icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/PasteToSelected', 'PasteToSelected.PasteToSelected()', 'Ctrl+Alt+v',icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/frameTC', "nuke.createNode(\"frameTC\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/SprutEmitter', "nuke.createNode(\"SprutEmitter\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/SprutInspect', "nuke.createNode(\"SprutInspect\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/SprutSolver', "nuke.createNode(\"SprutSolver\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/DeepHoldoutSmoother', "nuke.createNode(\"DeepHoldoutSmoother\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/Bloom', "nuke.createNode(\"Bloom\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/GeoPoints', "nuke.createNode(\"GeoPoints\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/bakeLightMatrix', "nuke.createNode(\"bakeLightMatrix\")",icon='ddcmp.png')
DexterMenu_CMP.addCommand('ETC/loopRetime', "nuke.createNode(\"loopRetime\")",icon='ddcmp.png')

# stamp
DexterMenu_CMP.addCommand("STAMP/Stamp_DEFAULT", "stamp.stampDEFAULT()", icon='ddcmp.png' )
DexterMenu_CMP.addCommand("STAMP/Stamp_Netflix", "stamp.stampNetflix()", icon='ddcmp.png' )

nuke.menu("Nodes").addCommand("Time/AppendClip_SJ", 'frameAppendClip.frameAppendClip()', icon='AppendClip.png')

CompTeam = nuke.menu("Nuke").addMenu("CompTeam")
CompTeam.addCommand("Comp Shot Setup", "compShotSetupFunction()", icon='ddcmp.png')
CompTeam.addCommand("Publish Base Script", "renderHistory.publish_base_script()", icon='ddcmp.png')
CompTeam.addSeparator()
CompTeam.addCommand("branchOut", "branchOut.branchout()", "Ctrl+r", icon='ddcmp.png')
import Snippets
CompTeam.addCommand("make_Dot", "Snippets.make_Dot()", "Shift+d", icon='ddcmp.png')
# CompTeam.addCommand("platesArrange_v02", "plateArrange2()", icon='ddcmp.png')
# CompTeam.addCommand("Auto set RelativePath", "pathChange()", "Ctrl+Shift+p", icon='ddcmp.png')
CompTeam.addSeparator()
CompTeam.addCommand("HOT KEY Viewer", "hotKey()", icon='ddcmp.png')
CompTeam.addCommand("Knob Change", "knobChange.changeAll()", "f8", icon='ddcmp.png')
CompTeam.addCommand("Random Curve", "randomCurve.makeNoOp()", icon='ddcmp.png')
CompTeam.addCommand("SendTractor", "sendTractor.sendtoTractor()", "Ctrl+F12", icon='ddcmp.png')
CompTeam.addSeparator()
CompTeam.addCommand("compWrite/JPG/JPG_Linear", "compWrite.compWrite('jpg/sRGB')", "F11")
CompTeam.addCommand("compWrite/JPG/JPG_Log", "compWrite.compWrite('jpg/Cineon')")
CompTeam.addCommand("compWrite/JPG/JPG_sRGB", "compWrite.compWrite('jpg/default(sRGB)')")
CompTeam.addCommand("compWrite/EXR/EXR", "compWrite.compWrite('exr')")
CompTeam.addCommand("compWrite/TIFF/TIFF", "compWrite.compWrite('tiff')")
CompTeam.addCommand("compWrite/TIFF/TIFF_MASK", "compWrite.compWrite('tiff_mask')")
CompTeam.addCommand("compWrite/compPubWrite", "compPubWrite()", "F10")
CompTeam.addCommand("compWrite/RetimeWrite", "compWrite.compRetimeWrite()")
CompTeam.addCommand("compWrite/PRE_COMP", "compWrite.compWrite('precomp')")
CompTeam.addCommand("compWrite/PNG", "compWrite.compWrite('png')")
CompTeam.addCommand("compWrite/MXF", "compWrite.compWrite('mxf')")
CompTeam.addCommand("compWrite/LAYER", "layer_write()")
CompTeam.addSeparator()
CompTeam.addCommand('Batch_Render/Global', 'linuxBatch_global.linuxBatch_global()', 'Ctrl+F11')
CompTeam.addCommand('Batch_Render/Input', 'linuxBatch.linuxBatch()')
CompTeam.addSeparator()
CompTeam.addCommand('QC_GEN', 'QCGEN()', '')
CompTeam.addSeparator()
CompTeam.addCommand('Snap_To_QC', 'Snap_To_QC()', '')
CompTeam.addSeparator()
CompTeam.addCommand('Feedback_Topic', 'feedbackTopic()', '')


def importFromMeta():
    import import_from_meta
    # reload(import_from_meta)
    import_from_meta.import_from_meta()
CompTeam.addCommand('import from meta', 'importFromMeta()', "Ctrl+Shift+o")

import glt_reloadRange
CompTeam.addCommand('Reload Frames', 'glt_reloadRange.glt_reloadRange()', '')
CompTeam.addCommand('Reload all', 'reloadAll()', 'Shift+r')


# PROJECT SETTING MENU
Project_Setting = nuke.menu("Nuke").addMenu("Project_Setting")
Project_Setting.addCommand('Setting Nuke Default', 'projectSetting.settingNukeDefault()')
Project_Setting.addCommand('Setting CDH', 'projectSetting.settingCDH()')
Project_Setting.addCommand('Setting SLC', 'projectSetting.settingSLC()')
Project_Setting.addCommand('Setting 7ESC', 'projectSetting.setting7ESC()')
Project_Setting.addCommand('Setting DIE', 'projectSetting.settingDIE()')
Project_Setting.addCommand('Setting BY', 'projectSetting.settingBY()')
# Project_Setting.addCommand('Setting PRAT2', 'projectSetting.settingPRAT2()')
# Project_Setting.addCommand('Setting EMD', 'projectSetting.settingEMD()')
# Project_Setting.addCommand('Setting TMN', 'projectSetting.settingEMD()')


#### Toolbar menus ####
hslth_toolbar = nuke.toolbar( 'CompToolbar' )
hslth_toolbar.addCommand("compWrite/JPG/JPG_Linear", "compWrite.compWrite('jpg/sRGB')")
hslth_toolbar.addCommand("compWrite/JPG/JPG_Log", "compWrite.compWrite('jpg/Cineon')")
hslth_toolbar.addCommand("compWrite/JPG/JPG_sRGB", "compWrite.compWrite('jpg/default(sRGB)')")
hslth_toolbar.addCommand("compWrite/EXR/EXR", "compWrite.compWrite('exr')")
hslth_toolbar.addCommand("compWrite/TIFF/TIFF", "compWrite.compWrite('tiff')")
hslth_toolbar.addCommand("compWrite/TIFF/TIFF_MASK", "compWrite.compWrite('tiff_mask')")
hslth_toolbar.addCommand("compWrite/DPX/DPX", "compWrite.compWrite('dpx')")
hslth_toolbar.addCommand("compWrite/RetimeWrite", "compWrite.compRetimeWrite()")
hslth_toolbar.addCommand("compWrite/PRE_COMP", "compWrite.compWrite('precomp')")
hslth_toolbar.addCommand("compWrite/PNG", "compWrite.compWrite('png')")
hslth_toolbar.addCommand("compWrite/MXF", "compWrite.compWrite('mxf')")
hslth_toolbar.addCommand("compWrite/LAYER", "layer_write()")


collectMenu = hslth_toolbar.addMenu("Collect_Files")
collectMenu.addCommand('Collect Files', 'collectFiles.collectFiles()')

#Genie
import Opener_Genie

sbMenu = nuke.menu('Nodes').addMenu('sbMenu', icon="compTA.png")
sbMenu.addCommand('gizmo/OpticalGlow_vendor', "nuke.createNode(\"OpticalGlow.gizmo\")")
sbMenu.addCommand('gizmo/retime_warp', "nuke.createNode(\"retime_warp.gizmo\")")
sbMenu.addCommand('gizmo/pMatte', "nuke.createNode(\"P_Matte\")")
sbMenu.addCommand('gizmo/nan_FIX', "nuke.createNode(\"nan_FIX\")")
sbMenu.addCommand('gizmo/exponentialBlur', "nuke.createNode(\"exponentialBlur\")")
sbMenu.addCommand('gizmo/exponentialGlow', "nuke.createNode(\"exponentialGlow\")")
sbMenu.addCommand('gizmo/TX_3DRays', "nuke.createNode(\"TX_3DRays\")")
sbMenu.addCommand('gizmo/L_Grain_v05', "nuke.createNode(\"L_Grain_v05\")")
sbMenu.addCommand('gizmo/HeatWave', 'nuke.createNode(\"HeatWave\")', icon="HeatWave_Icon.png")
sbMenu.addCommand("gizmo/fake_volume", "nuke.createNode(\"fake_volume\")")
sbMenu.addCommand('gizmo/VR_4viewer', "nuke.createNode(\"VR_4_viewer\")",icon='VR_4viewer.png')
sbMenu.addCommand('gizmo/CameraShake', 'nuke.createNode("CameraShake_mv")')
sbMenu.addCommand('gizmo/shake_asset', "nuke.createNode(\"shake_asset\")")
sbMenu.addCommand('gizmo_cmp/FrameHold_set', "nuke.createNode(\"FrameHold_set.gizmo\")")

import RandomSwitch
sbMenu.addCommand('gizmo/RandomSwitch', 'RandomSwitch.createRandomSwitch()')
import createSticky
sbMenu.addCommand('gizmo/shot_name_make', 'createSticky.createSticky()')
import makePath
sbMenu.addCommand('gizmo/makePath', 'makePath.createNode()')
import axisSet
sbMenu.addCommand('gizmo/axisSet','axisSet.createAxisSet()')
import bm_AutoContactSheet
sbMenu.addCommand('gizmo/bm_AutoContactSheet', 'bm_AutoContactSheet.bm_AutoContactSheet()')
#sbMenu.addCommand('gizmo_vendor/PP_Mask_hub_imt', "nuke.createNode(\"PP_Mask_hub.gizmo\")")
#sbMenu.addCommand('gizmo_vendor/relight_normal_imt', "nuke.createNode(\"relight_normal.gizmo\")")
sbMenu.addCommand('gizmo/RainMaker4', "nuke.createNode(\"RainMaker4\")")
sbMenu.addCommand('gizmo/glitch', "nuke.createNode(\"glitch\")")
sbMenu.addCommand('gizmo/gw_despill_V2', "nuke.createNode(\"gw_despill_V2.gizmo\")")

# sbMenu.addCommand("gizmo_cmp/terRetimeWrite", "compWrite.terPlatePub()")
sbMenu.addCommand('gizmo_cmp/onoffMultiply', "nuke.createNode(\"onoffMultiply.gizmo\")")
sbMenu.addCommand('gizmo_cmp/RealHeatDistortion', "nuke.createNode(\"RealHeatDistortion.gizmo\")")
sbMenu.addCommand('gizmo_cmp/T_HeatDistortion1', "nuke.createNode(\"T_HeatDistortion1.gizmo\")")
sbMenu.addCommand('gizmo_cmp/ImagePlane', "nuke.createNode(\"ImagePlane.gizmo\")")
sbMenu.addCommand('gizmo_cmp/fxT_disableNodes4', "nuke.createNode(\"fxT_disableNodes4.gizmo\")")
sbMenu.addCommand('gizmo_cmp/randomMultiply', "nuke.createNode(\"randomMultiply.gizmo\")")
sbMenu.addCommand('gizmo_cmp/randomOnoff', "nuke.createNode(\"randomOnoff.gizmo\")")
sbMenu.addCommand('gizmo_cmp/normal_control', "nuke.createNode(\"normal_control.gizmo\")")
sbMenu.addCommand('gizmo_cmp/Breakdown_Tool', "nuke.createNode(\"Breakdown_Tool.gizmo\")")
sbMenu.addCommand('gizmo_cmp/Sparky', "nuke.createNode(\"Sparky.gizmo\")")
sbMenu.addCommand('gizmo_cmp/ColorAdvance', "nuke.createNode(\"ColorAdvance.gizmo\")","alt+shift+c")
sbMenu.addCommand('gizmo_cmp/WaveMaker', "nuke.createNode(\"WaveMaker.gizmo\")")
sbMenu.addCommand('gizmo_cmp/RotoPreview', "nuke.createNode(\"RotoPreview.gizmo\")","alt+o")
sbMenu.addCommand('gizmo_cmp/expoBloom', "nuke.createNode(\"expoBloom.gizmo\")")
sbMenu.addCommand('gizmo_cmp/Glitch_monitor', "nuke.createNode(\"Glitch_monitor.gizmo\")")
sbMenu.addCommand('gizmo_cmp/DasGrain', 'nuke.loadToolset("' + os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/DasGrain.nk")')
sbMenu.addCommand('gizmo_cmp/Air_mist', 'nuke.loadToolset("' + os.environ['REZ_NUKE_SCRIPTS_BASE'] + '/scripts/toolset/Air_mist.nk")')
import RandomSwitch
sbMenu.addCommand('gizmo_cmp/RandomSwitch', 'RandomSwitch.createRandomSwitch()')
import createSticky
sbMenu.addCommand('gizmo_cmp/shot_name_make', 'createSticky.createSticky()')
import makePath
sbMenu.addCommand('gizmo_cmp/makePath', 'makePath.createNode()')
import axisSet
sbMenu.addCommand('gizmo_cmp/axisSet','axisSet.createAxisSet()')
import bm_AutoContactSheet
sbMenu.addCommand('gizmo_cmp/bm_AutoContactSheet', 'bm_AutoContactSheet.bm_AutoContactSheet()')

#gizmo_sup
sbMenu.addCommand('gizmo_sup/Ag_AutoGrading', "nuke.createNode('Ag_AutoGrading')")
sbMenu.addCommand('gizmo_sup/mmColorTarget', "nuke.createNode('mmColorTarget')",icon='mmColorTarget.png')

sbMenu.addCommand('script/ID', 'ta_def.findId()')
import csvToDpx
sbMenu.addCommand('script/csv_to_renderDpx', 'csvToDpx.lastNodeContact()')
import csv_importer
sbMenu.addCommand('script/csv_importer', 'csv_importer.csv_importer()')
import csv_exporter
sbMenu.addCommand('script/csv_exporter', 'csv_exporter.csvFileMaker()')
import geoConstrain
sbMenu.addCommand('script/targetGeo', 'geoConstrain.constrain()')
import axisTo2D
sbMenu.addCommand('script/axisTo2D', 'axisTo2D.sb_axisTo2D()')
import axisToCard
sbMenu.addCommand('script/axisToCard', 'axisToCard.axisToCard()')

# W_SclaeTree
import W_scaleTree
nuke.menu('Nuke').addCommand('Edit/Node/W_scaleTree', 'W_scaleTree.scaleTreeFloatingPanel()', 'alt+`')


import bm_NodeComment
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_NodeComment', 'bm_NodeComment.bm_NodeComment()', 'alt+ctrl+c')
import bm_NodeSandwich
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_NodeSandwich/Premult_Sandwich', 'bm_NodeSandwich.bm_NodeSandwich("Unpremult", "Premult")', "ctrl+shift+p")
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_NodeSandwich/Log Sandwich', 'bm_NodeSandwich.bm_NodeSandwich("Log2Lin", "Log2Lin")', "ctrl+shift+l")
import bm_OperationSwitcher
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_OperationSwitcher', 'bm_OperationSwitcher.bm_OperationSwitcher()', 'ctrl+alt+s')
import bm_QuickKeys
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Quick Keys/On', 'bm_QuickKeys.bm_QuickKeys("on")', "alt+,")
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Quick Keys/Off', 'bm_QuickKeys.bm_QuickKeys("off")', "alt+.")
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Quick Keys/Off On Off', 'bm_QuickKeys.bm_QuickKeys("offonoff")', "ctrl+alt+,")
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Quick Keys/On Off On', 'bm_QuickKeys.bm_QuickKeys("onoffon")', "ctrl+alt+.")
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Quick Keys/Range', 'bm_QuickKeys.bm_QuickKeys("custom")', "alt+/")
import bm_SmartMerge
nuke.menu('Nuke').addCommand('Edit/Shortcuts/bm_Smart Merge', 'bm_SmartMerge.bm_SmartMerge()', 'ctrl+m')
import bm_ViewerToggle
nuke.menu("Nuke").addCommand("Edit/Shortcuts/bm_ViewerToggle", 'bm_ViewerToggle.bm_ViewerToggle()', "alt+ctrl+q")
import TrackerToRoto
nuke.menu("Nuke").addCommand("Edit/TrackerToRoto", 'TrackerToRoto.TrackerToRoto()', "alt+shift+v")

#SearchReplacePanel
import SearchReplacePanel
def addSRPanel():
    myPanel = SearchReplacePanel.SearchReplacePanel()
    return myPanel.addToPane()

nuke.menu('Pane').addCommand('SearchReplace', addSRPanel)
nukescripts.registerPanel('com.ohufx.SearchReplace', addSRPanel)

#Axis animated tracking
try:
    m = nuke.menu('Axis').findItem('Snap')
    m.addSeparator()
    m.addCommand('Match position - ANIMATED', 'animatedSnap3D.translateThisNodeToPointsAnimated()')
    m.addCommand('Match position, orientation - ANIMATED', 'animatedSnap3D.translateRotateThisNodeToPointsAnimated()')
    m.addCommand('Match position, orientation, scale - ANIMATED', 'animatedSnap3D.translateRotateScaleThisNodeToPointsAnimated()')
except:
    pass

nuke.menu("Nodes").addCommand("Draw/TextForGrid", 'textForGrid()', icon='Text.png')
nuke.menu("Nodes").addCommand("Draw/Text_old", 'nuke.createNode("Text")', icon='Text.png')
nuke.menu("Nuke").addCommand("Render/User/SmartRead", "nuke.tcl('SmartRead')","alt+r")

nuke.menu('Nodes').addCommand("Time/FrameHold", 'nuke.createNode("FrameHold")',"shift+h", icon='FrameHold.png')
nuke.menu('Nodes').menu('Time/FrameHold').setScript('fh = nuke.createNode("FrameHold")\nfh["first_frame"].setValue(nuke.frame())\nfh.setName("FrameHold_BOCK1", uncollide=True)')


# for Matchmove Team -----------------------------------------------------------
import retime_mmv
import timeoffsetTotimewarp
import panzoom_expression

DexterMenu_MMV = nuke.menu('Nodes').addMenu('Dexter MMV', icon='ddmmv.png')
DexterMenu_MMV.addCommand("3DE4/LD_3DE4_Anamorphic_Standard_Degree_4", "nuke.createNode('LD_3DE4_Anamorphic_Standard_Degree_4')")
DexterMenu_MMV.addCommand("3DE4/LD_3DE4_Anamorphic_Rescaled_Degree_4", "nuke.createNode('LD_3DE4_Anamorphic_Rescaled_Degree_4')")
DexterMenu_MMV.addCommand("3DE4/LD_3DE4_Anamorphic_Degree_6", "nuke.createNode('LD_3DE4_Anamorphic_Degree_6')")
DexterMenu_MMV.addCommand("3DE4/LD_3DE4_Radial_Standard_Degree_4", "nuke.createNode('LD_3DE4_Radial_Standard_Degree_4')")
DexterMenu_MMV.addCommand("3DE4/LD_3DE4_Radial_Fisheye_Degree_8", "nuke.createNode('LD_3DE4_Radial_Fisheye_Degree_8')")
DexterMenu_MMV.addCommand("3DE4/LD_3DE_Classic_LD_Model", "nuke.createNode('LD_3DE_Classic_LD_Model')")

DexterMenu_MMV.addCommand("Retime_MMV", "retime_mmv.retimeDistort()")
DexterMenu_MMV.addCommand("TimeOffset_To_TimeWarp", "timeoffsetTotimewarp.convert()")
DexterMenu_MMV.addCommand("2D_PANZOOM_EXPRESSION", "panzoom_expression.panzoom_expression()")



# ------------------------------------------------------------------------------
# for Lighting and Render Team

def read_setUp():
    selNode = [n.name() for n in nuke.selectedNodes()]
    k = []
    p = {}
    n = []
    for s in selNode:
        k.append(s)
        node = nuke.toNode(s)

        fileName = node.knob('file').getValue()
        fileNameSplit = fileName.split('.')
        frameRange = str(node.frameRange()).split('-')
        if fileNameSplit[1] == 'deep':
            create = nuke.createNode('DeepRead')
            create['file'].setValue(fileName)
            create.knob('first').setValue(int(frameRange[0]))
            create.knob('last').setValue(int(frameRange[1]))
            create.knob('origfirst').setValue(int(frameRange[0]))
            create.knob('origlast').setValue(int(frameRange[1]))

            delNode = nuke.toNode(s)
            nuke.delete(delNode)
            nodeName = create.name()
            k.remove(s)
            k.append(nodeName)
        if fileNameSplit[-1] == 'xml':
            k.remove(s)
            delNode = nuke.toNode(s)
            nuke.delete(delNode)
        if fileNameSplit[-1] == 'json':
            k.remove(s)
            delNode = nuke.toNode(s)
            nuke.delete(delNode)
    file = nuke.toNode(k[0]).knob('file').getValue()
    layer = file.split('/')[-3]
    version = file.split('/')[-2]
    for j in k:
        rp = nuke.toNode(j).knob('file').getValue()
        rps = rp.split('.')[-3]
        p[rps] = j

        keys = list(p.keys())
        keys.sort(reverse=True)

    for m in keys:
        n.append(p[m])

    for i in n:
        nuke.toNode(i).knob('selected').setValue(True)
        node = nuke.toNode(i)
        node.autoplace()
    autoBackdrop.autoBackdrop().knob('label').setValue(layer+"("+version+")")


def changeVersion():
	txt = nuke.getInput('Change Version','v000')
	if txt:
	    selNode = [n.name() for n in nuke.selectedNodes()]
	    for s in selNode:
    		name = nuke.toNode(s).knob('file').getValue()
    		v = txt
    		spl = name.split('/')
    		spl[-2] = v
    		spl = '/'.join(spl)
    		nuke.toNode(s).knob('file').setValue(spl)

def ACES_Workflow():
    nuke.root().knob('colorManagement').setValue('OCIO')
    nuke.root().knob('OCIO_config').setValue('custom')
    nuke.root().knob('customOCIOConfigPath').setValue(os.environ['REZ_OCIO_CONFIGS_BASE'] + '/nuke_config.ocio') #1.3

    # Setup Viewer Process
    allViewer = nuke.allNodes('Viewer')
    if allViewer:
        for i in allViewer:
            i['viewerProcess'].setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    else:
        nuke.createNode('Viewer').knob('viewerProcess').setValue("ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")

    nuke.knobDefault("Viewer.viewerProcess", "ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)")
    allRead = nuke.allNodes('Read')
    for i in allRead:
        if 'plates' in i['file'].value():
            i['colorspace'].setValue('ACES2065-1')

import stamp_info
def stampLNR():
    stampNode = nuke.createNode('LNR_Stamp_v01')
    stampNode['Project_name'].setValue(stamp_info.getPrjName())
    stampNode['Shotname'].setValue(stamp_info.getShotNameLNR())
    stampNode['Artist_name'].setValue(stamp_info.getUserName())
    stampNode['LetterBox'].setValue(0)

    return stampNode

def turntableTemplateLNR():
    templateFile = os.path.join(os.environ['REZ_NUKE_SCRIPTS_BASE'], 'scripts', 'template', 'LNR', 'Template_v002.nk')
    nuke.scriptReadFile(templateFile)

nodeToolBar = nuke.menu('Nodes')
DD_LNR = nodeToolBar.addMenu('Dexter LNR', icon='ddlnr.png')
# addCommand("Draw/Text_old", 'nuke.createNode("Text")', icon='Text.png')
DD_LNR.addCommand("0.General/ddee_TorchLight", 'nuke.createNode("ddee_TorchLight")')
DD_LNR.addCommand("0.General/EnvRelight", 'nuke.createNode("EnvRelight")', icon='EnvRelight.png')
DD_LNR.addCommand("0.General/P_Matte", 'nuke.createNode("P_Matte")')
DD_LNR.addCommand("0.General/Position_to_Mask", 'nuke.createNode("Position_to_Mask")')
# DD_LNR.addCommand("0.General/deep_to_depth", 'nuke.createNode("deep_to_depth")')
DD_LNR.addCommand("0.General/deep_vec_connect", 'nuke.createNode("deep_vec_connect")')
DD_LNR.addCommand("0.General/hdr_edit_box", 'nuke.createNode("hdr_edit_box")')
DD_LNR.addCommand("0.General/hdr_keyer_box", 'nuke.createNode("hdr_keyer_box")')
DD_LNR.addCommand("0.General/deep_vec_connect", 'nuke.createNode("deep_vec_connect")')
DD_LNR.addCommand("0.General/Torchlight_v02", 'nuke.createNode("Torchlight_v02")')
DD_LNR.addCommand("0.General/Cryptomatte_Select", 'nuke.createNode("Cryptomatte_Select")')
DD_LNR.addCommand("0.General/Cryptomatte_Ratio", 'nuke.createNode("Cryptomatte_Ratio")')
DD_LNR.addCommand("0.General/multi_uvs", 'nuke.createNode("multi_uvs")')
DD_LNR.addCommand("0.General/LNR_Write", 'nuke.createNode("LNR_Write")')


DD_LNR.addCommand("1.Stamp/GP_stamp", 'nuke.createNode("GP_stamp")')
DD_LNR.addCommand("1.Stamp/LookdevStamp_v10", 'nuke.createNode("LookdevStamp_v10")')
DD_LNR.addCommand("1.Stamp/prevStamp", 'nuke.createNode("prevStamp")')
DD_LNR.addCommand("1.Stamp/LNR_Stamp_v01", "stampLNR()", icon='ddlnr.png')

DD_LNR.addCommand('2.lnr_project/LAD_bg', 'nuke.createNode("LADBG")')

lnrMenu = nuke.menu("Nuke").addMenu("lnrTeam")
lnrMenu.addCommand("Read_setUp", "read_setUp()","Ctrl+L", icon='ddlnr.png')
lnrMenu.addCommand("Change_Version", "changeVersion()","Shift+V", icon='ddlnr.png')
lnrMenu.addSeparator()
lnrMenu.addCommand("ACES_workflow", "ACES_Workflow()", icon='ddlnr.png')
lnrMenu.addCommand("Turntable_template", "turntableTemplateLNR()", icon='ddlnr.png')

# FX --------------------------------------------------------------------------
def stampFX():
    stampNode = nuke.createNode('stamp_default')
    stampNode['Project_name'].setValue(stamp_info.getPrjName())
    stampNode['Shotname'].setValue(stamp_info.getShotNameLNR())
    stampNode['Artist_name'].setValue(stamp_info.getUserName())
    stampNode['LetterBox'].setValue(0)

    return stampNode

DD_FX = nodeToolBar.addMenu('Dexter FX', icon='ddfx.png')
DD_FX.addCommand("1.Stamp/FX_stamp", "stampFX()", icon='ddfx.png')

# BL Gizmos --------------------------------------------------------------------
toolbar = nuke.menu('Nodes')
toolbar.addMenu('BL', 'BL.png')

#IMAGE
toolbar.addCommand('BL/Image/Arc', 'nuke.createNode("bl_Arc")', '')
toolbar.addCommand('BL/Image/Line', 'nuke.createNode("bl_Line")', '')
toolbar.addCommand('BL/Image/Random', 'nuke.createNode("bl_Random")', '')
toolbar.addCommand('BL/Image/Shape', 'nuke.createNode("bl_Shape")', '')
toolbar.addCommand('BL/Image/Star', 'nuke.createNode("bl_Star")', '')

#TIME
toolbar.addCommand('BL/Time/ITime', 'nuke.createNode("bl_ITime")', '')

#CHANNEL
toolbar.addCommand('BL/Channel/ChannelBox', 'nuke.createNode("bl_ChannelBox")', '')

#COLOR
toolbar.addCommand('BL/Color/Bytes', 'nuke.createNode("bl_Bytes")', '')
toolbar.addCommand('BL/Color/Compress', 'nuke.createNode("bl_Compress")', '')
toolbar.addCommand('BL/Color/Expand', 'nuke.createNode("bl_Expand")', '')
toolbar.addCommand('BL/Color/Monochrome', 'nuke.createNode("bl_Monochrome")', '')
toolbar.addCommand('BL/Color/Normalizer', 'nuke.createNode("bl_Normalizer")', '')
toolbar.addCommand('BL/Color/SaturationRGB', 'nuke.createNode("bl_SaturationRGB")', '')
toolbar.addCommand('BL/Color/Slice', 'nuke.createNode("bl_Slice")', '')
toolbar.addCommand('BL/Color/Threshold', 'nuke.createNode("bl_Threshold")', '')

#KEYER
toolbar.addCommand('BL/Keyer/ColorSupress', 'nuke.createNode("bl_ColorSupress")', '')
toolbar.addCommand('BL/Keyer/Despillator', 'nuke.createNode("bl_Despillator")', '')
toolbar.addCommand('BL/Keyer/HSV Keyer', 'nuke.createNode("bl_HSVKeyer")', '')
toolbar.addCommand('BL/Keyer/Simple Spill Supress', 'nuke.createNode("bl_SpillSupress")', '')

#LAYER
toolbar.addCommand('BL/Layer/LayerAE', 'nuke.createNode("bl_LayerAE")', '')

#FILTER
toolbar.addCommand('BL/Filter/Morphological/Binary', 'nuke.createNode("bl_mf_Binary")', '')
toolbar.addCommand('BL/Filter/Morphological/Border', 'nuke.createNode("bl_mf_Border")', '')
toolbar.addCommand('BL/Filter/Morphological/DirectionalBlur', 'nuke.createNode("bl_mf_DirectionalBlur")', '')
toolbar.addCommand('BL/Filter/Morphological/Occlusion', 'nuke.createNode("bl_mf_Occlusion")', '')
toolbar.addCommand('BL/Filter/Morphological/ShapeSofter', 'nuke.createNode("bl_mf_ShapeSofter")', '')

toolbar.addCommand('BL/Filter/BlurChroma', 'nuke.createNode("bl_BlurChroma")', '')
toolbar.addCommand('BL/Filter/Bokeh', 'nuke.createNode("bl_Bokeh")', '')
toolbar.addCommand('BL/Filter/IBokeh', 'nuke.createNode("bl_IBokeh")', '')
toolbar.addCommand('BL/Filter/ColorEdge', 'nuke.createNode("bl_ColorEdge")', '')
toolbar.addCommand('BL/Filter/Convolve', 'nuke.createNode("bl_Convolve")', '')
toolbar.addCommand('BL/Filter/CurveFilter', 'nuke.createNode("bl_CurveFilter")', '')
toolbar.addCommand('BL/Filter/EdgeExtend', 'nuke.createNode("bl_EdgeExtend2")', '')
toolbar.addCommand('BL/Filter/Emboss', 'nuke.createNode("bl_Emboss")', '')
toolbar.addCommand('BL/Filter/IBlur', 'nuke.createNode("bl_IBlur")', '')
toolbar.addCommand('BL/Filter/IDilateErode', 'nuke.createNode("bl_IDilateErode")', '')

#STYLISE
toolbar.addCommand('BL/Stylise/Mosaic', 'nuke.createNode("bl_Mosaic")', '')
toolbar.addCommand('BL/Stylise/Randomizer', 'nuke.createNode("bl_Randomizer")', '')
toolbar.addCommand('BL/Stylise/ScanLines', 'nuke.createNode("bl_ScanLines")', '')
toolbar.addCommand('BL/Stylise/Scatterize', 'nuke.createNode("bl_Scatterize")', '')
toolbar.addCommand('BL/Stylise/Solarize', 'nuke.createNode("bl_Solarize")', '')
toolbar.addCommand('BL/Stylise/TileMosaic', 'nuke.createNode("bl_TileMosaic")', '')
toolbar.addCommand('BL/Stylise/Zebrafy', 'nuke.createNode("bl_Zebrafy")', '')

#TRANSFORM
toolbar.addCommand('BL/Transform/Scroll', 'nuke.createNode("bl_Scroll")', '')
toolbar.addCommand('BL/Transform/ToBBOX', 'nuke.createNode("bl_ToBBOX")', '')

#WARP
toolbar.addCommand('BL/Warp/Bulge', 'nuke.createNode("bl_Bulge")', '')
toolbar.addCommand('BL/Warp/ChromaticAberation', 'nuke.createNode("bl_ChromaticAberation")', '')
toolbar.addCommand('BL/Warp/IDisplace', 'nuke.createNode("bl_IDisplace")', '')
toolbar.addCommand('BL/Warp/Twirl', 'nuke.createNode("bl_Twirl")', '')
toolbar.addCommand('BL/Warp/Wave', 'nuke.createNode("bl_Wave")', '')

#PIPE
toolbar.addCommand('BL/Pipe/GUI Switch', 'nuke.createNode("bl_GUISwitch")', '')
toolbar.addCommand('BL/Pipe/CleanOUT', 'nuke.createNode("bl_CleanOUT")', '')

#OTHER
toolbar.addCommand('BL/Other/Filler', 'nuke.createNode("bl_Filler")', '')
toolbar.addCommand('BL/Other/Match', 'nuke.createNode("bl_Match")', '')
toolbar.addCommand('BL/Other/Sample', 'nuke.createNode("bl_Sample")', '')
toolbar.addCommand('BL/Other/Scanner', 'nuke.createNode("bl_Scanner")', '')
toolbar.addCommand('BL/Other/ScanSlice', 'nuke.createNode("bl_ScanSlice")', '')
toolbar.addCommand('BL/Other/SetBBOXColor', 'nuke.createNode("bl_SetBBOXColor")', '')


'''
#NOT USE


# def plateArrange2():
#     plateAr_v02 = plateArrange_v02.MainWidget_v02(QtWidgets.QApplication.activeWindow())
#     plateAr_v02.show()

# def mmvSearchShow():
#     mmvSearchWidget = mmvSearch.MmvSearch(QtWidgets.QApplication.activeWindow())
#     mmvSearchWidget.setWindowFlags(QtCore.Qt.Dialog)
#     mmvSearchWidget.show()

# def compWizard():
#     reload(shotWizard)
#     wizardWidget = shotWizard.ClassWizard(QtWidgets.QApplication.activeWindow())
#     wizardWidget.show()

# def autoplaceV():
#     nodes = nuke.selectedNodes()
#     minX = min([ n.xpos()  for n in nodes])
#     minY = min([ n.ypos() for n in nodes])
#     offset = 0
#
#     for nd in reversed(nodes):
#         nd.setXYpos(minX, minY + offset)
#         offset += 50 + nd.screenHeight()

# ------------------------------------------------------------------------------

# def slate_4th_vendor_yys():
#     snode = nuke.createNode('slate_vendor')
#
#     fullPath = nuke.root().name()
#     if fullPath.startswith('/netapp/dexter/show'):
#         fullPath = fullPath.replace('/netapp/dexter', '')
#
#     seq = fullPath.split('/')[4]
#     shot = fullPath.split('/')[5]
#
#     snode['seq'].setValue(seq)
#     snode['shot'].setValue(shot.split('_')[-1])
#     verScript = "[python {'v' + os.path.basename(nuke.root().name()).split('_v')[-1][:3]}]"
#
#     snode['version'].setValue(verScript)
#     snode['artist'].setValue(getpass.getuser().split('.')[0])
#
#
#     start = int(nuke.knob("first_frame"))
#     end = int(nuke.knob("last_frame"))
#
#     snode['input.first_1'].setValue(start)
#     snode['input.last_1'].setValue(end)
#
# def slate_4th_vendor():
#     snode = nuke.createNode('slate')
#
#     fullPath = nuke.root().name()
#     if fullPath.startswith('/netapp/dexter/show'):
#         fullPath = fullPath.replace('/netapp/dexter', '')
#
#     seq = fullPath.split('/')[4]
#     shot = fullPath.split('/')[5]
#
#     snode['seq'].setValue(seq)
#     snode['shot'].setValue(shot.split('_')[-1])
#     verScript = "[python {'v' + os.path.basename(nuke.root().name()).split('_v')[-1][:3]}]"
#
#     snode['version'].setValue(verScript)
#
#     start = int(nuke.knob("first_frame"))
#     end = int(nuke.knob("last_frame"))
#
#     snode['input.first_1'].setValue(start)
#     snode['input.last_1'].setValue(end)
#
# def autoContactSheet():
#     cs = nuke.createNode("ContactSheet")
#     cs['knobChanged'].setValue("""
# if nuke.thisKnob().name() == 'inputChange':
#     import math
#     theNode = nuke.thisNode()
#     inputs = theNode.dependencies()
#     row = math.sqrt(len(inputs))
#     if row.is_integer():
#         column = row
#     else:
#         if (row - int(row)) > 0.5:
#             row = row + 1
#             column = row
#         else:
#             column = row + 1
#
#     hw = [0,0]
#     for i in inputs:
#         hw[0] += i.width()
#         hw[1] += i.height()
#
#     theNode['width'].setValue(hw[0]/int(row))
#     theNode['height'].setValue(hw[1]/int(column))
#     theNode['rows'].setValue(int(row))
#     theNode['columns'].setValue(int(column))""")


#TA(kim giuk) -- "Absolute PATH -> Relative PATH"
# def pathChange():
#     allRead = nuke.allNodes('Read')
#     try:
#         for filePath in allRead:
#             orgPath = filePath.knob('file').getValue()
#             startPath = orgPath.split('/')[0]
#             addValue = '[file dirname [knob root.name]]/'
#             if orgPath.startswith('[file'):
#                 pass
#             else:
#                 replacePath = addValue + orgPath
#                 changedPath = filePath.knob('file').setValue(replacePath)
#         return changedPath
#     except:
#         pass

'''
