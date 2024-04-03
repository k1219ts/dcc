# 3DE4.script.name:	1. dxExport Maya Mel ...
#
# 3DE4.script.version:	v2.0.0
#
# 3DE4.script.gui:	Main Window::dx_Export
#
# 3DE4.script.comment:	Creates a MEL script file that contains all project data, which can be imported into Autodesk Maya.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import dxExportMaya
from imp import reload
reload(dxExportMaya)
import dxUIcommon
reload(dxUIcommon)
import DD_common


windowTitle = 'dxExport Maya ...'
# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #

# if DD_common.checkProject():
if tde4.getCameraList(1):
    for cam in tde4.getCameraList(1):
        if not tde4.getCameraLens(cam):
            camName = tde4.getCameraName(cam)
            tde4.postQuestionRequester(windowTitle, 'please connect the lens to %s!' % camName, 'Ok')
            break
    req = tde4.createCustomRequester()
    dxEM = dxExportMaya.dxExportMel(req)
    dxEM.windowTitle = windowTitle

    tde4.addFileWidget(req,'file_browser','Exportfile...','*.mel', dxEM.file_name)
    tde4.addTextFieldWidget(req, 'start_frame', 'Start Frame', str(tde4.getCameraSequenceAttr(dxEM.cameraList[0])[0]))
    tde4.addOptionMenuWidget(req,'camera_selection','Export', 'Current Camera Only', 'Selected Cameras Only', 'Sequence Cameras Only', 'Reference Cameras Only', 'All Cameras')
    tde4.setWidgetValue(req,'camera_selection','2')

    tde4.addToggleWidget(req,'stereo','Stereo Camera', 0)
    tde4.addToggleWidget(req,'hide_ref_frames','Hide Reference Frames',0)
    tde4.addSeparatorWidget(req, 'sep01')

    fileName = tde4.getWidgetValue(req, 'file_browser')
    if 'pmodel' in fileName.lower():
        tde4.setWidgetSensitiveFlag(req, 'stereo', 0)
        tde4.setWidgetSensitiveFlag(req, 'hide_ref_frames', 0)

    dxUIovr = dxUIcommon.setOverscanWidget(req, dxEM.cameraList[0])
    dxUIovr.doIt()

    tde4.addSeparatorWidget(req, 'sep02')
    tde4.addToggleWidget(req,'export_3dmodel','Export 3D Model', 1)
    # tde4.addOptionMenuWidget(req, 'model_selection','Export', 'No 3D Models At All', 'Selected 3D Models Only','All 3D Models')
    # tde4.setWidgetValue(req, 'model_selection', '3')
    # tde4.addToggleWidget(req, 'export_texture', 'Export UV Textures' ,0)

    for c in tde4.getCameraList():
        stereo_status = tde4.getCameraStereoMode(c)
    if not stereo_status == 'STEREO_OFF':
        tde4.setWidgetValue(req,'stereo', '1')

    ret = tde4.postCustomRequester(req, dxEM.windowTitle, 800, 0, 'Ok','Cancel')
    if ret==1:
        dxEM.doIt()
else:
    tde4.postQuestionRequester(windowTitle, 'Only selected cameras will be exported.', 'Ok')
