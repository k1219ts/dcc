# 3DE4.script.name:  4. dxExport Lens Distortion Nuke Node ...
#
# 3DE4.script.version:  v2.0.0
#
# 3DE4.script.gui:  Main Window::dx_Export
#
# 3DE4.script.comment:  Create Lens Distortion Node for Nuke
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import tde4
import dxExportNuke
reload(dxExportNuke)
import DD_common
reload(DD_common)

# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #

if DD_common.checkProject():
    if tde4.getCameraList(1):
        req = tde4.createCustomRequester()
        dxELD = dxExportNuke.createNukeLD(req)

        tde4.addFileWidget(req, 'userInput', 'Filename: ', '*.nk', dxELD.nuke_path)
        tde4.addToggleWidget(req, "stereo", "Stereo")
        tde4.setWidgetCallbackFunction(req, "stereo", "dxELD._export_LD_callback")
        tde4.addOptionMenuWidget(req, "left_camera", "Main Camera", *dxELD.camNames)
        tde4.addOptionMenuWidget(req, "right_camera", "", "")
        tde4.setWidgetSensitiveFlag(req, "right_camera", 0)
        tde4.addTextFieldWidget(req, 'start_frame', "Start Frame", str(tde4.getCameraSequenceAttr(dxELD.camList[0])[0]))

        ret = tde4.postCustomRequester(req, "dxExport Lens Distortion Nuke Node", 700, 0, "Ok", "Cancel")
        if ret==1:
            dxELD.doIt()
    else:
        tde4.postQuestionRequester('', 'Only selected cameras will be exported.', 'Ok')
