# 3DE4.script.name:  2. dxExport Imageplanes for Shot ...
#
# 3DE4.script.version:  v3.0.0
#
# 3DE4.script.gui:  Main Window::dx_Export
#
# 3DE4.script.comment:  Export undistorted imageplanes for Shot.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim

import tde4
import dxExportImageplane
from imp import reload
reload(dxExportImageplane)
import dxUIcommon
reload(dxUIcommon)
import DD_common
reload(DD_common)


windowTitle = 'dxExport Imageplanes for Shot ...'
# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #

camList = tde4.getCameraList(1)
prjPath = tde4.getProjectPath()

if DD_common.checkProject():
    if not prjPath:
        tde4.postQuestionRequester(windowTitle, 'Save a project first!', 'Ok')
    elif not camList:
        tde4.postQuestionRequester(windowTitle, 'Only selected cameras will be exported.', 'Ok')
    else:
        for cam in camList:
            if not tde4.getCameraLens(cam):
                camName = tde4.getCameraName(cam)
                tde4.postQuestionRequester(windowTitle, 'please connect the lens to %s!' % camName, 'Ok')
                break
            if tde4.getCameraProxyFootage(cam) == 0:
                req = tde4.createCustomRequester()

                dxEI = dxExportImageplane.dxExportImp(req, cam)
                dxEI.windowTitle = windowTitle

                tde4.addFileWidget(req, 'file_path', 'Location...', '*', DD_common.find_target_path(dxEI.seqPath) + '/hi')
                tde4.addTextFieldWidget(req, 'file_name', 'File Name', dxEI.jpgFile)
                tde4.addTextFieldWidget(req, 'start_frame', 'Start Frame', str(dxEI.seqAttr[0]))

                tde4.addSeparatorWidget(req, 'sep01')
                tde4.addTextFieldWidget(req, 'colorspaceR', 'Input Colorspace', str(dxEI.colorspaceR))
                tde4.addTextFieldWidget(req, 'colorspaceW', 'Output Colorspace', str(dxEI.colorspaceW))
                tde4.setWidgetSensitiveFlag(req, 'colorspaceR', 0)
                tde4.setWidgetSensitiveFlag(req, 'colorspaceW', 0)

                tde4.addSeparatorWidget(req, 'sep02')
                dxUIovr = dxUIcommon.setOverscanWidget(req, cam)
                dxEI.overscanValue = dxUIovr.doIt()

                tde4.addTextFieldWidget(req, 'os_width', 'Overscan Width', str(dxEI.seqWidth))
                tde4.addTextFieldWidget(req, 'os_height', 'Overscan Height', str(dxEI.seqHeight))
                tde4.setWidgetSensitiveFlag(req, 'os_width', 0)
                tde4.setWidgetSensitiveFlag(req, 'os_height', 0)
                tde4.setWidgetValue(req, 'os_width', str(int(dxEI.seqWidth * dxEI.overscanValue)))
                tde4.setWidgetValue(req, 'os_height', str(int(dxEI.seqHeight * dxEI.overscanValue)))

                tde4.addSeparatorWidget(req, 'sep03')
                tde4.addOptionMenuWidget(req, 'size', 'Reformat Size', 'Full', 'Half', 'Quarter')
                tde4.setWidgetCallbackFunction(req, 'size', 'dxUIovr._imageReformatScale')
                tde4.addToggleWidget(req, 'burnin', 'Burn In Info', 1)

                tde4.addSeparatorWidget(req, 'sep04')
                tde4.addToggleWidget(req, 'send_tractor', 'Send Tractor', 1)
                tde4.addToggleWidget(req, 'only_script', 'Only Script', 0)
                tde4.addToggleWidget(req, 'database_publish', 'DB Publish', 1)

                ret = tde4.postCustomRequester(req, windowTitle, 600, 0, 'Ok', 'Cancel')
                if ret==1:
                    dxEI.doIt()
