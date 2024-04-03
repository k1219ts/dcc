# 3DE4.script.name:  3. dxExport Imageplanes for pmodel ...
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
reload(dxExportImageplane)
import dxUIcommon
reload(dxUIcommon)
import DD_common
reload(DD_common)


windowTitle = 'dxExport Imageplanes for pmodel ...'
# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #

if DD_common.checkProject(type='pmodel'):
    camList = tde4.getCameraList(1)
    prjPath = tde4.getProjectPath()

    if not prjPath:
        tde4.postQuestionRequester(windowTitle, 'Save a project first!', 'Ok')
    elif not camList:
        tde4.postQuestionRequester(windowTitle, 'Select cameras Please!', 'Ok')
    else:
        for cam in camList:
            if not tde4.getCameraLens(cam):
                camName = tde4.getCameraName(cam)
                tde4.postQuestionRequester(windowTitle, 'please connect the lens to %s!' % camName, 'Ok')
                break
        req = tde4.createCustomRequester()

        dxEI = dxExportImageplane.dxExportImp(req, camList)
        dxEI.windowTitle = windowTitle

        tde4.addFileWidget(req, 'file_path', 'Location...', '*', DD_common.find_target_path(dxEI.seqPath))

        tde4.addSeparatorWidget(req, 'sep03')
        tde4.addOptionMenuWidget(req, 'size', 'Reformat Size', 'Full', 'Half', 'Quarter')

        tde4.addSeparatorWidget(req, 'sep04')
        tde4.addToggleWidget(req, 'only_script', 'Only Script', 0)

        ret = tde4.postCustomRequester(req, windowTitle, 600, 0, 'Ok', 'Cancel')
        if ret==1:
            dxEI.doItPmodel()
