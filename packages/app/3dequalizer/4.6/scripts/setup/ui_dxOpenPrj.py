# 3DE4.script.name:  1-1. dxOpen Project ...
#
# 3DE4.script.version:  v2.0.0
#
# 3DE4.script.gui:  Main Window::dx_Setup
#
# 3DE4.script.comment:  Open Project.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import os
import getpass
import DD_common
reload(DD_common)
import dxOpenProject
reload(dxOpenProject)


# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #

req = tde4.createCustomRequester()
dxOP = dxOpenProject.dxOpenPrj(req)
dxOP.windowTitle = 'dxOpenProject ...'

tde4.addTextFieldWidget(req, 'keyword', 'Shot', '')
tde4.setWidgetCallbackFunction(req, 'keyword', 'dxOP._searchShotKeyword')

tde4.addListWidget(req, 'shotlist', '', 0, 200)
tde4.setWidgetCallbackFunction(req, 'shotlist', 'dxOP._setListWidgetPlateTypeTask')

tde4.addListWidget(req, 'platetype', 'Plate Type', 0, 60)
tde4.insertListWidgetItem(req, 'platetype', 'Select Shot.', 0)
tde4.setWidgetCallbackFunction(req, 'platetype', 'dxOP._setListWidgetFile')

tde4.addListWidget(req, 'filelist', 'File', 0, 100)
tde4.insertListWidgetItem(req, 'filelist', 'Select Plate Type.', 0)

tde4.addToggleWidget(req, 'import_cache', 'Import Image Cache', 0)

ret = tde4.postCustomRequester(req, dxOP.windowTitle, 600, 490, 'Open', 'Cancel')

if ret == 1:
    dxOP.doIt()
