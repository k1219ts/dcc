#
#
# 3DE4.script.name:  4. dxImport Project Asset...
#
# 3DE4.script.version:  v1.3.0
#
# 3DE4.script.gui:  Main Window::dx_Setup
#
# 3DE4.script.comment:  Import Project Asset OBJ.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import os
import DD_common
from imp import reload
reload(DD_common)
import dxImportProjectDummy
reload(dxImportProjectDummy)


windowTitle = 'Import Project Asset ...'

# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #
if DD_common.checkProject():
    req = tde4.createCustomRequester()

    tde4.addTextFieldWidget(req, 'show', 'Show', os.environ['show'])
    tde4.setWidgetSensitiveFlag(req, 'show', 0)

    tde4.addListWidget(req, 'assetList', 'Asset List', 0, 200)
    tde4.insertListWidgetItem(req, 'assetList', 'global', 0)

    dxID = dxImportProjectDummy.dxImportDummy(req)
    dxID.windowTitle = windowTitle

    tde4.setWidgetCallbackFunction(req, 'assetList', 'dxID._assetListCallback')

    tde4.addListWidget(req, 'assetType', 'Asset Type', 0, 80)
    tde4.insertListWidgetItem(req, 'assetType', 'Select Asset.', 0)
    tde4.setWidgetCallbackFunction(req, 'assetType', 'dxID._assetTypeCallback')

    tde4.addListWidget(req, 'fileList', 'Asset File', 0, 150)
    tde4.insertListWidgetItem(req, 'fileList', 'Select Asset Type.', 0)
    tde4.setWidgetCallbackFunction(req, 'fileList', 'dxID._fileListCallback')

    tde4.addButtonWidget(req, 'btnDelete', 'Delete', 100, 470)
    tde4.setWidgetCallbackFunction(req, 'btnDelete', 'dxID._btnDeleteCallback')

    tde4.addListWidget(req, 'selList', 'selected Asset File', 0, 100)

    ret = tde4.postCustomRequester(req, windowTitle, 600, 0, 'Import', 'Cancel')

    if ret == 1:
        dxID.doIt()
