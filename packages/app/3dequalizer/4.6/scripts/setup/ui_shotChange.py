#
# 3DE4.script.name:  shotChange Tool ...
#
# 3DE4.script.version:  v1.1.0
#
# 3DE4.script.gui:  Main Window::Dexter
#
# 3DE4.script.comment:  Shot change Tool.
#
# DXTER STUDIOS:  CG Supervisor Kwantae.Kim


import DD_common
reload(DD_common)
import dxShotChange
reload(dxShotChange)


windowTitle = "shotChange Tool v1.1.0..."

# --------------------------------------------------------------------------- #
# open requester                                                              #
# --------------------------------------------------------------------------- #


if DD_common.checkProject():
    req = tde4.createCustomRequester()

    tde4.addTextFieldWidget(req, "show", "show", os.environ["show"])
    tde4.setWidgetSensitiveFlag(req, "show", 0)

    tde4.addTextFieldWidget(req, "fromShot", "from shot", os.environ["shot"])
    tde4.setWidgetSensitiveFlag(req, "fromShot", 0)

    tde4.addSeparatorWidget(req, "sep01")

    dxSC = dxShotChange.dxShotChange(req, windowTitle)

    tde4.addTextFieldWidget(req, "toShot", "to shot", "")
    tde4.setWidgetCallbackFunction(req, "toShot", "dxSC._toShot_callback")

    tde4.addTextAreaWidget(req, "shotInfo", "shot Info", 100, 0)

    tde4.addListWidget(req, "platetype", "Plate Type", 0, 70)
    tde4.insertListWidgetItem(req, "platetype", "Input Shot.", 0)

    tde4.addToggleWidget(req, 'add_plate', 'add Plate', 1)
    #tde4.addToggleWidget(req, 'make_dir', 'make Dir', 1)
    tde4.addToggleWidget(req, 'copy_mayaScene', 'Copy mayaScene', 1)

    # tde4.addSeparatorWidget(req, 'sep02')
    # tde4.addTextAreaWidget(req, "result", "Result", 100, 0)

    ret = tde4.postCustomRequester(req, windowTitle, 600, 420, "Save", "Cancel")

    if ret == 1:
        dxSC.doIt()
