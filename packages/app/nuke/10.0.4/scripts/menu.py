import nuke
import os
import W_hotbox, W_hotboxManager

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

import nuke_inven_pub
import slate
import shotSetup.compShotSetup


curDir = os.path.dirname(__file__)

# Title Menu
titleMenuBar = nuke.menu("Nuke")
titleMenuBar.addCommand("DexterDigital/0. Comp Asset Tool", "compAssetTool()", icon='ddcmp.png')
# titleMenuBar.addCommand("DexterDigital/1. Matchmove Search", "mmvSearchShow()", icon='ddcmp.png')
# titleMenuBar.addCommand("DexterDigital/1-1. Matchmove Search(DATABASE)", "camera_search()", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/2. Render History", "renderHistoryShow()", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/3. Revealfolder", "Revealfolder.Revealfolder()", "F6", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/4. sendTractor", "sendTractor.sendtoTractor()", "Ctrl+F12", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/5. Jpeg Write", "compWrite.compWrite('jpg/sRGB')", "F11", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/6. Nuke Inventory Pub", "invenPubShow()", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/7. Collect Files", "collectFiles.collectFiles()", icon='ddcmp.png')
titleMenuBar.addCommand("DexterDigital/8. abc center pivot", "abcCenterPivot()", icon='ddcmp.png')

# New Menu?
compMenu = nuke.menu("Nuke").addMenu("CompTeam")
compMenu.addCommand("Comp Shot Setup by Garam.kim", "compShotSetupFunction()", icon='ddcmp.png')

# Recent File
fileTab = titleMenuBar.addMenu("File")
fileTab.addCommand("Open Recent Comp/@recent_file7", "nuke.scriptOpen(nuke.recentFile(7))", "#+7")
fileTab.addCommand("Open Recent Comp/@recent_file8", "nuke.scriptOpen(nuke.recentFile(8))", "#+8")
fileTab.addCommand("Open Recent Comp/@recent_file9", "nuke.scriptOpen(nuke.recentFile(9))", "#+9")
for i in range(10, 26):
    fileTab.addCommand("Open Recent Comp/@recent_file%d" % i, lambda: nuke.scriptOpen(nuke.recentFile(i)))

# add Nodes
nodesMenu = nuke.menu('Nodes')
nodesMenu.findItem('Deep').addCommand('DeepOpenEXRId', lambda: nuke.createNode('DeepOpenEXRId')) # in plugins

nodesMenu.findItem('3D').addCommand('FishEye', lambda: nuke.createNode('FishEye')) # in plugins
nodesMenu.findItem('3D').addCommand('OmnidirectionalStereo', lambda: nuke.createNode('OmnidirectionalStereo')) # in plugins

xToolMenu = nodesMenu.addMenu("X_Tools", icon='X_Tools.png')
xToolMenu.addCommand("X_Distort", lambda: nuke.createNode('X_Distort'), icon='X_Distort.png')
xToolMenu.addCommand("X_Distort", lambda: nuke.createNode('X_Tesla'), icon='X_Tesla.png')

vendorMenu = nodesMenu.addMenu("Vendor", icon='4th.png')
vendorMenu.addCommand('4th/slate', lambda: slate.slate_4th_vendor())
vendorMenu.addCommand('4th/slate_vendor', lambda: slate.slate_4th_vendor_yys())

# nudeToolBar CompMenu
DexterMenu_CMP = nuke.menu('Nodes').addMenu('Dexter CMP', icon='ddcmp.png')

DexterMenu_CMP.addCommand("STAMP/Stamp_DEFAULT", "stamp.stampDEFAULT()", icon='ddcmp.png')
DexterMenu_CMP.addCommand("STAMP/Stamp_Netflix", "stamp.stampNetflix()", icon='ddcmp.png')

DexterMenu_CMP.addCommand("ETC/QC/QC_HSVSat", lambda: nuke.loadToolset('%s/toolset/QC_HSVSat.nk' % curDir))
DexterMenu_CMP.addCommand("ETC/QC/QC_PlayAB", lambda: nuke.loadToolset('%s/toolset/QC_PlayAB.nk' % curDir))

DexterMenu_CMP.addCommand("ETC/Auto Backdrop" )

def compShotSetupFunction():
    reload(compShotSetup)
    widget = compShotSetup.CompShotSetup(QtWidgets.QApplication.activeWindow())
    widget.setWindowFlags(QtCore.Qt.Dialog)
    widget.show()