####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, getpass
import Qt
from Qt import QtWidgets
from Qt import QtGui
import os
from inventory import inventory
import LayInventory
reload(LayInventory)
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )

try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

if "Side" in Qt.__binding__:
    import maya.OpenMayaUI as mui

    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken

    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

    def main(): ## editable team leader and rnd department
        userlist = ['youkyoung.kim', 'youngkyu.choi', 'jeho.choi', 'jihoon.lee']
        user = getpass.getuser()
        if user in userlist:
            window_name = 'Layout_Inventory by RND youkyoung.kim'
            if cmds.window(window_name, exists=True):
                cmds.showWindow(window_name)
            else:
                window = LayInventory.InventoryMain(getMayaWindow())
                window.move(QtWidgets.QDesktopWidget().availableGeometry().center() - window.frameGeometry().center())
                window.setObjectName(window_name)
                window.show()
        else:
            window = inventory.Inventory()
            window.show()

elif "PyQt" in Qt.__binding__:
    def main():
        app = QtWidgets.QApplication(sys.argv)
        mainVar = LayInventory.InventoryMain(None)
        mainVar.show()
        sys.exit(app.exec_())

else:
    print "No Qt binding available"

if __name__ == "__main__":
    main()

