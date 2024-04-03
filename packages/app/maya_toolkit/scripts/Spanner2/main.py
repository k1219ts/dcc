from PySide2 import QtWidgets
from PySide2 import QtGui
import os
CURRENTPATH = os.path.dirname( os.path.abspath( __file__ ) )
import time
import MainForm
reload(MainForm)
try:
    import maya.cmds as cmds
    import maya.mel as mel
except ImportError:
    pass

import maya.OpenMayaUI as mui
import shiboken2 as shiboken

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

def main():
    # show splash
    try:
        splash = QtGui.QSplashScreen()
    except:
        splash = QtWidgets.QSplashScreen()
    splash.show()
    image = os.path.join( CURRENTPATH, 'resource/splash.jpg' )
    image = QtGui.QPixmap(image)
    splash.setPixmap(image)
    start = time.time()
    while time.time() - start < 0.05:
        QtWidgets.QApplication.processEvents()

    # show window
    window_name = 'Spanner2'
    if cmds.window(window_name, exists=True):
        cmds.showWindow(window_name)
    else:
        window = MainForm.MainForm(getMayaWindow())
        window.move(QtWidgets.QDesktopWidget().availableGeometry().center() - window.frameGeometry().center())
        window.setObjectName(window_name)
        window.show()

if __name__ == "__main__":
    main()
