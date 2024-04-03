import maya.OpenMayaUI as apiUI

from PySide2 import QtWidgets

try:
    import shiboken as shiboken
except:
    import shiboken2 as shiboken

def getMayaWindow():
    ptr = apiUI.MQtUtil.mainWindow()
    if ptr:
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    return None