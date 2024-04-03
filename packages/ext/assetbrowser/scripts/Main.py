import sys

from PySide2 import QtWidgets
from MainForm import MainForm

DCC = "Standalone"
platform = "Linux"


try:
    import maya.OpenMayaUI as mui
    DCC = "Maya"
except:
    try:
        import hou
        DCC = "Houdini"
    except:
        try:
            import nuke
            DCC = "Nuke"
        except:
            DCC = "Standalone"

if DCC != "Standalone":
    if DCC == "Maya":
        def getWindow():
            import shiboken2
            import maya.cmds as cmds
            ptr = mui.MQtUtil.mainWindow()
            return shiboken2.wrapInstance(long(ptr), QtWidgets.QWidget)
    elif DCC == "Houdini":
        def getWindow():
            return hou.qt.mainWindow()
    else:
        def getWindow():
            return None


    def main():
        mainView = MainForm(getWindow())
        mainView.show()
else:
    def main():
        app = QtWidgets.QApplication(sys.argv)
        # if cmds.window('Form', exists=True, q=True):
        #     cmds.deleteUI('Form')
        mainView = MainForm(None)
        mainView.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
