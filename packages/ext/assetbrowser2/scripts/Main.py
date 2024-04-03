# -*- coding: utf-8 -*-
import sys

from pymodule import Qt
from pymodule.Qt import QtWidgets

from widgets.mainform import MainForm

DCC = "Standalone"

if sys.platform == "linux2":
    if "Side" in Qt.__binding__:
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
                    assert False, "Not Support Software"

    elif "PyQt" in Qt.__binding__:
        DCC = "Standalone"
    else:
        assert False, "Not Qt Binding available"

if DCC != "Standalone":
    if DCC == "Maya":
        def getWindow():
            import shiboken2
            import dxsUsd
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
        app.setStyleSheet("QLineEdit {padding-left: 5;}")
        # if cmds.window('Form', exists=True, q=True):
        #     cmds.deleteUI('Form')
        mainView = MainForm(None)
        mainView.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
