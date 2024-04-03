import maya.cmds as cmds
from MainForm import MainForm
from pymodule.Qt import QtWidgets

def main():
    if cmds.window('RoundingConfirm', q = 1, ex = 1):
        cmds.deleteUI('RoundingConfirm')

    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    lookdevMain = MainForm()
    lookdevMain.show()
    
if __name__ == "__main__":
    main()