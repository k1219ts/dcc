import sys

# import pymodule.Qt as Qt

# from pymodule.Qt import QtWidgets
from PySide2 import QtWidgets

import MainForm as mf
reload(mf)

from MainForm import MainForm


# if "Side" in Qt.__binding__:
#     if Qt.__qt_version__ > "5.0.0":
import shiboken2 as shiboken
#     else:
#         import shiboken as shiboken
#
import maya.OpenMayaUI as mui

def getMayaWindow():
    ptr = mui.MQtUtil.mainWindow()
    return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)

def main():
    mainVar = MainForm(getMayaWindow())
    mainVar.show()
# else:
    # def main():
    #     app = QtWidgets.QApplication(sys.argv)
    #     mainVar = MainForm(None)
    #     mainVar.show()
    #     sys.exit(app.exec_())

if __name__ == "__main__":
    main()
