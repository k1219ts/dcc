################################################
#
# author        : Dexter RND daeseok.chae
# create        : 2017.06.14
# filename      : Main.py
# last update   : 2017.06.14
#
################################################

import sys

MY_PYPATH_MODUEL = "/netapp/backstage/pub/apps/maya2/versions/2017/global/linux/lib/site-packages"
import site

site.addsitedir(MY_PYPATH_MODUEL)

from Qt import QtWidgets
from Qt import QtGui
from Qt import QtCore
import Qt

from MainForm import MainForm

if "Side" in Qt.__binding__:
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken

    import maya.OpenMayaUI as mui


    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)


    def main():
        mainVar = MainForm(getMayaWindow())
        mainVar.show()
else:
    def main():
        app = QtWidgets.QApplication(sys.argv)
        fd = MainForm(None)
        fd.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
