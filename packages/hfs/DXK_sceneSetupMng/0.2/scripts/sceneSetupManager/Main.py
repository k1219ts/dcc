import sys
from pymodule import Qt
from pymodule.Qt import QtWidgets
from SceneSetup import MainForm
import SceneSetup
reload (SceneSetup)

mainWindowName = "None"

try :
    import maya.OpenMayaUI as mui
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken
    mainWindowName = "Maya"
except:
    try:
        import hou
        mainWindowName = "Houdini"
    except:
        pass

if "Side" in Qt.__binding__:
    
    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    
    def main():
        parentWindow = None
        if mainWindowName == "Maya":
            parentWindow = getMayaWindow()
        elif mainWindowName == "Houdini":
            parentWindow = hou.qt.mainWindow()
#            QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("windows"))

        mainVar = MainForm(parentWindow)
        mainVar.show()
        
elif "PyQt" in Qt.__binding__:
    
    def main():
        app = QtWidgets.QApplication(sys.argv)
        mainVar = MainForm(None)
        mainVar.show()
        sys.exit(app.exec_())
    
else:
    print "No Qt binding available"

if __name__ == "__main__":
    main()
