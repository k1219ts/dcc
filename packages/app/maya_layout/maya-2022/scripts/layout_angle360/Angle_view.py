#encoding=utf-8
####################################################
########## coding by RND youkyoung.kim #############
####################################################
import sys, os, math, subprocess
import Qt
from Qt import QtGui
from Qt import QtWidgets
from Qt import QtCore
from angleui import Ui_Form
import maya.cmds as cmds
import maya.OpenMaya as om
import maya.mel as mel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if "Side" in Qt.__binding__:
    import maya.OpenMayaUI as mui
    if Qt.__qt_version__ > "5.0.0":
        import shiboken2 as shiboken
    else:
        import shiboken as shiboken
    def getMayaWindow():
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    def main():
        mainVar = AngleMain(getMayaWindow())
        mainVar.move(QtWidgets.QDesktopWidget().availableGeometry().center() - mainVar.frameGeometry().center())
elif "PyQt" in Qt.__binding__:
    def main():
        app = QtWidgets.QApplication(sys.argv)
        mainVar = AngleMain(None)
        sys.exit(app.exec_())
else:
    print "No Qt binding available"
if __name__ == "__main__":
    main()

class AngleMain(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.startSet()
        self.show()
        self.Connection()

    def startSet(self):
        # help icon > help document link
        icon = Qt.QtGui.QIcon()
        helpimg = "resource/help.png"
        icon.addPixmap(Qt.QtGui.QPixmap(os.path.join(CURRENT_DIR, helpimg)),
                                        Qt.QtGui.QIcon.Normal, Qt.QtGui.QIcon.Off)
        self.ui.helpbtn.setIcon(icon)
        self.ui.helpbtn.setObjectName("helpbtn")
        self.ui.helpbtn.setCursor(QtCore.Qt.WhatsThisCursor)
        self.ui.helpbtn.setToolTip("help")
        # plugin load and text output reload setting
        if not cmds.pluginInfo('AngleDis_ZNumToString', q=True, l=True):
            cmds.loadPlugin('AbcExport')

    def Connection(self):
        self.ui.firstbtn.clicked.connect(lambda : self.Selectlocator(self.ui.firsttxt))
        self.ui.centerbtn.clicked.connect(lambda :self.Selectlocator(self.ui.centertxt))
        self.ui.endbtn.clicked.connect(lambda: self.Selectlocator(self.ui.endtxt))
        self.ui.importbtn.clicked.connect(self.ImportP)
        self.ui.runbtn.clicked.connect(self.Runloof)
        self.ui.stopbtn.clicked.connect(self.Endloof)
        self.ui.helpbtn.clicked.connect(self.Help)

    def Help(self):
        pdf = "/usr/bin/evince" #pdf viewer
        #odp = "/usr/bin/libreoffice"  # odp viewer
        helpdoc = "resource/angle360_help.pdf"
        fileName = os.path.join(CURRENT_DIR, helpdoc)
        subprocess.Popen([pdf, fileName])

    def ImportP(self):
        # locator and text mb import
        mel.eval('file -import -type "mayaBinary"  -ignoreVersion -ra true -mergeNamespacesOnClash false -rpr "AngleDis" -options "v=0;"  -pr -loadReferenceDepth "all" "/netapp/backstage/pub/apps/maya2/versions/2017/team/layout/linux/scripts/layout_angle360/resource/Angle_360.mb";')
        #mel.eval('file -import -type "mayaBinary"  -ignoreVersion -ra true -mergeNamespacesOnClash false -rpr "AngleDis" -options "v=0;"  -pr -loadReferenceDepth "all" "/dexter/Cache_DATA/RND/youkyoung/mb_speed_angle/angle/Angle_360.mb";')

    def Selectlocator(self, labeltxt):
        # three point input
        point = cmds.ls(sl=True)[0]
        labeltxt.setText(point)
        print point

    def Runloof(self):
        # text node angle output connect
        self.firstp = self.ui.firsttxt.text()
        self.centerp = self.ui.centertxt.text()
        self.endp = self.ui.endtxt.text()

        try:
            cmds.connectAttr('AngleDis_ZNumToString1.output',
                             'AngleDis_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_textForBevel5.text',
                             force=True)
            cmds.connectAttr('AngleDis_ZNumToString2.output',
                             'AngleDis_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_textForBevel6.text',
                             force=True)
        except:
            pass
        self.frameCallback = om.MEventMessage.addEventCallback("timeChanged", self.Angleloof)

    def Angleloof(self, msg):
        # tangent 360 angle -> angle text print
        line = cmds.xform(self.firstp, query=True, ws=True, t=True)
        cen = cmds.xform(self.centerp, query=True, ws=True, t=True)
        car = cmds.xform(self.endp, query=True, ws=True, t=True)
        X = om.MVector(line[0] - cen[0], 0, line[2] - cen[2])  # tangent vector
        X.normalize()
        Y = om.MVector(0, 1, 0)
        Z = X ^ Y
        Z.normalize()

        look = om.MVector(car[0] - cen[0], 0, car[2] - cen[2])  # look vector
        look.normalize()
        x = look * X
        z = look * Z
        degrees = math.atan2(z, x) * 180 / math.pi
        if (degrees < 0):
            degrees += 360
        result = round(degrees) % 360
        result2 = -(360 - result)
        if result2 == -360:
            result2 = 0
        print result, result2
        cmds.setAttr('AngleDis_ZNumToString1.input', result)
        cmds.setAttr('AngleDis_ZNumToString2.input', result2)
        cmds.dgdirty('AngleDis_ZNumToString1')  # c plugin -> compute all exec
        cmds.dgdirty('AngleDis_ZNumToString2')  # c plugin -> compute all exec

    def Endloof(self):
        # event callback delete
        try:
            cmds.disconnectAttr('AngleDis_ZNumToString1.output',
                                'AngleDis_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_textForBevel5.text')
            cmds.disconnectAttr('AngleDis_ZNumToString2.output',
                                'AngleDis_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_ZDistanceMeasureMesh_v01_textForBevel6.text')
        except:
            pass
        om.MEventMessage.removeCallback(self.frameCallback)
