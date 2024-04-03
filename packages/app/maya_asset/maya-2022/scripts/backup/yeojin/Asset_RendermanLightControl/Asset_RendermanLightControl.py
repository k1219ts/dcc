import sys

from RendermanLightControl import Ui_Form
import os
#
# from PyQt4 import QtGui
# PySide == PyQt4
# PySide2 == PyQt5
from pymodule.Qt  import QtWidgets
from pymodule.Qt  import QtCore


import maya.cmds as cmds
import sgComponent as sgc
import maya.mel as mel
import maya.OpenMayaUI as mui
import shiboken2 as shiboken


def getMayaWindow():
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    except:
        return None


class Asset_RendermanLightControl(QtWidgets.QWidget):

    def __init__(self):
        QtWidgets.QWidget.__init__(self, getMayaWindow())
        print "init"

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #Intensity
        self.ui.lineEdit.returnPressed.connect(self.light_Intensity)
        #Exposure
        self.ui.lineEdit_2.returnPressed.connect(self.light_Exposure)
        #Color
        self.ui.lineEdit_3.returnPressed.connect(self.light_Color)

        #Enable Temperature
        self.ui.checkBox.clicked.connect(self.light_EnableTemperature)
        #Temperature
        self.ui.lineEdit_6.returnPressed.connect(self.light_Temperature)
        #Normalize
        self.ui.checkBox_2.clicked.connect(self.light_Normalize)
        #Light Group
        self.ui.lineEdit_7.returnPressed.connect(self.light_LightGroup)
        #visibility_camera
        self.ui.checkBox_3.clicked.connect(self.light_VisibilityCamera)


    def light_Intensity(self):

        lightIntensity = self.ui.lineEdit.text()

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.intensity' % i, float(lightIntensity))


    def light_Exposure(self):

        lightExposure = self.ui.lineEdit_2.text()

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.exposure' % i, float(lightExposure) )

    def light_Color(self):

        lightColorText = self.ui.lineEdit_3.text()
        lightColor = lightColorText.split(',')


        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)
        selPxrDistantLight = cmds.ls(sl=1, type='PxrDistantLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight + selPxrDistantLight

        for i in sel:
            cmds.setAttr('%s.lightColor' % i, float(lightColor[0]), float(lightColor[1]),float(lightColor[2]), type='double3')

    def light_EnableTemperature(self):

        if self.ui.checkBox.isChecked():

            ETchecked = '1'

        else:
            ETchecked = '0'


        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.enableTemperature' % i, int(ETchecked))

    def light_Temperature(self):

        lightTemperature = self.ui.lineEdit_6.text()

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.temperature' % i, int(lightTemperature))

    def light_Normalize(self):

        if self.ui.checkBox_2.isChecked():
            Nchecked = '1'

        else:
            Nchecked = '0'

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.areaNormalize' % i, int(Nchecked))

    def light_LightGroup(self):

        lightID = self.ui.lineEdit_7.text()

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.lightGroup' % i, lightID ,type ='string')

    def light_VisibilityCamera(self):

        if self.ui.checkBox_3.isChecked():
            Vchecked = '1'

        else:
            Vchecked = '0'

        selPxrDiskLight = cmds.ls(sl=1, type='PxrDiskLight', dag=1)
        selPxrSphereLight = cmds.ls(sl=1, type='PxrSphereLight', dag=1)
        selPxrRectLight = cmds.ls(sl=1, type='PxrRectLight', dag=1)
        selPxrDomeLight = cmds.ls(sl=1, type='PxrDomeLight', dag=1)

        sel = selPxrDiskLight + selPxrSphereLight + selPxrRectLight + selPxrDomeLight

        for i in sel:
            cmds.setAttr('%s.rman__riattr__visibility_camera' % i, int(Vchecked))





def main():
    # app = QtWidgets.QApplication(sys.argv)
    mainVar = Asset_RendermanLightControl()
    mainVar.show()
    # sys.exit(app.exec_())

if __name__ == "__main__":
    main()
