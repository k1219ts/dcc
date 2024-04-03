from TaneToolUI import Ui_Form
import os
from PySide2 import QtWidgets
from PySide2 import QtCore
import maya.cmds as cmds
import dxCommon

class Asset_TaneControlTool_v01(QtWidgets.QWidget):

    def __init__(self,parent = dxCommon.getMayaWindow()):
        QtWidgets.QWidget.__init__(self, parent)
        print "init"

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #TN display  Mode Change
        self.ui.pushButton_display0.clicked.connect(self.TN_displayMode_none)
        self.ui.pushButton_display1.clicked.connect(self.TN_displayMode_point)
        self.ui.pushButton_display2.clicked.connect(self.TN_displayMode_box)
        #Duplicate  TN_Tane
        self.ui.pushButton_copy.clicked.connect(self.TN_copyTane)
        #TN set
        self.ui.pushButton_set0.clicked.connect(self.TN_setBaseMesh)
        self.ui.pushButton_set1.clicked.connect(self.TN_setTN_tane)
        #TNsource rename
        self.ui.pushButton_sc.clicked.connect(self.TN_scRename)
        #TN bake rib
        self.ui.pushButton_bake1.clicked.connect(self.TN_bakeON)
        self.ui.pushButton_bake0.clicked.connect(self.TN_bakeOFF)
        # TN density Reload
        self.ui.pushButton_reload.clicked.connect(self.TN_reloadMaskmap)
        #TN Environment Auto update
        self.ui.pushButton_update1.clicked.connect(self.TN_autoUpdateON)
        self.ui.pushButton_update0.clicked.connect(self.TN_autoUpdateOFF)


    def TN_displayMode_none(self):
        selTN = cmds.ls(sl=1, type='TN_Tane', dag=1)
        # none
        for TN in selTN:
            cmds.setAttr('%s.displayMode' % TN, 0)
    def TN_displayMode_point(self):
        selTN = cmds.ls(sl=1, type='TN_Tane', dag=1)
        # none
        for TN in selTN:
            cmds.setAttr('%s.displayMode' % TN, 1)
    def TN_displayMode_box(self):
        selTN = cmds.ls(sl=1, type='TN_Tane', dag=1)
        # none
        for TN in selTN:
            cmds.setAttr('%s.displayMode' % TN, 2)
    def TN_copyTane(self):
        selTane = cmds.ls(sl=1)
        copyTane = cmds.ls(cmds.duplicate(rr=True, renameChildren=True, un=True), dag=1)

        taneInput = cmds.listConnections(copyTane[1], type='TN_Environment')
        enInput = cmds.listConnections(taneInput, type='mesh')
        cmds.delete(enInput)
        cmds.parent(w=1)
    def TN_setBaseMesh(self):
        set = cmds.ls(type='objectSet')
        if 'TN_baseMesh' in set:
            pass
        else:

            cmds.sets(n='TN_baseMesh')

        tanebaseMesh = cmds.ls(sl=1)
        for t in tanebaseMesh:
            checkMesh = t.split('_')[:1]
            checkName = checkMesh[0]
            cmds.sets(t, addElement='TN_baseMesh')

            if checkName != 'TN':
                tanebaseMeshRename = 'TN_baseMesh_' + t
                cmds.rename(t, tanebaseMeshRename)
                print 'yes'
            else:
                pass


    def TN_setTN_tane(self):
        set = cmds.ls(type='objectSet')
        if 'TN_Tane' in set:
            pass
        if 'TN_baseMesh' in set:
            pass
        else:
            cmds.sets(n='TN_baseMesh')
            cmds.sets(n='TN_Tane')

        selNode = cmds.ls(type='TN_Tane')
        cmds.sets(selNode, addElement='TN_Tane')
        selMesh = cmds.ls('TN_baseMesh*', type='transform')
        cmds.sets(selMesh, addElement='TN_baseMesh')


    def TN_scRename(self):
        sel = cmds.ls(sl=1)

        for abc in sel:
            print abc
            abcPath = cmds.getAttr('%s.filepath' % abc)
            print abcPath
            getName = abcPath.split('/')[-1]
            print getName
            assetName = '_' + (getName.split('_model')[0])
            print assetName
            TN_AbcName = abc + assetName
            print TN_AbcName
            cmds.rename(abc, TN_AbcName)

    def TN_bakeON(self):
        sel = cmds.ls(sl=1, type='TN_Tane', dag=1)

        for i in sel:
            print i
            cmds.setAttr('%s.bakeRib' % i, 1)

    def TN_bakeOFF(self):
        sel = cmds.ls(sl=1, type='TN_Tane', dag=1)

        for i in sel:
            print i
            cmds.setAttr('%s.bakeRib' % i, 0)

    def TN_reloadMaskmap(self):
        selTNs = cmds.ls(sl=1, type='TN_Tane', dag=1)
        print selTNs

        for selTN in selTNs:
            selEnv = cmds.listConnections(selTN, destination=True)[1]
            print selEnv

            densityMap = cmds.getAttr('%s.useDensityMap' % selEnv)
            removeMap = cmds.getAttr('%s.useRemoveMap' % selEnv)
            scaleMap = cmds.getAttr('%s.useScaleMap' % selEnv)

            if densityMap == True:
                cmds.setAttr("%s.useDensityMap" % selEnv, 0)
                cmds.setAttr("%s.useDensityMap" % selEnv, 1)
            if removeMap == True:
                cmds.setAttr("%s.useRemoveMap" % selEnv, 0)
                cmds.setAttr("%s.useRemoveMap" % selEnv, 1)
            if scaleMap == True:
                cmds.setAttr("%s.useScaleMap" % selEnv, 0)
                cmds.setAttr("%s.useScaleMap" % selEnv, 1)



    def TN_autoUpdateON(self):
        selTNs = cmds.ls(sl=1, type='TN_Tane', dag=1)
        print selTNs

        for selTN in selTNs:
            selEnv = cmds.listConnections(selTN, destination=True)[1]

            cmds.setAttr("%s.autoUpdate" % selEnv, 1)
    def TN_autoUpdateOFF(self):
        selTNs = cmds.ls(sl=1, type='TN_Tane', dag=1)
        print selTNs

        for selTN in selTNs:
            selEnv = cmds.listConnections(selTN, destination=True)[1]
            cmds.setAttr("%s.autoUpdate" % selEnv, 0)



def main():
    # app = QtWidgets.QApplication(sys.argv)
    mainVar = Asset_TaneControlTool_v01()
    mainVar.show()
    # sys.exit(app.exec_())

if __name__ == "__main__":
    main()
