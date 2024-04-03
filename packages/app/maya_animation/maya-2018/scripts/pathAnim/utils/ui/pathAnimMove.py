from PySide2 import QtWidgets
import maya.cmds as cmds

class pathAnimWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(pathAnimWidget, self).__init__(parent)
        self.setWindowTitle("Move pathAnim")
        self.btnLayout = QtWidgets.QHBoxLayout(self)
        self.pickBtn = QtWidgets.QPushButton('Pick')
        self.pickBtn.resize(120, 40)
        self.setBtn = QtWidgets.QPushButton('Del Const')
        self.setBtn.resize(120, 40)
        self.btnLayout.addWidget(self.pickBtn)
        self.btnLayout.addWidget(self.setBtn)
        self.pickBtn.clicked.connect(self.pickDF)
        self.setBtn.clicked.connect(self.setDF)

    def pickDF(self):
        sel = str(cmds.ls(sl=True)[0])
        tar = str(cmds.ls(sl=True)[1])
        nsChar = tar.split("_PathAnim_")[0]
        if nsChar:
            cmds.parentConstraint(sel, nsChar + "_PathAnimCurves", mo=True, w=1)
            try:
                cmds.parentConstraint(sel, nsChar + "_curveF_crtl_GRP", mo=True, w=1)
            except:
                pass
        cmds.select(sel)

    def setDF(self):
        sel = str(cmds.ls(sl=True)[0])
        nsChar = sel.split("_PathAnim_")[0]
        if nsChar:
            tempA = str(cmds.listRelatives(nsChar + "_PathAnimCurves", c=True, type="parentConstraint")[0])
            try:
                tempB = str(cmds.listRelatives(nsChar + "_curveF_crtl_GRP", c=True, type="parentConstraint")[0])
            except:
                pass
            cmds.delete(tempA)
            try:
                cmds.delete(tempB)
            except:
                pass