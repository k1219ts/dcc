__author__ = 'gyeongheon.jeong'

import os
import math
import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya
from PySide2 import QtGui, QtCore, QtWidgets, QtUiTools
import aniCommon

currentpath = os.path.abspath(__file__)
UIROOT = os.path.dirname(currentpath)

MAYAVERSION = "2017"
UIFILE = os.path.join(os.path.dirname(currentpath), "GHGHSnapKeyUI.ui")


def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class SnapKeyTool(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(SnapKeyTool, self).__init__(parent)

        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)

        self.setWindowTitle("SnapKey UI {}".format(MAYAVERSION))
        self.move(QtCore.QPoint(1600 / 2, 500))

        self.Sample_SpinBox.setValue(1)
        self.Sample_SpinBox.setMinimum(0)
        self.Sample_SpinBox.setMaximum(1)
        self.Sample_SpinBox.setSingleStep(0.0001)
        self.selectedObj_lineEdit.setReadOnly(True)
        # self.selectedObj_lineEdit.setEnabled(False)
        self.targetObj_lineEdit.setEnabled(False)

        self.init()
        self.connectSignals()
        self.mayaSelectionChangeCallback()
        self.setWindowFlags(QtCore.Qt.WindowTitleHint|QtCore.Qt.CustomizeWindowHint)

    def init(self):
        self.selList = cmds.ls(sl=1)
        self.MaxTime = cmds.playbackOptions(q=True, max=True)
        self.MinTime = cmds.playbackOptions(q=True, min=True)
        self.isFast = self.fastBake_checkBox.isChecked()
        self.isSmart = self.smartBake_checkBox.isChecked()

    def connectSignals(self):
        self.fastBake_checkBox.toggled.connect(self.fastBakeChanged)
        self.toAnother_checkBox.stateChanged.connect(self.updateObjectnameLineEdit)
        self.C2L_pushButton.clicked.connect(self.bakeLocators)
        self.L2C_pushButton.clicked.connect(self.loc2ctrl)
        self.Close_pushButton.clicked.connect(self.closeWidget)

    def closeWidget(self):
        OpenMaya.MMessage.removeCallback(self.selChangedCallback)
        self.deleteLater()

    def fastBakeChanged(self):
        state = self.fastBake_checkBox.isChecked()

        if state:
            self.smartBake_checkBox.setChecked(not state)
            self.smartBake_checkBox.setEnabled(False)
        else:
            self.smartBake_checkBox.setEnabled(True)

    def updateObjectnameLineEdit(self, *args, **kwargs):
        sel = cmds.ls(sl=True)
        if sel:
            sel_absName = sel[0].split(":")[-1]
            self.selectedObj_lineEdit.clear()

            if self.toAnother_checkBox.isChecked():
                self.selectedObj_lineEdit.setText(sel_absName)
                self.targetObj_lineEdit.setEnabled(True)
            else:
                self.targetObj_lineEdit.clear()
                self.targetObj_lineEdit.setEnabled(False)

    def mayaSelectionChangeCallback(self):
        self.selChangedCallback = OpenMaya.MEventMessage.addEventCallback("SelectionChanged",
                                                                          self.updateObjectnameLineEdit)

    def bakeCommand(self, objects, **kwargs):
        cmds.bakeResults(objects,
                         simulation=True,
                         t=(self.MinTime - 1, self.MaxTime + 1),
                         disableImplicitControl=True,
                         preserveOutsideKeys=True,
                         sparseAnimCurveBake=False,
                         removeBakedAttributeFromLayer=False,
                         bakeOnOverrideLayer=False,
                         minimizeRotation=True,
                         controlPoints=False,
                         shape=True,
                         **kwargs)

    def getMatrix(self, node, time, attr='worldMatrix'):
        sel = OpenMaya.MSelectionList()
        sel.add(node)

        mobj = sel.getDependNode(0)
        mfn = OpenMaya.MFnDependencyNode(mobj)

        mtxAttr = mfn.attribute(attr)
        mtxPlug = OpenMaya.MPlug(mobj, mtxAttr)
        mtxPlug = mtxPlug.elementByLogicalIndex(0)

        time = OpenMaya.MTime(time)
        timeContext = OpenMaya.MDGContext(time)
        mtxObj = mtxPlug.asMObject(timeContext)
        mtxData = OpenMaya.MFnMatrixData(mtxObj)
        mtxValue = mtxData.matrix()

        return mtxValue

    def getWorldTransform(self, matrix):
        mTransformMtx = OpenMaya.MTransformationMatrix(matrix)
        trans = mTransformMtx.translation(OpenMaya.MSpace.kWorld)
        eulerRot = mTransformMtx.rotation()

        angles = [math.degrees(angle) for angle in (eulerRot.x, eulerRot.y, eulerRot.z)]
        return trans, angles

    @aniCommon.undo
    def bakeLocators(self):
        self.init()
        toAnother = self.toAnother_checkBox.isChecked()
        newTarget = str(self.targetObj_lineEdit.text())
        LocatorList = []
        tempConstraintList = []
        smartBake_kwargs = dict()
        sampleBy = float(self.Sample_SpinBox.value())

        if self.isSmart:
            smartBake_kwargs['smart'] = True
        else:
            smartBake_kwargs['sampleBy'] = sampleBy

        for i in self.selList:
            if toAnother and newTarget:
                locatorName = ":".join(i.split(":")[:-1] + [newTarget + "_GHsnapKey_LOC"])
            else:
                locatorName = i + "_GHsnapKey_LOC"
            locator = cmds.spaceLocator(n=locatorName)[0]
            LocatorList.append(locator)

            if self.isFast:
                for t in range(int(self.MinTime - 1), int(self.MaxTime + 1), 1):
                    matrix = self.getMatrix(node=i, time=t)
                    pos, angles = self.getWorldTransform(matrix)
                    cmds.setKeyframe(locator, at="tx", t=t, v=pos.x)
                    cmds.setKeyframe(locator, at="ty", t=t, v=pos.y)
                    cmds.setKeyframe(locator, at="tz", t=t, v=pos.z)
                    cmds.setKeyframe(locator, at="rx", t=t, v=angles[0])
                    cmds.setKeyframe(locator, at="ry", t=t, v=angles[1])
                    cmds.setKeyframe(locator, at="rz", t=t, v=angles[2])
            else:
                tempConstraintList.append(cmds.parentConstraint(i, locatorName, w=1, mo=False)[0])

        if self.isFast:
            return

        self.bakeCommand(LocatorList, **smartBake_kwargs)
        cmds.delete(tempConstraintList)

    @aniCommon.undo
    def loc2ctrl(self, *args):
        self.init()
        sampleBy = float(self.Sample_SpinBox.value())
        tempConstList2 = []
        consList = []

        locList = cmds.ls(type="locator")
        bakedlocs = []
        smartBake_kwargs = dict()

        if self.isSmart:
            smartBake_kwargs['smart'] = True
        else:
            smartBake_kwargs['sampleBy'] = sampleBy

        for i in locList:
            if i.count("_GHsnapKey_LOC") != 0:
                bakedlocs.append(cmds.listRelatives(i, p=1)[0])

        for i in bakedlocs:
            try:
                consName = i.split("_GHsnapKey_LOC")[0]
                consList.append(consName)

                TlockStatus = cmds.getAttr(consName + ".tx", l=True)
                RlockStatus = cmds.getAttr(consName + ".rx", l=True)

                if self.isFast:
                    for t in range(int(self.MinTime - 1), int(self.MaxTime + 1), 1):
                        srcMatrix = self.getMatrix(i, t)
                        srcPInvMatrix = self.getMatrix(consName, t, attr="parentInverseMatrix")
                        absMatrix = srcMatrix * srcPInvMatrix
                        pos, angles = self.getWorldTransform(absMatrix)
                        cmds.setKeyframe(consName, at="tx", t=t, v=pos.x)
                        cmds.setKeyframe(consName, at="ty", t=t, v=pos.y)
                        cmds.setKeyframe(consName, at="tz", t=t, v=pos.z)
                        cmds.setKeyframe(consName, at="rx", t=t, v=angles[0])
                        cmds.setKeyframe(consName, at="ry", t=t, v=angles[1])
                        cmds.setKeyframe(consName, at="rz", t=t, v=angles[2])
                else:
                    if not TlockStatus and not RlockStatus:
                        tempConstList2.append(cmds.parentConstraint(i, consName, w=1, mo=False)[0])
                    elif TlockStatus:
                        tempConstList2.append(cmds.orientConstraint(i, consName, w=1, mo=False)[0])
                    elif RlockStatus:
                        tempConstList2.append(cmds.pointConstraint(i, consName, w=1, mo=False)[0])
            except:
                pass

        if not self.isFast:
            self.bakeCommand(consList, **smartBake_kwargs)
            cmds.delete(tempConstList2)

        cmds.delete(bakedlocs)


def runUI():
    global _win
    try:
        if _win != None:
            _win.closeWidget()
            _win.deleteLater()
            _win = None
    except:
        pass
    _win = SnapKeyTool()
    _win.show()
