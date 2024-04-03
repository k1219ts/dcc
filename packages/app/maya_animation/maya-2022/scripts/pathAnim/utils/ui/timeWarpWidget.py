
from PySide2 import QtCore, QtGui, QtWidgets
import logging
import maya.cmds as cmds
import pathAnim.utils.timeWarp as timeWarp
import aniCommon
reload(timeWarp)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class PATimeWarpWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PATimeWarpWidget, self).__init__(parent)
        self.setWindowTitle("PathAnim Time Warp")

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 3, 0, 3)
        self.main_layout.setSpacing(3)

        self.pathAnimNodesComboBox = QtWidgets.QComboBox()

        button_layout = QtWidgets.QGridLayout()
        self.selectWarpBtn = QtWidgets.QPushButton('Select Warp')
        self.selectAppliedBtn = QtWidgets.QPushButton('Select Applied')
        self.applyBtn = QtWidgets.QPushButton('Apply Warp To Selection')
        self.removeBtn = QtWidgets.QPushButton('Remove Warp from Selection')
        self.createBtn = QtWidgets.QPushButton('Create Warp')
        self.createBtn.setStyleSheet("color: rgb(0,243,0)")
        self.deleteBtn = QtWidgets.QPushButton('Delete Warp')
        self.deleteBtn.setStyleSheet("background-color: rgb(216, 33, 33)")

        button_layout.addWidget(self.selectWarpBtn, 0, 0)
        button_layout.addWidget(self.selectAppliedBtn, 0, 1)
        button_layout.addWidget(self.applyBtn, 1, 0)
        button_layout.addWidget(self.removeBtn, 1, 1)
        button_layout.addWidget(self.createBtn, 2, 0)
        button_layout.addWidget(self.deleteBtn, 2, 1)

        self.main_layout.insertWidget(0, self.pathAnimNodesComboBox)
        self.main_layout.addLayout(button_layout)
        self.reloadComboBox()
        self.setButtonEnabled()
        self.connectSignals()

    def connectSignals(self):
        self.pathAnimNodesComboBox.currentIndexChanged.connect(self.comboBoxAction)
        self.selectWarpBtn.clicked.connect(self.selectWarp)
        self.selectAppliedBtn.clicked.connect(self.selectApplied)
        self.applyBtn.clicked.connect(self.applyToSelection)
        self.removeBtn.clicked.connect(self.removeFromSelection)
        self.createBtn.clicked.connect(self.create)
        self.deleteBtn.clicked.connect(self.delete)


    def reloadComboBox(self):
        self.pathAnimNodesComboBox.clear()
        cls = timeWarp.PATimeWarp()
        nodes = cls.getPathAnimNodes()

        if nodes:
            self.pathAnimNodesComboBox.addItem('- Select PathAnim Node -')
            self.pathAnimNodesComboBox.insertSeparator(1)
            self.pathAnimNodesComboBox.addItems(nodes)
        else:
            self.pathAnimNodesComboBox.addItem('- Select PathAnim Node -')

    def setButtonEnabled(self):
        node = str(self.pathAnimNodesComboBox.currentText())
        if node != '- Select PathAnim Node -':
            self.applyBtn.setEnabled(True)
            self.removeBtn.setEnabled(True)
            self.createBtn.setEnabled(True)
            self.deleteBtn.setEnabled(True)
            self.selectAppliedBtn.setEnabled(True)
            self.selectWarpBtn.setEnabled(True)
        else:
            self.applyBtn.setEnabled(False)
            self.removeBtn.setEnabled(False)
            self.createBtn.setEnabled(False)
            self.deleteBtn.setEnabled(False)
            self.selectAppliedBtn.setEnabled(False)
            self.selectWarpBtn.setEnabled(False)

    def comboBoxAction(self):
        self.setButtonEnabled()

    def findKey(self, object):
        animCurve = cmds.listConnections(object, scn=True, d=False, type='animCurve')
        return animCurve

    def getAnimObjects(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        prefix = timeWarp.PATimeWarp.getPrefix(paNode)
        #animLocs = timeWarp.PATimeWarp.getAnimLocators(prefix)
        #animLocs.append(prefix + "_PathAnim_Ctrl")
        plma = cmds.listConnections(prefix + "_PathAnim_Ctrl", scn=True, s=False, type="plusMinusAverage")[0]
        mpl = str(cmds.listConnections(plma, scn=True, s=False, type="multiplyDivide")[0])
        mpt = str(cmds.listConnections(mpl, scn=True, s=False, type="motionPath")[0])
        nsChar = str(cmds.listConnections(mpt, scn=True, s=False, type="transform")[0]).split(":")[0]
        char = nsChar.replace(prefix + "_", "")
        en = list()
        for i in cmds.ls(prefix + "_" + char + ":*_AnimLoc"):
            a = str(i).replace("_AnimLoc", "")
            en.append(a.replace(prefix + "_" + char + ":", ""))
        en.append("place_CON")
        en.append("direction_CON")
        en.append("move_CON")
        cmds.select(prefix + "_" + char + ":*_AnimLoc")
        cmds.select(char + ":*_CON", add=True)
        for j in en:
            cmds.select(char + ":" + str(j), d=True)
        animLocs = cmds.ls(sl=True)
        animLocs.append(prefix + "_PathAnim_Ctrl")
        #
        animCurve = self.findKey(prefix + "_PathAnim_Ctrl")
        if not animCurve:
            cmds.setKeyframe(prefix + "_PathAnim_Ctrl", t=cmds.playbackOptions(q=True, min=True))
        return animLocs

    def selectWarp(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        cls = timeWarp.PATimeWarp()
        cls.selectTimeCurve(paNode)

    def selectApplied(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        cls = timeWarp.PATimeWarp()
        cls.selectAppliedObject(paNode)

    def applyToSelection(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        selection = cmds.ls(sl=True)

        cls = timeWarp.PATimeWarp()
        cls.selection = selection
        cls.timeCurve = cls.getConnectedTimewarp(paNode)
        cls.apply()

    def removeFromSelection(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        selection = cmds.ls(sl=True)

        cls = timeWarp.PATimeWarp()
        timeCurve = cls.getConnectedTimewarp(paNode)

        if timeCurve:
            timeWarp.PATimeWarp.remove(timeCurve[0], selection)


    @aniCommon.undo
    def create(self):
        animLocs = self.getAnimObjects()

        cls = timeWarp.PATimeWarp()
        cls.selection = animLocs
        timeCurve = cls.PA_createNodes()

        cls.timeCurve = timeCurve
        cls.apply()

        self.setButtonEnabled()
        logger.debug('Create Timewarp : {0}'.format(timeCurve))

    @aniCommon.undo
    def delete(self):
        paNode = str(self.pathAnimNodesComboBox.currentText())
        cls = timeWarp.PATimeWarp()
        timeCurve = cls.getConnectedTimewarp(paNode)
        if timeCurve:
            timeWarp.PATimeWarp.delete(timeCurve[0])
            logger.debug('Delete Timewarp : {0}'.format(timeCurve))
