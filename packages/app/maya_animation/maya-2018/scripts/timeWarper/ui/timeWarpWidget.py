
import logging

import aniCommon
import maya.cmds as cmds
import timeWarper.timeWarper as timeWarp
from PySide2 import QtCore, QtWidgets

reload(timeWarp)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COMBOBOX_ITEMS = [
    'Select a warp',
    'Create New Warp',
    'Create Anti-Warp'
]

class TimeWarpWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TimeWarpWidget, self).__init__(parent)
        self.setWindowTitle("Time Warp GUI")

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 3, 0, 3)
        self.main_layout.setSpacing(3)

        comboBox_layout = QtWidgets.QHBoxLayout()
        self.comboBox = QtWidgets.QComboBox()
        self.enableWarp_checkBox = QtWidgets.QCheckBox('Enable This Warp')
        comboBox_layout.addWidget(self.comboBox)
        comboBox_layout.addWidget(self.enableWarp_checkBox)

        warpedTimeLayout = QtWidgets.QHBoxLayout()
        warpedTime_label = QtWidgets.QLabel('Warped Time :')
        warpedTime_label.setFixedWidth(140)
        warpedTime_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.warpedTim_lineEdit = QtWidgets.QLineEdit()
        self.warpedTim_lineEdit.setReadOnly(True)
        horizontalSpacer_1 = QtWidgets.QSpacerItem(20, 10,
                                                   QtWidgets.QSizePolicy.Expanding,
                                                   QtWidgets.QSizePolicy.Minimum)
        warpedTimeLayout.addWidget(warpedTime_label)
        warpedTimeLayout.addWidget(self.warpedTim_lineEdit)
        warpedTimeLayout.addItem(horizontalSpacer_1)

        unwarpedTimeLayout = QtWidgets.QHBoxLayout()
        unwarpedTime_label = QtWidgets.QLabel('Unwarped Time :')
        unwarpedTime_label.setFixedWidth(140)
        unwarpedTime_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.unwarpedTim_lineEdit = QtWidgets.QLineEdit()
        self.unwarpedTim_lineEdit.setReadOnly(True)
        horizontalSpacer_3 = QtWidgets.QSpacerItem(20, 10,
                                                   QtWidgets.QSizePolicy.Expanding,
                                                   QtWidgets.QSizePolicy.Minimum)
        unwarpedTimeLayout.addWidget(unwarpedTime_label)
        unwarpedTimeLayout.addWidget(self.unwarpedTim_lineEdit)
        unwarpedTimeLayout.addItem(horizontalSpacer_3)

        button_layout_0 = QtWidgets.QHBoxLayout()
        self.selectWarp_btn = QtWidgets.QPushButton('Select This Warp')
        self.selectWarp_btn.setMinimumWidth(150)
        self.selectApplied_btn = QtWidgets.QPushButton('Select Applied')
        button_layout_0.addWidget(self.selectWarp_btn)
        button_layout_0.addWidget(self.selectApplied_btn)

        button_layout_1 = QtWidgets.QHBoxLayout()
        self.applyWarp_btn = QtWidgets.QPushButton('Apply Warp to Selection')
        self.applyWarp_btn.setStyleSheet("color: rgb(0,243,0)")
        self.applyWarp_btn.setMinimumWidth(150)
        self.removeWarp_btn = QtWidgets.QPushButton('Remove Warp from Selection')
        button_layout_1.addWidget(self.applyWarp_btn)
        button_layout_1.addWidget(self.removeWarp_btn)

        button_layout_2 = QtWidgets.QHBoxLayout()
        self.deleteWarp_btn = QtWidgets.QPushButton('Delete This Warp')
        self.deleteWarp_btn.setStyleSheet("background-color: rgb(216, 33, 33)")
        self.deleteWarp_btn.setMinimumWidth(150)
        self.close_btn = QtWidgets.QPushButton('Close')
        button_layout_2.addWidget(self.deleteWarp_btn)
        button_layout_2.addWidget(self.close_btn)


        self.main_layout.addLayout(comboBox_layout)
        self.main_layout.addLayout(warpedTimeLayout)
        self.main_layout.addLayout(unwarpedTimeLayout)
        self.main_layout.addLayout(button_layout_0)
        self.main_layout.addLayout(button_layout_1)
        self.main_layout.addLayout(button_layout_2)

        self.setDefualt()
        self.setEnableUI(False)
        self.connectSignals()
        self.reloadComboBox()

    def connectSignals(self):
        self.comboBox.currentIndexChanged.connect(self.comboBoxAction)
        self.enableWarp_checkBox.stateChanged.connect(self.enableTimewarp)
        self.selectWarp_btn.clicked.connect(self.selectWarp)
        self.selectApplied_btn.clicked.connect(self.selectApplied)
        self.applyWarp_btn.clicked.connect(self.applyWarp)
        self.removeWarp_btn.clicked.connect(self.removeWarp)
        self.deleteWarp_btn.clicked.connect(self.deleteWarp)
        self.close_btn.clicked.connect(self.close)

    def setDefualt(self):
        self.enableWarp_checkBox.setChecked(False)
        self.warpedTim_lineEdit.setText('0.000')
        self.unwarpedTim_lineEdit.setText('0.000')

    def setEnableUI(self, state):
        self.enableWarp_checkBox.setEnabled(state)
        self.warpedTim_lineEdit.setEnabled(state)
        self.unwarpedTim_lineEdit.setEnabled(state)
        self.selectWarp_btn.setEnabled(state)
        self.selectApplied_btn.setEnabled(state)
        self.applyWarp_btn.setEnabled(state)
        self.removeWarp_btn.setEnabled(state)
        self.deleteWarp_btn.setEnabled(state)

    def getTimewarpCurves(self):
        timeCurves = cmds.ls("*DD_TimeCurve*")
        return timeCurves

    def enableTimewarp(self):
        cls = timeWarp.TimeWarp()
        cls.timeCurve = str(self.comboBox.currentText())
        state = self.enableWarp_checkBox.isChecked()
        cls.enable(state)

    def updateTimeInfo(self, timeCurve):
        cls = timeWarp.TimeWarp()
        cls.timeCurve = timeCurve
        timeInfo = cls.getTimeInfo()
        warpedTime = timeInfo.get('warpedTime', 0.000)
        unWarpedTime = timeInfo.get('unWarpedTime', 0.000)
        self.warpedTim_lineEdit.setText(str(warpedTime))
        self.unwarpedTim_lineEdit.setText(str(unWarpedTime))

    def reloadComboBox(self):
        self.comboBox.clear()
        timeCurves = self.getTimewarpCurves()
        if timeCurves:
            self.comboBox.addItems(COMBOBOX_ITEMS)
            self.comboBox.insertSeparator(3)
            self.comboBox.addItems(timeCurves)
        else:
            self.comboBox.addItems(COMBOBOX_ITEMS)

    def comboBoxAction(self):
        currentItem = str(self.comboBox.currentText())
        if currentItem == COMBOBOX_ITEMS[0]:
            self.setDefualt()
            self.setEnableUI(False)
        elif currentItem == COMBOBOX_ITEMS[1]:
            logger.debug("Create Warp")

            antiName = None
            self.createWarp(antiName)
        elif currentItem == COMBOBOX_ITEMS[2]:
            logger.debug("Create Anti-Warp")

            antiName = "DD_TimeCurve_AntiWarp"
            self.createWarp(antiName)
        elif currentItem:
            self.updateTimeInfo(currentItem)
            self.setEnableUI(True)

    def createWarp(self, antiName):
        cls = timeWarp.TimeWarp()
        timeCurve = cls.createNodes(antiName)

        self.reloadComboBox()
        index = self.comboBox.findText(timeCurve)
        self.comboBox.setCurrentIndex(index)
        self.setEnableUI(True)
        self.enableWarp_checkBox.setChecked(True)

    def selectWarp(self):
        timeCurve = str(self.comboBox.currentText())
        cmds.select(timeCurve)

    def selectApplied(self):
        pass

    @aniCommon.undo
    def applyWarp(self):
        cls = timeWarp.TimeWarp()
        cls.timeCurve = str(self.comboBox.currentText())
        cls.selection = cmds.ls(sl=True)
        cls.apply()

    @aniCommon.undo
    def removeWarp(self):
        timeCurve = str(self.comboBox.currentText())
        selection = cmds.ls(sl=True)

        if not selection:
            raise Exception("Please Select Object")

        timeWarp.TimeWarp.remove(timeCurve, selection)

    @aniCommon.undo
    def deleteWarp(self):
        timeCurve = str(self.comboBox.currentText())
        timeWarp.TimeWarp.delete(timeCurve)


