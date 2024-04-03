# encoding:utf-8

import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya
import switch; reload(switch)
from PySide2 import QtCore, QtGui, QtWidgets

class Fk2Ik(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(Fk2Ik, self).__init__(parent)
        self.setWindowTitle('Fk Ik Switch')
        self.move(QtCore.QPoint(1920 / 2, 500))

        main_widget = QtWidgets.QWidget(self)
        main_vbox = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        self.handLocalSpace_checkBox = QtWidgets.QCheckBox('Hand Local Space')
        self.fk2ik_Btn = QtWidgets.QPushButton('SNAP')
        self.fk2ik_Btn.clicked.connect(lambda : self.switch(command='switch'))

        main_vbox.addWidget(self.handLocalSpace_checkBox)
        main_vbox.addWidget(self.fk2ik_Btn)

        verticalSpacer = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        main_vbox.addWidget(self.HLine())

        self.bakeToCurrentTimeCheckBox = QtWidgets.QCheckBox('Bake to current time')
        self.bakeToCurrentTimeCheckBox.setChecked(True)
        self.bakeToCurrentTimeCheckBox.stateChanged.connect(self.endtime)
        main_vbox.addWidget(self.bakeToCurrentTimeCheckBox)

        baketime_layout = QtWidgets.QHBoxLayout()
        bakeTime_start_label = QtWidgets.QLabel('start')
        self.bakeTime_start = QtWidgets.QLineEdit(str(int(cmds.playbackOptions(q=True, min=True))))
        self.bakeTime_start.setFixedWidth(60)
        bakeTime_end_label = QtWidgets.QLabel('end')
        self.bakeTime_end = QtWidgets.QLineEdit(str(int(cmds.playbackOptions(q=True, max=True))))
        self.bakeTime_end.setFixedWidth(60)
        baketime_layout.addWidget(bakeTime_start_label)
        baketime_layout.addWidget(self.bakeTime_start)
        baketime_layout.addWidget(bakeTime_end_label)
        baketime_layout.addWidget(self.bakeTime_end)
        baketime_layout.addItem(verticalSpacer)
        main_vbox.addLayout(baketime_layout)

        self.bakeButton = QtWidgets.QPushButton('Bake')
        self.bakeButton.clicked.connect(lambda : self.switch(command='bake'))
        main_vbox.addWidget(self.bakeButton)

        self.endtime()
        self.mayaTimeChangeCallback()

    def closeEvent(self, event):
        OpenMaya.MMessage.removeCallback(self.currentTimeChangedCallback)
        event.accept()


    def mayaTimeChangeCallback(self):
        self.currentTimeChangedCallback = OpenMaya.MEventMessage.addEventCallback("timeChanged",
                                                                                  self.endtime)


    def HLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.HLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def endtime(self, *args, **kwargs):
        checkState = self.bakeToCurrentTimeCheckBox.isChecked()
        if checkState:
            currentTime = str(int(cmds.currentTime(q=True)))
            self.bakeTime_end.setText(currentTime)
        else:
            self.bakeTime_end.setText(str(int(cmds.playbackOptions(q=True, max=True))))
        self.bakeTime_end.setEnabled(not checkState)


    def selection(self):
        fullObjectNameList = list()
        _selection = OpenMaya.MGlobal.getActiveSelectionList()
        for i in range( _selection.length() ):
            node = _selection.getDagPath( i )
            fullObjectName = node.fullPathName()
            fullObjectNameList.append(fullObjectName)
        return fullObjectNameList


    def switch(self, command='switch'):
        nodes = self.selection()
        switchCls = switch.SnapKinematics()
        bakeControlers = list()

        for node in nodes:
            mode = "IK"
            if not node.find("FK") == -1:
                mode = "FK"
            switchCls.mode = mode
            switchCls.node = node
            if command == 'switch':
                switchCls.setHandLocalSpace = self.handLocalSpace_checkBox.isChecked()
                switchCls.snap()
            elif command == 'bake':
                switchCls.startTime = int(self.bakeTime_start.text())
                switchCls.endTime = int(self.bakeTime_end.text())
                bakeControlers += switchCls.getControlers()

        if command == 'bake':
            switchCls.bake(bakeControlers)

