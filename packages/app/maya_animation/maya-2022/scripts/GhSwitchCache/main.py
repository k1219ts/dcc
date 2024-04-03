# -*- coding: utf-8 -*-
__author__ = 'gyeongheon.jeong'

import os
from PySide2 import QtCore, QtGui, QtWidgets
import logging
import maya.OpenMaya as OpenMaya
import maya.cmds as cmds
import modules

reload(modules)

logger = logging.getLogger(__name__)

_win = None

def showUI():
    global _win
    if _win:
        _win.close()
        # _win.deleteLater()
    _win = SwitchCache()
    _win.show()
    _win.resize(600, 500)


class SwitchCache(QtWidgets.QMainWindow):
    #selChangedCallback = None

    def __init__(self, parent=None):
        super(SwitchCache, self).__init__(parent)
        self.setWindowTitle("Dexter Animation - Switch Cache")

        # main widget
        main_widget = QtWidgets.QWidget(self)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        radioBtnLayout = QtWidgets.QHBoxLayout()
        self.fromSel_radioBtn = QtWidgets.QRadioButton()
        self.fromSel_radioBtn.setText("From Selection")
        self.fromAll_radioBtn = QtWidgets.QRadioButton()
        self.fromAll_radioBtn.setText("All Character")
        self.fromAll_radioBtn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                            QtWidgets.QSizePolicy.Fixed)
        self.fromAll_radioBtn.setEnabled(False)
        verticalSpacer1 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        emailLabel = QtWidgets.QLabel("gyeongheon.jeong@tactic.com")
        radioBtnLayout.addWidget(self.fromSel_radioBtn)
        radioBtnLayout.addWidget(self.fromAll_radioBtn)
        radioBtnLayout.addItem(verticalSpacer1)
        radioBtnLayout.addWidget(emailLabel)
        main_layout.addLayout(radioBtnLayout)

        # table widget
        self.nodeListTableWidget = QtWidgets.QTableWidget()
        self.nodeListTableWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                               QtWidgets.QSizePolicy.Expanding)
        self.nodeListTableWidget.setColumnCount(3)
        self.nodeListTableWidget.setHorizontalHeaderLabels(['Name', 'Type', 'Cache'])
        self.nodeListTableWidgetHeader = self.nodeListTableWidget.horizontalHeader()
        # self.nodeListTableWidgetHeader.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.nodeListTableWidgetHeader.setStretchLastSection(True)
        self.nodeListTableWidgetHeader.resizeSection(0, 380)
        self.nodeListTableWidget.setColumnWidth(1, 100)
        main_layout.addWidget(self.nodeListTableWidget)

        # tab widget
        self.main_tab_widget = QtWidgets.QTabWidget()
        self.main_tab_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Minimum)

        """Create tab"""
        create_widget = QtWidgets.QWidget()
        create_layout = QtWidgets.QVBoxLayout(create_widget)

        # start, end, step options
        options_layout = QtWidgets.QHBoxLayout()
        startframeLabel = QtWidgets.QLabel('Start : ')
        self.startFrameLineEdit = QtWidgets.QLineEdit()
        endframeLabel = QtWidgets.QLabel('End : ')
        self.endFrameLineEdit = QtWidgets.QLineEdit()
        stepLabel = QtWidgets.QLabel('Step : ')
        self.stepLineEdit = QtWidgets.QLineEdit('1.0')
        verticalSpacer2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Minimum)
        self.startFrameLineEdit.setFixedWidth(60)
        self.endFrameLineEdit.setFixedWidth(60)
        self.stepLineEdit.setFixedWidth(60)
        options_layout.addWidget(startframeLabel)
        options_layout.addWidget(self.startFrameLineEdit)
        options_layout.addWidget(endframeLabel)
        options_layout.addWidget(self.endFrameLineEdit)
        options_layout.addWidget(stepLabel)
        options_layout.addWidget(self.stepLineEdit)
        options_layout.addItem(verticalSpacer2)
        create_layout.addLayout(options_layout)
        create_layout.addWidget(self.HLine())
        # --------------------------------------------------------------------------- #

        # file browser widgets
        fileBrowse_layout = QtWidgets.QHBoxLayout()
        outpathLabel = QtWidgets.QLabel('Output Path : ')
        self.outCachePathLineEdit = QtWidgets.QLineEdit()
        self.browseFilePathButton = QtWidgets.QPushButton()
        self.browseFilePathButton.setText('. . .')
        self.browseFilePathButton.setFixedSize(30, 20)
        fileBrowse_layout.addWidget(outpathLabel)
        fileBrowse_layout.addWidget(self.outCachePathLineEdit)
        fileBrowse_layout.addWidget(self.browseFilePathButton)
        create_layout.addLayout(fileBrowse_layout)
        create_layout.addWidget(self.HLine())
        # --------------------------------------------------------------------------- #

        # create cache button
        self.createCacheBtn = QtWidgets.QPushButton()
        self.createCacheBtn.setText("Create Cache")
        self.createCacheBtn.setMinimumHeight(50)
        self.createCacheBtn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Fixed)
        create_layout.addWidget(self.createCacheBtn)

        """Switch tab"""
        switch_widget = QtWidgets.QWidget()
        switch_layout = QtWidgets.QVBoxLayout(switch_widget)

        # gcd rig list
        gcdRigList_layout = QtWidgets.QHBoxLayout()
        gcdRigSelectLabel = QtWidgets.QLabel('Rigging File Path')
        #        self.gcdRigListComboBox = QtWidgets.QComboBox()
        #        self.gcdRigListComboBox.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
        #                                              QtWidgets.QSizePolicy.Minimum)
        self.rigFilePathLineEdit = QtWidgets.QLineEdit()
        self.rigBrowseButton = QtWidgets.QPushButton('. . .')
        self.rigBrowseButton.setFixedSize(30, 20)
        gcdRigList_layout.addWidget(gcdRigSelectLabel)
        gcdRigList_layout.addWidget(self.rigFilePathLineEdit)
        gcdRigList_layout.addWidget(self.rigBrowseButton)
        switch_layout.addLayout(gcdRigList_layout)
        switch_layout.addWidget(self.HLine())

        # cache file browser widgets
        cacheFileBrowse_layout = QtWidgets.QHBoxLayout()
        cachePathLabel = QtWidgets.QLabel('Cache Path : ')
        self.cachePathLineEdit = QtWidgets.QLineEdit()
        self.browseCachePathButton = QtWidgets.QPushButton()
        self.browseCachePathButton.setText('. . .')
        self.browseCachePathButton.setFixedSize(30, 20)
        cacheFileBrowse_layout.addWidget(cachePathLabel)
        cacheFileBrowse_layout.addWidget(self.cachePathLineEdit)
        cacheFileBrowse_layout.addWidget(self.browseCachePathButton)
        switch_layout.addLayout(cacheFileBrowse_layout)
        switch_layout.addWidget(self.HLine())

        # switch button
        self.switchCacheBtn = QtWidgets.QPushButton()
        self.switchCacheBtn.setText("Switch")
        self.switchCacheBtn.setMinimumHeight(50)
        self.switchCacheBtn.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Expanding)
        switch_layout.addWidget(self.switchCacheBtn)

        # add tabs to main layout
        self.main_tab_widget.addTab(switch_widget, "Switch")
        self.main_tab_widget.addTab(create_widget, "Create")
        main_layout.addWidget(self.main_tab_widget)

        # match ground widget
        mg_groupBox = QtWidgets.QGroupBox("Match Rig To Ground")
        mg_groupBox_mainLayout = QtWidgets.QVBoxLayout(mg_groupBox)
        mg_groupBox_HLayout = QtWidgets.QHBoxLayout()
        mg_label = QtWidgets.QLabel("Ground Mesh : ")
        self.mg_groundMesh_lineEdit = QtWidgets.QLineEdit()
        self.mg_groundMesh_addButton = QtWidgets.QPushButton("Add Selection")
        self.mg_doItButton = QtWidgets.QPushButton('Match to ground')
        mg_groupBox_HLayout.addWidget(mg_label)
        mg_groupBox_HLayout.addWidget(self.mg_groundMesh_lineEdit)
        mg_groupBox_HLayout.addWidget(self.mg_groundMesh_addButton)
        mg_groupBox_mainLayout.addLayout(mg_groupBox_HLayout)
        mg_groupBox_mainLayout.addWidget(self.mg_doItButton)
        main_layout.addWidget(mg_groupBox)

        self.initUI()
        self.connectSignals()
        self.mayaSelectionChangeCallback()

    def HLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.HLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def VLine(self):
        toto = QtWidgets.QFrame()
        toto.setFrameShape(QtWidgets.QFrame.VLine)
        toto.setFrameShadow(QtWidgets.QFrame.Sunken)
        return toto

    def initUI(self):
        minTime = int(cmds.playbackOptions(q=True, min=True))
        maxTime = int(cmds.playbackOptions(q=True, max=True))
        dataPath = modules.getBackstageDataPath()
        gcdRigs = modules.gcdRigList()

        self.fromSel_radioBtn.setChecked(True)
        self.startFrameLineEdit.setText(str(minTime))
        self.endFrameLineEdit.setText(str(maxTime))
        self.outCachePathLineEdit.setText(str(dataPath))
        self.rigFilePathLineEdit.setText(str(gcdRigs[0]))
        self.cachePathLineEdit.setText(str(dataPath))

#        self.gcdRigListComboBox.addItems(gcdRigs)


    def connectSignals(self):
        self.fromAll_radioBtn.toggled.connect(self.fromSelChanged)
        self.browseFilePathButton.clicked.connect(lambda: self.browseOutPath(self.outCachePathLineEdit, isDir=True))
        self.rigBrowseButton.clicked.connect(lambda: self.browseOutPath(self.rigFilePathLineEdit, isDir=False))
        self.browseCachePathButton.clicked.connect(lambda: self.browseOutPath(self.cachePathLineEdit, isDir=True))
        self.createCacheBtn.clicked.connect(self.create)
        self.switchCacheBtn.clicked.connect(self.switch)
        self.mg_groundMesh_addButton.clicked.connect(self.addSelectedGround)
        self.mg_doItButton.clicked.connect(self.matchGround)

    def closeEvent(self, event):
        OpenMaya.MMessage.removeCallback(self.selChangedCallback)
        reply = QtWidgets.QMessageBox.question(self, 'Message', 'Are you sure to quit?',
                                               QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def fromSelChanged(self):
        """Selection query radio button change event
        """
        if self.fromAll_radioBtn.isChecked():
            dxRigs = cmds.ls(type='dxRig')
            self.nodeListTableWidget.setRowCount(len(dxRigs))
            for i, item in enumerate(dxRigs):
                itemType = cmds.objectType(item)
                self.nodeListTableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(item))
                self.nodeListTableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(itemType))
                self.nodeListTableWidget.item(i, 1).setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.nodeListTableWidget.resizeRowsToContents()
        else:
            self.updateSelectionList()

    def mayaSelectionChangeCallback(self):
        """Create Maya's selection changed callback
        """
        self.selChangedCallback = OpenMaya.MEventMessage.addEventCallback("SelectionChanged",
                                                                          self.updateSelectionList)

    def browseOutPath(self, lineEdit, isDir=True):
        startPath = modules.getBackstageDataPath()
        if isDir:
            outPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'select directory', startPath)
        else:
            outPath = QtWidgets.QFileDialog.getOpenFileName(self, 'select file')[0]
        if outPath:
            lineEdit.setText(outPath)

    def updateSelectionList(self, *args, **kwargs):
        """Add selection list to table widget
        """
        selectedObject = cmds.ls(sl=True)

        if self.fromAll_radioBtn.isChecked():
            return
        rowCount = 1
        nodeList = list()
        if not selectedObject:
            self.nodeListTableWidget.setRowCount(0)
            return
        for i, item in enumerate(selectedObject):
            node = modules.getRootNode(item, type='dxNode')
            if (not node) and (len(selectedObject) == 1):
                self.nodeListTableWidget.setRowCount(0)
                return
            if (not node) or (node in nodeList):
                continue
            nodeList.append(node)
            self.nodeListTableWidget.setRowCount(rowCount)
            objectType = cmds.objectType(node)
            self.nodeListTableWidget.setItem(rowCount - 1, 0,
                                             QtWidgets.QTableWidgetItem(node))
            self.nodeListTableWidget.setItem(rowCount - 1, 1,
                                             QtWidgets.QTableWidgetItem(objectType))
            self.nodeListTableWidget.item(rowCount - 1, 1).setTextAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            rowCount += 1
        self.nodeListTableWidget.resizeRowsToContents()

        if not nodeList:
            self.nodeListTableWidget.setRowCount(0)

    def getSelectionList(self):
        """Get selection list from table widget
        
        :return: A list of table widget items
        """
        selection = list()
        rowCount = self.nodeListTableWidget.rowCount()
        for row in range(rowCount):
            item = str(self.nodeListTableWidget.item(row, 0).text())
            selection.append(item)
        return selection

    def getOptionsFromUI(self):
        dataPath = str(self.outCachePathLineEdit.text())
        filePath = os.sep.join([dataPath, 'geoCache/crowd'])
        cachePath = str(self.cachePathLineEdit.text())
        crowdCachePath = os.sep.join([cachePath, 'geoCache/crowd'])

        options = dict()
        options['filepath'] = filePath
        options['crowdCachePath'] = crowdCachePath
        options['nodes'] = self.getSelectionList()
        options['startFrame'] = int(self.startFrameLineEdit.text())
        options['endFrame'] = int(self.endFrameLineEdit.text())
        options['step'] = float(self.stepLineEdit.text())
        return options

    def create(self):
        """Create Cache from dxRig
        """
        fromSelection = self.fromSel_radioBtn.isChecked()
        options = self.getOptionsFromUI()
        modules.exportCache(selection=fromSelection, options=options)

    def switch(self):
        """Switch cache to rig, or rig to cache
        """
        opts = self.getOptionsFromUI()
        fromSelection = self.fromSel_radioBtn.isChecked()

        # ------------------------- gcd only ---------------------------
#        currentGcdRig = str(self.gcdRigListComboBox.currentText())
        currentGcdRig = str(self.rigFilePathLineEdit.text())
        opts.update({'rigfile': currentGcdRig})
        # --------------------------------------------------------------

        if os.path.isfile(currentGcdRig):
            logger.debug("switch to : {0}".format(currentGcdRig))
            modules.switchCache(selection=fromSelection, options=opts)


    def addSelectedGround(self):
        object = cmds.ls(sl=True)[0]
        objectShape = cmds.listRelatives(object, s=True)[0]
        if cmds.objectType(objectShape) == "mesh":
            self.mg_groundMesh_lineEdit.setText(object)


    def matchGround(self):
        nodes = self.getSelectionList()
        ground = str(self.mg_groundMesh_lineEdit.text())
        for node in nodes:
            nameSpace = modules.getNameSpace(node)
            modules.matchRigToGround(nameSpace, ground)