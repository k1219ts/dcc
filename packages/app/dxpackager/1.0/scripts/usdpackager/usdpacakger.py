# coding: utf-8
import os, sys, shutil, re, time, datetime, subprocess, threading, getpass, logging, csv, yaml, glob, ftplib

from PySide2 import QtWidgets
from PySide2 import QtGui
from PySide2 import QtCore
import xlrd2

import tractor.api.author as author
from tractor.TractorEngine import TractorEngine

import assetpackage, shotpackage
import assetconvert
from usd2maya import PUtils as putl

import DXRulebook.Interface as rb

appver = '1.2.0'
runmode = 'r'
if runmode == 't':
    appver = appver+'(Test)'

FTPHOST = '10.0.0.61'
FTPPORT = 3000

extWdgWidth = 0
defaultFontSize = 10
defaultFont = QtGui.QFont()
defaultFont.setPointSize(defaultFontSize)

tablePreviewWidth = 200
tableRowHeight = 40
tablePreviewRowHeight = 112

previewFrame = '1001'

DEV = re.match("/works/", __file__)
PACKAGE_FOLDER = "/backstage/dcc/packages/app/dxpackager/1.0/scripts"
scriptsDir = os.path.join(PACKAGE_FOLDER, "usdpackager") if DEV else os.path.dirname(__file__)

def setShowConfig(show):
    showRbPath = '/show/{SHOW}/_config/DXRulebook.yaml'.format(SHOW=show)

    if os.path.exists(showRbPath):
        print ('>> showRbPath:{}'.format(showRbPath))
        os.environ['DXRULEBOOKFILE'] = showRbPath
    else:
        if os.environ.has_key('DXRULEBOOKFILE'):
            del os.environ['DXRULEBOOKFILE']

    rb.Reload()

def openFile(path):
    if sys.platform == 'win32':
        os.startfile(path)
    else:
        opener = 'xdg-open'
        if sys.platform == 'darwin':
            opener = 'open'
        subprocess.call([opener, path])

def createLogger(logName, logPath):
    if logPath == '':
        logPath = '/tmp/log/'+logName+'.log'
    logDir = os.path.dirname(logPath)
    if not os.path.isdir(logDir):
        try: os.makedirs(logDir)
        except: pass

    newLogger = logging.getLogger(logName)
    newLogger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)-15s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if os.path.isdir(logDir):
        fileHandler = logging.FileHandler(logPath)
        fileHandler.setFormatter(formatter)
        newLogger.addHandler(fileHandler)

    # else:
    # streamHander = logging.StreamHandler(sys.stdout)
    # streamHander.setFormatter(formatter)
    # newLogger.addHandler(streamHander)

    return newLogger

class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent=None, imgPath=None, imgWidth=0, imgHeight=0):
        super(ImageLabel, self).__init__(parent)

        self.parent = parent
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

        self.updateImage(imgPath)


    def updateImage(self, imgPath):
        self.imgPath = imgPath
        image = QtGui.QImage(imgPath)
        if image.isNull():
            image = QtGui.QImage(defaultImagePath)
        if self.imgWidth != 0:
            image = image.scaledToWidth(self.imgWidth, transformMode=QtCore.Qt.SmoothTransformation)
        if self.imgHeight != 0:
            image = image.scaledToWidth(self.imgHeight, transformMode=QtCore.Qt.SmoothTransformation)
        pixmap = QtGui.QPixmap(image)
        self.setPixmap(pixmap)

gMsgBox = None
def closeNotificationBox():
    global gMsgBox

    if gMsgBox != None:
        gMsgBox.close()
        QtWidgets.QApplication.processEvents()
        gMsgBox = None

def notificationBoxProgress(progressText):
    global gMsgBox

    if gMsgBox != None:
        prevText = gMsgBox.text()
        prevProgressText = prevText.split(' >> ')[0]
        if progressText == '100%':
            gMsgBox.setText(prevProgressText+' -- '+progressText)
        else:
            gMsgBox.setText(prevProgressText+' >> '+progressText)
        QtWidgets.QApplication.processEvents()

def notificationBox(parent, message, title='Notification', iconStyle='', autoClose=0):
    global gMsgBox

    maxMsgs = 20

    icon = QtWidgets.QMessageBox.Information
    if iconStyle == 'warning':
        icon = QtWidgets.QMessageBox.Warning
        if title == 'Notification': title='Warning'
    elif iconStyle == 'error':
        icon = QtWidgets.QMessageBox.Critical
        if title == 'Notification': title='Error'
    elif iconStyle == 'question':
        icon = QtWidgets.QMessageBox.Question
        if title == 'Notification': title='Question'

    prevText = ''
    if gMsgBox == None:
        gMsgBox = QtWidgets.QMessageBox(parent)
        gMsgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
        gMsgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        gMsgBox.setMaximumWidth(960)
        gMsgBox.setMaximumHeight(640)
        buttonO = gMsgBox.button(QtWidgets.QMessageBox.Ok)
        buttonO.setText('Ok')
        buttonO.clicked.connect(closeNotificationBox)
        gMsgBox.show()

    else:
        prevText = gMsgBox.text()

    gMsgBox.setIcon(icon)
    gMsgBox.setWindowTitle(title)
    prevMsgs = prevText.split('\n')
    if len(prevMsgs) > maxMsgs:
        prevText = '(...)\n'+'\n'.join(prevMsgs[-maxMsgs:])
    gMsgBox.setText(prevText+'\n'+message)
    QtWidgets.QApplication.processEvents()

    if autoClose > 0:
        time.sleep(autoClose)
        gMsgBox.deleteLater()
        gMsgBox = None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.title = 'USD Packager '+appver
        self.closed = False
        self.menuBar = QtWidgets.QMenuBar(self)
        self.menuBar.setFont(defaultFont)
        self.setMenuBar(self.menuBar)
        self.fileMenu = self.menuBar.addMenu('&File')
        self.fileMenu.setFont(defaultFont)
        self.fileMenu.addAction("&Open", self.openDataFile)
        self.fileMenu.addAction("E&xit", QtWidgets.qApp.closeAllWindows)

        self.mainWidget = QtWidgets.QWidget(self)

        mainLayout = QtWidgets.QVBoxLayout()

        projectLayout = QtWidgets.QHBoxLayout()
        projectLayout.setAlignment(QtCore.Qt.AlignVCenter)
        self.projectLabel = QtWidgets.QLabel('Project:')
        self.projectLabel.setFont(defaultFont)
        projectLayout.addWidget(self.projectLabel)

        self.projectComboBox = QtWidgets.QComboBox(self)
        self.projectComboBox.setFont(defaultFont)
        self.projectComboBox.setFixedSize(120+extWdgWidth, 22)
        projectListView = QtWidgets.QListView()
        projectListView.setFont(defaultFont)
        self.projectComboBox.setView(projectListView)
        projectLayout.addWidget(self.projectComboBox)

        self.ptypeSpaceLabel = QtWidgets.QLabel('    ')
        self.ptypeSpaceLabel.setFont(defaultFont)
        projectLayout.addWidget(self.ptypeSpaceLabel)

        self.ptypeLabel = QtWidgets.QLabel('Type:')
        self.ptypeLabel.setFont(defaultFont)
        projectLayout.addWidget(self.ptypeLabel)

        self.ptypeComboBox = QtWidgets.QComboBox(self)
        self.ptypeComboBox.setFont(defaultFont)
        self.ptypeComboBox.setFixedSize(60+extWdgWidth, 22)
        ptypeListView = QtWidgets.QListView()
        ptypeListView.setFont(defaultFont)
        self.ptypeComboBox.addItem('_3d')
        self.ptypeComboBox.addItem('_2d')
        self.ptypeComboBox.setView(ptypeListView)
        self.ptypeComboBox.currentTextChanged.connect(self.ptypeComboBoxChanged)
        projectLayout.addWidget(self.ptypeComboBox)

        self.pcatgSpaceLabel = QtWidgets.QLabel('    ')
        self.pcatgSpaceLabel.setFont(defaultFont)
        projectLayout.addWidget(self.pcatgSpaceLabel)

        self.pcatgLabel = QtWidgets.QLabel('Category:')
        self.pcatgLabel.setFont(defaultFont)
        projectLayout.addWidget(self.pcatgLabel)

        self.pcatgComboBox = QtWidgets.QComboBox(self)
        self.pcatgComboBox.setFont(defaultFont)
        self.pcatgComboBox.setFixedSize(80+extWdgWidth, 22)
        pcatgListView = QtWidgets.QListView()
        pcatgListView.setFont(defaultFont)
        self.pcatgComboBox.addItem('asset')
        self.pcatgComboBox.addItem('shot')
        self.pcatgComboBox.setView(pcatgListView)
        self.pcatgComboBox.currentTextChanged.connect(self.pcatgComboBoxChanged)
        projectLayout.addWidget(self.pcatgComboBox)

        findButton = QtWidgets.QPushButton('Find')
        findButton.setFont(defaultFont)
        findButton.setFixedSize(80+extWdgWidth, 24)
        findButton.clicked.connect(self.findButtonClicked)
        projectLayout.addWidget(findButton)

        taskCheckLayout = QtWidgets.QVBoxLayout()
        self.p2dAssetTaskCheckWidget = QtWidgets.QWidget()
        p2dAssetTaskCheckLayout = QtWidgets.QHBoxLayout(self.p2dAssetTaskCheckWidget)
        p2dAssetLookdevCheckBox = QtWidgets.QCheckBox('lookdev')
        p2dAssetLookdevCheckBox.setFont(defaultFont)
        p2dAssetTaskCheckLayout.addWidget(p2dAssetLookdevCheckBox)
        taskCheckLayout.addWidget(self.p2dAssetTaskCheckWidget)

        self.p2dShotTaskCheckWidget = QtWidgets.QWidget()
        p2dShotTaskCheckLayout = QtWidgets.QHBoxLayout(self.p2dShotTaskCheckWidget)
        p2dShotPlatesCheckBox = QtWidgets.QCheckBox('plates')
        p2dShotPlatesCheckBox.setFont(defaultFont)
        p2dShotTaskCheckLayout.addWidget(p2dShotPlatesCheckBox)
        p2dShotImageplaneCheckBox = QtWidgets.QCheckBox('imageplane')
        p2dShotImageplaneCheckBox.setFont(defaultFont)
        p2dShotTaskCheckLayout.addWidget(p2dShotImageplaneCheckBox)
        p2dShotLightingCheckBox = QtWidgets.QCheckBox('lighting')
        p2dShotLightingCheckBox.setFont(defaultFont)
        p2dShotTaskCheckLayout.addWidget(p2dShotLightingCheckBox)
        taskCheckLayout.addWidget(self.p2dShotTaskCheckWidget)

        self.p3dAssetTaskCheckWidget = QtWidgets.QWidget()
        p3dAssetTaskCheckLayout = QtWidgets.QHBoxLayout(self.p3dAssetTaskCheckWidget)
        p3dAssetModelCheckBox = QtWidgets.QCheckBox('model')
        p3dAssetModelCheckBox.setFont(defaultFont)
        p3dAssetTaskCheckLayout.addWidget(p3dAssetModelCheckBox)
        # p3dAssetTextureCheckBox = QtWidgets.QCheckBox('texture')
        # p3dAssetTextureCheckBox.setFont(defaultFont)
        # p3dAssetTaskCheckLayout.addWidget(p3dAssetTextureCheckBox)
        p3dAssetRigCheckBox = QtWidgets.QCheckBox('rig')
        p3dAssetRigCheckBox.setFont(defaultFont)
        p3dAssetTaskCheckLayout.addWidget(p3dAssetRigCheckBox)
        p3dAssetLidarCheckBox = QtWidgets.QCheckBox('lidar')
        p3dAssetLidarCheckBox.setFont(defaultFont)
        p3dAssetTaskCheckLayout.addWidget(p3dAssetLidarCheckBox)
        p3dAssetAgentCheckBox = QtWidgets.QCheckBox('agent')
        p3dAssetAgentCheckBox.setFont(defaultFont)
        p3dAssetTaskCheckLayout.addWidget(p3dAssetAgentCheckBox)
        p3dAssetClipCheckBox = QtWidgets.QCheckBox('clip')
        p3dAssetClipCheckBox.setFont(defaultFont)
        p3dAssetTaskCheckLayout.addWidget(p3dAssetClipCheckBox)
        # prevtex texture: 벤더에 애니메이션용 텍스처를 보내기 위한 디퓨즈맵(diffC) 패키징
        #   /show/slc/_3d/asset/miran/texture/images/v001/miran_diffC.tif
        # p3dAssetPreviewTextureCheckBox = QtWidgets.QCheckBox('prevtex')
        # p3dAssetPreviewTextureCheckBox.setFont(defaultFont)
        # p3dAssetTaskCheckLayout.addWidget(p3dAssetPreviewTextureCheckBox)
        # p3dAssetExrTextureCheckBox = QtWidgets.QCheckBox('exrtex')
        # p3dAssetExrTextureCheckBox.setFont(defaultFont)
        # p3dAssetTaskCheckLayout.addWidget(p3dAssetExrTextureCheckBox)
        self.p3dAssetTextureComboBox = QtWidgets.QComboBox(self)
        self.p3dAssetTextureComboBox.setFont(defaultFont)
        self.p3dAssetTextureComboBox.setFixedSize(80+extWdgWidth, 22)
        p3dAssetTextureView = QtWidgets.QListView()
        p3dAssetTextureView.setFont(defaultFont)
        self.p3dAssetTextureComboBox.addItem('texture')
        self.p3dAssetTextureComboBox.addItem('jpg')
        self.p3dAssetTextureComboBox.addItem('exr')
        self.p3dAssetTextureComboBox.addItem('tif')
        self.p3dAssetTextureComboBox.setView(p3dAssetTextureView)
        p3dAssetTaskCheckLayout.addWidget(self.p3dAssetTextureComboBox)
        taskCheckLayout.addWidget(self.p3dAssetTaskCheckWidget)

        self.p3dShotTaskCheckWidget = QtWidgets.QWidget()
        p3dShotTaskCheckLayout = QtWidgets.QHBoxLayout(self.p3dShotTaskCheckWidget)
        p3dCamCheckBox = QtWidgets.QCheckBox('cam')
        p3dCamCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dCamCheckBox)
        p3dShotAniCheckBox = QtWidgets.QCheckBox('ani')
        p3dShotAniCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dShotAniCheckBox)
        p3dShotLayoutCheckBox = QtWidgets.QCheckBox('layout')
        p3dShotLayoutCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dShotLayoutCheckBox)
        p3dShotGroomCheckBox = QtWidgets.QCheckBox('groom')
        p3dShotGroomCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dShotGroomCheckBox)
        p3dShotSimCheckBox = QtWidgets.QCheckBox('sim')
        p3dShotSimCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dShotSimCheckBox)
        p3dShotCrowdCheckBox = QtWidgets.QCheckBox('crowd')
        p3dShotCrowdCheckBox.setFont(defaultFont)
        p3dShotTaskCheckLayout.addWidget(p3dShotCrowdCheckBox)
        taskCheckLayout.addWidget(self.p3dShotTaskCheckWidget)

        addressLayout = QtWidgets.QHBoxLayout()
        addressLeftLayout = QtWidgets.QHBoxLayout()
        addressLeftLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        addressLayout.addLayout(addressLeftLayout)

        addressRightLayout = QtWidgets.QHBoxLayout()
        addressRightLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        addressLayout.addLayout(addressRightLayout)

        packageDirLabel = QtWidgets.QLabel('Package Dir: ')
        packageDirLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        packageDirLabel.setFont(defaultFont)
        addressLeftLayout.addWidget(packageDirLabel)

        self.packageDir = QtWidgets.QLineEdit()
        self.packageDir.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.packageDir.setFont(defaultFont)
        # self.packageDir.textChanged.connect(self.packageDirChanged)
        addressLeftLayout.addWidget(self.packageDir)

        browsePackageDirButton = QtWidgets.QPushButton('Browse')
        browsePackageDirButton.setFont(defaultFont)
        browsePackageDirButton.setFixedSize(80+extWdgWidth, 24)
        browsePackageDirButton.clicked.connect(self.browsePackageDirButtonClicked)
        addressRightLayout.addWidget(browsePackageDirButton)

        optionLayout = QtWidgets.QHBoxLayout()
        optionLeftLayout = QtWidgets.QHBoxLayout()
        optionLeftLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        optionLayout.addLayout(optionLeftLayout)

        optionRightLayout = QtWidgets.QHBoxLayout()
        optionRightLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        optionLayout.addLayout(optionRightLayout)

        optionBottomLayout = QtWidgets.QHBoxLayout()
        optionBottomLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        searchKeywordLabel = QtWidgets.QLabel('Search: ')
        searchKeywordLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        searchKeywordLabel.setFont(defaultFont)
        optionBottomLayout.addWidget(searchKeywordLabel)

        self.searchKeyword = QtWidgets.QLineEdit()
        self.searchKeyword.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.searchKeyword.setFont(defaultFont)
        self.searchKeyword.textChanged.connect(self.searchKeywordChanged)
        optionBottomLayout.addWidget(self.searchKeyword)

        checkAllButton = QtWidgets.QPushButton('Check All')
        checkAllButton.setFont(defaultFont)
        checkAllButton.setFixedSize(100+extWdgWidth, 24)
        checkAllButton.clicked.connect(self.checkAllButtonButtonClicked)
        optionBottomLayout.addWidget(checkAllButton)

        uncheckAllButton = QtWidgets.QPushButton('Uncheck All')
        uncheckAllButton.setFont(defaultFont)
        uncheckAllButton.setFixedSize(100+extWdgWidth, 24)
        uncheckAllButton.clicked.connect(self.uncheckAllButtonButtonClicked)
        optionBottomLayout.addWidget(uncheckAllButton)

        middleLayout = QtWidgets.QVBoxLayout()
        self.tableWidget = QtWidgets.QTableWidget(0, 10, self)
        self.tableWidget.setFont(defaultFont)
        # self.tableWidget.setStyleSheet("QTableWidget::item { color: rgba(0, 0, 0, 0); selection-color: rgba(0, 0, 0, 0); }")
        self.tableWidget.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.setAcceptDrops(True)
        self.tableWidget.setDragEnabled(False)
        self.tableWidget.viewport().installEventFilter(self)
        types = ['text/uri-list']
        types.extend(self.tableWidget.mimeTypes())
        self.tableWidget.mimeTypes = lambda: types

        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.doubleClicked.connect(self.tableDoubleClicked)

        self.twhHeader = self.tableWidget.horizontalHeader()
        # self.twhHeader.setStretchLastSection(True)
        self.twhHeader.setFont(defaultFont)
        self.twhHeader.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.twhHeader.setLineWidth(0)
        self.twhHeader.setStyleSheet("::section { color: black; background-color: lightgray; }")
        self.twvHeader = self.tableWidget.verticalHeader()
        self.twvHeader.setDefaultSectionSize(tableRowHeight)
        self.twvHeader.setVisible(False)
        # self.twvHeader.setSortIndicator(0, QtCore.Qt.AscendingOrder)

        self.tableHeader = ['#', 'NAME', 'CHECK', 'INFO']
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.setHorizontalHeaderLabels(self.tableHeader)
        self.twhHeader.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setColumnWidth(0, 60)
        self.tableWidget.setColumnWidth(1, 400)
        self.tableWidget.setColumnWidth(2, 80)
        self.tableWidget.setColumnWidth(3, 80)
        self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.showTableContextMenu)
        middleLayout.addWidget(self.tableWidget)

        # self.overwriteCheckBox = QtWidgets.QCheckBox('Overwrite Version')
        # self.overwriteCheckBox.setFont(defaultFont)
        # middleLayout.addWidget(self.overwriteCheckBox)

        bottomLayout = QtWidgets.QHBoxLayout()
        bottomLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.makePreviewButton = QtWidgets.QPushButton('Make Preview')
        self.makePreviewButton.setFont(defaultFont)
        self.makePreviewButton.setFixedSize(100+extWdgWidth, 24)
        self.makePreviewButton.clicked.connect(self.makePreviewButtonClicked)
        bottomLayout.addWidget(self.makePreviewButton)

        self.packagefmtComboBox = QtWidgets.QComboBox(self)
        self.packagefmtComboBox.setFont(defaultFont)
        self.packagefmtComboBox.setFixedSize(100+extWdgWidth, 22)
        self.packagefmtComboBox.currentTextChanged.connect(self.packagefmtComboBoxChanged)
        bottomLayout.addWidget(self.packagefmtComboBox)

        self.withAssetCheckBox = QtWidgets.QCheckBox('With Asset')
        self.withAssetCheckBox.setFont(defaultFont)
        bottomLayout.addWidget(self.withAssetCheckBox)
        self.TempRemoveCheckBox = QtWidgets.QCheckBox('TempRemove')
        self.TempRemoveCheckBox.setChecked(True)
        self.TempRemoveCheckBox.setFont(defaultFont)
        bottomLayout.addWidget(self.TempRemoveCheckBox)
        self.SmartBakeCheckBox = QtWidgets.QCheckBox('SmartBake')
        self.SmartBakeCheckBox.setChecked(False)
        self.SmartBakeCheckBox.setFont(defaultFont)
        bottomLayout.addWidget(self.SmartBakeCheckBox)
        self.tractorCheckBox = QtWidgets.QCheckBox('Tractor')
        self.tractorCheckBox.setChecked(True)
        self.tractorCheckBox.setFont(defaultFont)
        bottomLayout.addWidget(self.tractorCheckBox)
        self.packageButton = QtWidgets.QPushButton('Package')
        self.packageButton.setFont(defaultFont)
        self.packageButton.setFixedSize(140+extWdgWidth, 24)
        self.packageButton.clicked.connect(self.packageButtonClicked)
        bottomLayout.addWidget(self.packageButton)

        logLayout = QtWidgets.QVBoxLayout()
        # logLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.logText = QtWidgets.QPlainTextEdit(self)
        self.logText.setStyleSheet("QPlainTextEdit { color: grey; background-color: black; }")
        self.logText.setReadOnly(True)
        self.logText.setFont(defaultFont)
        self.logText.setFixedHeight(200)
        logLayout.addWidget(self.logText)

        mainLayout.addLayout(projectLayout)
        mainLayout.addLayout(taskCheckLayout)
        mainLayout.addLayout(addressLayout)
        mainLayout.addLayout(optionLayout)
        mainLayout.addLayout(optionBottomLayout)
        mainLayout.addLayout(middleLayout)
        mainLayout.addLayout(bottomLayout)
        mainLayout.addLayout(logLayout)

        self.setCentralWidget(self.mainWidget)
        self.mainWidget.setLayout(mainLayout)

        self.setWindowTitle(self.title)
        self.mainWidget.setMinimumWidth(640)
        self.mainWidget.setMinimumHeight(800)
        # self.setWindowIcon(QtGui.QIcon(scriptDir+'/images/icon.ico'))

        self.setAcceptDrops(True)

        self.show()
        self.initUi()

    def initUi(self):
        if os.path.isdir('/show'):
            try: fList = os.listdir('/show')
            except: fList = []

            fList = sorted(fList)

            for f in fList:
                if f.startswith('.'):
                    continue

                if os.path.isdir('/show/'+f):
                    if os.path.isdir('/show/'+f+'/_2d') or os.path.isdir('/show/'+f+'/_3d'):
                        self.projectComboBox.addItem(f)

            self.pcatgComboBoxChanged(self.pcatgComboBox.currentText())

        self.loadUiStatus()

    def loadUiStatus(self):
        homeDir = os.path.expanduser('~')
        settingsDir = homeDir+'/.config'
        settingsFile = settingsDir+'/dxpackager.yml'

        settingsDict = {}
        if os.path.isfile(settingsFile):
            with open(settingsFile, 'r') as f:
                settingsDict = yaml.load(f, Loader=yaml.SafeLoader)

            try:
                self.projectComboBox.setCurrentText(settingsDict['ui']['project'])
                self.ptypeComboBox.setCurrentText(settingsDict['ui']['pType'])
                self.pcatgComboBox.setCurrentText(settingsDict['ui']['pCatg'])
                self.packageDir.setText(settingsDict['ui']['outDir'])
            except: pass

        for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            try:
                if bool(settingsDict['ui']['tasks']['p2dAsset'][str(cb.text())]):
                    cb.setCheckState(QtCore.Qt.Checked)
                else: cb.setCheckState(QtCore.Qt.Unchecked)
            except: pass
        for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            try:
                if bool(settingsDict['ui']['tasks']['p2dShot'][str(cb.text())]):
                    cb.setCheckState(QtCore.Qt.Checked)
                else: cb.setCheckState(QtCore.Qt.Unchecked)
            except: pass
        for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            try:
                if bool(settingsDict['ui']['tasks']['p3dAsset'][str(cb.text())]):
                    cb.setCheckState(QtCore.Qt.Checked)
                else: cb.setCheckState(QtCore.Qt.Unchecked)
            except: pass
        for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            try:
                if bool(settingsDict['ui']['tasks']['p3dShot'][str(cb.text())]):
                    cb.setCheckState(QtCore.Qt.Checked)
                else: cb.setCheckState(QtCore.Qt.Unchecked)
            except: pass

    def saveUiStatus(self):
        homeDir = os.path.expanduser('~')
        settingsDir = homeDir+'/.config'
        settingsFile = settingsDir+'/dxpackager.yml'

        try: os.makedirs(settingsDir)
        except: pass

        settingsDict = {}
        if os.path.isdir(settingsDir):
            if os.path.isfile(settingsFile):
                with open(settingsFile, 'r') as f:
                    settingsDict = yaml.load(f, Loader=yaml.SafeLoader)

            if not 'ui' in settingsDict:
                settingsDict['ui'] = {}

            settingsDict['ui']['project'] = str(self.projectComboBox.currentText())
            settingsDict['ui']['pType'] = str(self.ptypeComboBox.currentText())
            settingsDict['ui']['pCatg'] = str(self.pcatgComboBox.currentText())

            if not 'tasks' in settingsDict['ui']:
                settingsDict['ui']['tasks'] = {}
            for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                if not 'p2dAsset' in settingsDict['ui']['tasks']:
                    settingsDict['ui']['tasks']['p2dAsset'] = {}
                settingsDict['ui']['tasks']['p2dAsset'][str(cb.text())] = cb.isChecked()
            for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                if not 'p2dShot' in settingsDict['ui']['tasks']:
                    settingsDict['ui']['tasks']['p2dShot'] = {}
                settingsDict['ui']['tasks']['p2dShot'][str(cb.text())] = cb.isChecked()
            for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                if not 'p3dAsset' in settingsDict['ui']['tasks']:
                    settingsDict['ui']['tasks']['p3dAsset'] = {}
                settingsDict['ui']['tasks']['p3dAsset'][str(cb.text())] = cb.isChecked()
            for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                if not 'p3dShot' in settingsDict['ui']['tasks']:
                    settingsDict['ui']['tasks']['p3dShot'] = {}
                settingsDict['ui']['tasks']['p3dShot'][str(cb.text())] = cb.isChecked()

            settingsDict['ui']['outDir'] = str(self.packageDir.text().strip())

            with open(settingsFile, 'w+') as f:
                yaml.dump(settingsDict, f)

    def eventFilter(self, object, event):
        if (object is self.tableWidget.viewport()):
            if (event.type() == QtCore.QEvent.Drop and event.mimeData().hasUrls()):
                row = self.tableWidget.indexAt(event.pos()).row()
                col = self.tableWidget.indexAt(event.pos()).column()

                for url in event.mimeData().urls():
                    filePath = url.toLocalFile()
                    if filePath.endswith('.csv'):
                        self.addTableRowsFromCsv(filePath)

                    elif filePath.endswith('.xls') or filePath.endswith('.xlsx'):
                        self.addTableRowsFromXls(filePath)

                return True
            return False
        return False

    def showTableContextMenu(self, position):
        slRows = self.tableWidget.selectionModel().selectedRows()
        cursorAt = self.tableWidget.itemAt(position)
        try:
            row = cursorAt.row()
            col = cursorAt.column()
        except:
            return

        contexMenu = QtWidgets.QMenu(self)
        contexMenu.setFont(defaultFont)
        checkAct = contexMenu.addAction('Check')
        pcatg = self.pcatgComboBox.currentText()
        envCheckAct = contexMenu.addAction('Check Env Asset')
        if pcatg != 'asset':
            envCheckAct.setVisible(False)
        uncheckAct = contexMenu.addAction('Uncheck')

        openProjectDirAct = contexMenu.addAction('Open Project Dir')
        openPackageDirAct = contexMenu.addAction('Open Package Dir')
        showProjectUsdAct = contexMenu.addAction('Show Project Usd')
        showPackageUsdAct = contexMenu.addAction('Show Package Usd')
        infoProjectUsdAct = contexMenu.addAction('Info Project Usd')
        infoPackageUsdAct = contexMenu.addAction('Info Package Usd')

        action = contexMenu.exec_(self.tableWidget.mapToGlobal(position))
        if action == checkAct:
            self.tableWidget.setSortingEnabled(False)
            for slRow in slRows:
                row = slRow.row()
                if self.tableWidget.isRowHidden(row):
                    continue
                self.tableWidget.item(row, 2).setBackground(QtGui.QColor(255, 0, 0, 32))
                self.tableWidget.item(row, 2).setText('V')
            self.tableWidget.setSortingEnabled(True)

        elif action == envCheckAct:
            self.tableWidget.setSortingEnabled(False)
            for slRow in slRows:
                row = slRow.row()
                self.tableWidget.item(row, 2).setBackground(QtGui.QColor(255, 0, 0, 32))
                self.tableWidget.item(row, 2).setText('Env')
            self.tableWidget.setSortingEnabled(True)

        elif action == uncheckAct:
            self.tableWidget.setSortingEnabled(False)
            for slRow in slRows:
                row = slRow.row()
                self.tableWidget.item(row, 2).setBackground(self.tableWidget.item(row, 0).background())
                self.tableWidget.item(row, 2).setText('')
            self.tableWidget.setSortingEnabled(True)

        elif action == openProjectDirAct:
            projectDir = '/show/'+self.projectComboBox.currentText()
            ptypeDir = projectDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName

            openFile(rowDir)

        elif action == openPackageDirAct:
            packageDir = self.packageDir.text().strip()
            ptypeDir = packageDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName

            openFile(rowDir)

        elif action == showProjectUsdAct:
            projectDir = '/show/'+self.projectComboBox.currentText()
            ptypeDir = projectDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName
            rowUsd = rowDir+'/'+rowName+'.usd'

            cmd = [
                '/backstage/dcc/DCC',
                'rez-env',
                'usdtoolkit',
                'renderman-23.3',
                '--',
                'usdviewer',
                rowUsd
            ]

            if os.path.isfile(rowUsd):
                subprocess.call(cmd)

        elif action == showPackageUsdAct:
            packageDir = self.packageDir.text().strip()
            ptypeDir = packageDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName
            rowUsd = rowDir+'/'+rowName+'.usd'

            cmd = [
                '/backstage/dcc/DCC',
                'rez-env',
                'usdtoolkit',
                'renderman-23.3',
                '--',
                'usdviewer',
                rowUsd
            ]

            if os.path.isfile(rowUsd):
                subprocess.call(cmd)

        elif action == infoProjectUsdAct:
            projectDir = '/show/'+self.projectComboBox.currentText()
            ptypeDir = projectDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName
            rowUsd = rowDir+'/'+rowName+'.usd'

            cmd = [
                '/backstage/dcc/DCC',
                'rez-env',
                'usdmanager',
                '--',
                'usdmanager',
                rowUsd
            ]

            if os.path.isfile(rowUsd):
                subprocess.call(cmd)

        elif action == infoPackageUsdAct:
            packageDir = self.packageDir.text().strip()
            ptypeDir = packageDir+'/'+self.ptypeComboBox.currentText()
            pcatg = self.pcatgComboBox.currentText()
            pcatgDir = ptypeDir+'/'+pcatg
            rowName = self.tableWidget.item(row, 1).text()
            rowDir = pcatgDir+'/'+rowName
            if pcatg == 'shot':
                seq = rowName.split('_')[0]
                rowDir = pcatgDir+'/'+seq+'/'+rowName
            rowUsd = rowDir+'/'+rowName+'.usd'

            cmd = [
                '/backstage/dcc/DCC',
                'rez-env',
                'usdmanager',
                '--',
                'usdmanager',
                rowUsd
            ]

            if os.path.isfile(rowUsd):
                subprocess.call(cmd)

    def tableDoubleClicked(self, event):
        row = event.row()
        col = event.column()

        if self.ptypeComboBox.currentText() == '_3d' and self.pcatgComboBox.currentText() == 'shot' and self.packagefmtComboBox.currentText() == 'usd':
            project = self.projectComboBox.currentText()
            shot = self.tableWidget.item(row, 1).text()
            seq = shot.split('_')[0]
            shotRoot = '/show/'+project+'/_3d/shot'
            shotUsd = '%s/%s/%s/%s.usd'%(shotRoot, seq, shot, shot)
            relPath = shotUsd.split('/'+project+'/')[-1]
            packageDir = self.packageDir.text().strip()

            previewFile = ''
            if col == 4:
                packageJpg = '%s/preview/%s.%s.package.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                previewFile = packageJpg
            elif col == 5:
                previewJpg = '%s/preview/%s.%s.show.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                previewFile = previewJpg
            elif col == 6:
                tacticJpg = '%s/preview/%s.%s.tactic_ani.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                previewFile = tacticJpg

            if os.path.isfile(previewFile):
                openFile(previewFile)

    def searchKeywordChanged(self, keyword):
        keywords = [k for k in keyword.strip().split() if k]
        pattern = re.compile('|'.join(keywords), re.IGNORECASE)
        for row in range(self.tableWidget.rowCount()):
            name = self.tableWidget.item(row, 1).text()
            self.tableWidget.setRowHidden(row, pattern.match(name) is None)

    def browsePackageDirButtonClicked(self):
        fileDialog = QtWidgets.QFileDialog(self)
        fileDialog.setWindowTitle('Select Vendor Directory')
        fileDialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if fileDialog.exec_() == QtWidgets.QDialog.Accepted:
            slDir = fileDialog.selectedFiles()[0]
            self.packageDir.setText(slDir)

    def checkAllButtonButtonClicked(self):
        self.tableWidget.setSortingEnabled(False)
        rows = self.tableWidget.rowCount()
        for row in range(rows):
            if not self.tableWidget.isRowHidden(row):
                self.tableWidget.item(row, 2).setBackground(QtGui.QColor(255, 0, 0, 32))
                self.tableWidget.item(row, 2).setText('V')
        self.tableWidget.setSortingEnabled(True)

    def uncheckAllButtonButtonClicked(self):
        self.tableWidget.setSortingEnabled(False)
        rows = self.tableWidget.rowCount()
        for row in range(rows):
            self.tableWidget.item(row, 2).setBackground(self.tableWidget.item(row, 0).background())
            self.tableWidget.item(row, 2).setText('')
        self.tableWidget.setSortingEnabled(True)

    def findButtonClicked(self):
        self.saveUiStatus()

        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.clearSelection()
        self.tableWidget.setRowCount(0)
        self.tableHeader = ['#', 'NAME', 'CHECK', 'INFO']
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setHorizontalHeaderLabels(self.tableHeader)
        self.mainWidget.setMinimumWidth(640)
        self.twvHeader = self.tableWidget.verticalHeader()
        self.twvHeader.setDefaultSectionSize(tableRowHeight)
        QtWidgets.QApplication.processEvents()

        # set showConfig
        setShowConfig(self.projectComboBox.currentText())

        projectDir = '/show/'+self.projectComboBox.currentText()
        ptypeDir = projectDir+'/'+self.ptypeComboBox.currentText()
        pcatgDir = ptypeDir+'/'+self.pcatgComboBox.currentText()

        packageDir = self.packageDir.text().strip()
        pkgTypeDir = packageDir+'/'+self.ptypeComboBox.currentText()
        pkgCatgDir = pkgTypeDir+'/'+self.pcatgComboBox.currentText()

        cellColor = QtGui.QColor(255, 255, 255, 255)

        try: pcatgList = os.listdir(pcatgDir)
        except: pcatgList = []
        pcatgList = sorted(pcatgList)
        for pcf in pcatgList:
            if pcf.startswith('.') or pcf[0].isnumeric():
                continue
            if not os.path.isdir(pcatgDir+'/'+pcf):
                continue

            if self.pcatgComboBox.currentText() == 'shot':
                try: pcatgSubList = os.listdir(pcatgDir+'/'+pcf)
                except: pcatgSubList = []
                pcatgSubList = sorted(pcatgSubList)
                for pcsf in pcatgSubList:
                    try:
                        coder = rb.Coder()
                        arg = coder.N.SHOTNAME.Decode(pcsf)
                    except:
                        continue
                    if not os.path.isdir(pcatgDir+'/'+pcf+'/'+pcsf):
                        continue

                    pkgExists = ''
                    if os.path.isfile(pkgCatgDir+'/'+pcf+'/'+pcsf+'/'+pcsf+'.usd'):
                        pkgExists = 'Exists'

                    self.addTableRow(['#', pcsf, '', pkgExists], 1)

            else:
                pkgExists = ''
                if os.path.isfile(pkgCatgDir+'/'+pcf+'/'+pcf+'.usd'):
                    pkgExists = 'Exists'

                self.addTableRow(['#', pcf, '', pkgExists], 1)

        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.sortItems(0)

        keyword = self.searchKeyword.text().strip()
        if keyword:
            self.searchKeywordChanged(keyword)

    def taskCheckBoxToggle(self, ptype, pctag):
        self.packagefmtComboBox.clear()
        if ptype == '_2d':
            if pctag == 'asset':
                self.p2dAssetTaskCheckWidget.setVisible(True)
                self.p2dShotTaskCheckWidget.setVisible(False)
                self.p3dAssetTaskCheckWidget.setVisible(False)
                self.p3dShotTaskCheckWidget.setVisible(False)
                self.withAssetCheckBox.setVisible(False)
                self.TempRemoveCheckBox.setVisible(False)
                self.SmartBakeCheckBox.setVisible(False)
                self.packagefmtComboBox.setVisible(False)
            elif pctag == 'shot':
                self.p2dAssetTaskCheckWidget.setVisible(False)
                self.p2dShotTaskCheckWidget.setVisible(True)
                self.p3dAssetTaskCheckWidget.setVisible(False)
                self.p3dShotTaskCheckWidget.setVisible(False)
                self.withAssetCheckBox.setVisible(False)
                self.TempRemoveCheckBox.setVisible(False)
                self.SmartBakeCheckBox.setVisible(False)
                self.packagefmtComboBox.setVisible(False)

        elif ptype == '_3d':
            if pctag == 'asset':
                self.p2dAssetTaskCheckWidget.setVisible(False)
                self.p2dShotTaskCheckWidget.setVisible(False)
                self.p3dAssetTaskCheckWidget.setVisible(True)
                self.p3dShotTaskCheckWidget.setVisible(False)
                self.withAssetCheckBox.setVisible(False)
                self.TempRemoveCheckBox.setVisible(False)
                self.SmartBakeCheckBox.setVisible(False)
                packagefmtListView = QtWidgets.QListView()
                packagefmtListView.setFont(defaultFont)
                self.packagefmtComboBox.addItem('usd')
                self.packagefmtComboBox.addItem('mb')
                # self.packagefmtComboBox.addItem('mb(abc)')
                # self.packagefmtComboBox.addItem('usd(sym)')
                # self.packagefmtComboBox.addItem('mb(sym)')
                # self.packagefmtComboBox.addItem('abc(sym)')
                self.packagefmtComboBox.setView(packagefmtListView)
                self.packagefmtComboBox.setVisible(True)
            elif pctag == 'shot':
                self.p2dAssetTaskCheckWidget.setVisible(False)
                self.p2dShotTaskCheckWidget.setVisible(False)
                self.p3dAssetTaskCheckWidget.setVisible(False)
                self.p3dShotTaskCheckWidget.setVisible(True)
                self.withAssetCheckBox.setVisible(True)
                self.TempRemoveCheckBox.setVisible(True)
                self.SmartBakeCheckBox.setVisible(True)
                packagefmtListView = QtWidgets.QListView()
                packagefmtListView.setFont(defaultFont)
                self.packagefmtComboBox.addItem('usd')
                self.packagefmtComboBox.addItem('mb')
                self.packagefmtComboBox.addItem('mb(low)')
                self.packagefmtComboBox.addItem('mb(abc)')
                # self.packagefmtComboBox.addItem('usd(sym)')
                # self.packagefmtComboBox.addItem('mb(sym)')
                # self.packagefmtComboBox.addItem('abc(sym)')
                self.packagefmtComboBox.setView(packagefmtListView)
                self.packagefmtComboBox.setVisible(True)

    def ptypeComboBoxChanged(self, value):
        self.taskCheckBoxToggle(value, self.pcatgComboBox.currentText())

    def pcatgComboBoxChanged(self, value):
        self.taskCheckBoxToggle(self.ptypeComboBox.currentText(), value)

    def packagefmtComboBoxChanged(self, text):
        if self.ptypeComboBox.currentText() == '_3d' and self.pcatgComboBox.currentText() == 'shot' and 'usd' in text:
            self.makePreviewButton.setVisible(True)
            self.withAssetCheckBox.setDisabled(False)
            self.TempRemoveCheckBox.setVisible(False)
            self.SmartBakeCheckBox.setVisible(False)
        else:
            self.makePreviewButton.setVisible(False)
            self.withAssetCheckBox.setDisabled(True)
            self.TempRemoveCheckBox.setVisible(True)
            self.SmartBakeCheckBox.setVisible(True)
            

        if self.ptypeComboBox.currentText() == '_3d' and self.pcatgComboBox.currentText() == 'asset' and text == 'mb':
            self.p3dAssetTextureComboBox.setDisabled(False)
        else:
            self.p3dAssetTextureComboBox.setDisabled(True)

        if '(sym)' in text:
            if text == 'usd(sym)':
                self.withAssetCheckBox.setDisabled(False)
            else:
                self.withAssetCheckBox.setDisabled(True)
                self.tractorCheckBox.setChecked(False)

            self.tractorCheckBox.setDisabled(True)
            self.tractorCheckBox.setChecked(False)
            self.logText.setVisible(True)

        else:
            self.tractorCheckBox.setDisabled(False)
            self.tractorCheckBox.setChecked(True)
            self.logText.setVisible(False)

        if not text.startswith('usd'):
            self.withAssetCheckBox.setChecked(False)

    def openDataFile(self):
        fileDialog = QtWidgets.QFileDialog(self)
        fileDialog.setWindowTitle('Select Data File')
        fileDialog.setNameFilter('Data Files (*.csv)')
        if fileDialog.exec_() == QtWidgets.QDialog.Accepted:
            for filePath in fileDialog.selectedFiles():
                self.addTableRowsFromCsv(filePath)
                break

    def addTableRow(self, dataList, altBgBaseCol=-1, rowColor=QtGui.QColor(255, 255, 255, 255)):
        rowCnt = self.tableWidget.rowCount()
        self.tableWidget.insertRow(rowCnt)

        rowId = str(rowCnt+1).rjust(5)
        if dataList[0] == '#':
            dataList[0] = rowId

        self.tableWidget.setItem(rowCnt, 0, QtWidgets.QTableWidgetItem(dataList[0]))
        self.tableWidget.setItem(rowCnt, 1, QtWidgets.QTableWidgetItem(dataList[1]))
        self.tableWidget.setItem(rowCnt, 2, QtWidgets.QTableWidgetItem(dataList[2]))
        self.tableWidget.setItem(rowCnt, 3, QtWidgets.QTableWidgetItem(dataList[3]))

        if altBgBaseCol > -1:
            if rowCnt > 0:
                prevData = self.tableWidget.item(rowCnt-1, altBgBaseCol).text()
                currData = dataList[altBgBaseCol]
                prevColor = self.tableWidget.item(rowCnt-1, altBgBaseCol).background()
                rowColor = prevColor
                changeRowColor = False
                if prevData.count('_') > 0 and currData.count('_') > 0:
                    prevPrefix = prevData.split('_')[0]
                    currPrefix = currData.split('_')[0]

                    if len(prevPrefix) != len(currPrefix) or prevPrefix != currPrefix:
                        changeRowColor = True

                else:
                    if prevData[0] != currData[0]:
                        changeRowColor = True

                if changeRowColor:
                    if prevColor == QtGui.QColor(255, 255, 255, 255):
                        rowColor = QtGui.QColor(0, 0, 128, 16)
                    else:
                        rowColor = QtGui.QColor(255, 255, 255, 255)

        self.tableWidget.item(rowCnt, 0).setBackground(rowColor)
        self.tableWidget.item(rowCnt, 1).setBackground(rowColor)
        self.tableWidget.item(rowCnt, 2).setBackground(rowColor)
        self.tableWidget.item(rowCnt, 3).setBackground(rowColor)

    def addTableRowsFromCsv(self, csvFile):
        csvFn = os.path.basename(csvFile)
        fnTok = re.split(r'[_\-.]', csvFn.lower())

        forBreak = False
        for i in range(self.projectComboBox.count()):
            for t in fnTok:
                if t == self.projectComboBox.itemText(i):
                    self.projectComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        forBreak = False
        for i in range(self.ptypeComboBox.count()):
            for t in fnTok:
                t = '_'+t
                if t == self.ptypeComboBox.itemText(i):
                    self.ptypeComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        forBreak = False
        for i in range(self.pcatgComboBox.count()):
            for t in fnTok:
                if t == self.pcatgComboBox.itemText(i):
                    self.pcatgComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)

        for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)

        with open(csvFile, 'r') as f:
            csvReader = csv.reader(f)

            self.tableWidget.setSortingEnabled(False)
            self.tableWidget.clearSelection()
            self.tableWidget.setRowCount(0)

            pcatg = ''
            rCnt = 0
            dataCol = -1
            for line in csvReader:
                if rCnt == 0:
                    tCnt = 0
                    for t in line:
                        tl = t.lower()
                        if tl.count('asset') > 0 or tl.count('ast') > 0 or tl.count('mod') > 0 or tl.count('rig') > 0:
                            if pcatg == '':
                                pcatg = 'asset'

                        if tl.count('seq') > 0 or tl.count('shot') > 0 or tl.count('match') > 0 or tl.count('ani') > 0 or tl.count('plate') > 0:
                            if pcatg == '':
                                pcatg = 'shot'

                        if tl.count('shot') > 0 or tl.count('num') > 0 or tl.count('code') > 0 or tl.count('asset') > 0 or tl.count('name') > 0:
                            if dataCol == -1:
                                dataCol = tCnt

                        tCnt = tCnt + 1

                    if pcatg == '':
                        pcatg = 'asset'

                        if dataCol == -1:
                            dataCol = 0
                            self.addTableRow(['#', line[dataCol], '', ''], 1)

                    self.pcatgComboBox.setCurrentText(pcatg)

                else:
                    self.addTableRow(['#', line[dataCol], '', ''], 1)

                rCnt = rCnt + 1

            self.tableWidget.setSortingEnabled(True)
            self.tableWidget.sortItems(0)

    def addTableRowsFromXls(self, xlsFile):
        csvFn = os.path.basename(xlsFile)
        fnTok = re.split(r'[_\-.]', csvFn.lower())

        forBreak = False
        for i in range(self.projectComboBox.count()):
            for t in fnTok:
                if t == self.projectComboBox.itemText(i):
                    self.projectComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        forBreak = False
        for i in range(self.ptypeComboBox.count()):
            for t in fnTok:
                t = '_' + t
                if t == self.ptypeComboBox.itemText(i):
                    self.ptypeComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        forBreak = False
        for i in range(self.pcatgComboBox.count()):
            for t in fnTok:
                if t == self.pcatgComboBox.itemText(i):
                    self.pcatgComboBox.setCurrentText(t)
                    forBreak = True
                    if forBreak: break
            if forBreak: break

        for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)
        for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            cb.setCheckState(QtCore.Qt.Unchecked)

        for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)
        for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
            for t in fnTok:
                if t == cb.text():
                    cb.setCheckState(QtCore.Qt.Checked)

        wBook = xlrd2.open_workbook(xlsFile)
        wSheetList = wBook.sheet_names()
        selectedSheet = wSheetList[0]

        if len(wSheetList) > 1:
            selectSheetDialog = QtWidgets.QDialog()
            mainLayout = QtWidgets.QVBoxLayout()
            for ws in wSheetList:
                sheetButton = QtWidgets.QRadioButton(ws, self)
                mainLayout.addWidget(sheetButton)

            sheetButton = QtWidgets.QPushButton('Select')
            sheetButton.clicked.connect(lambda:(selectSheetDialog.close()))
            mainLayout.addWidget(sheetButton)

            selectSheetDialog.setLayout(mainLayout)
            selectSheetDialog.setWindowTitle('Select Sheet')
            selectSheetDialog.setMinimumWidth(220)
            selectSheetDialog.setMinimumHeight(120)
            selectSheetDialog.exec_()

            for rButton in selectSheetDialog.findChildren(QtWidgets.QRadioButton):
                if rButton.isChecked():
                    selectedSheet = rButton.text()
                    break

        wSheet = wBook.sheet_by_name(selectedSheet)

        ncol = wSheet.ncols
        nrow = wSheet.nrows

        sheetData = []
        for i in range(nrow):
            sheetData.append(wSheet.row_values(i))

        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.clearSelection()
        self.tableWidget.setRowCount(0)

        pcatg = ''
        rCnt = 0
        dataCol = -1
        for line in sheetData:
            if rCnt == 0:
                tCnt = 0
                for t in line:
                    tl = t.lower()
                    if tl.count('asset') > 0 or tl.count('ast') > 0 or tl.count('mod') > 0 or tl.count('rig') > 0:
                        if pcatg == '':
                            pcatg = 'asset'

                    if tl.count('seq') > 0 or tl.count('shot') > 0 or tl.count('match') > 0 or tl.count(
                            'ani') > 0 or tl.count('plate') > 0:
                        if pcatg == '':
                            pcatg = 'shot'

                    if tl.count('shot') > 0 or tl.count('num') > 0 or tl.count('code') > 0 or tl.count(
                            'asset') > 0 or tl.count('name') > 0:
                        if dataCol == -1:
                            dataCol = tCnt

                    tCnt = tCnt + 1

                if pcatg == '':
                    pcatg = 'asset'

                    if dataCol == -1:
                        dataCol = 0
                        self.addTableRow(['#', line[dataCol], '', ''], 1)

                self.pcatgComboBox.setCurrentText(pcatg)

            else:
                self.addTableRow(['#', line[dataCol], '', ''], 1)

            rCnt = rCnt + 1

        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.sortItems(0)

    def getSelectedTaskList(self):
        taskList = []
        if self.ptypeComboBox.currentText() == '_2d':
            if self.pcatgComboBox.currentText() == 'asset':
                for cb in self.p2dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                    if cb.isChecked():
                        taskList.append(str(cb.text()))
            elif self.pcatgComboBox.currentText() == 'shot':
                for cb in self.p2dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                    if cb.isChecked():
                        taskList.append(str(cb.text()))

        elif self.ptypeComboBox.currentText() == '_3d':
            if self.pcatgComboBox.currentText() == 'asset':
                for cb in self.p3dAssetTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                    if cb.isChecked():
                        taskList.append(str(cb.text()))
            elif self.pcatgComboBox.currentText() == 'shot':
                for cb in self.p3dShotTaskCheckWidget.findChildren(QtWidgets.QCheckBox):
                    if cb.isChecked():
                        taskList.append(str(cb.text()))
        return taskList

    def makePreviewButtonClicked(self):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setWindowTitle('Create preview')
        msgBox.setText(u'모든 프리뷰 이미지를 새로 만들겠습니까?')
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        buttonY = msgBox.button(QtWidgets.QMessageBox.Yes)
        buttonY.setText('Yes')
        buttonN = msgBox.button(QtWidgets.QMessageBox.No)
        buttonN.setText('No')

        createAll = False
        msgBox.exec_()
        if msgBox.clickedButton() == buttonY:
            createAll = True

        self.mainWidget.setMinimumWidth(1200)
        self.tableHeader = ['#', 'NAME', 'CHECK', 'INFO', 'PACKAGE', 'SHOW', 'TACTIC']
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(self.tableHeader)
        self.tableWidget.setColumnWidth(4, tablePreviewWidth)
        self.tableWidget.setColumnWidth(5, tablePreviewWidth)
        self.tableWidget.setColumnWidth(6, tablePreviewWidth)
        self.twvHeader = self.tableWidget.verticalHeader()
        self.twvHeader.setDefaultSectionSize(tablePreviewRowHeight)
        QtWidgets.QApplication.processEvents()

        def getTacticProjectCode(projName):
            import dxConfig
            import requests

            API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
            user = {}
            user['api_key'] = API_KEY

            projectDict = {}

            infos = requests.get("http://%s/dexter/search/project.php" %(dxConfig.getConf('TACTIC_IP')), params=user).json()
            for idx, i in enumerate(infos):
                    # print idx, i['code'], i['name']
                    projectDict[str(i['name'])] = str(i['code'])

            projCode = ''
            try: projCode = projectDict[projName]
            except: pass
            if projCode == '':
                projNameOrig = re.sub(r'[0-9]+$', '', projName)
                try: projCode = projectDict[projNameOrig]
                except: pass

            return projCode

        project = self.projectComboBox.currentText()
        tacticProject = getTacticProjectCode(project)

        tacticRoot = '/tactic/assets/'+tacticProject
        if self.ptypeComboBox.currentText() == '_3d' and self.pcatgComboBox.currentText() == 'shot' and self.packagefmtComboBox.currentText() == 'usd':
            tacticShotRoot = tacticRoot+'/shot'
            shotRoot = '/show/'+project+'/_3d/shot'
            packageDir = self.packageDir.text().strip()
            shotList = []
            notPkgdShot = []
            rows = self.tableWidget.rowCount()
            for row in range(rows):
                if self.tableWidget.item(row, 2).text().strip() == 'V':
                    shot = self.tableWidget.item(row, 1).text()
                    shotList.append(shot)
                    seq = shot.split('_')[0]
                    shotDir = '%s/%s/%s'%(shotRoot, seq, shot)
                    shotUsd = '%s/%s/%s/%s.usd'%(shotRoot, seq, shot, shot)
                    relPath = shotUsd.split('/'+project+'/')[-1]
                    packageUsd = '%s/%s'%(packageDir, relPath)

                    packageJpg = '%s/preview/%s.%s.package.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                    packageThumbJpg = '%s/preview/%s.%s.package_thumb.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                    if os.path.isfile(packageUsd):
                        needConv = True
                        if os.path.isfile(packageJpg):
                            if os.path.getmtime(packageUsd) < os.path.getmtime(packageJpg):
                                needConv = False
                        previewDir = os.path.dirname(packageJpg)
                        try: os.makedirs(previewDir)
                        except: pass

                        if self.tableWidget.cellWidget(row, 4) != None:
                            self.tableWidget.removeCellWidget(row, 4)
                            QtWidgets.QApplication.processEvents()

                        if createAll or needConv:
                            os.system('/backstage/dcc/DCC rez-env usd_core-20.08 pyside2 -- usdrecord --imageWidth 1920 --purposes render --frames '+previewFrame+':'+previewFrame+' '+packageUsd+' '+packageJpg.replace(previewFrame.zfill(4), '####'))
                            os.system('/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg -y -i '+packageJpg+' -vf "scale=\'min(200,iw)\':-1" -vframes 1 '+packageThumbJpg)
                            # os.system('/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg -y -i '+previewJpg+' -vf "crop=w=\'min(iw\\,ih)\':h=\'min(iw\\,ih)\',scale=200:200,setsar=1" -vframes 1 '+previewThumbJpg)

                        if os.path.isfile(packageThumbJpg):
                            self.tableWidget.setCellWidget(row, 4, ImageLabel(self, packageThumbJpg, tablePreviewWidth))
                            QtWidgets.QApplication.processEvents()

                        previewJpg = '%s/preview/%s.%s.show.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                        previewThumbJpg = '%s/preview/%s.%s.show_thumb.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                        if os.path.isfile(shotUsd):
                            needConv = True
                            if os.path.isfile(previewJpg):
                                if os.path.getmtime(shotUsd) < os.path.getmtime(previewJpg):
                                    needConv = False

                            if self.tableWidget.cellWidget(row, 5) != None:
                                self.tableWidget.removeCellWidget(row, 5)
                                QtWidgets.QApplication.processEvents()

                            if createAll or needConv:
                                os.system('/backstage/dcc/DCC rez-env usd_core-20.08 pyside2 -- usdrecord --imageWidth 1920 --purposes render --frames '+previewFrame+':'+previewFrame+' '+shotUsd+' '+previewJpg.replace(previewFrame.zfill(4), '####'))
                                os.system('/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg -y -i '+previewJpg+' -vf "scale=\'min(200,iw)\':-1" -vframes 1 '+previewThumbJpg)

                            if os.path.isfile(previewThumbJpg):
                                self.tableWidget.setCellWidget(row, 5, ImageLabel(self, previewThumbJpg, tablePreviewWidth))
                                QtWidgets.QApplication.processEvents()

                        tacticShotDir = '%s/%s'%(tacticShotRoot, shot)
                        tacticShotAnimDir = tacticShotDir+'/animation/animation'
                        tacticShotMmvDir = tacticShotDir+'/matchmove/matchmove'

                        latestMov = ''
                        tacticJpg = ''
                        tacticThumbJpg = ''
                        if os.path.isdir(tacticShotAnimDir):
                            movList = sorted(glob.glob(tacticShotAnimDir+'/*_animation_*.mov'), reverse=True)
                            if len(movList) > 0:
                                latestMov = movList[0]
                                tacticJpg = '%s/preview/%s.%s.tactic_ani.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                                tacticThumbJpg = '%s/preview/%s.%s.tactic_ani_thumb.jpg'%(packageDir, relPath, previewFrame.zfill(4))

                        elif os.path.isdir(tacticShotMmvDir):
                            movList = sorted(glob.glob(tacticShotMmvDir+'/*matchmove*.mov'), reverse=True)
                            if len(movList) > 0:
                                latestMov = movList[0]
                                tacticJpg = '%s/preview/%s.%s.tactic_mmv.jpg'%(packageDir, relPath, previewFrame.zfill(4))
                                tacticThumbJpg = '%s/preview/%s.%s.tactic_mmv_thumb.jpg'%(packageDir, relPath, previewFrame.zfill(4))

                        if latestMov != '':
                            needConv = True
                            if os.path.isfile(tacticJpg):
                                if os.path.getmtime(latestMov) < os.path.getmtime(tacticJpg):
                                    needConv = False

                            if self.tableWidget.cellWidget(row, 6) != None:
                                self.tableWidget.removeCellWidget(row, 6)
                                QtWidgets.QApplication.processEvents()

                            if createAll or needConv:
                                os.system('/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg -y -i '+latestMov+' -ss 00:00:00 -vframes 1 '+tacticJpg)
                                os.system('/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg -y -i '+tacticJpg+' -vf "scale=\'min(200,iw)\':-1" -vframes 1 '+tacticThumbJpg)

                            if os.path.isfile(tacticThumbJpg):
                                self.tableWidget.setCellWidget(row, 6, ImageLabel(self, tacticThumbJpg, tablePreviewWidth))
                                QtWidgets.QApplication.processEvents()

                    else:
                        notPkgdShot.append(shot)

            if len(shotList) < 1:
                notificationBox(self, u'선택된 샷이 없습니다.')

            else:
                notificationBox(self, u'프리뷰 생성이 완료되었습니다.')
                if len(notPkgdShot) > 0:
                    notificationBox(self, u'(패키징 데이터가 없는 일부 샷의 프리뷰 생성 실패)')


    def packageButtonClicked(self):
        self.saveUiStatus()
        taskList = self.getSelectedTaskList()
        packageDir = self.packageDir.text().strip()
        if packageDir == '':
            notificationBox(self, u'패키지 될 폴더를 지정해야 합니다.')
            return

        ftpLogin = ['', '']
        if '(sym)' in self.packagefmtComboBox.currentText():
            loginDialog = QtWidgets.QDialog()
            mainLayout = QtWidgets.QVBoxLayout()

            idLayout = QtWidgets.QHBoxLayout()
            idLayout.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            idLabel = QtWidgets.QLabel('ID:')
            idLabel.setFont(defaultFont)
            idLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            idLabel.setFixedWidth(100+extWdgWidth)
            idLayout.addWidget(idLabel)
            idInput = QtWidgets.QLineEdit()
            idInput.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            idInput.setFixedWidth(140+extWdgWidth)
            idLayout.addWidget(idInput)

            pwLayout = QtWidgets.QHBoxLayout()
            pwLayout.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            pwLabel = QtWidgets.QLabel('PW:')
            pwLabel.setFont(defaultFont)
            pwLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            pwLabel.setFixedWidth(100+extWdgWidth)
            pwLayout.addWidget(pwLabel)
            pwInput = QtWidgets.QLineEdit()
            pwInput.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            pwInput.setEchoMode(QtWidgets.QLineEdit.Password)
            pwInput.setFixedWidth(140+extWdgWidth)
            pwLayout.addWidget(pwInput)

            btnLayout = QtWidgets.QHBoxLayout()
            btnLayout.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            loginLabel = QtWidgets.QLabel('')
            loginLabel.setFont(defaultFont)
            loginLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            loginLabel.setFixedWidth(140+extWdgWidth)
            btnLayout.addWidget(loginLabel)

            loginButton = QtWidgets.QPushButton('Login')
            loginButton.setFont(defaultFont)
            loginButton.setFixedWidth(80+extWdgWidth)
            btnLayout.addWidget(loginButton)

            msgLabel = QtWidgets.QLabel()
            msgLabel.setStyleSheet('color: red;')
            msgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            msgLabel.setFont(defaultFont)

            mainLayout.addLayout(idLayout)
            mainLayout.addLayout(pwLayout)
            mainLayout.addWidget(msgLabel)
            mainLayout.addLayout(btnLayout)

            loginDialog.setLayout(mainLayout)
            loginDialog.setWindowTitle(u'FTP 계정을 입력하십시오.')

            def checkFtpAccount(loginResult):
                ftpId = idInput.text()
                ftpPw = pwInput.text()
                ftpd = ftplib.FTP()
                ftpd.connect(host=FTPHOST, port=FTPPORT)
                try: ftpd.login(ftpId, ftpPw)
                except:
                    msgLabel.setText('FTP login failed.')
                    return
                ftpd.quit()
                loginDialog.close()
                loginResult[0] = ftpId
                loginResult[1] = ftpPw
                self.logText.setPlainText('')

            idInput.textChanged.connect(lambda:  msgLabel.setText(''))
            pwInput.textChanged.connect(lambda:  msgLabel.setText(''))
            pwInput.returnPressed.connect(lambda: checkFtpAccount(ftpLogin))
            loginButton.clicked.connect(lambda: checkFtpAccount(ftpLogin))

            loginDialog.exec_()
            packageDir = ftpLogin[0]+'#'+packageDir
            print ('FTP login {}'.format(ftpLogin[0]))

        table = self.tableWidget
        if self.pcatgComboBox.currentText() == 'asset':
            assetList = []
            envList = []

            rows = table.rowCount()
            for row in range(rows):
                if not table.isRowHidden(row) and table.item(row, 2).text().strip() == 'V':
                    assetList.append(table.item(row, 1).text())

                if 'Env' in table.item(row, 2).text().strip():
                    envList.append(table.item(row, 1).text())

            if len(assetList) < 1:
                notificationBox(self, u'선택된 애셋이 없습니다.')

            if self.packagefmtComboBox.currentText() == 'mb':
                print ('Start asset packaging...')
                machineType = 'Local'
                if self.tractorCheckBox.isChecked():
                    machineType = 'Tractor'

                assetconvert.AssetConvertPack(dst=packageDir,
                                              show=self.projectComboBox.currentText(),
                                              list=assetList,
                                              tasklist=taskList,
                                              envlist=envList,
                                              texFmt=self.p3dAssetTextureComboBox.currentText(),
                                              machineType=machineType)
            else:
                packageFmt = 'usd'
                if self.packagefmtComboBox.isVisible():
                    packageFmt = self.packagefmtComboBox.currentText()

                if self.tractorCheckBox.isChecked():
                    spoolAsset = SpoolAssetPackage(self.projectComboBox.currentText(), self.ptypeComboBox.currentText(), taskList, assetList, packageDir, 'vendorCode', packageFmt)
                    spoolAsset.spool()

                else:
                    assetPack = assetpackage.AssetPack(self.projectComboBox.currentText(), self.ptypeComboBox.currentText(), taskList, assetList, packageDir, 'vendorCode', packageFmt)
                    assetPack.startPackage()
                    print ('End asset packaging')

        elif self.pcatgComboBox.currentText() == 'shot':
            print ('Start shot packaging...')
            shotList = []
            rows = table.rowCount()
            for row in range(rows):
                if not table.isRowHidden(row) and table.item(row, 2).text().strip() == 'V':
                    shotList.append(table.item(row, 1).text())

            if len(shotList) < 1:
                notificationBox(self, u'선택된 샷이 없습니다.')

            withAsset = False
            if self.withAssetCheckBox.isChecked():
                withAsset = True

            TempRemove = False
            if self.TempRemoveCheckBox.isChecked():
                TempRemove = True

            SmartBake = False
            if self.SmartBakeCheckBox.isChecked():
                SmartBake = True

            packageFmt = 'usd'
            if self.packagefmtComboBox.isVisible():
                packageFmt = self.packagefmtComboBox.currentText()

            if self.tractorCheckBox.isChecked():
                spoolShot = SpoolShotPackage(self.projectComboBox.currentText(), self.ptypeComboBox.currentText(), taskList, shotList, packageDir, 'vendorCode', packageFmt, withAsset, TempRemove, SmartBake)
                if spoolShot.spool():
                    self.summarize(shotList)
                notificationBox(self, u'패키징 Tractor Job Spool 완료.')

            else:
                shotPack = shotpackage.ShotPack(self.projectComboBox.currentText(), self.ptypeComboBox.currentText(), taskList, shotList, packageDir, 'vendorCode', packageFmt, withAsset, TempRemove, SmartBake)
                if shotPack.startPackage():
                    self.summarize(shotList)
                print ('End shot packaging')
                notificationBox(self, u'패키징 완료.')

        if self.packagefmtComboBox.currentText() == 'mb':
            toVendorScriptDir = scriptsDir+'/usd2maya/toVendor'
            try: toVendorScripts = os.listdir(toVendorScriptDir)
            except: toVendorScripts = []

            packageScriptDir = packageDir+'/scripts'
            try: os.makedirs(packageScriptDir)
            except: pass

            if os.path.isdir(packageScriptDir):
                for f in toVendorScripts:
                    if f.endswith('.py'):
                        shutil.copy2(toVendorScriptDir+'/'+f, packageScriptDir+'/'+f)
            else:
                print ('Package Script Dir does not exists. {}'.format(packageScriptDir))

        if ftpLogin[0] != '' and ftpLogin[1] != '' :
            ftpId =  ftpLogin[0]
            ftpPw =  ftpLogin[1]
            ftpd = ftplib.FTP()
            ftpd.connect(host=FTPHOST, port=FTPPORT)
            try:
                ftpd.login(ftpId, ftpPw)
                ftpd.cwd('/from_dexter/'+packageDir.split('/from_dexter/')[-1])
                localLogFile = '/show/pipe/stuff/logs/sympkg-'+ftpId+'-'+packageDir.split('/from_dexter/')[-1].replace('#', '-').replace('/', '$')+'.log'
                fd = open(localLogFile, 'wb')
                ftpd.retrbinary('RETR sympkg.log', fd.write)
                fd.close()
                ftpd.quit()
                # os.system('xdg-open "'+localLogFile+'"')
                with open(localLogFile, 'r') as lf:
                    logPretty = []
                    successCount = 1
                    data = lf.read()
                    for dl in data.split('\n'):
                        if '->' in dl:
                            dl = dl.replace('`', '').replace("'", '')
                            srcDst = re.split(r'[ ]*\-\>[ ]*', dl)
                            if not srcDst[0].startswith('/show/'):
                                srcDst.reverse()
                                srcDst[1] = '@'+srcDst[1]
                            src, dst = srcDst
                            logPretty.append(str(successCount)+': '+src+'\n    -> '+dst)
                            successCount += 1

                        else:
                            logPretty.append(dl)

                    self.logText.setPlainText('\n'.join(logPretty))
                    self.logText.verticalScrollBar().setValue(self.logText.verticalScrollBar().maximum())

            except:
                print ('FTP sympkg.log download failed.')

    COLOR_SUCCESS = QtGui.QColor(0, 255, 0, 32)
    COLOR_FAIL = QtGui.QColor(255, 0, 0, 32)

    def summarize(self, shot_list):
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.clearSelection()
        self.tableWidget.setRowCount(0)

        log_folder = os.path.join(self.packageDir.text().strip(), "logs")
        for shot in shot_list:
            path = os.path.join(log_folder, "{}.log".format(shot))
            if os.path.exists(path):
                with open(path, 'r') as log:
                    result = log.readlines()[-1]
                if result == "Succeeded.":
                    self.addTableRow(['#', shot, 'V', result], 1, self.COLOR_SUCCESS)
                else:
                    self.addTableRow(['#', shot, 'V', "Check {}.log".format(shot)], 1, self.COLOR_FAIL)
                continue
            self.addTableRow(['#', shot, 'V', "Failed"], 1, self.COLOR_FAIL)


class SpoolAssetPackage:
    def __init__(self, projectCode, pipeType, taskList, assetList, packageDir, vendorCode, packageFmt):
        self.TRACTOR_IP = '10.0.0.25'
        self.PORT = 80

        self.packageDir = packageDir
        self.projectCode = projectCode.lower()
        self.vendorCode = vendorCode.lower()
        self.packageFmt = packageFmt.lower()
        self.pipeType = pipeType.lower()
        self.taskList = taskList
        self.projectDir = '/show/'+self.projectCode
        self.findDir = '%s/%s/asset'%(self.projectDir, self.pipeType)
        self.foundList = []
        self.assetList = assetList

        self.user = getpass.getuser()

    def subTask(self, parent):
        # if 'prevtex' in self.taskList:
        #     self.taskList.remove('prevtex')
        # if 'exrtex' in self.taskList:
        #     self.taskList.remove('exrtex')

        for s in self.assetList:
            task = author.Task(title=str(s))
            parent.addChild(task)
            for t in self.taskList:
                subtask = author.Task(title=str(t))
                task.addChild(subtask)
                command = ['/backstage/dcc/DCC', 'rez-env',
                    'usd_core-20.08', 'dxusd', 'pylibs-2', 'python-2',
                    '--', 'python',
                    scriptsDir + '/assetpackage.py',
                    self.projectCode, self.pipeType, t, s, self.packageDir, 'vendorCode', self.packageFmt
                ]

                subtask.addCommand(
                    author.Command(argv=command)
                )

    def spool(self):
        title = '(PACKAGE) '+str(self.projectCode+' '+self.pipeType+' '+self.packageDir.split('/ftp/')[-1])
        job = author.Job(
            title=title,
            tier='cache',
            priority=100,
            projects=['package'],
            service='Cache',
            comment='',
            metadata='',
            maxactive=5
        )

        # post script
        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        JobTask = author.Task(title='Package')
        # JobTask.serialsubtasks = 1
        job.addChild(JobTask)
        self.subTask(JobTask)
        print(job)
        # return
        engine = TractorEngine(hostname=self.TRACTOR_IP, port=self.PORT, user=self.user, debug=True)
        state, msg = engine.spool(job)
        print('# STATE')
        print('\t', state)
        print('# MESSAGE')
        print('\t', msg)


class SpoolShotPackage:
    def __init__(self, projectCode, pipeType, taskList, shotList, packageDir, vendorCode, packageFmt, withAsset, TempRemove, SmartBake):
        self.TRACTOR_IP = '10.0.0.25'
        self.PORT = 80

        self.packageDir = packageDir
        self.projectCode = projectCode.lower()
        self.vendorCode = vendorCode.lower()
        self.packageFmt = packageFmt.lower()
        self.pipeType = pipeType.lower()
        self.taskList = taskList
        self.projectDir = '/show/'+self.projectCode
        self.findDir = '%s/%s/shot'%(self.projectDir, self.pipeType)
        self.foundList = []
        self.shotList = shotList
        self.withAsset = str(withAsset)
        self.TempRemove = str(TempRemove)
        self.SmartBake = str(SmartBake)

        self.user = getpass.getuser()

    def subTask(self, parent):
        if self.packageFmt == "usd":
            for s in self.shotList:
                task = author.Task(title=str(s))
                parent.addChild(task)
                for t in self.taskList:
                    subtask = author.Task(title=str(t))
                    task.addChild(subtask)
                    command = ['/backstage/dcc/DCC', 'rez-env',
                               'usd_core-20.08', 'dxusd', 'pylibs-2', 'python-2',
                               '--', 'python',
                               scriptsDir+'/shotpackage.py',
                               self.projectCode, self.pipeType, t, s, self.packageDir,
                               'vendorCode', self.packageFmt, self.withAsset, self.TempRemove, self.SmartBake]

                    subtask.addCommand(
                        author.Command(argv=command)
                    )
        else:
            task_string = '_'.join(self.taskList)
            for s in self.shotList:
                task = author.Task(title=str(s))
                parent.addChild(task)
                command = ['/backstage/dcc/DCC', 'rez-env',
                           'usd_core-20.08', 'dxusd', 'pylibs-2', 'python-2',
                           '--', 'python', scriptsDir + '/shotpackage.py',
                           self.projectCode, self.pipeType, task_string, s, self.packageDir,
                           'vendorCode', self.packageFmt, self.withAsset, self.TempRemove, self.SmartBake]

                task.addCommand(
                    author.Command(argv=command)
                )

    def spool(self):
        title = '(PACKAGE) '+str(self.projectCode+' '+self.pipeType+' '+self.packageDir.split('/ftp/')[-1])
        job = author.Job(
            title=title,
            tier='cache',
            priority=100,
            projects=['package'],
            service='Cache',
            comment='',
            metadata='',
            maxactive=5
        )

        JobTask = author.Task(title='Package')
        # JobTask.serialsubtasks = 1
        job.addChild(JobTask)
        self.subTask(JobTask)
        print(job)
        # return
        engine = TractorEngine(hostname=self.TRACTOR_IP, port=self.PORT, user=self.user, debug=True)
        state, msg = engine.spool(job)
        print('# STATE')
        print('\t', state)
        print('# MESSAGE')
        print('\t', msg)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainwindow = MainWindow()
    sys.exit(app.exec_())

# cp -rf /home/jungsup.han/homestage/dcc/packages/app/dxpackager/1.0/scripts/usdpackager/* /stdrepo/CSP/jungsup.han/scripts/usdpackager/; /backstage/dcc/DCC rez-env usd_core dxusd pyside2-5.12 -- python /stdrepo/CSP/jungsup.han/scripts/usdpackager/usdpacakger.py
# /backstage/dcc/DCC rez-env usd_core-20.08 dxusd pyside2-5.12 -- python /backstage/dcc/packages/app/dxpackager/1.0/scripts/usdpackager/usdpacakger.py
