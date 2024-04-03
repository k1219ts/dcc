#coding:utf-8
import os, sys, re, time, datetime, subprocess, threading, getpass, logging, yaml, glob, hashlib, json, shutil

from PySide2 import QtWidgets
from PySide2 import QtGui
from PySide2 import QtCore

# STYLESHEET
import qdarkstyle

import vendorSpool
reload(vendorSpool)

import DXRulebook.Interface as rb

######################################## Moon ####################################################

# import DXRulebook.Interface as rb
import DXUSD.Utils as utl


BATCHSCENESCRIPT = '{DCC} rez-env {PACKAGES} -- DXBatchMain {OPTS}'




##################################################################################################

extWdgWidth = 0
defaultFontSize = 9
defaultFont = QtGui.QFont()
defaultFont.setPointSize(defaultFontSize)

tablePreviewWidth = 150
tableRowHeight = 84

scriptsDir = os.path.dirname(__file__)
jobDir = '/stuff/pipe/stuff/ftp/_vendor'

sleepInt = 10

EXEdefault = 'xdg-open'
EXErv = '/backstage/dcc/DCC rez-env rv-7.9.2 -- rv'
EXEffmpeg = '/backstage/dcc/DCC rez-env ffmpeg -- ffmpeg'

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

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        darkStyleSheet = qdarkstyle.load_stylesheet()
        self.setStyleSheet(darkStyleSheet)

        self.title = 'DX Publisher'
        self.closed = False
        # self.menuBar = QtWidgets.QMenuBar(self)
        # self.menuBar.setFont(defaultFont)
        # self.setMenuBar(self.menuBar)
        # self.fileMenu = self.menuBar.addMenu('&File')
        # self.fileMenu.setFont(defaultFont)
        # self.fileMenu.addAction("E&xit", QtWidgets.qApp.closeAllWindows)

        self.idCol = 0
        # self.vndPrevCol = 1
        self.chkPrevCol = 1
        self.nameCol = 2
        self.checkCol = 3
        self.messageCol = 4

        self.mainWidget = QtWidgets.QWidget(self)

        mainLayout = QtWidgets.QVBoxLayout()

        addressLayout = QtWidgets.QHBoxLayout()
        addressLeftLayout = QtWidgets.QHBoxLayout()
        addressLeftLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        addressLayout.addLayout(addressLeftLayout)

        addressRightLayout = QtWidgets.QHBoxLayout()
        addressRightLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        addressLayout.addLayout(addressRightLayout)

        self.vendorDirLabel = QtWidgets.QLabel('Vendor Dir: ')
        self.vendorDirLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.vendorDirLabel.setFixedWidth(70+extWdgWidth)
        self.vendorDirLabel.setFont(defaultFont)
        addressLeftLayout.addWidget(self.vendorDirLabel)

        self.vendorDir = QtWidgets.QLineEdit()
        self.vendorDir.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.vendorDir.setFont(defaultFont)
        self.vendorDir.textChanged.connect(self.vendorDirChanged)
        addressLeftLayout.addWidget(self.vendorDir)

        browseVendorDirButton = QtWidgets.QPushButton('Browse')
        browseVendorDirButton.setFont(defaultFont)
        browseVendorDirButton.setFixedSize(80+extWdgWidth, 24)
        browseVendorDirButton.clicked.connect(self.browseVendorDirButtonClicked)
        addressRightLayout.addWidget(browseVendorDirButton)

        optionLayout = QtWidgets.QHBoxLayout()
        optionLeftLayout = QtWidgets.QHBoxLayout()
        optionLeftLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        optionLayout.addLayout(optionLeftLayout)

        optionRightLayout = QtWidgets.QHBoxLayout()
        optionRightLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        optionLayout.addLayout(optionRightLayout)

        # self.vendorScriptLabel = QtWidgets.QLabel('Vendor Script: ')
        # self.vendorScriptLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.vendorScriptLabel.setFont(defaultFont)
        # optionLeftLayout.addWidget(self.vendorScriptLabel)
        #
        # self.vendorScript = QtWidgets.QLineEdit()
        # self.vendorScript.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # self.vendorScript.setFont(defaultFont)
        # optionLeftLayout.addWidget(self.vendorScript)
        # self.vendorScript.setText('/stdrepo/CSP/jungsup.han/scripts/dxpackager/vendorMaya.py')
        # self.vendorScript.setText('/stdrepo/CSP/jungsup.han/scripts/dxpublisher/1.0/scripts/vendorMaya.py')
        #
        # browseVendorScriptButton = QtWidgets.QPushButton('Browse')
        # browseVendorScriptButton.setFont(defaultFont)
        # browseVendorScriptButton.setFixedSize(80+extWdgWidth, 24)
        # browseVendorScriptButton.clicked.connect(self.browseVendorScriptButtonClicked)
        # optionRightLayout.addWidget(browseVendorScriptButton)

        # self.vendorScriptLabel.setVisible(False)
        # self.vendorScript.setVisible(False)
        # browseVendorScriptButton.setVisible(False)

        optionBottomLayout = QtWidgets.QHBoxLayout()
        optionBottomLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        optionBottomLeftLayout = QtWidgets.QHBoxLayout()
        optionBottomLeftLayout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        optionBottomLayout.addLayout(optionBottomLeftLayout)

        optionBottomRightLayout = QtWidgets.QHBoxLayout()
        optionBottomRightLayout.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        optionBottomLayout.addLayout(optionBottomRightLayout)

        self.userNameLabel = QtWidgets.QLabel('User Name: ')
        self.userNameLabel.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.userNameLabel.setFixedWidth(70+extWdgWidth)
        self.userNameLabel.setFont(defaultFont)
        optionBottomLeftLayout.addWidget(self.userNameLabel)

        userList = self.getUserInfo()

        userStringModel = QtCore.QStringListModel()
        userStringModel.setStringList(sorted(userList))
        userCompleter = QtWidgets.QCompleter()
        userCompleter.setModel(userStringModel)
        userCompleter.setMaxVisibleItems(10)

        self.userName = QtWidgets.QLineEdit()
        self.userName.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.userName.setFixedWidth(200+extWdgWidth)
        self.userName.setFont(defaultFont)
        self.userName.setCompleter(userCompleter)
        optionBottomLeftLayout.addWidget(self.userName)

        self.playerLabel = QtWidgets.QLabel('Player:')
        self.playerLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.playerLabel.setFont(defaultFont)
        self.playerLabel.setFixedWidth(60+extWdgWidth)
        optionBottomLeftLayout.addWidget(self.playerLabel)

        self.playerComboBox = QtWidgets.QComboBox(self)
        self.playerComboBox.setFont(defaultFont)
        self.playerComboBox.setFixedSize(100+extWdgWidth, 22)
        playerListView = QtWidgets.QListView()
        playerListView.setFont(defaultFont)
        self.playerComboBox.addItem('Default')
        self.playerComboBox.addItem('RV')
        self.playerComboBox.setView(playerListView)
        optionBottomLeftLayout.addWidget(self.playerComboBox)

        self.jobLabel = QtWidgets.QLabel('Job:')
        self.jobLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.jobLabel.setFont(defaultFont)
        self.jobLabel.setFixedWidth(40+extWdgWidth)
        optionBottomLeftLayout.addWidget(self.jobLabel)

        self.jobCheckBox = QtWidgets.QCheckBox()
        self.jobCheckBox.setFixedWidth(80+extWdgWidth)
        self.jobCheckBox.setChecked(False)
        optionBottomLeftLayout.addWidget(self.jobCheckBox)

        self.jobLabel.setVisible(False)
        self.jobCheckBox.setVisible(False)

        refreshButton = QtWidgets.QPushButton('Refresh List')
        refreshButton.setFont(defaultFont)
        refreshButton.setFixedSize(160+extWdgWidth, 24)
        refreshButton.clicked.connect(self.refreshStatusButtonClicked)
        optionBottomRightLayout.addWidget(refreshButton)

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
        # self.tableWidget.itemChanged.connect(self.tableItemChanged)

        self.twhHeader = self.tableWidget.horizontalHeader()
        # self.twhHeader.setStretchLastSection(True)
        self.twhHeader.setFont(defaultFont)
        self.twhHeader.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        self.twhHeader.setLineWidth(0)
        # self.twhHeader.setStyleSheet("::section { color: black; background-color: lightgray; }")
        self.twvHeader = self.tableWidget.verticalHeader()
        self.twvHeader.setDefaultSectionSize(tableRowHeight)
        self.twvHeader.setVisible(False)
        # self.twvHeader.setSortIndicator(0, QtCore.Qt.AscendingOrder)

        self.tableHeader = ['No', 'CHKPREV', 'NAME', 'CHECK', 'MESSAGE']
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setSortingEnabled(True)
        self.tableWidget.setHorizontalHeaderLabels(self.tableHeader)
        self.twhHeader.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setColumnWidth(0, 60)
        # self.tableWidget.setColumnWidth(1, tablePreviewWidth)
        self.tableWidget.setColumnWidth(1, tablePreviewWidth)
        self.tableWidget.setColumnWidth(2, 340)
        self.tableWidget.setColumnWidth(3, 80)
        self.tableWidget.setColumnWidth(4, 350)
        self.tableWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        # self.tableWidget.customContextMenuRequested.connect(self.showTableContextMenu)
        middleLayout.addWidget(self.tableWidget)

        self.overwriteCheckBox = QtWidgets.QCheckBox('Overwrite Version')
        self.overwriteCheckBox.setFont(defaultFont)
        self.overwriteCheckBox.setVisible(False)
        middleLayout.addWidget(self.overwriteCheckBox)

        bottomLayout = QtWidgets.QHBoxLayout()
        bottomLayout.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        # submitSelectionButton = QtWidgets.QPushButton('Submit Selection')
        # submitSelectionButton.setFont(defaultFont)
        # submitSelectionButton.setFixedSize(140+extWdgWidth, 24)
        # submitSelectionButton.clicked.connect(self.submitSelectionButtonClicked)
        # bottomLayout.addWidget(submitSelectionButton)

        checkSelectionButton = QtWidgets.QPushButton('Check spool')
        checkSelectionButton.setFont(defaultFont)
        checkSelectionButton.setFixedSize(140+extWdgWidth, 40)
        checkSelectionButton.clicked.connect(self.checkSelectionButtonClicked)
        bottomLayout.addWidget(checkSelectionButton)

        ################################################################## Moon ########

        exportCacheButton = QtWidgets.QPushButton('Export Cache')
        exportCacheButton.setFont(defaultFont)
        exportCacheButton.setFixedSize(140+extWdgWidth, 40)
        exportCacheButton.clicked.connect(self.exportCacheClicked)
        bottomLayout.addWidget(exportCacheButton)

        #################################################################################

        # exportSelectionButton = QtWidgets.QPushButton('Export Selection')
        # exportSelectionButton.setFont(defaultFont)
        # exportSelectionButton.setFixedSize(140+extWdgWidth, 24)
        # exportSelectionButton.clicked.connect(self.exportSelectionButtonClicked)
        # bottomLayout.addWidget(exportSelectionButton)

        mainLayout.addLayout(addressLayout)
        mainLayout.addLayout(optionLayout)
        mainLayout.addLayout(optionBottomLayout)
        mainLayout.addLayout(middleLayout)
        mainLayout.addLayout(bottomLayout)

        self.setCentralWidget(self.mainWidget)
        self.mainWidget.setLayout(mainLayout)

        self.setWindowTitle(self.title)
        self.mainWidget.setMinimumWidth(1000)
        self.mainWidget.setMinimumHeight(800)
        # self.setWindowIcon(QtGui.QIcon(scriptDir+'/images/icon.ico'))
        self.show()

        self.loadSettings()

        self.refreshStatus()

        self.refreshTh = MainWindowTh(self)
        self.refreshTh.refreshStatusSignal.connect(self.refreshStatus)
        self.refreshTh.start()

    def loadSettings(self):
        homeDir = os.path.expanduser('~')
        settingsDir = homeDir+'/.config'
        settingsFile = settingsDir+'/dxpublisher.yml'

        settingsDict = {}
        if os.path.isfile(settingsFile):
            with open(settingsFile, 'r') as f:
                settingsDict = yaml.load(f, Loader=yaml.SafeLoader)

        try: self.vendorDir.setText(settingsDict['ui']['vendor_dir'])
        except: pass

    def saveSettings(self):
        homeDir = os.path.expanduser('~')
        settingsDir = homeDir+'/.config'
        settingsFile = settingsDir+'/dxpublisher.yml'

        try: os.makedirs(settingsDir)
        except: pass

        settingsDict = {}
        if os.path.isdir(settingsDir):
            if os.path.isfile(settingsFile):
                with open(settingsFile, 'r') as f:
                    settingsDict = yaml.load(f, Loader=yaml.SafeLoader)

        if not 'ui' in settingsDict:
            settingsDict['ui'] = {}

        settingsDict['ui']['vendor_dir'] = str(self.vendorDir.text())

        with open(settingsFile, 'w+') as f:
            yaml.dump(settingsDict, f)

    def getUserName(self):
        userName = self.userName.text()
        if userName.count('.') != 1 or len(userName) < 7:
            userName = getpass.getuser()

        return userName

    def getUserInfo(self):
        import dxConfig
        import requests

        API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
        user = {}
        user['api_key'] = API_KEY

        infos = requests.get("http://%s/dexter/search/user.php" %(dxConfig.getConf('TACTIC_IP')), params=user).json()

        codeList = []
        for idx, i in enumerate(infos):
            codeList.append(i['code'])

        return codeList

    def closeEvent(self, *args, **kwargs):
        self.closed = True

    def tableDoubleClicked(self, event):
        # row = int(self.tableWidget.item(event.row(), self.idCol).text())
        row = event.row()
        col = event.column()

        vendorDir = self.vendorDir.text()
        rowFn = self.tableWidget.item(row, self.nameCol).text()

        fileName = rowFn
        fileNameNoext = os.path.splitext(fileName)[0]

        showName = fileName.split('_')[0]
        showRulebookFile = os.path.join('/show', showName, '_config', 'DXRulebook.yaml')
        if os.path.exists(showRulebookFile):
            os.environ['DXRULEBOOKFILE'] = showRulebookFile
            rb.Reload()

        # if col == self.vndPrevCol:
        #     EXEplayer = EXEdefault
        #     if self.playerComboBox.currentText() == 'RV': EXEplayer = EXErv
        #     try: vfMov = self.tableWidget.item(row,  self.vndPrevCol).text()
        #     except: vfMov = ''
        #     if os.path.isfile(vfMov):
        #         subprocess.Popen(EXEplayer+' '+vfMov, shell=True)

        if col == self.chkPrevCol:
            EXEplayer = EXEdefault
            if self.playerComboBox.currentText() == 'RV': EXEplayer = EXErv
            try: cfMov = self.tableWidget.item(row,  self.chkPrevCol).text()
            except: cfMov = ''
            if os.path.isfile(cfMov):
                subprocess.Popen(EXEplayer+' '+cfMov, shell=True)

        elif col == self.nameCol:
            try:
                coder = rb.Coder()
                tmpv = coder.F.MAYA.vendor.Decode(fileName)
                tmpv.departs = 'ANI'
                showCode = tmpv.show

                aniWorks = coder.D.ANI.WORKS.Encode(**tmpv)
                aniFile = coder.F.MAYA.BASE.Encode(**tmpv)
                snPathAs = os.path.join(aniWorks, aniFile)
            except:
                msgBox = QtWidgets.QMessageBox(self)
                msgBox.setText('파일 이름 형식이 올바르지 않습니다.')
                msgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msgBox.setMaximumWidth(960)
                msgBox.setMaximumHeight(640)
                msgBox.exec_()
                return

            if not os.path.isfile(snPathAs):
                msgBox = QtWidgets.QMessageBox(self)
                msgBox.setText(unicode(snPathAs)+u'\n파일이 존재하지 않습니다.')
                msgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msgBox.setMaximumWidth(960)
                msgBox.setMaximumHeight(640)
                msgBox.exec_()
                return

            checkStatus = self.tableWidget.item(row, self.checkCol).text()
            if unicode(checkStatus) != u'CHECKED' and not unicode(checkStatus).startswith(u'통과'):
                msgBox = QtWidgets.QMessageBox(self)
                msgBox.setIcon(QtWidgets.QMessageBox.Warning)
                msgBox.setWindowTitle('Confirm')
                msgBox.setText('체크가 완료되지 않았습니다. 그래도 여시겠습니까?')
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
                buttonY = msgBox.button(QtWidgets.QMessageBox.Yes)
                buttonY.setText('Yes')
                buttonN = msgBox.button(QtWidgets.QMessageBox.No)
                buttonN.setText('No')
                msgBox.exec_()
                if msgBox.clickedButton() == buttonN:
                    return

            os.environ['NAUTILUS_SCRIPT_SELECTED_FILE_PATHS'] = snPathAs
            subprocess.Popen('/backstage/dcc/packages/ext/dxusd_maya/1.0.0/share/nautilus/scripts/UsdCacheExportTool3', shell=True)
            os.environ['NAUTILUS_SCRIPT_SELECTED_FILE_PATHS'] = ''

        elif col == self.checkCol:
            logMayaPath = self.vendorDir.text()+'/logs/'+self.tableWidget.item(row, self.nameCol).text()+'.log'
            if os.path.isfile(logMayaPath):
                subprocess.Popen('/bin/gedit '+logMayaPath, shell=True)

        elif col == self.messageCol:
            # logUsdPath = self.vendorDir.text()+'/logs/'+os.path.splitext(self.tableWidget.item(row, self.nameCol).text())[0]+'.usd.log'
            # if os.path.isfile(logUsdPath):
            #     subprocess.Popen('/bin/gedit '+logUsdPath, shell=True)

            logMayaPath = self.vendorDir.text()+'/logs/'+self.tableWidget.item(row, self.nameCol).text()+'.log'
            if os.path.isfile(logMayaPath):
                subprocess.Popen('/bin/nautilus '+logMayaPath, shell=True)
    
    ###################################################################### Moon ###############################################

    def setShowConfig(self, dirPath):
        tmp = dirPath.split('/')
        self.showName = tmp[tmp.index('show')+1]
        showRbPath = '/show/{SHOW}/_config/DXRulebook.yaml'.format(SHOW=self.showName)

        if os.path.exists(showRbPath):
            print '>> showRbPath:', showRbPath
            os.environ['DXRULEBOOKFILE'] = showRbPath
        else:
            if os.environ.has_key('DXRULEBOOKFILE'):
                del os.environ['DXRULEBOOKFILE']

        rb.Reload()





    def exportCacheClicked(self):


        slRows = self.tableWidget.selectionModel().selectedRows()

        vndSnPathList = []
        for slRow in slRows:
            rown = slRow.row()
            fileName = self.tableWidget.item(rown, self.nameCol).text()
            fileNameNoext = os.path.splitext(fileName)[0]
            showName = fileName.split('_')[0]
            showRulebookFile = os.path.join('/show', showName, '_config', 'DXRulebook.yaml')


            if os.path.exists(showRulebookFile):
                os.environ['DXRULEBOOKFILE'] = showRulebookFile
                rb.Reload()



            coder = rb.Coder()
            tmpv = coder.F.MAYA.vendor.Decode(fileName)

            tmpv.departs = 'ANI'
            showCode = tmpv.show

            aniWorks = coder.D.ANI.WORKS.Encode(**tmpv)
            aniFile = coder.F.MAYA.BASE.Encode(**tmpv)
            snPathAs = os.path.join(aniWorks, aniFile)

            if os.path.isfile(snPathAs):
                print "export scene :", snPathAs
                    
            jsonFile = snPathAs.replace(".mb", ".json")
            print jsonFile
            sceneName = os.path.basename(snPathAs)
            dirPath = os.path.dirname(snPathAs)                    



            flags  = rb.Flags(pub='_3d')
            flags.D.SetDecode(dirPath, 'ROOTS')
            flags.F.MAYA.SetDecode(sceneName, 'BASE')
            if flags.has_key('departs'):
                flags.pop("departs")
            outDir = flags.D.SHOT


            flags = rb.Flags()
            flags.D.SetDecode(outDir)
            outDirItem = flags.copy()


            nameSpaceErrorList = []

            with open(jsonFile, "r") as f:
                sceneGraph = json.load(f)
                
                # Base Setup
                if sceneGraph.has_key('rezRequest'):
                    self.rezRequestConfig = sceneGraph['rezRequest']
                else:
                    self.rezRequestConfig = ['maya-' + sceneGraph['mayaVersion'], 'dxusd_maya', 'usd_maya']     # default packages

                self.mayaVersion = sceneGraph["mayaVersion"]
                self.artistName = sceneGraph['artist']
 

                # Camera
                self.cameras = []
                if len(sceneGraph['camera']) >= 1:
                    for i in range(len(sceneGraph['camera'])):
                        self.cameras.append(sceneGraph['camera'][i][0])



                # Layout 
                self.layouts = []
                if len(sceneGraph['layout']) >= 1:
                    for i in range(len(sceneGraph['layout'])):
                        self.layouts.append(sceneGraph['layout'][i][0])                    



                ''' 
                if len(sceneGraph["layout"]) >= 1:
                    nsLayerItem = {}
                    flags = rb.Flags()

                    for node in sceneGraph['layout']:

                        # [nodeName, export type, GetViz, nodeType]
                        if node[3] == 'extra':
                            outDirItem['nslyr'] = 'extra'
                            outDirItem['desc'] = node[0]
                        else:
                            ret = flags.N.USD.layout.Decode(node[0])
                            outDirItem['nslyr'] = ret['nslyr']
                            outDirItem['desc'] = node[0]
                '''



                # Animation Cache
                self.geoCache = []
                if len(sceneGraph["geoCache"]) >= 1:
                     
                    for i in range(len(sceneGraph['geoCache'])):
                        self.geoCache.append(sceneGraph['geoCache'][i][0])

                        if len(sceneGraph['geoCache'][i][0].split(":")) > 2:
                            nameSpaceErrorList.append(sceneGraph['geoCache'][i][0])
                            continue




                if nameSpaceErrorList:
                    print nameSpaceErrorList
                    text = "\n".join(nameSpaceErrorList)
                    text += "\ntoo many namespace, please clean up namespace"
                    showfinishedDialog("info", text)




                opts = ""
                if os.environ.has_key('DXRULEBOOKFILE'):
                    opts += ' --show %s' % showName
            
                if snPathAs:
                    #srcfile = outDir + "/" + sceneName
                    if os.path.splitext(snPathAs)[-1] in ['.mb']:
                        opts += ' --file %s' % snPathAs
                        
                    if outDir:
                        opts += ' --outDir %s' % outDir                        
                        

                    opts += ' --user %s' % getpass.getuser()
                    opts += " --mayaver %s" % self.mayaVersion                    




                optstr = self.sceneItemToString(self.cameras, 'cam', outDirItem)
                if optstr:
                    opts += ' --camera "%s"' % optstr
 
 
                optstr = self.sceneItemToString(self.layouts, 'layout', outDirItem)
                if optstr:
                    opts += ' --layout "%s"' % optstr
 
 
                optstr = self.sceneItemToString(self.geoCache, 'ani', outDirItem)
                if optstr:
                    opts += ' --mesh "%s"' % optstr
                    
                start = 0
                end = 0
                step = 1.0

                opts += ' --frameRange %s %s' % (start, end)                    
                opts += ' --step %s' % step
                
                    
                    
                opts += " --host spool"
                #text = " Spool Completed "
                command = BATCHSCENESCRIPT.format(DCC=os.getenv('DCCPROC'), PACKAGES=' '.join(self.rezRequestConfig), OPTS=opts)
                print 'batch cmd >>', command
                p = subprocess.Popen(command, shell=True)
                p.wait()





    def sceneItemToString(self, eleList, category, outDirItem):
        data = list()

        eleNum = len(eleList)
        coder = rb.Coder()
        outDirItem['task'] = category                        


        if(category == 'cam'):
            
            for node in eleList:
                outDirItem['nslyr'] = ""
                nsLayerDir = coder.D.Encode(**outDirItem)
                
                nextVersion = utl.GetNextVersion(nsLayerDir)
                data.append('{VER}={NODE}'.format(VER=nextVersion, NODE=node))

            if data:
                return ' '.join(data)



        if(category == 'layout'):
            
            for node in eleList:
                outDirItem['nslyr'] = node
                nsLayerDir = coder.D.Encode(**outDirItem)

                nextVersion = utl.GetNextVersion(nsLayerDir)
                data.append('{VER}={NODE}'.format(VER=nextVersion, NODE=node))

            if data:
                return ' '.join(data)



        if(category == 'ani'):

            for node in eleList:
                nsLayer, nodeName = node.split(":")
                outDirItem['nslyr'] = nsLayer
                nsLayerDir = coder.D.Encode(**outDirItem)

                nextVersion = utl.GetNextVersion(nsLayerDir)
                #print nextVersion

                data.append('{VER}={NODE}'.format(VER=nextVersion, NODE=node))

            if data:
                return ' '.join(data)



    ############################################################################################################################

    def browseVendorDirButtonClicked(self):
        fileDialog = QtWidgets.QFileDialog(self)
        fileDialog.setWindowTitle('Select Vendor Directory')
        fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        fileDialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        fileDialog.setNameFilter('Maya Files (*.ma *.mb)')
        if fileDialog.exec_() == QtWidgets.QDialog.Accepted:
            slDir = os.path.dirname(fileDialog.selectedFiles()[0])
            self.vendorDir.setText(slDir)

    # def browseVendorScriptButtonClicked(self):
    #     fileDialog = QtWidgets.QFileDialog(self)
    #     fileDialog.setWindowTitle('Select Vendor Script File')
    #     fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    #     fileDialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    #     fileDialog.setNameFilter('Python Files (*.py)')
    #     if fileDialog.exec_() == QtWidgets.QDialog.Accepted:
    #         slFile = fileDialog.selectedFiles()[0]
    #         self.vendorScript.setText(slFile)

    def loadPreview(self, vendorDir, vf, rowCnt):
        vfNoExt = os.path.splitext(vf)[0]
        vfMov = os.path.join(vendorDir, vfNoExt+'.mov')
        if os.path.isfile(vfMov):
            vfMovTime =  os.path.getmtime(vfMov)
            vfThumbDir = os.path.join(vendorDir, 'thumb')
            if not os.path.isdir(vfThumbDir):
                os.makedirs(vfThumbDir)

            vfThumb =  os.path.join(vfThumbDir,  vfNoExt+'.thumb.jpg')
            vfThumbTime = 0
            if os.path.isfile(vfThumb):
                vfThumbTime = os.path.getmtime(vfThumb)


            if vfMovTime > vfThumbTime:
                os.system(EXEffmpeg+' -y -i '+vfMov+' -ss 00:00:00 -vf "scale=\'min(200,iw)\':-1" -vframes 1 '+vfThumb)

            if os.path.isfile(vfThumb):
                self.tableWidget.setCellWidget(rowCnt, self.vndPrevCol, ImageLabel(self,  vfThumb, tablePreviewWidth))
                self.tableWidget.setItem(rowCnt, self.vndPrevCol, QtWidgets.QTableWidgetItem(vfMov))

        else:
            try:
                self.tableWidget.item(rowCnt, self.vndPrevCol).setText('')
                self.tableWidget.removeCellWidget(rowCnt, self.vndPrevCol)
            except: pass

        shotName = '_'.join(vfNoExt.split('_')[2:4])

        cfPreviewDir = os.path.join(vendorDir, 'preview')
        cfMovList = glob.glob(cfPreviewDir+'/*'+shotName+'*.mov')
        cfMov = ''
        if len(cfMovList) > 0:
            cfMov = cfMovList[0]

        if os.path.isfile(cfMov):
            cfMovTime =  os.path.getmtime(cfMov)
            cfThumbDir = os.path.join(cfPreviewDir, 'thumb')
            if not os.path.isdir(cfThumbDir):
                os.makedirs(cfThumbDir)

            cf = os.path.basename(cfMov)
            cfNoExt = os.path.splitext(cf)[0]
            cfThumb =  os.path.join(cfThumbDir, cfNoExt+'.thumb.jpg')
            cfThumbTime = 0
            if os.path.isfile(cfThumb):
                cfThumbTime = os.path.getmtime(cfThumb)

            if cfMovTime > cfThumbTime:
                os.system(EXEffmpeg+' -y -i '+cfMov+' -ss 00:00:00 -vf "scale=\'min(200,iw)\':-1" -vframes 1 '+cfThumb)

            if os.path.isfile(cfThumb):
                self.tableWidget.setCellWidget(rowCnt, self.chkPrevCol, ImageLabel(self, cfThumb, tablePreviewWidth))
                self.tableWidget.setItem(rowCnt, self.chkPrevCol, QtWidgets.QTableWidgetItem(cfMov))

        else:
            try:
                self.tableWidget.item(rowCnt, self.chkPrevCol).setText('')
                self.tableWidget.removeCellWidget(rowCnt, self.chkPrevCol)
            except: pass

    def vendorDirChanged(self):
        vendorDir = self.vendorDir.text()
        if os.path.dirname(vendorDir):
            try: venderFileList = os.listdir(vendorDir)
            except: venderFileList = []
            venderFileList = sorted(venderFileList)
            if len(venderFileList) > 0:
                self.tableWidget.setSortingEnabled(False)
                self.tableWidget.clearSelection()
                self.tableWidget.setRowCount(0)
                for vf in venderFileList:
                    if bool(re.search(r'\.m(a|b)$', vf)):
                        rowCnt = self.tableWidget.rowCount()
                        self.tableWidget.insertRow(rowCnt)

                        self.tableWidget.setItem(rowCnt, self.idCol, QtWidgets.QTableWidgetItem(format(rowCnt+1, ' 4d')))
                        self.tableWidget.setItem(rowCnt, self.nameCol, QtWidgets.QTableWidgetItem(vf))
                        self.tableWidget.setItem(rowCnt, self.checkCol, QtWidgets.QTableWidgetItem(''))
                        self.tableWidget.setItem(rowCnt, self.messageCol, QtWidgets.QTableWidgetItem(''))

                self.tableWidget.setSortingEnabled(True)
                self.tableWidget.sortItems(0)

                self.saveSettings()

        # self.refreshStatus()

    def checkSelectionButtonClicked(self):
        slRows = self.tableWidget.selectionModel().selectedRows()
        # vndScript = self.vendorScript.text().strip()
        # if vndScript != '' and not os.path.isfile(vndScript):
        #     msgBox = QtWidgets.QMessageBox(self)
        #     msgBox.setText('Vendor script file does not exists.')
        #     msgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
        #     msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
        #     msgBox.setMaximumWidth(960)
        #     msgBox.setMaximumHeight(640)
        #     msgBox.exec_()
        #     return

        vndSnPathList = []
        for slRow in slRows:
            rown = slRow.row()
            vndSnPath = self.vendorDir.text()+'/'+self.tableWidget.item(rown, self.nameCol).text()
            vndSnPathList.append(vndSnPath)

        if len(vndSnPathList) > 0:
            if not self.jobCheckBox.isChecked():
                coder = rb.Coder()
                for vsp in vndSnPathList:
                    vspFn = os.path.basename(vsp)
                    try:
                        tmpv = coder.F.MAYA.vendor.Decode(vspFn)
                    except:
                        msgBox = QtWidgets.QMessageBox(self)
                        msgBox.setText(u'파일 이름 형식이 올바르지 않습니다.\n%s\n리네임 후 다시 시도 해 주세요!' % vspFn)
                        msgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
                        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                        msgBox.setMaximumWidth(960)
                        msgBox.setMaximumHeight(640)
                        msgBox.exec_()
                        shutil.rmtree(self.vendorDir.text()+'/logs')
                        self.refreshStatus()
                        return

                    logMayaPath = self.vendorDir.text()+'/logs/'+vspFn+'.log'
                    if os.path.isfile(logMayaPath):
                        os.remove(logMayaPath)

                    mayaLogger = createLogger('VENDOR_CHECK', logMayaPath)
                    mayaLogger.info('[ mayapy ] < START >')
                    logging.shutdown()
                    del mayaLogger

                vndChk = vendorSpool.VendorCheck(','.join(vndSnPathList), vendorScript='', spoolCache=False, overwrite=bool(self.overwriteCheckBox.isChecked()), user=self.getUserName())
                vndChk.spool()

                msgBox = QtWidgets.QMessageBox(self)
                msgBox.setText(u'Job spool 완료.')
                msgBox.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
                msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msgBox.setMaximumWidth(960)
                msgBox.setMaximumHeight(640)
                msgBox.exec_()

            else:
                jobDict = {
                    'vendorDir': self.vendorDir.text(),
                    'snPathList': vndSnPathList,
                    'vendorScript': vndScript,
                    'userName': self.getUserName()
                }
                jobDictStr = json.dumps(jobDict, indent=2)
                snListStr = json.dumps(jobDict['snPathList'])
                jobId = hashlib.md5(snListStr).hexdigest()
                vspFn = os.path.basename(vndSnPathList[0])
                vendorCode = vspFn.split('_')[1]
                vendorJobDir = jobDir+'/'+vendorCode
                vendorChkWaitDir = vendorJobDir+'/check/wait'

                if not os.path.isdir(vendorChkWaitDir):
                    try: os.makedirs(vendorChkWaitDir)
                    except: pass

                if os.path.isdir(vendorChkWaitDir):
                    vsnChkJob = vendorChkWaitDir+'/'+jobId+'.job'
                    with open(vsnChkJob, 'w+') as fd:
                        fd.write(jobDictStr)
                        print 'check submitted', jobId+'.job'

                else:
                    print vendorChkWaitDir, 'does not exists.'

        self.refreshStatus()

    def refreshStatusButtonClicked(self):
        self.refreshStatus()

    def refreshStatus(self):
        try:
            for rown in range(self.tableWidget.rowCount()):
                self.tableWidget.item(rown, self.checkCol).setText('')
                self.tableWidget.item(rown, self.checkCol).setBackground(self.tableWidget.item(rown, self.nameCol).background())

                self.tableWidget.item(rown, self.messageCol).setText('')
                self.tableWidget.item(rown, self.messageCol).setBackground(self.tableWidget.item(rown, self.nameCol).background())

                self.loadPreview(self.vendorDir.text(), self.tableWidget.item(rown, self.nameCol).text(), rown)

                logMayaPath = self.vendorDir.text()+'/logs/'+self.tableWidget.item(rown, self.nameCol).text()+'.log'
                if os.path.isfile(logMayaPath):
                    fileTime = datetime.datetime.fromtimestamp(os.path.getmtime(logMayaPath))
                    expTstr = fileTime + datetime.timedelta(minutes=30)
                    nowTime = datetime.datetime.now()

                    fdata = ''
                    with open(logMayaPath, 'r') as fd:
                        fdata = fd.read()

                    if fdata.count('- ERROR -') > 0:
                        self.tableWidget.item(rown, self.checkCol).setText('오류')
                        self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(255, 0, 0, 64))

                        msg = fdata.split('- ERROR -')[-1].split('\n')[0].split(']')[-1].strip()
                        self.tableWidget.item(rown, self.messageCol).setText(msg)
                        self.tableWidget.item(rown, self.messageCol).setBackground(QtGui.QColor(255, 0, 0, 64))

                    elif fdata.count('- WARNING -') > 0:
                        if fdata.count('< PREVIEW >') > 0:  self.tableWidget.item(rown, self.checkCol).setText('경고')
                        else: self.tableWidget.item(rown, self.checkCol).setText('경고(프리뷰)')
                        self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(255, 128, 0, 64))

                        msg = fdata.split('- WARNING -')[-1].split('\n')[0].split(']')[-1].strip()
                        self.tableWidget.item(rown, self.messageCol).setText(msg)
                        self.tableWidget.item(rown, self.messageCol).setBackground(QtGui.QColor(255, 128, 0, 64))

                    elif fdata.count('< END >') > 0:
                        if fdata.count('< PREVIEW >') > 0:  self.tableWidget.item(rown, self.checkCol).setText('완료')
                        else: self.tableWidget.item(rown, self.checkCol).setText('완료(프리뷰)')
                        self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(128, 255, 0, 64))

                    elif fdata.count('< CHECKED >') > 0:
                        if fdata.count('< PREVIEW >') > 0:  self.tableWidget.item(rown, self.checkCol).setText('통과')
                        else: self.tableWidget.item(rown, self.checkCol).setText('통과(프리뷰)')
                        self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(128, 255, 0, 64))

                    else:
                        if fdata.strip() != '':
                            if expTstr < nowTime:
                                self.tableWidget.item(rown, self.checkCol).setText('트랙터 확인')
                                self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(128, 64, 255, 64))

                            else:
                                self.tableWidget.item(rown, self.checkCol).setText('처리 중')
                                self.tableWidget.item(rown, self.checkCol).setBackground(QtGui.QColor(0, 0, 255, 64))

                sleepInt = 10
                QtWidgets.QApplication.processEvents()

        except:
            sleepInt = 5
            print '# Refresh status error'

class MainWindowTh(QtCore.QThread):
    refreshStatusSignal = QtCore.Signal()
    def __init__(self, mainWindow):
        QtCore.QThread.__init__(self)
        self.mainWindow = mainWindow

    def run(self):
        while 1:
            if self.mainWindow.closed or not self.mainWindow.isVisible():
                break

            time.sleep(sleepInt)
            self.refreshStatusSignal.emit()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    mainWindow = MainWindow()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            path = os.path.dirname(path)
        mainWindow.vendorDir.setText(path)

    sys.exit(app.exec_())

# /backstage/dcc/DCC dxpublisher  -- dxpublisher
# cp -rf /home/jungsup.han/homestage/dcc/packages/ext/dxpublisher/* /stdrepo/CSP/jungsup.han/scripts/dxpublisher/test/; /backstage/dcc/DCC python-2 pyside2 -- python /stdrepo/CSP/jungsup.han/scripts/dxpublisher/test/scripts/dxpublisherUI.py
