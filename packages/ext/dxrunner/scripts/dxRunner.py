#coding : utf-8
import sys, os, re
from PyQt4 import QtGui, QtCore
import json
import subprocess
import getpass
import time
import glob

scriptsPath = os.path.dirname(__file__)
if not '/backstage/dcc/packages/ext/dxrulebook/1.0.0/python-2/scripts' in sys.path:
    sys.path.append('/backstage/dcc/packages/ext/dxrulebook/1.0.0/python-2/scripts')

if not '/backstage/libs/python/2.7.16/lib/python2.7/site-packages' in sys.path:
    sys.path.append('/backstage/libs/python/2.7.16/lib/python2.7/site-packages')

from fileItem import FileItem

# STYLESHEET
import qdarkstyle

# TACTIC
from resources.dxrunner_pyqt4 import Ui_Form
TACTIC_IP = '10.0.0.51'
API_KEY = "c70181f2b648fdc2102714e8b5cb344d"

# DXRULEBOOK
import DXRulebook.Interface as rb

parseDepart = {'animation': 'ANI',
               'creature': 'RIG',
               'model': 'AST',
               'texture': 'AST',
               'fx': 'PFX',
               'lighting': 'LNR',
               'comp': 'CMP',
               'matchmove': 'MMV',
               'layout': 'LAY',
               'previz': 'LAY',
               'crowd':'CRD'
}

departToDCC = {
    "ANI": "MAYA",
    "AST": "MAYA",
    "RIG": "MAYA",
    "CMP": "NUKE",
    "PFX": "HOUDINI",
    "LAY": "MAYA",
    "LNR": "KATANA"
}

# FileName
contextMapping = {
    # ASSET
    'model': 'model',
    'texture': 'texture',
    'hair': 'groom',
    # RIGGING
    'creature': 'rig',
    'rigging': 'rig',
    'muscle': 'rig',
    'facial': 'rig',
    'cloth': 'sim',
    'simulation': 'sim',
    'finalize': 'sim',
    # ANIMATION
    'animation': 'ani',
    'animation/layout': 'ani',
    'crowd': 'crowd',
    'mocap': 'mocap',
    # LAYOUT
    'layout': 'layout',
    # LIGHTING
    'lighting': 'lighting',
    # FX
    'fx': 'fx',
    # MATCHMOVE
    "matchmove": 'matchmove',
    # COMP
    "comp": "comp",
    "keying": "keying",
    "retime": "retime",
    "roto": "roto",
    "remove": "remove",
    "precomp": "precomp",
    'merge': 'merge'
}

# Directory
taskDirMapping = {
    # ASSET
    'model': 'model',
    'texture': 'texture',
    'hair': 'groom',
    # RIGGING
    'creature': 'rig',
    'rigging': 'rig',
    'muscle': 'ziva',
    'facial': 'rig',
    'cloth': 'cloth',
    'finalize': 'finalize',
    'simulation': 'cloth',
    'hairSim': 'groom',
    # ANIMATION
    'animation': 'ani',
    'animation/layout': 'ani',
    'crowd': 'crowd',
    'mocap': 'mocap',
    # LAYOUT
    'layout': 'layout',
    # LIGHTING
    'lighting': 'lighting',
    # FX
    'fx': 'fx',
    # MATCHMOVE
    "matchmove": 'matchmove',
    # COMP
    "comp": "comp",
    "keying": "keying",
    "retime": "retime",
    "roto": "roto",
    "remove": "remove",
    "precomp": "precomp",
    'merge': 'merge'
}

class dxRunnerMain(QtGui.QDialog):
    def __init__(self, parent=None, taskInfo=None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        darkStyleSheet = qdarkstyle.load_stylesheet_pyqt()
        self.setStyleSheet(darkStyleSheet)

        self.taskInfo = taskInfo
        print taskInfo

        self.move(QtGui.QDesktopWidget().availableGeometry(parent).center() - self.frameGeometry().center())
        self.setWindowTitle('DXRunner')
        iconpath = os.path.join(scriptsPath, 'resources', 'dxrunner_icon.png')
        self.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(iconpath)))

        self.ui.dccListWidget.itemDoubleClicked.connect(self.launchDCC)
        self.ui.devTreeWidget.itemClicked.connect(self.fileSelectFunc)
        self.ui.pubTreeWidget.itemClicked.connect(self.fileSelectFunc)
        self.ui.fileBrowserTreeWidget.itemClicked.connect(self.fileSelectFunc2)
        self.ui.devFileMakeBtn.clicked.connect(lambda: self.makeWorkFile('dev'))
        self.ui.devOpenDirBtn.clicked.connect(lambda: self.openWorkDir('dev'))
        self.ui.pubFileMakeBtn.clicked.connect(lambda: self.makeWorkFile('pub'))
        self.ui.pubOpenDirBtn.clicked.connect(lambda: self.openWorkDir('pub'))

        self.ui.directoryEdit.returnPressed.connect(self.lineEditPressedReturn)
        self.ui.fileBrowserTreeWidget.itemDoubleClicked.connect(self.itemDoubleClicked)
        self.ui.fileBrowserTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.fileBrowserTreeWidget.customContextMenuRequested.connect(self.rmbClicked)

        self.ui.previousBtn.clicked.connect(self.previousBtnClicked)
        self.ui.nextBtn.clicked.connect(self.nextBtnClicked)

        # Shortcut
        self.altDownSC = QtGui.QShortcut(QtGui.QKeySequence("Alt+Down"), self).activated.connect(self.shortCutAltDown)
        self.altUpSC = QtGui.QShortcut(QtGui.QKeySequence("Alt+Up"), self).activated.connect(self.shortCutAltUp)

        # Parse TaskInfo
        self.showName = self.taskInfo['project_name'] # responseData[0]['name']
        # Temp Convention
        if self.showName == 'testshot':
            self.showName = 'pipe'

        if self.showName:
            showRulebookFile = os.path.join('/show', self.showName, '_config', 'DXRulebook.yaml')
            if os.path.exists(showRulebookFile):
                os.environ['DXRULEBOOKFILE'] = showRulebookFile
                rb.Reload()

        self.taskName = self.taskInfo['extra_name'] # SEQ_0010 or AssetName
        self.taskGroup = self.taskInfo['parent_category']

        # rig asset, shot
        if parseDepart[self.taskInfo['process']] != 'RIG':
            try:
                coder = rb.Coder()
                argv = coder.N.SHOTNAME.Decode(self.taskName)
                
                if argv.seq:
                    self.taskGroup = argv.seq
            except:
                pass

        self.ui.showEdit.setText(self.showName)
        self.ui.shotEdit.setText(self.taskName)

        self.makeDCCIcons()

        # Setup Working Directory
        self.setupWorkingDirectory()
        self.workFile = ''
        self.fileStack = []

    def shortCutAltDown(self):
        for item in self.ui.fileBrowserTreeWidget.selectedItems():
            self.updateFileBrowser(item.getPwd())
            break

    def shortCutAltUp(self):
        self.previousBtnClicked()

    def rmbClicked(self, pos):
        departRuleFile = os.path.join('/show', self.showName, '_config', 'DepartRule.config')
        with open(departRuleFile, 'r') as f:
            departRule = json.load(f)


        menu = QtGui.QMenu(self)
        menu.addAction("New Folder", self.makeNewFolder)
        menu.addAction("Open Works Directory", self.openWorkDir2)
        context = ''
        if taskDirMapping.has_key(self.task):
            context = taskDirMapping[self.task]
        elif taskDirMapping.has_key(self.process):
            context = taskDirMapping[self.process]
        else:
            assert False, "Not find context PROCESS[%s]/TASK[%s]" % (self.task, self.process)

        if departRule[self.department].has_key('_2d/%s' % self.entityType):
            _2dMenu = menu.addMenu("Open _2d Directory")
            for ruleFormat in departRule[self.department]['_2d/%s' % self.entityType]:
                departmentDirRulePath = ruleFormat.format(TYPE=self.taskGroup, ASSETNAME=self.taskName,  # ASSET
                                                          SEQ=self.taskGroup, SHOT=self.taskName,  # SHOT
                                                          CONTEXT=context,
                                                          ARTIST=self.taskInfo['login'])  # TRASH

                _2dMenu.addAction(os.path.basename(departmentDirRulePath), lambda: self.open_2dDir(departmentDirRulePath))

        if departRule[self.department].has_key('_3d/%s' % self.entityType):
            _3dMenu = menu.addMenu("Open _3d Directory")
            for ruleFormat in departRule[self.department]['_3d/%s' % self.entityType]:
                departmentDirRulePath = ruleFormat.format(TYPE=self.taskGroup, ASSETNAME=self.taskName,  # ASSET
                                                          SEQ=self.taskGroup, SHOT=self.taskName,  # SHOT
                                                          CONTEXT=context,
                                                          ARTIST=self.taskInfo['login'])  # TRASH

                _3dMenu.addAction(os.path.basename(departmentDirRulePath), lambda: self.open_3dDir(departmentDirRulePath))

        menu.addAction("Make Default File", self.makeWorkFile2)
        menu.exec_(QtGui.QCursor.pos())

    def exeSubprocess(self, command, shell=False):
        subprocess.Popen(command, shell=shell)
        # if os.path.isfile('/etc/centos-release'):
        #     f = open('/etc/centos-release', 'r')
        #     ver = f.readline()
        #     print ver
        #     if '7.' in ver:
        #         subprocess.Popen(command, shell=shell)
        #     else:
        #         subprocess.Popen(command, shell=shell, env={'USER': getpass.getuser()})

    def open_2dDir(self, ruleDir):
        departmentDirPath = os.path.join('/show', self.showName, '_2d')
        _2dDirPath = os.path.join(departmentDirPath, ruleDir)
        if not os.path.exists(_2dDirPath):
            errorMsg = QtGui.QErrorMessage()
            errorMsg.showMessage('not found %s' % _2dDirPath)
        self.exeSubprocess(["/usr/bin/nautilus", _2dDirPath])

    def open_3dDir(self, ruleDir):
        departmentDirPath = os.path.join('/show', self.showName, '_3d')
        _3dDirPath = os.path.join(departmentDirPath, ruleDir)
        if not os.path.exists(_3dDirPath):
            errorMsg = QtGui.QErrorMessage()
            errorMsg.showMessage('not found %s' % _3dDirPath)
        self.exeSubprocess(["/usr/bin/nautilus", _3dDirPath])

    def openWorkDir2(self):
        self.exeSubprocess(["/usr/bin/nautilus", self.ui.directoryEdit.text()])

    def openWorkDir(self, process):
        workingDirPath = eval('self.ui.%sTreeWidget.dirPath' % process)

        self.exeSubprocess(["/usr/bin/nautilus", workingDirPath])

    def getFileList(self, searchPath):
        value = os.popen('ls -lh %s' % searchPath).read()
        tempFiles = value.split('\n')
        length = len(tempFiles)
        fileIndex = 0
        for index, text in enumerate(tempFiles):
            fileIndex = index
            if "total " in text:
                tempFiles = tempFiles[index + 1:-1]
                break
        if fileIndex == length:
            return None

        files = []
        for file in tempFiles:
            testList = file.split(' ')
            pack = []
            isLink = False
            for index, x in enumerate(testList):
                if x:
                    pack.append(x)
                    if index == 0 and x[0] == "l":
                        isLink = True
            if isLink == False and len(pack) == 9:
                files.append(pack)
            elif isLink == True and len(pack) == 11:
                files.append(pack)

        if len(files) == 1:
            fileInfo = files[0]
            if fileInfo[0][0] == 'd' or fileInfo[0][0] == 'l':  # directory
                return files, searchPath
            else:
                filename = fileInfo[-1]
                r = re.compile("\033\[[0-9;]+m")
                filename = r.sub("", filename)
                filename = os.path.basename(filename)
                searchFileName = os.path.basename(searchPath)
                if searchFileName == filename:
                    return self.getFileList(os.path.dirname(searchPath))

        return files, searchPath

    def itemDoubleClicked(self, item, column):
        # if not item.isDirectory:
        #     # files
        #     self.getFileDownload(item.getPwd())
        # else:
        if item.isDirectory:
            self.updateFileBrowser(item.getPwd())
            # self.ui.fileBrowserTreeWidget.addTopLevelItem(item)

    def lineEditPressedReturn(self):
        self.updateFileBrowser(self.ui.directoryEdit.text())

    def updateFileBrowser(self, searchPath):
        searchPath = str(searchPath)
        fileList, searchPath = self.getFileList(searchPath)
        self.ui.directoryEdit.setText(searchPath)
        self.ui.fileBrowserTreeWidget.clear()
        for File in fileList:
            if "~" in File[-1]:
                continue
            FileItem(self.ui.fileBrowserTreeWidget, searchPath, File)

    def previousBtnClicked(self):
        self.ui.nextBtn.setEnabled(True)
        currentPath = str(self.ui.directoryEdit.text())
        self.fileStack.append(currentPath)
        dirPath = os.path.dirname(currentPath)
        self.updateFileBrowser(dirPath)

    def nextBtnClicked(self):
        nextPath = self.fileStack.pop(len(self.fileStack) - 1)
        if len(self.fileStack) == 0:
            self.ui.nextBtn.setEnabled(False)
        self.updateFileBrowser(nextPath)

    def makeNewFolder(self):
        # workingDirPath = eval('self.ui.%sTreeWidget.dirPath' % process)
        workingDirPath = str(self.ui.directoryEdit.text())

        dialog = QtGui.QInputDialog.getText(self, "Folder Name", "Folder Name")
        ret = dialog[1]
        if not ret:
            return

        description = str(dialog[0])
        os.makedirs(os.path.join(workingDirPath, description))

        self.updateFileBrowser(workingDirPath)

    def makeWorkFile2(self):
        # workingDirPath = eval('self.ui.%sTreeWidget.dirPath' % process)
        workingDirPath = str(self.ui.directoryEdit.text())

        dialog = QtGui.QInputDialog.getText(self, "description", "description")
        ret = dialog[1]
        if not ret:
            return

        description = str(dialog[0])
        DCC = departToDCC[self.department]

        context = ''
        if contextMapping.has_key(self.task):
            context = contextMapping[self.task]
        elif contextMapping.has_key(self.process):
            context = contextMapping[self.process]
        else:
            assert False, "Not find context PROCESS[%s]/TASK[%s]" % (self.task, self.process)

        flags = rb.Coder('F', DCC)
        filename = ''
        for index in range(1, 1000):
            if self.entityType == 'asset':
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    asset=self.taskName, desc=description)
            else:
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    seq=self.taskGroup, shot=self.taskName.split('_')[-1],
                                                    desc=description)

            if not os.path.exists(os.path.join(workingDirPath, filename)):
                break

        cmd = 'touch %s/%s' % (workingDirPath, filename)
        os.system('touch %s' % cmd)

        self.updateFileBrowser(workingDirPath)

    def makeWorkFile(self, process):
        workingDirPath = eval('self.ui.%sTreeWidget.dirPath' % process)

        if process == 'dev':
            dialog = QtGui.QInputDialog.getText(self, "description", "description")
            ret = dialog[1]
            if not ret:
                return

            description = str(dialog[0])
        else:
            description = ''

        DCC = departToDCC[self.department]

        context = ''
        if contextMapping.has_key(self.task):
            context = contextMapping[self.task]
        elif contextMapping.has_key(self.process):
            context = contextMapping[self.process]
        else:
            assert False, "Not find context PROCESS[%s]/TASK[%s]" % (self.task, self.process)

        flags = rb.Coder('F', DCC)
        filename = ''

        findLastVer = 1
        for index in range(1, 1000):
            if self.entityType == 'asset':
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    asset=self.taskName)
            else:
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    seq=self.taskGroup, shot=self.taskName.split('_')[-1])

            if not glob.glob('%s/%s' % (workingDirPath, filename.replace('.', '*.'))):
                findLastVer = index
                break

        if findLastVer == 1:
            findLastVer = 2

        for index in range(findLastVer - 1, 1000):
            if self.entityType == 'asset':
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    asset=self.taskName, desc=description)
            else:
                filename = flags.BASE.Encode(task=context, ver='v%03d' % index,
                                                    seq=self.taskGroup, shot=self.taskName.split('_')[-1],
                                                    desc=description)

            if not os.path.exists(os.path.join(workingDirPath, filename)):
                break

        cmd = '%s/%s' % (workingDirPath, filename)
        os.system('touch %s' % cmd)

        self.setupWorkingDirectory()


    def fileSelectFunc(self, item):
        self.workFile = os.path.join(item.dirPath, str(item.text(0)))

    def fileSelectFunc2(self, item):
        self.workFile = os.path.join(str(self.ui.directoryEdit.text()), str(item.text(0)))

    # Make Items
    def makeDCCIcons(self):
        self.ui.dccListWidget.clear()

        configPath = os.path.join('/show', str(self.ui.showEdit.text()), '_config', 'DCCLaunchList.config')
        if not os.path.exists(configPath):
            configPath = os.path.join(scriptsPath, 'resources', 'DCCLaunchList.config')

        with open(configPath, 'r') as f:
            dccJson = json.load(f)

            for dccName in dccJson.keys():
                item = QtGui.QListWidgetItem()
                iconName = dccJson[dccName]['icon']
                icon = QtGui.QIcon(QtGui.QPixmap('/backstage/share/icons/%s_icon.png' % iconName))
                item.setIcon(icon)
                item.setText(dccJson[dccName]['title'])
                item.command = dccJson[dccName]['cmd']
                self.ui.dccListWidget.addItem(item)

    def launchDCC(self, item):
        # os.system(item.command)

        # env = dict(os.environ)
        # env['TASK_INFO'] = dict(self.taskInfo)
        #
        # import pprint
        # pprint.pprint(env)
        # env = {}
        # env['TASK_INFO'] = str(self.taskInfo)

        # subprocess.Popen(item.command + self.workFile, shell=True, env=env)
        self.exeSubprocess(item.command + self.workFile, shell=True)
        self.close()
        # proc = subprocess.Popen(item.command + self.workFile, shell=True)
        # # # widProc = subprocess.Popen('echo "ibase=16; `wmctrl -l  | grep "Autodesk Maya" | cut -c 3-11 | tr a-z A-Z`"|bc', shell=True)
        # # print proc.pid
        # # widProc = subprocess.Popen('xdotool search --pid %d' % proc.pid, shell=True)
        # # while widProc.poll() == None:
        # #     output = widProc.stdout.readline()
        # #     if output:
        # #         print output.strip()
        # #
        # # proc.wait()
        # self.close()

    def setupWorkingDirectory(self):
        # find entity
        self.entityType = self.taskInfo['search_type'].split('/')[-1].split('?')[0] # asset or shot
        self.process = self.taskInfo['process']
        department = parseDepart[self.process]
        self.department = department
        
        task = self.taskInfo['context']
        if '/' in task:
            for splitTask in reversed(task.split('/')[1:]):
                task = ''.join([i for i in splitTask if not i.isdigit()])
                if task != '':
                    break

            if task is '':
                task = self.taskInfo['context'].split('/')[0]
        self.task = task

        context = ''
        if taskDirMapping.has_key(self.task):
            context = taskDirMapping[self.task]
        elif taskDirMapping.has_key(self.process):
            context = taskDirMapping[self.process]
        else:
            assert False, "Not find context PROCESS[%s]/TASK[%s]" % (self.task, self.process)

        self.ui.contextEdit.setText(context)

        if self.task == 'crowd':
            department = parseDepart['crowd']

        # open config
        departRuleFile = os.path.join('/show', self.showName, '_config', 'DepartRule.config')
        with open(departRuleFile, 'r') as f:
            departRule = json.load(f)

        ruleFormat = departRule[department][self.entityType]
        templateRuleFormat = ruleFormat
        if departRule[department].has_key('%s_template' % self.entityType):
            templateRuleFormat = departRule[department]['%s_template' % self.entityType]
        departmentDirPath = os.path.join('/show', self.showName, 'works', department)

        devpubList = ['pub']
        if '{DEVPUB}' in ruleFormat:
            devpubList.insert(0, 'dev')
        # else: # Not Using DEVPUB => Disable DEV
        #     self.ui.devGroupBox.setVisible(False)
        #     self.ui.line_3.setVisible(False)

        for devpub in devpubList:
            departmentDirRulePath = ruleFormat.format(TYPE=self.taskGroup, ASSETNAME=self.taskName, # ASSET
                                            SEQ=self.taskGroup, SHOT=self.taskName, # SHOT
                                            DEVPUB=devpub, CONTEXT=context, ARTIST=self.taskInfo['assigned']) # TRASH
            eval('self.ui.%sTreeWidget.clear()' % devpub)

            workingDirPath = os.path.join(departmentDirPath, departmentDirRulePath)
            if not os.path.exists(workingDirPath):
                print self.taskGroup, self.taskName, devpub, context, self.taskInfo['assigned']
                templateDirRulePath = templateRuleFormat.format(TYPE=self.taskGroup, ASSETNAME=self.taskName, # ASSET
                                            SEQ=self.taskGroup, SHOT=self.taskName, # SHOT
                                            DEVPUB=devpub, CONTEXT=context, ARTIST=self.taskInfo['assigned']) # TRASH
                print departmentDirPath, templateDirRulePath
                templateDirPath = os.path.join(departmentDirPath, templateDirRulePath)

                if not os.path.exists(os.path.dirname(templateDirPath)):
                    if not self.department in ['RIG', 'ANI', 'CRD']: # rig add
                        os.makedirs(os.path.dirname(templateDirPath))
                templateDir = os.path.join('/show', 'pipe', 'template', 'showFolderTemplate', 'works', department, self.entityType)
                if not templateDirPath: # rig add
                    os.system('cp -rf %s/* %s/' % (templateDir, os.path.dirname(templateDirPath)))

            if not os.path.exists(workingDirPath):
                if not self.department in ['RIG', 'ANI', 'CRD']: # rig add
                    os.makedirs(workingDirPath)

            for filename in sorted(os.listdir(workingDirPath)):
                if os.path.isdir(os.path.join(workingDirPath, filename)):
                    continue
                if filename.endswith('.json'):
                    continue
                if "~" in filename:
                    continue

                item = QtGui.QTreeWidgetItem()
                item.setText(0, filename)
                item.setText(1, 'unknown')
                if '.mb' in filename:
                    sceneLog = filename.replace('.mb', '.json')
                    if os.path.exists(os.path.join(workingDirPath, sceneLog)):
                        try:
                            with open(os.path.join(workingDirPath, sceneLog), 'r') as f:
                                sceneLogData = json.load(f)
                                if sceneLogData.has_key('artist'):
                                    item.setText(1, sceneLogData['artist'])
                        except:
                            pass
                item.dirPath = workingDirPath
                eval('self.ui.%sTreeWidget.addTopLevelItem(item)' % devpub)

            if devpub == 'dev':
                self.ui.devTreeWidget.dirPath = workingDirPath
            else:
                self.updateFileBrowser(workingDirPath)
                self.ui.pubTreeWidget.dirPath = workingDirPath
            # eval('self.ui.%sTreeWidget.workingDirPath = "%s"' % (devpub, workingDirPath))

    def closeEvent(self, QCloseEvent):
        print "called closeEvent"
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainVar = dxRunnerMain(None)
    mainVar.show()
    sys.exit(app.exec_())
