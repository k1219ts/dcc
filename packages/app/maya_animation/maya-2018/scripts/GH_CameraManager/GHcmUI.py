
try:
    from PySide import QtGui, QtCore
    import pysideuic
    import shiboken
    import xml.etree.ElementTree as xml
    from cStringIO import StringIO
    from PySide.phonon import Phonon
except:
    import sip
    from PyQt4 import QtGui, QtCore
    from PyQt4 import uic

import sys
import os
import json
import string
import cameraPresetUI
import getpass

currentpath = os.path.abspath(__file__)
UIROOT = os.path.join(os.path.dirname(currentpath), 'ui')
currentDir = os.path.dirname(currentpath)
UIFILE = os.path.join(UIROOT, "mainWindow.ui")
#css = open(os.path.join(UIROOT, 'studioLibrary.css'), 'r').read()
#rcss = css.replace("RESOURCE_DIRNAME", currentDir)
#rcss = rcss.replace("BACKGROUND_COLOR", "rgb(40,40,40)")
#rcss = rcss.replace("ACCENT_COLOR", "rgb(255,90,40)")
rcss = open(os.path.join(UIROOT, 'darkorange.css'), 'r').read()

SHOW_PATH = "/show"
BASEPATH = "/show/{project}/works/MMV/asset/camera"
SHOTCAM_INFO = os.path.join(BASEPATH, "cameraInfo.json")
PRESET_JSON = "/stdrepo/MMV/asset/camera/cameraInfo_preset.json"
CAM_TYPES_JSON = "/stdrepo/MMV/asset/camera/cameraType_preset.json"
LETTERBOX_PREFIX = "letterbox"

#USER = getpass.getuser()
#master = [ "dongho.cha", "gyeongheon.jeong" ]

def loadUiType(uiFile):
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text

    with open(uiFile, 'r') as f:
        o = StringIO()
        frame = {}

        pysideuic.compileUi(f, o, indent=0)
        pyc = compile(o.getvalue(), '<string>', 'exec')
        exec pyc in frame

        form_class = frame['Ui_%s' %form_class]
        base_class = eval('QtGui.%s' % widget_class)

    return form_class, base_class

try:
    formclass, baseclass = uic.loadUiType(UIFILE)
except:
    formclass, baseclass = loadUiType(UIFILE)

class camManagerMain(formclass, baseclass):
    projectInfo = dict()

    def __init__(self, parent=None):
        super(camManagerMain, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Camera Info Manager For MMV")
        self.move(QtCore.QPoint(1000/2, 200))
        self.setStyleSheet(rcss)
        self.infoPreview_textBrowser.setLineWrapMode(QtGui.QTextBrowser.NoWrap)
        self.splitter.setStretchFactor(1, 10)

#        if USER not in master:
#            self.editTypes_pushButton.setEnabled(False)
        self.aspectRatio_lineEdit.setReadOnly(True)
        self.removeType_toolButton.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(currentDir, 'icons/delete.png'))))
        self.addCamera_toolButton.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(currentDir, 'icons/add.png'))))
        self.clearPreview_toolButton.setIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(currentDir, 'icons/cache.png'))))
        self.connectSignals()

        self.refreshPreset()
        self.project_comboBox.addItems(self.getDirList(SHOW_PATH))
        self.type_comboBox.addItems(self.getJsonData(CAM_TYPES_JSON)["TYPES"])

    def connectSignals(self):
        self.presetList_comboBox.currentIndexChanged.connect(self.presetChanged)
        self.project_comboBox.currentIndexChanged.connect(self.getProjectInfo)
        #self.type_comboBox.currentIndexChanged.connect(self.typeChanged)
        self.editTypes_pushButton.clicked.connect(self.editCamTypes)
        self.resGateW_lineEdit.textChanged.connect(self.resGateEditEvent)
        self.resGateH_lineEdit.textChanged.connect(self.resGateEditEvent)
        self.removeType_toolButton.clicked.connect(self.removeSelectedCamType)
        self.filmBackmmW_lineEdit.textChanged.connect(self.filmBackEditEvent)
        self.filmBackmmH_lineEdit.textChanged.connect(self.filmBackEditEvent)
        self.savePreset_pushButton.clicked.connect(self.savePreset)
        self.removePreset_pushButton.clicked.connect(self.removePreset)
        #self.letterBoxBrowse_pushButton.clicked.connect(self.browseFile)
        self.addCamera_toolButton.clicked.connect(self.addNewCam)
        self.clearPreview_toolButton.clicked.connect(self.clearPreview)
        self.saveInfo_pushButton.clicked.connect(self.saveInfo)

    def refreshPreset(self):
        self.presetList_comboBox.clear()
        self.presetList_comboBox.addItem("----------- select ------------")
        self.presetList_comboBox.addItems(self.getJsonData(PRESET_JSON).keys())

    def presetChanged(self):
        if self.presetList_comboBox.currentIndex() != 0:
            camName = str(self.presetList_comboBox.currentText())
            presetInfo = self.getJsonData(PRESET_JSON)

            if not camName:
                return

            lensVal = presetInfo[camName]["lens"]
            filmBackMM = presetInfo[camName]["filmBack_mm"]
            originalRes = presetInfo[camName]["Original_Resolution"]
            resolutionGate = presetInfo[camName]["resolution_gate"]
            previewSize = presetInfo[camName]["preview_size"]

            self.cameraName_lineEdit.setText(camName)
            self.lens_lineEdit.setText( map(str, lensVal)[0] )
            self.filmBackmmW_lineEdit.setText( map(str, filmBackMM)[0] )
            self.resGateW_lineEdit.setText( map(str, resolutionGate)[0] )
            self.prvSizeW_lineEdit.setText( map(str, previewSize)[0] )
            self.prvSizeH_lineEdit.setText(map(str, previewSize)[1])
            self.plateSizeW_lineEdit.setText( map(str, originalRes)[0] )
            self.plateSizeH_lineEdit.setText( map(str, originalRes)[1] )
            self.comment_textEdit.setText(presetInfo[camName]["Note"])
            try:
                self.filmBackmmH_lineEdit.setText( map(str, filmBackMM)[1] )
                self.resGateH_lineEdit.setText( map(str, resolutionGate)[1] )
            except:
                pass

    def getDirList(self, path, isFile=False, fileType=None):
        finalList = []
        if not os.path.exists(path):
            return finalList

        fileList = os.listdir(path)

        for F in fileList:
            if not isFile:
                tmp = path + os.sep + F
                if os.path.isdir(tmp) and '.' not in F and '_' not in F:
                    finalList.append(F)
            else:
                fileName =path + os.sep + F
                if os.path.isfile(fileName) and fileName.endswith(fileType):
                    finalList.append(F)

        return sorted(finalList)

    def getProjectInfo(self):
        self.resGateW_lineEdit.clear()
        self.resGateH_lineEdit.clear()
        self.aspectRatioRG_lineEdit.clear()
        self.prvSizeW_lineEdit.clear()
        self.prvSizeH_lineEdit.clear()
        self.letterBox_comboBox.clear()
        
        project = self.project_comboBox.currentText()
        infoJson = SHOTCAM_INFO.format(project=project)
        self.projectInfo = self.getJsonData(infoJson)

#        if self.projectInfo:
#            previewsize = map( str, self.projectInfo["PREVIEW_SIZE"])
#            resgatesize = map( str, self.projectInfo["RESOLUTION_GATE"])
#            self.prvSizeW_lineEdit.setText(previewsize[0])
#            self.prvSizeH_lineEdit.setText(previewsize[1])
#            self.resGateW_lineEdit.setText(resgatesize[0])
#            self.resGateH_lineEdit.setText(resgatesize[1])

        letterBox = self.getDirList(BASEPATH.format(project=project),
                                    isFile=True,
                                    fileType="png")
        for lb in letterBox:
            if lb.startswith(LETTERBOX_PREFIX):
                self.letterBox_comboBox.addItem(lb)
        self.appendToPreview(self.projectInfo)

    def editCamTypes(self):
        #cType = self.type_comboBox.currentText()
        allTypes = self.getJsonData(CAM_TYPES_JSON)["TYPES"]

        cpUI = cameraPresetUI.camPresetWindow(parent=self, jsonfile=CAM_TYPES_JSON)
        cpUI.show()
        cpUI.move(QtCore.QPoint(1400/2, 400))

        cpUI.typeListWidget.addItems(allTypes)

    def removeSelectedCamType(self):
        currentType = str(self.type_comboBox.currentText())

        if self.projectInfo:
            if self.projectInfo["CAMERAS"].has_key(currentType):
                self.projectInfo["CAMERAS"].pop(currentType)

        self.appendToPreview(self.projectInfo)

    def resGateEditEvent(self):
        resGateWStr = self.resGateW_lineEdit.text()
        resGateHStr = self.resGateH_lineEdit.text()

        if resGateHStr and resGateWStr:
            aspectRatio = round(float(resGateWStr) / float(resGateHStr), 4)
            self.aspectRatioRG_lineEdit.setText("{:.4f} : 1".format(aspectRatio))

    def filmBackEditEvent(self):
        mmWStr = self.filmBackmmW_lineEdit.text()
        mmHStr = self.filmBackmmH_lineEdit.text()
        if mmWStr:
            mmW = round(float(mmWStr), 6)
            self.filmBackInchW_lineEdit.setText("{:.6f}".format(mmW/25.4))
        if mmHStr:
            mmH = round(float(mmHStr), 6)
            self.filmBackInchH_lineEdit.setText("{:.6f}".format(mmH/25.4))
        if mmWStr and mmHStr:
            aspectRatio = round(float(mmWStr)/float(mmHStr), 4)
            self.aspectRatio_lineEdit.setText("{:.4f} : 1".format(aspectRatio))

    def resetTextLines(self):
        self.cameraName_lineEdit.clear()
        self.lens_lineEdit.clear()
        self.filmBackmmW_lineEdit.clear()
        self.filmBackmmH_lineEdit.clear()
        self.filmBackInchW_lineEdit.clear()
        self.filmBackInchH_lineEdit.clear()
        self.plateSizeW_lineEdit.clear()
        self.plateSizeH_lineEdit.clear()
        self.prvSizeW_lineEdit.clear()
        self.prvSizeH_lineEdit.clear()
        self.letterBox_comboBox.clear()

    def getCamDataFromUI(self):
        self.project = str(self.project_comboBox.currentText())
        self.cameraType = str(self.type_comboBox.currentText())
        self.cameraName = str(self.cameraName_lineEdit.text())
        self.filmBackMM = [float(self.filmBackmmW_lineEdit.text()),
                           float(self.filmBackmmH_lineEdit.text())]
        self.filmBackInch = [float(self.filmBackInchW_lineEdit.text()),
                             float(self.filmBackInchH_lineEdit.text())]
        self.previewSize = [int(self.prvSizeW_lineEdit.text()),
                            int(self.prvSizeH_lineEdit.text())]
        self.plateSize = [int(self.plateSizeW_lineEdit.text()),
                          int(self.plateSizeH_lineEdit.text())]
        self.resGate = [int(self.resGateW_lineEdit.text()),
                        int(self.resGateH_lineEdit.text())]

        lensListStr = str(self.lens_lineEdit.text())
        if lensListStr:
            if lensListStr.find(",") != -1:
                lensListNoWhiteSpace = lensListStr.translate(None, string.whitespace).strip(",")
                self.lensList = map(int, lensListNoWhiteSpace.split(","))
            else:
                self.lensList = map(int, lensListStr.split(" "))
        else:
            self.lensList = None

        self.letterBoxPath = os.sep.join([BASEPATH.format(project=self.project),
                                          str(self.letterBox_comboBox.currentText())])

        self.comments = str(self.comment_textEdit.toPlainText())

    def getJsonData(self, jsonFile):
        if os.path.exists(jsonFile):
            with open(jsonFile, 'r') as f:
                presetData = json.load(f)
        else:
            presetData = dict()

        return presetData

    def savePreset(self):
        presetInfo = self.getJsonData(PRESET_JSON)

        self.getCamDataFromUI()

        cameraInfo = dict()
        cameraInfo["filmBack_inch"] = self.filmBackInch
        cameraInfo["filmBack_mm"] = self.filmBackMM
        cameraInfo["Original_Resolution"] = self.plateSize
        cameraInfo["lens"] = self.lensList
        cameraInfo["resolution_gate"] = self.resGate
        cameraInfo["preview_size"] = self.previewSize
        cameraInfo["Note"] = self.comments

        presetInfo[self.cameraName] = cameraInfo

        #self.revData(PRESET_JSON)
        self.writeJson(presetInfo, PRESET_JSON)
        self.refreshPreset()

    def removePreset(self):
        presetInfo = self.getJsonData(PRESET_JSON)
        currentPreset = self.presetList_comboBox.currentText()

        if __name__ == '__main__':
            currentPresetConv = str(currentPreset.toUtf8())
        else:
            currentPresetConv = currentPreset

        presetInfo.pop(currentPresetConv)
        #self.revData(PRESET_JSON)
        self.writeJson(presetInfo, PRESET_JSON)

        self.refreshPreset()

    def camInfoUpdate(self):
        self.getCamDataFromUI()
        cameraInfo = dict()
        cameraInfo["name"] = self.cameraName
        cameraInfo["filmBack_inch"] = self.filmBackInch
        cameraInfo["filmBack_mm"] = self.filmBackMM
        cameraInfo["Original_Resolution"] = self.plateSize
        cameraInfo["lens"] = self.lensList
        cameraInfo["resolution_gate"] = self.resGate
        cameraInfo["preview_size"] = self.previewSize
        cameraInfo["letter_box"] = self.letterBoxPath
        cameraInfo["Note"] = self.comments

        if not self.projectInfo:
            self.projectInfo = dict()
            self.projectInfo["PROJECT"] = self.project
            self.projectInfo["CAMERAS"] = dict()
            #self.projectInfo["RESOLUTION_GATE"] = self.resGate
            #self.projectInfo["PREVIEW_SIZE"] = self.previewSize
            #self.projectInfo["LETTER_BOX"] = self.letterBoxPath
        self.projectInfo["CAMERAS"][self.cameraType] = cameraInfo

    def browseFile(self):
        startPath = ''
        letterBoxFile = QtGui.QFileDialog.getOpenFileName(self, "Select LetterBox Imagefile",
                                                          startPath, "png (*.*)")
        try:
            fileName = str(letterBoxFile.toUtf8())
        except:
            fileName = letterBoxFile[0]

        if not fileName:
            return
        else:
            self.letterBoxPath_lineEdit.setText(fileName)

    def appendToPreview(self, projectInfo):
        info = json.dumps(projectInfo, indent=4)
        info = info.replace("\\n", "\n\t")
        self.infoPreview_textBrowser.clear()
        self.infoPreview_textBrowser.append(info)

    def refreshPreview(self):
        self.infoPreview_textBrowser.clear()

    def addNewCam(self):
        self.camInfoUpdate()
        self.appendToPreview(self.projectInfo)

    def writeJson(self, camInfo, outJson):
        with open(outJson, 'w') as f:
            json.dump(camInfo, f, indent=4)
            f.close()

    def saveInfo(self):
        #self.camInfoUpdate()
        self.getCamDataFromUI()
        jsonFile = SHOTCAM_INFO.format(project=self.project)
        ##self.revData(jsonFile)
        self.writeJson(self.projectInfo, jsonFile)

    def clearPreview(self):
        self.projectInfo = dict()
        self.infoPreview_textBrowser.clear()

    def revData(self, filePath=None):
        addnum = 1
        revPath = filePath

        while os.path.exists(revPath):
            if os.path.isfile(revPath):
                revPath = os.path.splitext(filePath)[0]
                revPath += "_rev_{:02d}".format(addnum)
                revPath += os.path.splitext(filePath)[1]
            else:
                revPath = string.join(filePath.split("_")[:-1], "_")
                revPath += "_rev_{:02d}".format(addnum)
            addnum += 1

        if not filePath == revPath:
            os.rename(filePath, revPath)

        return revPath

def showUI():
    global app
    # Use a shared instance of QApplication
    import maya.OpenMayaUI as mui
    app = QtGui.QApplication.instance()
    # Get a pointer to the maya main window
    ptr = mui.MQtUtil.mainWindow()
    # Use sip to wrap the pointer into a QObject
    try:
        win = shiboken.wrapInstance(long(ptr), QtGui.QWidget)
    except:
        win = sip.wrapinstance(long(ptr), QtCore.QObject)

    form = camManagerMain(win)
    try:
        form.close()
    except:
        pass

    form.show()
    form.resize(1500, 600)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setApplicationName("GHCameraManager")
    win = camManagerMain()
    win.resize(1500, 600)
    win.show()
    sys.exit(app.exec_())
