# -*- coding:utf-8 -*-
__author__ = 'gyeongheon.jeong'

import os
import json
import string
from collections import OrderedDict
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools
import maya.cmds as cmds
import CamTools.CamToolsFucntions as CamToolsFucntions
reload(CamToolsFucntions)
import aniCommon
reload(aniCommon)

currentpath = os.path.abspath(__file__)
UIROOT = os.path.dirname(currentpath)
UIFILE = os.path.join(os.path.dirname(currentpath), "DDani_CamToos.ui")

SHOW_PATH = "/show"
SHOTCAM_INFO = "{showPath}/{project}/works/MMV/asset/camera/cameraInfo.json"

LENS = ["12", "25", "35", "50", "75", "135"]

def hconv(text):
    return unicode(text, 'utf-8')


def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetObject':
            setattr(base_instance, member, getattr(ui, member))


class DDaniCamToosUI(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(DDaniCamToosUI, self).__init__(parent)

        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)

        setup_ui(ui, self)
        self.setWindowTitle("CamTools UI")
        self.initGUI()
        self.connectSignals()

    def connectSignals(self):
        self.project_comboBox.currentIndexChanged.connect(self.updateCamType)
        self.CamTool_tabWidget.currentChanged.connect(self.tabChangeCmd)
        self.camType_comboBox.currentIndexChanged.connect(self.updateProjectInfo)
        self.DDcamTool_autoUVButton.clicked.connect(self.AutoUV)
        self.DDcamTool_DoItButton.clicked.connect(self.doIt)
        self.DDcamTool_DoBakeButton.clicked.connect(self.DoBakeTextures)
        self.textureResComboBox.currentIndexChanged.connect(self.DoTextureRes)
        self.DDcamTool_AddImpButton.clicked.connect(self.addImpToList)
        self.DDcamTool_AddImpClearButton.clicked.connect(self.clearImpList)
        self.CreateNew_checkBox.stateChanged.connect(self.CamListEnable)
        self.DeleteFCNodes_Btn.clicked.connect(self.deleteFcNodes)
        self.AddLocator_Button.clicked.connect(self.addLocBtn)       # def addLocBtn
        self.BarShowPushButton.clicked.connect(self.openDirectory)
        self.BarShowComboBox.currentIndexChanged.connect(self.barShowChange)

    def openDirectory(self):
        showName = self.BarShowComboBox.currentText()
        barDefaultPath = "/show/%s/works/MMV/asset/camera" % showName
        os.system("nautilus %s" % barDefaultPath)

    def barShowChange(self):
        self.GHcreateImageplane_comboBox.clear()
        showName = self.BarShowComboBox.currentText()
        barDefaultPath = "/show/%s/works/MMV/asset/camera" % showName
        if not os.path.exists(barDefaultPath):
            return

        for imgFile in os.listdir(barDefaultPath):
            if imgFile.startswith('.') or not imgFile.endswith('.png'):
                continue

            imgFileFullPath = os.path.join(barDefaultPath, imgFile)
            self.GHcreateImageplane_comboBox.addItem(imgFileFullPath)

    def initGUI(self):
        ScenePath = cmds.file(q=1, sceneName=1)
        ShortSceneName = os.path.splitext(cmds.file(q=1, sceneName=1, shortName=1))[0]
        shotName = string.join(ShortSceneName.split("_")[:2], "_")
        shotPath = os.sep.join( ScenePath.split(os.sep)[:-2] )
        retimeDataPath = os.sep.join([shotPath, "data", "retime", ShortSceneName])
        camList = cmds.ls(type='camera')
        SstartFrame_ = cmds.playbackOptions(q=1,min=1)
        SendFrame_ = cmds.playbackOptions(q=1,max=1)
        showList = aniCommon.getListDirs(SHOW_PATH)
        self.project_comboBox.addItems(showList)
        self.BarShowComboBox.addItems(showList)
        self.project_comboBox.setCurrentIndex(cmds.optionVar(q='CurPrjNm'))
        self.BarShowComboBox.setCurrentIndex(cmds.optionVar(q='CurPrjNm'))
        self.updateCamType()
        self.prefix_lineEdit.setText(shotName)

        self.RefreshCamList_btn.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(UIROOT, 'resource/refreshBtn.png')))
        )
        self.AddLocator_Button.setIcon(
            QtGui.QIcon(QtGui.QPixmap(os.path.join(UIROOT, 'resource/locator.svg')))
        )
        self.DDcamTool_AddImplineEdit.setText(retimeDataPath)
        self.DDcamTool_CamList.addItems( camList )
        self.StartFrame_SpinBox.setValue( SstartFrame_ )
        self.EndFrame_SpinBox.setValue( SendFrame_ )
        self.barShowChange()
        self.DDcamTool_CamList.setEnabled(False)

    def updateData(self):
        shotCamInfoJson = SHOTCAM_INFO.format(showPath=SHOW_PATH, project=self.project)
        if not os.path.exists( shotCamInfoJson ):
            return False

        with open(shotCamInfoJson, 'r') as f:
            self.shotCamInfo = json.loads(f.read())
        '''
        map integer list to string
        '''
        #self.shotCamLens = map( str, self.shotCamInfo["project"][self.project]["lens_others"] )
        return True

    def updateCamType(self):
        self.project = str( self.project_comboBox.currentText())
        cmds.optionVar(iv=('CurPrjNm', self.project_comboBox.currentIndex()))
        infoQ = self.updateData()
        if not infoQ:
            self.projectInfo_textBrowser.clear()
            self.camType_comboBox.clear()
            return

        shotCamList = self.shotCamInfo["CAMERAS"].keys()
        self.camType_comboBox.clear()
        self.camType_comboBox.addItems(shotCamList)
        self.updateProjectInfo()

    def updateProjectInfo(self):
        camType = str(self.camType_comboBox.currentText())
        previewSize = self.shotCamInfo["CAMERAS"][camType]["preview_size"]
        self.projectInfo_textBrowser.clear()
        noteC = self.shotCamInfo["CAMERAS"][camType]["Note"]

        try:
            self.projectInfo_textBrowser.clearHistory()
            self.projectInfo_textBrowser.append("project         :  {}".format(self.project))
            self.projectInfo_textBrowser.append("preview size :  {} x {}".format(previewSize[0],
                                                                                previewSize[1]))
            self.projectInfo_textBrowser.append("Note : {}".format(noteC))
            self.lens_comboBox.clear()
            self.lens_comboBox.addItems(LENS)
            self.lens_comboBox.setCurrentIndex(2)
        except OSError:
            pass

    def baseTree(self, parent, nodeName, icoPath):

        self.NodeItem = QtGui.QTreeWidgetItem(parent)
        self.NodeItem.setText(0, nodeName)
        self.NodeItem.setIcon(0, QtGui.QIcon(os.path.join(UIROOT, icoPath)))
        self.NodeItem.setExpanded(True)

    def reloadShotList(self):
        sequence = str(self.seq_comboBox.currentText())
        try:
            shotPath = os.sep.join([self.seqPath, sequence])
            shotList = aniCommon.getListDirs(shotPath)
            shotList.sort()
            self.shot_comboBox.clear()
            self.shot_comboBox.addItem("select")
            self.shot_comboBox.addItems(shotList)
        except OSError:
            pass

    def CamListEnable(self):
        checkStt = self.CreateNew_checkBox.isChecked()

        if checkStt:
            self.followCam_Label.setEnabled(False)
            self.CamList_comboBox.setEnabled(False)
            self.RefreshCamList_btn.setEnabled(False)
        else:
            self.followCam_Label.setEnabled(True)
            self.CamList_comboBox.setEnabled(True)
            self.RefreshCamList_btn.setEnabled(True)

        self.AddFollowCamList()

    def AddFollowCamList(self):
        #FC_list = cmds.ls("GH_followCAM*", type="transform")
        FC_list = list()
        camList = cmds.listCameras()

        for cam in camList:
            if cam.find("GH_followCAM") > -1:
                FC_list.append(cam)
        self.CamList_comboBox.clear()
        self.CamList_comboBox.addItems(FC_list)

    def tabChangeCmd(self):
        tabIndex = self.CamTool_tabWidget.currentIndex()

        if tabIndex == 1 or tabIndex == 3 or tabIndex == 5:
            self.DDcamTool_CamList.setEnabled(True)
        else:
            self.DDcamTool_CamList.setEnabled(False)

    def doCreateShotCam(self):
        self.shotName = str(self.prefix_lineEdit.text())
        camType = str(self.camType_comboBox.currentText())
        self.isRenderCam = self.RenderCam_checkBox.isChecked()
        f_lens = self.lens_comboBox.currentText()

        CamToolsFucntions.createShotCam(shotName=self.shotName, camInfoJson=self.shotCamInfo, camtype=camType, flens=f_lens, isRenderCam=self.isRenderCam)

    def DoProjections(self):
        selObj = cmds.ls(sl=1)
        camName=None

        try:
            camName = str(self.DDcamTool_CamList.currentItem().text())
        except:
            cmds.error("select camera first")

        if len(selObj) == 0:
            cmds.error("select object first")
        if camName:
            CamToolsFucntions.doCamProjection(selObj, camName)

    def DoTextureRes(self):
        objectName = cmds.ls(sl=1)
        resolution = int(self.textureResComboBox.currentText())

        if objectName:
            CamToolsFucntions.DoTextureResolution(objectName, resolution)

    def DoBakeTextures(self):
        SelMaterial = cmds.ls(sl=1)
        BakeRes = int(self.DDcamTool_baketexRes.text())
        CamToolsFucntions.DoBakeTexture(SelMaterial, BakeRes)

    def AutoUV(self):
        SelMaterial = cmds.ls(sl=1)[0]
        FaceNum = cmds.polyEvaluate(f=True) - 1
        cmds.selectMode(component=True)

        cmds.select("%s.f[0:%d]" % (SelMaterial, FaceNum), r=True )
        cmds.polyAutoProjection(
            "%s.f[0:%d]" % (SelMaterial, FaceNum),
            lm=0, pb=0, ibd=1, cm=0, l=2,
            sc=1, o=1, p=6, ps=0.2, ws=0
        )

        cmds.selectMode(object = True)


    def DoRetime(self):
        startFrame_ =  self.StartFrame_SpinBox.value()
        endFrame = self.EndFrame_SpinBox.value()
        TimeScale_ = 24.0 / self.FPS_SpinBox.value()
        camName = str(self.DDcamTool_CamList.currentItem().text())

        CamToolsFucntions.DoRetime(startFrame_, endFrame, TimeScale_, camName)


    def addImpToList(self):
        sel = cmds.ls(sl=1)
        impList = []
        for que in sel:
            shapeName = cmds.listRelatives(que, s=1)[0]
            objType = cmds.objectType(shapeName)

            if objType == "camera":
                imp = cmds.listConnections(shapeName + ".imagePlane[0]", type="imagePlane")
                impShape = cmds.listRelatives(imp, s=1)[0]
                impList.append(impShape)
            else:
                impList.append( cmds.listRelatives(que, s=1)[0] )

        self.DDcamTool_AddImplistWidget.addItems(impList)



    def clearImpList(self):
        self.DDcamTool_AddImplistWidget.clear()


    def doExportTimeWarp(self):
        retimeDataPath = str(self.DDcamTool_AddImplineEdit.text())
        items = []
        for index in xrange( self.DDcamTool_AddImplistWidget.count() ):
            items.append( self.DDcamTool_AddImplistWidget.item(index) )

        ip = [ str(i.text()) for i in items ]

        if not os.path.exists(retimeDataPath):
            os.makedirs(retimeDataPath)

        aniCommon.exportRetime2Nuke(ip, retimeDataPath)


    def DoCreateImageBar(self):
        imagePlaneName = str(self.GHcreateImageplane_comboBox.currentText())
        print imagePlaneName

        try:
            camName = str(self.DDcamTool_CamList.currentItem().text())
        except:
            cmds.error("select camera first")

        CamToolsFucntions.DoCreateImageplane(imagePlaneName, camName)


    def doCreateFollowCam(self):
        selCon = cmds.ls(sl=1)
        Qtx = self.Qtx_checkBox.isChecked()
        Qty = self.Qty_checkBox.isChecked()
        Qtz = self.Qtz_checkBox.isChecked()
        NewCamQ = self.CreateNew_checkBox.isChecked()

        print Qtx, Qty, Qtz

        if not selCon:
            QtGui.QMessageBox.warning(self, "Warning!", hconv("오브젝트를 선택해 주세요"))
            return
        if NewCamQ:
            for selCon_I in selCon:
                CamToolsFucntions.createFollowCam(selCon_I, [Qtx, Qty, Qtz], NewCamQ )
        else:
            if len(selCon) > 1:
                QtGui.QMessageBox.warning(self, "Warning!", hconv("하나만 선택하세요"))
            else:
                SelCam = str( self.CamList_comboBox.currentText() )
                CamToolsFucntions.createFollowCam( selCon[0], [Qtx, Qty, Qtz], NewCamQ, SelCam )


    def deleteFcNodes(self):
        FC_Nodes = cmds.ls("*_GHFC_*")
        FC_Cam = cmds.ls("GH_followCAM_GRP*")
        muteNodes = cmds.ls("mute_GH_followCAM_GRP*_t*")

        allNodes = FC_Nodes + FC_Cam + muteNodes

        if allNodes:
            cmds.delete(allNodes)

    def addLocBtn(self):

        selectN = str(cmds.ls(sl=1)[0])
        if selectN[-3:].count("GRP") == 1:
            selectNc = str(cmds.listRelatives(selectN, c=True)[0])
        else:
            selectNc = str(cmds.listRelatives(selectN, c=True)[1])

        selectNl = str(cmds.spaceLocator(p=(0,0,0))[0])
        if len(selectNl) == 8:
            comN = int(selectNl[-1])
        elif len(selectNl) == 9:
            comN = int(selectNl[-2:])
        else:
            comN = int(selectNl[-3:])

        addLocList = []

        if len(cmds.ls("*_ani_renderCam_LOC*", type="transform")) != 0:

            for i in cmds.ls("*_ani_renderCam_LOC*", type="transform"):
                addLocList.append(i[(i.index("_LOC") + 4):])

            while addLocList.count(str(comN)) == True:
                comN += 1
            addedLocName = self.shotName + "_ani_renderCam_LOC" + str(comN)

        else:
            addedLocName = self.shotName + "_ani_renderCam_LOC" + selectNl[-1]

        cmds.rename(selectNl, addedLocName)
        selectNl = addedLocName

        cmds.parent(selectNl, selectN)

        attS = [".tx", ".ty", ".tz", ".rx", ".ry", ".rz"]

        for i in attS:
            cmds.setAttr(selectNl + i, 0)

        cmds.parent(selectNc, selectNl)
        cmds.select(selectNl)

    def doIt(self):
        if self.CamTool_tabWidget.currentIndex() == 0:
            self.doCreateShotCam()
        elif self.CamTool_tabWidget.currentIndex() == 1:
            self.DoProjections()
        elif self.CamTool_tabWidget.currentIndex() == 2:
            pass
        elif self.CamTool_tabWidget.currentIndex() == 3:
            self.DoRetime()
        elif self.CamTool_tabWidget.currentIndex() == 4:
            self.doExportTimeWarp()
        elif self.CamTool_tabWidget.currentIndex() == 5:
            self.DoCreateImageBar()
        elif self.CamTool_tabWidget.currentIndex() == 6:
            self.doCreateFollowCam()

def runUI():
    global win
    try:
        win.close()
    except:
        pass
    win = DDaniCamToosUI()

    win.show()
    win.resize(500, 700)
