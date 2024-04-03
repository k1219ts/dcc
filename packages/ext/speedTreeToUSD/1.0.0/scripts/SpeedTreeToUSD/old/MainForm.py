# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#
#   kwantae.Kim
#
#	2019.11.
#
#-------------------------------------------------------------------------------

import os
import subprocess
import getpass
import pprint
import glob

# QT
from PySide2 import QtWidgets, QtCore, QtGui

from SpeedTreeToUSD.speedTreeToUSDUI import Ui_Form
import SpeedTreeToUSD.treeCommon as treeCommon

# ScriptRoot = os.path.dirname(os.path.abspath(__file__))
#
# COLORSTYLE={
#     'red'   :'color: rgb(250, 30, 50);',
#     'green' :'color: rgb(50, 250, 30);',
#     'gray'  :'color: rgb(120, 120, 120);',
#     'orange':'color: rgb(250, 150, 30);',
#     'white' :'color: white;'
# }
#
# MSGRESULT  = 0
# MSGWARNING = 1
# MSGERROR   = 2

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        # QtWidgets.QWidget.__init__(self, parent)
        # self.ui = Ui_Form()
        # self.ui.setupUi(self)

        self.showName  = None
        self.assetName = None
        self.isclip    = False
        self.abcFps    = None
        self.abcPath   = None
        self.jsonPath  = None
        self.clipRange = []
        self.astpath   = None
        self.outVer    = None
        self.texVer    = None
        self.newVers   = {'out':None, 'tex':None}
        self.verpaths  = {'out':None, 'tex':None}
        self.outpaths  = {'out':None, 'tex':None}

        # ui
        self.assetCompleter  = None
        self.versCompleter   = {'out':None, 'tex':None}
        self.versUIs = {
            'out':self.ui.ver_lineEdit,
            'tex':self.ui.texVer_lineEdit
        }

        # self.styleSetting()
        # self.connections()
        # self.defaultSetting()
        #
        # # # GET TASK LIST #
        # self.setShowList()
        # # self.update_show_comboBox()
        #
        # self.tw_sceneGraph = self.ui.sceneGraph_treeWidget
        # self.tw_materialSet = self.ui.materialSet_treeWidget
        # self.tw_textureGroup = self.ui.textureGroup_treeWidget
        #
        # # default materialSet
        # treeCommon.MaterialSetWidgetItem(self.tw_materialSet, self.tw_sceneGraph, 'wood')
        # treeCommon.MaterialSetWidgetItem(self.tw_materialSet, self.tw_sceneGraph, 'leaf')

    '''
    # def closeEvent(self, evnt):
    #     self.assetCompleter.deleteLater()
    #     self.assetCompleter = None
    #
    #     for k in self.versCompleter.keys():
    #         if self.versCompleter[k]:
    #             self.versCompleter[k].deleteLater()
    #             self.versCompleter[k] = None
    #
    #     super(MainForm, self).closeEvent(evnt)
    '''

    # --------------------------------------------------------------------------
    # UI INITIALIZE
    # --------------------------------------------------------------------------
    # def styleSetting(self):
    #     # main logo title image
    #     # imagePath = '%s/ui/pxr_usd_w.png'%ScriptRoot
    #     # image = QtGui.QPixmap(imagePath).scaled(30, 30, QtCore.Qt.KeepAspectRatioByExpanding, QtCore.Qt.SmoothTransformation)
    #     # self.ui.titleLogo_label.setPixmap(image)
    #
    #     # open dir button image
    #     imagePath_folder = '%s/ui/folder.png'%ScriptRoot
    #     self.ui.abcFile_openDir_pushButton.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))
    #     self.ui.xmlFile_openDir_pushButton.setIcon(QtGui.QIcon(QtGui.QPixmap(imagePath_folder)))
    #
    #     # machineType comboBox style
    #     style = '''
    #         QComboBox QAbstractItemView::item { min-height: 30px; min-width: 120px;}
    #         background-color: rgb(90,90,90);
    #         color : white;
    #     '''
    #
    #     self.ui.machineType_comboBox.setView(QtWidgets.QListView())
    #     self.ui.machineType_comboBox.setStyleSheet(style)
    #
    #     # font to white style
    #     self.ui.show_comboBox.setStyleSheet(COLORSTYLE['white'])
    #     self.ui.ver_lineEdit.setStyleSheet(COLORSTYLE['white'])
    #     self.ui.texVer_lineEdit.setStyleSheet(COLORSTYLE['white'])
    #     self.ui.pivot_comboBox.setStyleSheet(COLORSTYLE['white'])
    #     self.ui.asset_lineEdit.setStyleSheet(COLORSTYLE['white'])

    def connections(self):
        # clip
        self.ui.clip_checkBox.clicked.connect(self.clip_activeUI)

        # OUTPUT
        self.ui.show_comboBox.activated.connect(self.update_show_comboBox)
        #self.ui.asset_lineEdit.cursorPositionChanged.connect(self.updateTextEdited_asset_lineEdit)
        self.ui.asset_lineEdit.editingFinished.connect(self.update_asset_lineEdit)
        self.ui.ver_lineEdit.cursorPositionChanged.connect(self.update_ver_lineEdit)
        self.ui.texVer_lineEdit.cursorPositionChanged.connect(self.update_texVer_lineEdit)
        self.ui.clip_checkBox.stateChanged.connect(self.update_show_comboBox)

        # INPUT FILES
        self.ui.abcFile_find_pushButton.clicked.connect(self.openDialog_selectAbcFile)
        self.ui.xmlFile_find_pushButton.clicked.connect(self.openDialog_selectXmlFile)
        self.ui.abcFile_openDir_pushButton.clicked.connect(self.openDirectory)
        self.ui.xmlFile_openDir_pushButton.clicked.connect(self.openDirectory)
        self.ui.abcFile_lineEdit.textChanged.connect(self.update_allGraphs)
        self.ui.xmlFile_lineEdit.textChanged.connect(self.update_allGraphs)

        self.ui.materialSet_add_pushButton.clicked.connect(self.addMaterialSet)
        self.ui.materialSet_treeWidget.clicked.connect(self.updateMaterialSet)
        self.ui.export_pushButton.clicked.connect(self.export)
        self.ui.textureGroup_add_pushButton.clicked.connect(self.addTextureGroup)

    def defaultSetting(self):
        # self.ui.user_name_label.setText(getpass.getuser())
        #
        # self.ui.frame_start_label.setText('0')
        # self.ui.frame_end_label.setText('0')
        #
        # self.ui.export_start_lineEdit.setText('0')
        # self.ui.export_end_lineEdit.setText('0')
        #
        # # self.ui.proxy_checkBox.setCheckState(QtCore.Qt.Checked)
        # self.ui.loop_checkBox.setCheckState(QtCore.Qt.Checked)
        # self.ui.clip_checkBox.setCheckState(QtCore.Qt.Unchecked)
        #
        # self.clip_activeUI()
        self.update_allGraphs()


    # def clip_activeUI(self):
    #     enable = self.ui.clip_checkBox.checkState() == QtCore.Qt.Checked
    #     style  = None
    #
    #     self.setEnabled(self.ui.clip_checkBox, enable, uiEnabled=False)
    #
    #     self.setEnabled(self.ui.frame_label, enable)
    #     self.setEnabled(self.ui.frame_start_label, enable)
    #     self.setEnabled(self.ui.frame_end_label, enable)
    #
    #     self.setEnabled(self.ui.export_label, enable)
    #     self.setEnabled(self.ui.export_start_lineEdit, enable)
    #     self.setEnabled(self.ui.export_end_lineEdit, enable)
    #     self.setEnabled(self.ui.loop_checkBox, enable)
    #
    #     self.setEnabled(self.ui.fps_label, enable)
    #     self.setEnabled(self.ui.fps_lineEdit, enable)
    #
    #     self.setEnabled(self.ui.speed_m0_8_checkBox, enable)
    #     self.setEnabled(self.ui.speed_m1_0_checkBox, enable)
    #     self.setEnabled(self.ui.speed_m1_5_checkBox, enable)


    # def setEnabled(self, ui, enable, uiEnabled=True):
    #     if enable:
    #         style = COLORSTYLE['white']
    #     else:
    #         style = COLORSTYLE['gray']
    #
    #     ui.setStyleSheet(style)
    #
    #     if uiEnabled:
    #         ui.setEnabled(enable)


    # def message(self, text='', type=MSGRESULT):
    #     style = COLORSTYLE['green']
    #
    #     if text:
    #         if type == MSGERROR:
    #             style = COLORSTYLE['red']
    #             text = 'ERROR : %s'%text
    #         elif type == MSGWARNING:
    #             style = COLORSTYLE['yellow']
    #             text = 'WARNING : %s'%text
    #         else:
    #             text = 'RESULT : %s'%text
    #
    #     self.ui.message_label.setStyleSheet(style)
    #     self.ui.message_label.setText(text)


    # --------------------------------------------------------------------------
    # OUTPUT UI FUNCTIONS
    # --------------------------------------------------------------------------

    def GetShowDir(self, show):
        _showDir= '/show/%s' % show
        showDir = _showDir
        if showDir.find('_pub') == -1:
            showDir += '_pub'

        pathRuleFile = '{SHOW}/_config/maya/pathRule.json'.format(SHOW=_showDir)
        if os.path.exists(pathRuleFile):
            showDir = _showDir
            try:
                ruleData = json.load(open(pathRuleFile))
                if ruleData.has_key('showDir') and ruleData['showDir']:
                    __showDir = ruleData['showDir']
                    if os.path.isabs(__showDir):
                        showDir = __showDir
                    else:
                        showDir = os.path.join('/show', __showDir)
            except:
                pass
        return showDir

    def setShowList(self):
        titleList =[]
        self.ui.show_comboBox.clear()
        for i in os.listdir('/show'):
            if not i.startswith('.') and not "_pub" in i:
                titleList.append(i.upper())

        self.ui.show_comboBox.addItems(sorted(titleList))


    def setOutputInfo(self):
        self.showName  = self.ui.show_comboBox.currentText().lower()
        self.assetName = self.ui.asset_lineEdit.text()
        self.isclip    = self.ui.clip_checkBox.checkState() == QtCore.Qt.Checked
        self.abcFps    = float(self.ui.fps_lineEdit.text())
        self.abcPath   = self.ui.abcFile_lineEdit.text()
        self.jsonPath  = self.abcPath.replace('abc','json')
        self.outVer    = self.ui.ver_lineEdit.text()
        self.texVer    = self.ui.texVer_lineEdit.text()
        self.clipRange = [self.ui.export_start_lineEdit.text(),
                          self.ui.export_end_lineEdit.text()]

        self.toppath = '{SHOWDIR}/asset'.format(
            SHOWDIR=self.GetShowDir(self.showName)
        )

        self.astpath = '{TOPPATH}/{ASSET}'.format(
            TOPPATH=self.toppath,
            ASSET=self.assetName
        )

        self.verpaths['out'] = '{ASTPATH}/{CATEGORY}'.format(
            ASTPATH=self.astpath,
            CATEGORY='clip' if self.isclip else 'model'
        )

        self.verpaths['tex'] = '{ASTPATH}/texture/tex'.format(
            ASTPATH=self.astpath
        )

        self.outpaths['out'] = '{VERPATH}/v{VER}'.format(
            VERPATH=self.verpaths['out'],
            VER=self.outVer
        )

        self.outpaths['tex'] = '{VERPATH}/v{VER}'.format(
            VERPATH=self.verpaths['tex'],
            VER=self.texVer
        )


    def setAssetCompleter(self, clear=False):
        if not self.assetCompleter:
            self.assetCompleter = QtWidgets.QCompleter(self)

        model = QtCore.QStringListModel()
        # get asset directories
        assets = []
        if not clear:
            assets = sorted([d for d in os.listdir(self.toppath)
                               if os.path.isdir('%s/%s'%(self.toppath, d))])

        model.setStringList(assets)
        self.assetCompleter.setModel(model)
        self.ui.asset_lineEdit.setCompleter(self.assetCompleter)


    def setVersCompleter(self, clear=False):
        for k in ['out', 'tex']:
            if not self.versCompleter[k]:
                self.versCompleter[k] = QtWidgets.QCompleter(self)

            model = QtCore.QStringListModel()

            # get versions
            vers = []
            if not clear:
                vers = sorted([d for d in glob.glob('%s/v*'%self.verpaths[k])
                                 if os.path.isdir(d)])
                vers = [os.path.basename(d)[1:] for d in vers]

                if not vers:
                    vers.append('001')
                else:
                    vers.append('%03d'%(int(vers[-1]) + 1))

                self.newVers[k] = vers[-1]

            model.setStringList(vers)
            self.versCompleter[k].setModel(model)

            self.versUIs[k].setCompleter(self.versCompleter[k])

            if self.ui.asset_lineEdit.text():
                self.versUIs[k].setText(self.newVers[k])
            else:
                self.versUIs[k].setText('')

    def setABCrange(self):
        startFrame, endFrame, fps = treeCommon.getABCrange(self.ui.abcFile_lineEdit.text())

        self.ui.frame_start_label.setText(str(startFrame))
        self.ui.frame_end_label.setText(str(endFrame))
        self.ui.fps_lineEdit.setText(str(fps))

        self.ui.export_start_lineEdit.setText(str(startFrame))
        self.ui.export_end_lineEdit.setText(str(endFrame))

    def __update_vers_lineEdit(self, k):
        self.setOutputInfo()

        ver = self.versUIs[k].text()

        if os.path.exists(self.outpaths[k]):
            self.versUIs[k].setStyleSheet(COLORSTYLE['orange'])
        elif ver == self.newVers[k]:
            self.versUIs[k].setStyleSheet(COLORSTYLE['green'])
        else:
            self.versUIs[k].setStyleSheet(COLORSTYLE['red'])


    def update_ver_lineEdit(self):
        self.__update_vers_lineEdit('out')


    def update_texVer_lineEdit(self):
        self.__update_vers_lineEdit('tex')


    def updateTextEdited_asset_lineEdit(self):
        self.setOutputInfo()

        if not (self.assetName and os.path.exists(self.astpath)):
            self.ui.asset_lineEdit.setStyleSheet(COLORSTYLE['orange'])
        else:
            self.ui.asset_lineEdit.setStyleSheet(COLORSTYLE['green'])


    def update_asset_lineEdit(self):
        self.updateTextEdited_asset_lineEdit()
        self.setVersCompleter()
        self.update_ver_lineEdit()
        self.update_texVer_lineEdit()
        self.setOutputInfo()
        self.updateSceneGraph()


    def update_show_comboBox(self):
        self.setOutputInfo()

        if not os.path.exists(self.toppath):
            msg = '"%s" show has no publish directory' % self.showName
            self.message(msg, type=MSGERROR)

            self.setEnabled(self.ui.asset_label, False)
            self.setEnabled(self.ui.asset_lineEdit, False)
            self.setEnabled(self.ui.ver_label, False)
            self.setEnabled(self.ui.ver_lineEdit, False)
            self.setEnabled(self.ui.texVer_label, False)
            self.setEnabled(self.ui.texVer_lineEdit, False)

            self.setAssetCompleter(clear=True)
        else:
            self.message()
            self.setEnabled(self.ui.asset_label, True)
            self.setEnabled(self.ui.asset_lineEdit, True)
            self.setEnabled(self.ui.ver_label, True)
            self.setEnabled(self.ui.ver_lineEdit, True)
            self.setEnabled(self.ui.texVer_label, True)
            self.setEnabled(self.ui.texVer_lineEdit, True)

            self.setAssetCompleter()
            #self.update_asset_lineEdit()


    def openDialog_selectAbcFile(self):
        dialog = FindFileDialog(self, "Find ABC File")
        dialog.setNameFilter("Alembic Files (*.abc)")
        result = dialog.exec_()

        if result == 1:
            path = dialog.selectedFiles()[-1]
            self.ui.abcFile_lineEdit.setText(path)
            assetName = 'Tree_%s' % os.path.basename(path).replace('.abc', '')
            self.ui.asset_lineEdit.setText(assetName)

            self.ui.clip_checkBox.setCheckState(QtCore.Qt.Checked)
            self.clip_activeUI()
            self.setABCrange()
            self.setOutputInfo()

            xmlpath = '%s_Mat.xml'%path.split('.')[-2]
            if os.path.exists(xmlpath):
                self.ui.xmlFile_lineEdit.setText(xmlpath)
                self.setTextureGroup(xmlpath)
                self.message()
            else:
                self.message('XML file not exist in the alembic directory or has different name.')


    def openDialog_selectXmlFile(self):
        dialog = FindFileDialog(self, "Find XML File")
        dialog.setNameFilter("XML Files (*.xml)")
        result = dialog.exec_()

        if result == 1:
            path = dialog.selectedFiles()[-1]
            self.ui.xmlFile_lineEdit.setText(path)


    def addMaterialSet(self):
        treeCommon.MaterialSetWidgetItem(self.tw_materialSet, self.tw_sceneGraph)

    def addTextureGroup(self):
        treeCommon.txGroupWidgetItem(self.tw_textureGroup, self.tw_sceneGraph)

    def setTextureGroup(self, path):
        self.tw_textureGroup.clear()
        txInfo = treeCommon.getTextureGroup(path)

        for k in txInfo:
            treeCommon.txGroupWidgetItem(self.tw_textureGroup, self.tw_sceneGraph, k)

    def openDirectory(self):
        path = ''
        if self.sender().objectName() == 'abcFile_openDir_pushButton':
            path = self.ui.abcFile_lineEdit.text()
        elif self.sender().objectName() == 'xmlFile_openDir_pushButton':
            path = self.ui.xmlFile_lineEdit.text()

        path = os.path.dirname(path)
        if os.path.exists(path):
            subprocess.Popen(['xdg-open', str(path)])

    def update_allGraphs(self):
        abcFile = self.ui.abcFile_lineEdit.text()
        xmlFile = self.ui.xmlFile_lineEdit.text()

        if os.path.exists(abcFile) and os.path.exists(xmlFile):
            self.setEnable_allGraphs(True)
            self.message()
        else:
            self.setEnable_allGraphs(False)
            self.message('Alembic or xml file does not exist.', MSGERROR)

    def setEnable_allGraphs(self, enable):
        frames = [
            self.ui.sceneGraph_frame,
            self.ui.materialSet_frame,
            self.ui.clip_frame,
            self.ui.textureGroup_frame
        ]
        uis = [
            self.ui.sceneGraph_label,
            self.ui.materialSet_label,
            self.ui.materialSet_add_pushButton,
            self.ui.textureGroup_label,
            self.ui.textureGroup_add_pushButton
        ]
        notUse = [
            self.ui.pivot_label,
            self.ui.pivot_comboBox,
            self.ui.proxy_checkBox,
            self.ui.withRiggingData_checkBox,
        ]
        for ui in notUse:
            ui.setStyleSheet('gray')
            ui.setEnabled(False)

        style = COLORSTYLE['white' if enable else 'gray']
        for ui in uis:
            ui.setStyleSheet(style)

        if not enable:
            self.ui.clip_checkBox.setCheckState(QtCore.Qt.CheckState(False))
            self.clip_activeUI()

        for frame in frames:
            frame.setEnabled(enable)

    # --------------------------------------------------------------------------
    # OUTPUT UI FUNCTIONS
    # --------------------------------------------------------------------------

    def updateSceneGraph(self):
        self.tw_sceneGraph.clear()

        xmlPath = self.ui.xmlFile_lineEdit.text()
        grpName = '%s_model_GRP' % self.assetName
        root = treeCommon.addWidgetItem(self.tw_sceneGraph, grpName)

        if self.assetName and xmlPath:
            prims = treeCommon.getSceneGraph(self.assetName, self.texVer, xmlPath)
            for p in prims:
                item = treeCommon.sceneGraphWidgetItem(root, p)

        self.tw_sceneGraph.expandItem(root)
        self.updateMaterialSet()
        self.updateTextureGroup()

    def updateTextureGroup(self):
        txGroup = []

        for index in range(self.tw_textureGroup.topLevelItemCount()):
            itemWidget = self.tw_textureGroup.topLevelItem(index)
            txGroup.append(itemWidget.txGroupName.text())

        for index in range(self.tw_sceneGraph.topLevelItemCount()):
            for c in range(self.tw_sceneGraph.topLevelItem(index).childCount()):
                itemWidget = self.tw_sceneGraph.topLevelItem(index).child(c)
                selectItem = itemWidget.child(1).txAssign.currentText()

                if itemWidget.child(1).txAssign.count():
                    itemWidget.child(1).txAssign.clear()

                for idx, tx in enumerate(txGroup):
                    itemWidget.child(1).txAssign.addItem(tx)
                    if tx in selectItem:
                        itemWidget.child(1).txAssign.setCurrentIndex(idx)

    def updateMaterialSet(self):
        material = []

        for index in range(self.tw_materialSet.topLevelItemCount()):
            itemWidget = self.tw_materialSet.topLevelItem(index)
            material.append(itemWidget.materialSetName.text())

        for index in range(self.tw_sceneGraph.topLevelItemCount()):
            for c in range(self.tw_sceneGraph.topLevelItem(index).childCount()):
                itemWidget = self.tw_sceneGraph.topLevelItem(index).child(c)

                selectItem = itemWidget.materialSet.currentText()

                if itemWidget.materialSet.count():
                    itemWidget.materialSet.clear()

                for idx, mat in enumerate(material):
                    itemWidget.materialSet.addItem(mat)
                    if mat in selectItem:
                        itemWidget.materialSet.setCurrentIndex(idx)

    def export(self):
        self.setOutputInfo()
        if self.tw_sceneGraph.topLevelItemCount():
            export = self.MessagePopupOkCancel('Do you want to File Export?')
            if export == QtWidgets.QMessageBox.Ok:
                grpName = '%s_model_GRP' % self.assetName
                usdPath = '%s/model/%s.usd' % (self.astpath, grpName)
                treeCommon.exportJson(self.tw_sceneGraph, self.tw_textureGroup, self.jsonPath)

                # houdini command excution
                houdini_cmd = 'DCC hython -b cgsup tree.py %s %s %s %s' % (self.abcPath, usdPath, '_'.join(self.clipRange), self.abcFps)
                run = subprocess.Popen(houdini_cmd, shell=True)
                run.wait()

                # add USD Primver
                result = treeCommon.addPrimvar(self.jsonPath, usdPath)
                if result:  # True
                    self.MessagePopup('USD Export Complate!!')
                else:   # False
                    self.MessagePopup('Error!!!')
        else:
            self.MessagePopup('import ABC file first!')

    def MessagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, self.ui.title_label.text(),
                                                msg,QtWidgets.QMessageBox.Ok)
    def MessagePopupOkCancel(self, msg):
        result = QtWidgets.QMessageBox.information(self, self.ui.title_label.text(),
                        msg,QtWidgets.QMessageBox.Ok |  QtWidgets.QMessageBox.Cancel)
        return result

class FindFileDialog(QtWidgets.QFileDialog):
    def __init__(self, parent=None, windowName='', dirPath=None):
        QtWidgets.QFileDialog.__init__(self, parent)
        self.setWindowTitle(windowName)
        self.setStyleSheet('''background-color: rgb(80, 80, 80); color:white;''')
        self.setMinimumSize(1200, 800)

        if dirPath:
            self.setDirectory(dirPath)
