# -*- coding: utf-8 -*-
import os
import subprocess
import getpass
import pprint

# QT
from PySide2 import QtWidgets, QtGui, QtCore

# STYLESHEET
import qdarkstyle

# USD
from pxr import Usd, Sdf

import dxConfig
from dxstats import inc_tool_by_user as log

from ui.ui_usdsgviewer import Ui_Form
from ui.ui_renderpopup import Ui_FormRender
import sgWidgetItem as sgItems
import sgCommon

# Mongo DB
from pymongo import MongoClient
DB_IP = dxConfig.getConf('DB_IP')
client = MongoClient(DB_IP)

class MainForm(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        darkStyleSheet = qdarkstyle.load_stylesheet()
        self.setStyleSheet(darkStyleSheet)

        branchless = 'QTreeView::branch { border-image: none;}'
        self.ui.variant_treeWidget.setStyleSheet(branchless)
        self.ui.metadata_treeWidget.setStyleSheet(branchless)

        # self.show = ''
        self.pubShow = ''
        self.seq = ''
        self.shot = ''
        self.shotUSDPath = ''
        self.stage = None
        self.popup = None

        self.variants = {}
        self.variantSpace = {}
        self.variantSave = {}
        self.variant4View = {}

        showList = sgCommon.getShowList()
        self.ui.show_comboBox.clear()
        for key, value in sorted(showList.items()):
            self.ui.show_comboBox.addItem(key, value['code'])

        completer = sgCommon.getShotList(self.ui.show_comboBox.currentText())
        if completer:
            self.ui.shot_lineEdit.setCompleter(completer)
        self.ui.shot_lineEdit.clear()

        self.ui.show_comboBox.activated.connect(self.setShotCompleter)
        self.ui.search_pushButton.clicked.connect(self.doIt)
        self.ui.shot_lineEdit.returnPressed.connect(self.doIt)
        self.ui.sceneGraph_treeWidget.itemClicked.connect(self.getSceneGraphItem)
        self.ui.save_pushButton.clicked.connect(self.saveUSD)
        self.ui.render_pushButton.clicked.connect(self.renderUSD)
        self.ui.usdviewer_pushButton.clicked.connect(self.openUsdViewer)

        self.ui.variant_treeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.variant_treeWidget.customContextMenuRequested.connect(self.menuContextVariant)

        self.ui.render_pushButton.setVisible(False)
        self.ui.usdviewer_pushButton.setVisible(False)
        self.ui.save_pushButton.setVisible(False)

        # test
        # currentIdx = self.ui.show_comboBox.findText('ncx')
        # self.ui.show_comboBox.setCurrentIndex(currentIdx)
        # self.ui.show_comboBox.activated.emit(currentIdx)
        # self.ui.shot_lineEdit.setText('CTC_0207')

    def menuContextVariant(self, point):
        sg = self.ui.sceneGraph_treeWidget.currentItem()
        name = sg.text(0)

        menu = QtWidgets.QMenu()
        menu.addAction('\'%s\' Save variantSelection' % name, self.saveVariantInfo)
        menu.exec_(self.ui.variant_treeWidget.mapToGlobal(point))

    def saveVariantInfo(self):
        sg = self.ui.sceneGraph_treeWidget.currentItem()
        primPath = sg.text(1)

        self.variantSave[primPath] = {}

        for index in range(self.ui.variant_treeWidget.topLevelItemCount()):
            itemWidget = self.ui.variant_treeWidget.topLevelItem(index)

            varName = itemWidget.variantName.text()
            varSelection = itemWidget.variantSelection.currentText()
            saveVer = ''
            if self.variantSpace[primPath].has_key(varName):
                saveVer = self.variantSpace[primPath][varName]

            if varSelection != saveVer:
                self.variantSave[primPath][varName] = varSelection
            itemWidget.setColor()

        self.ui.save_pushButton.setVisible(True)

        log.run('action.usdsgviewer.saveVariant', getpass.getuser())

    def getAllPrims(self, prim, item, keyword=''):
        pChildren = prim.GetAllChildren()
        if pChildren:
            for idx, i in enumerate(pChildren):
                pString = i.GetPath().pathString
                pName = i.GetPath().name

                if keyword in pString:
                    data = QtWidgets.QTreeWidgetItem()
                    data.setText(0, pName)
                    data.setText(1, pString)
                    item.addChild(data)

                    self.checkExpandItem(pString, item)
                    self.checkRigDate(i, pName, pString, data)

                    parent = item.child(idx)
                    self.getAllPrims(i, parent)

    def checkExpandItem(self, pString, item):
        notExpand = ['Agent', 'imageplanes', 'Looks', 'Geom', 'Proxy', 'Render',
                     'prototypes', 'dxCam', '_GRP', '_layout', 'extra']
        for ex in notExpand:
            if ex in pString:
                break
        else:
            self.ui.sceneGraph_treeWidget.expandItem(item)

    def checkRigDate(self, prim, pName, pString, data):
        if 'Rig/' not in pString:
            return

        customData = sgCommon.getAniCustomLayerData(prim, pName)
        if customData:
            assetName = os.path.basename(customData['rigFile']).split('_rig')[0]
            rigLastVer, rigTime = sgCommon.getRigPubTime(self.show, assetName)
            pubTime = sgCommon.getAniCacheOutTime(prim, pName)

            if rigTime and pubTime and rigTime > pubTime:
                data.setTextColor(0, QtGui.QColor(255, 0, 0, 255))
            # use debug
            elif rigTime and pubTime and rigTime < pubTime:
                data.setTextColor(0, QtGui.QColor(0, 255, 0, 255))

    def getPrimVariants(self, primPath, variant='', selection=''):
        self.variants['variant'] = {}
        self.variants['selection'] = {}

        exList = ['preview', 'WorldXform']
        prim = self.stage.GetPrimAtPath(primPath)

        if prim.HasVariantSets():
            if variant:
                prim.GetVariantSet(variant).SetVariantSelection(selection)

            vsets = prim.GetVariantSets().GetNames()
            for i in vsets:
                if i not in exList:
                    self.variants['variant'][i] = prim.GetVariantSets().GetVariantSet(i).GetVariantNames()
                    self.variants['selection'][i] = prim.GetVariantSets().GetVariantSelection(i)

                    if not self.variantSpace.has_key(primPath):
                        self.variantSpace[primPath] = {}
                    if not self.variantSpace[primPath].has_key(i):
                        self.variantSpace[primPath][i] = prim.GetVariantSets().GetVariantSelection(i)

    def getVariantRefList(self, vset, dir, vers):
        if vset:
            for var in vset.variantList:
                varPrim = var.primSpec
                reflist = varPrim.referenceList.prependedItems
                if reflist:
                    ver = var.name
                    vers[ver] = os.path.join(dir, reflist[0].assetPath.replace('./', ''))

                    if '/model' in vers[ver] and os.path.exists(vers[ver]):
                        layer = Sdf.Layer.FindOrOpen(vers[ver])
                        dPrim = layer.GetPrimAtPath(layer.defaultPrim)

                        for key, value in dPrim.variantSets.items():
                            vset2 = dPrim.variantSets.get(key)
                            if vset2:
                                dir = os.path.dirname(layer.realPath)
                                break
                        else:
                            break

                        self.getVariantRefList(vset2, dir, vers)
                else:
                    for key, value in var.variantSets.items():
                        vset = var.variantSets.get(key)
                        if vset:
                            dir = os.path.dirname(var.layer.realPath)
                            break

                    self.getVariantRefList(vset, dir, vers)

    def getVariantPath(self, primPath, variantSetName, ver=''):
        vset = None
        dir = None
        spec = None
        vers = {}

        prim = self.stage.GetPrimAtPath(primPath)
        for lyr in prim.GetPrimStack():
            if 'Layout' in primPath:
                if lyr.path.pathString in primPath:
                    spec = lyr.GetPrimAtPath(lyr.path.pathString)
            elif variantSetName.replace('Ver', '') in lyr.layer.realPath:
                spec = lyr.GetPrimAtPath(primPath)
            if not spec:
                continue

            for key, value in spec.variantSets.items():
                vset = spec.variantSets.get(key)
            if vset:
                dir = os.path.dirname(lyr.layer.realPath)
                break
        self.getVariantRefList(vset, dir, vers)

        if vers.has_key(ver):    return vers[ver]
        else:                    return vers

    def getSceneGraphItem(self, item, col):
        self.ui.metadata_treeWidget.clear()
        self.ui.variant_treeWidget.clear()

        nsLayer = item.text(0)
        primPath = item.text(1)

        self.variants['variantRefList'] = {}
        self.setVariantItem(primPath, nsLayer)

        # pprint.pprint(self.variantSpace)
        # pprint.pprint(self.variants)

    def setData(self, currentItem, primPath, varName, varVer=''):
        # print '-' * 50
        # print 'currentItem:', currentItem
        # print 'primPath:', primPath
        # print 'varName:', varName
        # print 'varVer:', varVer
        # print '-' * 50

        customLayerData = {}
        metadata = []
        aleatColumn = []
        if not varVer:  ver = self.variants['selection'][varName]
        else:           ver = varVer

        # cacheOut time
        cacheTime = sgCommon.getCacheOutTime(self.variants['variantRefList'][varName][ver])
        metadata.append(('cacheOut time', cacheTime.strftime('%Y-%m-%d %H:%M')))
        # metadata.append(('cacheOut path', self.variants['variantRefList'][varName][ver]))

        # custom Layer Data
        customLayerData = sgCommon.getCustomLayerData(self.variants['variantRefList'][varName][ver])
        for k, v in customLayerData.items():
            if k not in ['start', 'end', 'step', 'dxusd']:
                metadata.append((k, v))

        # USD pub_db
        g_DB = client['USD_PUBLISH']
        coll = g_DB[self.pubShow]
        find = {'shot': self.shot, 'task': 'shot'}

        type = varName.replace('Ver','')
        find['type'] = type

        if currentItem in ['Cam', 'main_cam']:
            type = 'cam'
            find['type'] = 'cam'
            currentItem = 'Cam'
        else:
            find['name'] = currentItem

        if varVer:
            find['version'] = varVer
        else:
            if self.variants['selection'].has_key(type+'Ver'):
                find['version'] = self.variants['selection'][type+'Ver']

        # print 'query:', varName, self.pubShow, find

        for i in coll.find(find).limit(1).sort([('$natural', -1)]):
            for key, value in i.items():
                if key in ['artist', 'user', 'version', 'outDirs']:
                    metadata.append((key, value))
                elif key == 'logs':
                    metadata.append(('artist', i['logs'][-1]['user']))
                    metadata.append(('comment', i['logs'][-1]['comment']))

        if 'ani' in varName:
            assetName = os.path.basename(customLayerData['rigFile']).split('_rig')[0]

            # LASTEST RIG
            rigLastVer, rigTime = sgCommon.getRigPubTime(self.show, assetName)
            if rigTime and cacheTime and rigTime > cacheTime:
                aleatColumn.append(cacheTime.strftime('%Y-%m-%d %H:%M'))
            if rigLastVer and rigTime:
                metadata.append(('rig latest ver', rigLastVer))
                metadata.append(('rig latest time', rigTime.strftime('%Y-%m-%d %H:%M')))

            # RIG ARTIST
            data = sgCommon.getRigInfoDB(self.show, assetName)
            if data:
                for i in data:
                    metadata.append(('rig artist', i['artist']))

        # pprint.pprint(metadata)
        return metadata, aleatColumn

    def createMetadataItem(self, data, title, ver='', aleatColumn=''):
        widgetItem = QtWidgets.QTreeWidgetItem()
        widgetItem.setText(0, title)
        if ver:
            widgetItem.setText(1, ver)
        self.ui.metadata_treeWidget.addTopLevelItem(widgetItem)

        tableModel = sgItems.MetaTableModel(data, aleatColumn, self)
        metaWidget = sgItems.metadataWidgetItem(widgetItem)
        metaWidget.metaTableView.setModel(tableModel)
        metaWidget.setWidgetSize()

        self.ui.metadata_treeWidget.expandItem(widgetItem)

    def setVariantItem(self, primPath, nsLayer, varName='', selVer=''):
        self.getPrimVariants(primPath, varName, selVer)
        self.createMetadataItem(self.variants['global'], 'global')

        for key, value in sorted(self.variants['variant'].items()):
            item = sgItems.variantWidgetItem(self, self.ui.variant_treeWidget, key, value, self.variants['selection'][key])
            currentIdx = item.variantSelection.findText(self.variants['selection'][key])

            if 'Ver' in key:
                value = sorted(self.variants['variant'][key], reverse=True)
                varPath = self.getVariantPath(primPath, key)
                if varPath:
                    self.variants['variantRefList'][key] = varPath
                    customLayerData, aleatColumn = self.setData(nsLayer, primPath, key)
                    if currentIdx != -1 and self.variants['selection'].has_key(key):
                        self.createMetadataItem(customLayerData, key.replace('Ver', ''), self.variants['selection'][key], aleatColumn)

            if not self.variant4View.has_key(primPath):
                self.variant4View[primPath] = {}
            self.variant4View[primPath][key] = self.variants['selection'][key]

        pprint.pprint(self.variants)

    def doIt(self):
        self.ui.sceneGraph_treeWidget.clear()
        self.ui.metadata_treeWidget.clear()
        self.ui.variant_treeWidget.clear()
        self.popup = None

        self.stage = None
        self.variants = {}
        self.variantSpace = {}
        self.varnantSave = {}
        self.variant4View = {}

        self.show = self.ui.show_comboBox.currentText().lower()
        self.pubShow = self.show + '_pub'
        self.shot = self.ui.shot_lineEdit.text()
        self.seq = self.shot.split('_')[0]

        self.shotUSDPath = '/show/{show}/_3d/shot/{seq}/{shot}/{shot}.usd'.format(show=self.show, seq=self.seq, shot=self.shot)
        if os.path.isfile(self.shotUSDPath):
            self.stage = Usd.Stage.Open(self.shotUSDPath)
            self.ui.usdviewer_pushButton.setVisible(True)
            self.ui.render_pushButton.setVisible(True)
        else:
            self.messagePopup('%s\n%s\nUSD file not found!' % (self.shot, self.shotUSDPath))
            return

        self.ui.save_pushButton.setVisible(False)

        dPrim = self.stage.GetDefaultPrim()
        data = QtWidgets.QTreeWidgetItem()
        data.setText(0, dPrim.GetName())
        data.setText(1, dPrim.GetPrimPath().pathString)
        self.ui.sceneGraph_treeWidget.addTopLevelItem(data)
        item = self.ui.sceneGraph_treeWidget.topLevelItem(0)
        self.getAllPrims(dPrim, item)

        self.variants['global'] = []
        self.variants['global'].append(('start', self.stage.GetStartTimeCode()))
        self.variants['global'].append(('end', self.stage.GetEndTimeCode()))
        self.variants['global'].append(('timecodePerSecond', self.stage.GetTimeCodesPerSecond()))
        self.variants['global'].append(('dxusd', self.stage.GetRootLayer().customLayerData['dxusd']))
        self.createMetadataItem(self.variants['global'], 'global')

        log.run('action.usdsgviewer.fileOpen', getpass.getuser())

    def saveUSD(self):
        result = QtWidgets.QMessageBox.information(self, "Save Selected Variants",
                                                   pprint.pformat(self.variantSave).replace('u\'', ''),
                                                   QtWidgets.QMessageBox.Save, QtWidgets.QMessageBox.No)
        if result == QtWidgets.QMessageBox.Save:
            for primPath in self.variantSave.keys():
                print primPath
                prim = self.stage.GetPrimAtPath(primPath)
                for varName, varSel in self.variantSpace[primPath].items():
                    varSelection = prim.GetVariantSet(varName).GetVariantSelection()
                    if varSel and varSel != varSelection:
                        prim.GetVariantSet(varName).SetVariantSelection(varSel)
            self.stage.GetRootLayer().Save()
            log.run('action.usdsgviewer.saveUSD', getpass.getuser())
            print '----------------------- save USD! -----------------------'

    def openUsdViewer(self):
        log.run('action.usdsgviewer.usdviewerOpen', getpass.getuser())

        varArgs = ''
        for primPath in self.variant4View.keys():
            for varName, varSel in self.variant4View[primPath].items():
                varArgs += ' --var \'%s:%s:%s\'' % (primPath, varName, varSel)

        cmd = '%s rez-env usdtoolkit -- usdview' % os.environ['DCCPROC']
        cmd += ' --camera main_cam --defaultsetting %s' % self.shotUSDPath
        cmd += varArgs
        run = subprocess.Popen(cmd, shell=True)

    def renderUSD(self):
        selectedPrims = self.ui.sceneGraph_treeWidget.selectedItems()

        if selectedPrims:
            if not self.popup:
                self.popup = RenderPopupWindow(selectedPrims,
                                               self.variants['global'], self)

            self.popup.ui.selPrims_listWidget.clear()
            self.popup.prims = selectedPrims
            for prim in self.popup.prims:
                item = QtWidgets.QListWidgetItem()
                item.setText(prim.text(1))
                self.popup.ui.selPrims_listWidget.addItem(item)

            self.popup.exec_()
        else:
            self.messagePopup('Prim을 선택 해 주세요.')

    def setShotCompleter(self, index):
        completer = sgCommon.getShotList(self.ui.show_comboBox.currentText())
        if completer:
            self.ui.shot_lineEdit.setCompleter(completer)
        self.ui.shot_lineEdit.clear()


    def messagePopup(self, msg):
        QtWidgets.QMessageBox.information(self, 'USD-sceneGraph', msg, QtWidgets.QMessageBox.Ok)


class RenderPopupWindow(QtWidgets.QDialog):
    def __init__(self, selectedPrims, infos, parent=None):
        super(RenderPopupWindow, self).__init__(parent)
        self.ui = Ui_FormRender()
        self.ui.setupUi(self)
        self.setWindowTitle('USD Recoder')
        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.parent = parent
        self.prims = selectedPrims
        self.infos = infos

        for i in self.infos:
            if 'start' in i[0]:
                self.ui.frameIn_lineEdit.setText(str(i[1]))
            elif 'end' in i[0]:
                self.ui.frameOut_lineEdit.setText(str(i[1]))

        self.ui.render_pushButton.clicked.connect(self.usdRecoder)

    def usdRecoder(self):
        cmd = '/backstage/dcc/DCC usdtoolkit -- usdrecorder'
        cmd += ' --mask '
        for idx in range(self.ui.selPrims_listWidget.count()):
            cmd += self.ui.selPrims_listWidget.item(idx).text() + ','
        cmd += '/World/Cam/main_cam'
        cmd += ' --purpose render'

        cmd += ' --frames ' + self.ui.frameIn_lineEdit.text()
        if self.ui.frameOut_lineEdit.text():
            cmd += ' ' + self.ui.frameOut_lineEdit.text()

        cmd += ' --camera /World/Cam/main_cam'
        cmd += ' --renderer Prman'
        cmd += ' --maxsamples ' + self.ui.maxSam_lineEdit.text()
        cmd += ' --pixelvariance ' + self.ui.pixelVal_lineEdit.text() + ' '
        cmd += self.parent.shotUSDPath.replace('.usd', '.prv.usd')

        print '### usdrecoder cmd:', cmd
        run = subprocess.Popen(cmd, shell=True)
