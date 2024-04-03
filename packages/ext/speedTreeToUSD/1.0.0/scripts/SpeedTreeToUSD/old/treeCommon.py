from PySide2 import QtWidgets, QtCore, QtGui

import xml.etree.ElementTree as ElemTree
import os
import pprint
import json
from pxr import Usd, Sdf, UsdGeom

import alembic


COLORSTYLE={
    'red'   :'color: rgb(250, 30, 50);',
    'green' :'color: rgb(50, 250, 30);',
    'gray'  :'color: rgb(120, 120, 120);',
    'orange':'colimport Zelosor: rgb(250, 150, 30);',
    'white' :'color: white;'
}

class addWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, value):
        super(addWidgetItem, self).__init__(parent)

        self.rootName = QtWidgets.QLineEdit(value)
        self.rootName.setMaximumSize(QtCore.QSize(350, 25))
        parent.setItemWidget(self, 0, self.rootName)

class sceneGraphWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, primInfo):
        super(sceneGraphWidgetItem, self).__init__(parent)

        # Column 0 -- prim Name
        self.primName = QtWidgets.QLineEdit(primInfo[0])
        self.primName.setMaximumSize(QtCore.QSize(350, 25))
        parent.treeWidget().setItemWidget(self, 0, self.primName)

        for key, value in primInfo[1].items():
            if key in 'assignTxGroup':      # ComboBox
                txAssignWidgetItem(self, value)
            else:
                txInfoWidgetItem(self, key, value)

        # Column 1 -- materialSet
        self.materialSet = QtWidgets.QComboBox()
        self.materialSet.setMaximumSize(QtCore.QSize(70, 25))
        parent.treeWidget().setItemWidget(self, 1, self.materialSet)

        # Column 2 -- shell ID
        self.sID = QtWidgets.QCheckBox('sID')
        self.sID.setMaximumSize(QtCore.QSize(50, 25))
        parent.treeWidget().setItemWidget(self, 2, self.sID)
        self.sID.stateChanged.connect(self.checked)
        self.sID.setCheckState(QtCore.Qt.Unchecked)

        # Column 3 -- subdvide
        self.subdvide = QtWidgets.QCheckBox('subd')
        self.subdvide.setMaximumSize(QtCore.QSize(55, 25))
        parent.treeWidget().setItemWidget(self, 3, self.subdvide)
        self.subdvide.stateChanged.connect(self.checked)

    def checked(self):
        font = QtGui.QFont()
        if self.sID.isChecked():
            font.setBold(True)
            self.sID.setFont(font)
            self.sID.setStyleSheet(COLORSTYLE['green'])
        else:
            font.setBold(False)
            self.sID.setFont(font)
            self.sID.setStyleSheet(COLORSTYLE['white'])

        if self.subdvide.isChecked():
            font.setBold(True)
            self.subdvide.setFont(font)
            self.subdvide.setStyleSheet(COLORSTYLE['green'])
        else:
            font.setBold(False)
            self.subdvide.setFont(font)
            self.subdvide.setStyleSheet(COLORSTYLE['white'])

class txInfoWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, key, value):
        super(txInfoWidgetItem, self).__init__(parent)

        self.txInfo = QtWidgets.QLabel('%s: %s' % (key, value))
        parent.treeWidget().setItemWidget(self, 0, self.txInfo)

class txAssignWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, value):
        super(txAssignWidgetItem, self).__init__(parent)

        self.baseWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.baseWidget)

        # Column 0 -- label
        self.txInfo = QtWidgets.QLabel('assignTxGroup: ')
        self.txInfo.setMaximumSize(QtCore.QSize(110, 25))
        self.layout.addWidget(self.txInfo)

        # Column 1 -- comboBox
        self.txAssign = QtWidgets.QComboBox()
        self.layout.addWidget(self.txAssign)
        self.txAssign.addItem(value)
        parent.treeWidget().setItemWidget(self, 0, self.baseWidget)

class MaterialSetWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, update, name=''):
        super(MaterialSetWidgetItem, self).__init__(parent)

        self.parent = parent
        self.update = update

        # Column 0 -- materialSetName
        self.materialSetName = QtWidgets.QLineEdit(name)
        self.materialSetName.setMinimumSize(QtCore.QSize(160, 25))
        parent.setItemWidget(self, 0, self.materialSetName)

        self.materialSetName.editingFinished.connect(self.editMaterialSetName)

        # Column 1 -- deleteMaterialSet
        self.btnDel = QtWidgets.QPushButton('-')
        self.btnDel.setMinimumSize(QtCore.QSize(23, 23))
        self.btnDel.setMaximumSize(QtCore.QSize(23, 23))
        parent.setItemWidget(self, 1, self.btnDel)

        self.btnDel.clicked.connect(self.delMaterial)

    def editMaterialSetName(self):
        material = []

        for index in range(self.parent.topLevelItemCount()):
            itemWidget = self.parent.topLevelItem(index)
            material.append(itemWidget.materialSetName.text())
        print material

        for index in range(self.update.topLevelItemCount()):
            for c in range(self.update.topLevelItem(index).childCount()):
                itemWidget = self.update.topLevelItem(index).child(c)

                selectItem = itemWidget.materialSet.currentText()

                if itemWidget.materialSet.count():
                    itemWidget.materialSet.clear()

                for idx, mat in enumerate(material):
                    itemWidget.materialSet.addItem(mat)
                    if mat in selectItem:
                        itemWidget.materialSet.setCurrentIndex(idx)

    def delMaterial(self):
        idx = self.parent.indexOfTopLevelItem(self)
        self.parent.takeTopLevelItem(idx)
        self.editMaterialSetName()

class txGroupWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, update, txInfo=''):
        super(txGroupWidgetItem, self).__init__(parent)

        self.parent = parent
        self.update = update

        # Column 0 -- txtureGroupName
        if txInfo:
            self.txGroupName = QtWidgets.QLineEdit(txInfo[0])
            for key, value in txInfo[1].items():
                if key in 'color':
                    txFileWidgetItem(self, 'diffC', value)
                elif key in 'specular':
                    txFileWidgetItem(self, 'specG', value)
                else:
                    txFileWidgetItem(self, key, value)
        else:
            self.txGroupName = QtWidgets.QLineEdit('')
        self.txGroupName.setMinimumSize(QtCore.QSize(380, 25))
        parent.setItemWidget(self, 0, self.txGroupName)

        self.txGroupName.editingFinished.connect(self.editTxGroupName)

        # Column 1 -- AddTxtureGroup
        self.btnAdd = QtWidgets.QPushButton('+')
        self.btnAdd.setMinimumSize(QtCore.QSize(23, 23))
        self.btnAdd.setMaximumSize(QtCore.QSize(23, 23))
        parent.setItemWidget(self, 1, self.btnAdd)

        self.btnAdd.clicked.connect(self.addTextureFile)

        # Column 2 -- deleteTxtureGroup
        self.btnDel = QtWidgets.QPushButton('-')
        self.btnDel.setMinimumSize(QtCore.QSize(23, 23))
        self.btnDel.setMaximumSize(QtCore.QSize(23, 23))
        parent.setItemWidget(self, 2, self.btnDel)

        self.btnDel.clicked.connect(self.delTexureGroup)

    def editTxGroupName(self):
        txGroup = []

        for index in range(self.parent.topLevelItemCount()):
            itemWidget = self.parent.topLevelItem(index)
            txGroup.append(itemWidget.txGroupName.text())
        print txGroup

        for index in range(self.update.topLevelItemCount()):
            for c in range(self.update.topLevelItem(index).childCount()):
                itemWidget = self.update.topLevelItem(index).child(c)
                selectItem = itemWidget.child(1).txAssign.currentText()

                if itemWidget.child(1).txAssign.count():
                    itemWidget.child(1).txAssign.clear()

                for idx, tx in enumerate(txGroup):
                    itemWidget.child(1).txAssign.addItem(tx)
                    if tx in selectItem:
                        itemWidget.child(1).txAssign.setCurrentIndex(idx)

    def addTextureFile(self):
        txFileWidgetItem(self, 'txType', 'fileName')

    def delTexureGroup(self):
        idx = self.parent.indexOfTopLevelItem(self)
        self.parent.takeTopLevelItem(idx)
        self.editTxGroupName()

class txFileWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent, type, file):
        super(txFileWidgetItem, self).__init__(parent)

        self.parent = parent

        self.baseWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.baseWidget)

        # Column 0 -- textureInfo
        self.txInfo = QtWidgets.QLabel('%s: ' % type)
        self.txInfo.setMaximumSize(QtCore.QSize(200, 25))
        self.layout.addWidget(self.txInfo)

        # Column 1 -- txtureFile
        self.txFile = QtWidgets.QLineEdit(file)
        self.layout.addWidget(self.txFile)

        # Column 2 -- txtureFileOpen
        self.btnOpen = QtWidgets.QPushButton('...')
        self.btnOpen.setMinimumSize(QtCore.QSize(23, 23))
        self.btnOpen.setMaximumSize(QtCore.QSize(23, 23))
        self.layout.addWidget(self.btnOpen)
        self.btnOpen.clicked.connect(self.openTxFile)

        # Column 3 -- deleteTxtureFile
        self.btnDel = QtWidgets.QPushButton('-')
        self.btnDel.setMinimumSize(QtCore.QSize(23, 23))
        self.btnDel.setMaximumSize(QtCore.QSize(23, 23))
        self.layout.addWidget(self.btnDel)
        self.btnDel.clicked.connect(self.delTexureFile)

        parent.treeWidget().setItemWidget(self, 0, self.baseWidget)

    def openTxFile(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setStyleSheet('background-color: rgb(80, 80, 80); color:white;')
        dialog.setMinimumSize(1200, 800)
        file = dialog.getOpenFileName()
        if file:
            self.txFile.setText(file)

    def delTexureFile(self):
        self.parent.takeChild(self)




def getTextureGroup(path):
    tree = ElemTree.parse(path)

    txGroup = []
    tmp = []

    attributeList = tree.findall('./attributeList')
    for atList in attributeList:
        txInfo = {}
        location = atList.get('location')
        if not location in tmp:
            attribute = atList.findall('attribute')
            for attr in attribute:
                passKey = ['__FACE_INDICES__', 'channels']
                if not attr.get('name') in passKey:
                    txInfo[attr.get('name')] = attr.get('value')

            txGroup.append([location, txInfo])
            tmp.append(location)

    return txGroup

def getSceneGraph(asset, txVer, path):
    tree = ElemTree.parse(path)

    info = []

    attributeList = tree.findall('./attributeList')
    for atList in attributeList:
        tmp = [asset]
        txInfo = {}

        location = atList.get('location')
        attribute = atList.findall('attribute')

        txInfo['originalName'] = location
        txInfo['assignTxGroup'] = location
        for attr in attribute:
            if '__FACE_INDICES__' in attr.get('name'):
                faceIndices = attr.get('value')
                tmp.append(faceIndices)
                txInfo['originalName'] = '%s_%s' %(faceIndices, location)
            else:
                txInfo['txLayerName'] = location
                txInfo['txBasePath'] = 'asset/%s/texture' % asset
                #txInfo[attr.get('name')] = attr.get('value')

        txInfo['txVersion'] = 'v%s' % txVer

        tmp.append(atList.get('location'))
        tmp.append('PLY')
        info.append(['_'.join(tmp), txInfo])

    return info

def exportJson(sceneTreeWidget, txGroupTreeWidget, path):
    data = {}
    for index in range(sceneTreeWidget.topLevelItemCount()):
        rootWidget = sceneTreeWidget.topLevelItem(index)
        rootName = rootWidget.rootName.text()
        data[rootName] = {}

        for index in range(sceneTreeWidget.topLevelItemCount()):
            for c in range(sceneTreeWidget.topLevelItem(index).childCount()):
                itemWidget = sceneTreeWidget.topLevelItem(index).child(c)

                primName = itemWidget.primName.text()
                data[rootName][primName] = {}

                data[rootName][primName]['materialSet'] = itemWidget.materialSet.currentText()
                data[rootName][primName]['sID'] = itemWidget.sID.isChecked()
                data[rootName][primName]['subd'] = itemWidget.subdvide.isChecked()

                txGroup = ''
                for idx in range(itemWidget.childCount()):
                    child = itemWidget.child(idx)
                    if idx != 1:
                        tmp = child.txInfo.text().split(':')
                        data[rootName][primName][tmp[0]] = tmp[1].strip()
                    else:
                        txGroup = child.txAssign.currentText()
                        data[rootName][primName][child.txInfo.text().replace(':','').strip()] = txGroup

                txFile = getTxFile(txGroupTreeWidget, txGroup)
                if txFile:
                    data[rootName][primName]['txFile'] = {}
                    for t in txFile:
                        data[rootName][primName]['txFile'][t[0]] = t[1]
    # pprint.pprint(data)
    f = open(path, 'w')
    json.dump(data, f, indent=4)
    f.close()

def getTxFile(txGroupTreeWidget, txGroup):
    txFile = []
    for index in range(txGroupTreeWidget.topLevelItemCount()):
        if txGroup in txGroupTreeWidget.topLevelItem(index).txGroupName.text():
            for c in range(txGroupTreeWidget.topLevelItem(index).childCount()):
                itemWidget = txGroupTreeWidget.topLevelItem(index).child(c)
                txFile.append([itemWidget.txInfo.text().replace(':','').strip(), itemWidget.txFile.text()])
                # print 'txFile: ', txFile
    return txFile

def getABCrange(path):

    iarch = alembic.Abc.IArchive(path)
    xform = alembic.AbcGeom.IPolyMesh(iarch.getTop().getChild(0).getChild(0),
                                      alembic.Abc.WrapExistingFlag.kWrapExisting)
    schema = xform.getSchema()
    ts = schema.getTimeSampling()
    tsType = schema.getTimeSampling().getTimeSamplingType()
    numTimeSample = schema.getNumSamples()

    minTime = ts.getSampleTime(0)
    maxTime = ts.getSampleTime(numTimeSample - 1)
    stepSize = float(tsType.getTimePerCycle())

    minFrame = minTime / stepSize
    maxFrame = maxTime / stepSize
    fps = 1 / stepSize

    print 'minFrame: %s, maxFrame: %s, fps: %s' % (minFrame, maxFrame, fps)
    return minFrame, maxFrame, fps


def addPrimvar(jsonFilePath, usdFilePath):
    # print jsonFilePath
    # print usdFilePath

    # try:
    stage = Usd.Stage.Open(usdFilePath)

    if os.path.isfile(jsonFilePath):
        f = open(jsonFilePath, "r")
        j = json.load(f)
        f.close()

    for groupName, prims in j.items():
        print groupName
        output = {}
        for prim, value in prims.items():
            #print prim
            output[prim] = {}
            for k, v in value.items():
                #print k, v
                output[prim][k] = v

        grpPrim = stage.GetPrimAtPath('/%s' % groupName)
        Usd.ModelAPI(grpPrim).SetKind('component')

        for prim in grpPrim.GetAllChildren():
            Usd.ModelAPI(prim).SetKind('component')
            primName = prim.GetName()
            for key, value in output[primName].items():
                if 'materialSet' == key:
                    prim.CreateAttribute('userProperties:MaterialSet', Sdf.ValueTypeNames.String).Set(value)
                elif 'txLayerName' == key or 'txBasePath' == key or 'txVersion' == key:
                    if 'txVersion' == key:
                        key = 'ri:attributes:user:txVersion'
                    print primName, '>>>', '%s: ' % key, value
                    geom = UsdGeom.Mesh(prim)
                    addPrimvar = geom.CreatePrimvar(key, Sdf.ValueTypeNames.String)
                    addPrimvar.Set(value)
                    addPrimvar.SetInterpolation(UsdGeom.Tokens.constant)

    stage.Save()
    print '-'*70
    print 'add primvar complate!!'
    return True

    #     return True
    # except:
    #     return False
