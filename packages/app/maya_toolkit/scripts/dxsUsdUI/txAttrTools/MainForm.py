#coding:utf-8

##########################################
__author__  = 'daeseok.chae in Dexter RND'
__date__ = '2019.01.04'
__comment__ = 'Texture Attribute Setup'
__windowName__ = "txAttrTools"
##########################################

import maya.OpenMayaUI as mui
import shiboken2 as shiboken
import maya.cmds as cmds

import txAttrToolsUI
reload(txAttrToolsUI)
from txAttrToolsUI import Ui_Form

import attributeItem
reload(attributeItem )
from attributeItem import AttritubeItem
from attributeItem import ShapeItem

from pymodule.Qt import QtWidgets
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui

def getMayaWindow():
    '''
    get Maya Window Process
    :return: Maya window Process
    '''
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QMainWindow)
    except:
        return None

class TxAttrTools(QtWidgets.QWidget):
    def __init__(self, parent = getMayaWindow()):

        # Load dependency plugin
        plugins = ['ZENNForMaya']
        for p in plugins:
            if not cmds.pluginInfo(p, q=True, l=True):
                cmds.loadPlugin(p)

        QtWidgets.QWidget.__init__(self, parent)

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.move(parent.frameGeometry().center() - self.frameGeometry().center())

        self.ui.treeWidget.header().resizeSection(0, 650)
        self.ui.treeWidget.header().resizeSection(1, 430)
        self.ui.treeWidget.header().resizeSection(2, 40)

        self.ui.treeWidget.itemSelectionChanged.connect(self.shapeNodeSelection)
        self.ui.treeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.treeWidget.customContextMenuRequested.connect(self.rmbClicked)
        self.ui.treeWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.attributeList = {"MaterialSet":{"orgAttr" : "",
                                             "dataType" : "string",
                                             "niceName" : "MaterialSet",
                                             "longName" : "MaterialSet",
                                             "default" : "",
                                             "must":False},
                             "USD_ATTR_subdivisionScheme":{"orgAttr" : "rman__torattr___subdivScheme",
                                                      "dataType" : "string",
                                                      "niceName": "subdivisionScheme",
                                                      "longName": "USD_ATTR_subdivisionScheme",
                                                      "default" : "none",
                                                      "must": False},
                             "txBasePath": {"orgAttr": "rman__riattr__user_txBasePath",
                                             "dataType": "string",
                                             "niceName": "txBasePath",
                                             "longName": "txBasePath",
                                             "default" : "",
                                             "must": True},
                             "txAssetName": {"orgAttr": "rman__riattr__user_txAssetName",
                                            "dataType" : "string",
                                            "niceName": "txBasePath",
                                            "longName": "txBasePath",
                                            "default": "",
                                            "must":True},
                             "txLayerName" : {"orgAttr":"rman__riattr__user_txLayerName",
                                              "dataType" : "string",
                                              "niceName": "txLayerName",
                                              "longName": "txLayerName",
                                              "default": "",
                                              "must" : True},
                             "txmultiUV" : {"orgAttr":"rman__riattr__user_txmultiUV",
                                            "attrType" : "long",
                                            "niceName": "txmultiUV",
                                            "longName": "txmultiUV",
                                            "default": "0",
                                            "must":False}
                             }

        USD_ATTR_subdivisionScheme = {0: 'catmullClark', 1: 'loop', 100: 'none'}

        selectNode = cmds.ls(sl = True)
        itemFont = QtGui.QFont()
        itemFont.setPointSize(13)
        if selectNode:
            for selNode in selectNode:
                # if "_model_" in selNode:
                shapeNode = cmds.ls(selNode, dag = True, type = 'mesh')

                rootNodeItem = QtWidgets.QTreeWidgetItem(self.ui.treeWidget, [selNode])
                rootNodeItem.setFont(0, itemFont)

                for node in shapeNode:
                    if not node.endswith("PLYShape"):
                        continue
                    item = ShapeItem(rootNodeItem, node, self.attributeList.keys())
                    for attr in sorted(self.attributeList.keys()):
                        orgValue = ""
                        error = False
                        # remove org attr
                        if self.attributeList[attr]["orgAttr"] and cmds.attributeQuery(self.attributeList[attr]["orgAttr"], n = node, exists = True):
                            orgValue = cmds.getAttr("%s.%s" % (node, self.attributeList[attr]["orgAttr"]))
                            cmds.deleteAttr("%s.%s" % (node, self.attributeList[attr]["orgAttr"]))
                            if attr == "txAssetName":
                                error = True
                            if not cmds.attributeQuery(self.attributeList[attr]["longName"], n = node, exists = True):
                                if self.attributeList[attr].has_key("dataType"):
                                    cmds.addAttr(node, ln=self.attributeList[attr]["longName"], nn=self.attributeList[attr]["niceName"], dt=self.attributeList[attr]["dataType"])
                                elif self.attributeList[attr].has_key("attrType"):
                                    cmds.addAttr(node, ln=self.attributeList[attr]["longName"], nn=self.attributeList[attr]["niceName"], at=self.attributeList[attr]["attrType"])

                        # must attr set
                        if self.attributeList[attr]["must"] and not cmds.attributeQuery(self.attributeList[attr]['longName'], n=node, exists=True):
                            if self.attributeList[attr].has_key("dataType"):
                                cmds.addAttr(node, ln=self.attributeList[attr]["longName"], nn=self.attributeList[attr]["niceName"], dt=self.attributeList[attr]["dataType"])
                            elif self.attributeList[attr].has_key("attrType"):
                                cmds.addAttr(node, ln=self.attributeList[attr]["longName"], nn=self.attributeList[attr]["niceName"], at=self.attributeList[attr]["attrType"])

                        if cmds.attributeQuery(self.attributeList[attr]['longName'], n=node, exists=True) and not item.alreadySetupAttr[self.attributeList[attr]['longName']]:
                            value = cmds.getAttr("%s.%s" % (node, self.attributeList[attr]["longName"]))
                            if orgValue != "":
                                value = orgValue
                                if self.attributeList[attr]["longName"] == "USD_ATTR_subdivisionScheme":
                                    value = USD_ATTR_subdivisionScheme[orgValue]
                                try:
                                    cmds.setAttr("%s.%s" % (node, self.attributeList[attr]["longName"]), value, type = self.attributeList[attr]["dataType"])
                                except:
                                    cmds.setAttr("%s.%s" % (node, self.attributeList[attr]["longName"]), int(value))
                            attrItem = AttritubeItem(item, node, self.attributeList[attr]["longName"], value, self.attributeList[attr], error)
                            item.alreadySetupAttr[self.attributeList[attr]["longName"]] = attrItem

            self.ui.treeWidget.expandAll()

    def shapeNodeSelection(self):
        shapeNameList = []
        for i in self.ui.treeWidget.selectedItems():
            if i.parent() is None:
                shapeNameList.append(i.text(0))
            elif type(i) is ShapeItem:
                shapeNameList.append(i.parent().text(0))
            else:
                shapeNameList.append(i.parent().parent().text(0))

        shapeNameList = list(set(shapeNameList))
        cmds.select(shapeNameList)

    def rmbClicked(self):
        item = self.ui.treeWidget.currentItem()

        if item.parent() is None: # rootNode
            menu = QtWidgets.QMenu(self)
            menu.setStyleSheet('''
                                                            QMenu::item:selected {
                                                            background-color: #81CF3E;
                                                            color: #404040; }
                                                           ''')
            menu.addAction(QtGui.QIcon(), "Set subdivisionScheme : catmullClark",
                           lambda: self.addAttributeForEveryShape(item, "USD_ATTR_subdivisionScheme", "catmullClark"))
            menu.addAction(QtGui.QIcon(), "Set subdivisionScheme : loop",
                           lambda: self.addAttributeForEveryShape(item, "USD_ATTR_subdivisionScheme", "loop"))
            menu.addAction(QtGui.QIcon(), "Set subdivisionScheme : none",
                           lambda: self.addAttributeForEveryShape(item, "USD_ATTR_subdivisionScheme", "none"))

            menu.addAction(QtGui.QIcon(), "Add txmultiUV",
                           lambda: self.addAttributeForEveryShape(item, "txmultiUV", "1"))
            menu.addAction(QtGui.QIcon(), "Delete txmultiUV",
                           lambda: self.deleteAttributeForEveryShape(item, "txmultiUV"))

        elif type(item) is ShapeItem:
            menu = QtWidgets.QMenu(self)
            menu.setStyleSheet('''
                                                    QMenu::item:selected {
                                                    background-color: #81CF3E;
                                                    color: #404040; }
                                                   ''')

            if not cmds.attributeQuery("MaterialSet", n=item.text(0), exists=True):
                menu.addAction(QtGui.QIcon(), "Add %s" % self.attributeList["MaterialSet"]['niceName'], lambda: self.addAttribute(item, "MaterialSet"))

            if not cmds.attributeQuery("USD_ATTR_subdivisionScheme", n=item.text(0), exists=True):
                menu.addAction(QtGui.QIcon(), "Add %s" % self.attributeList["USD_ATTR_subdivisionScheme"]['niceName'], lambda: self.addAttribute(item, "USD_ATTR_subdivisionScheme"))

            if not cmds.attributeQuery("txmultiUV", n=item.text(0), exists=True):
                menu.addAction(QtGui.QIcon(), "Add %s" % self.attributeList["txmultiUV"]['niceName'], lambda: self.addAttribute(item, "txmultiUV"))

        else:
            return

        menu.popup(QtGui.QCursor.pos())

    def deleteAttributeForEveryShape(self, rootItem, longName):
        for childIndex in range(rootItem.childCount()):
            childItem = rootItem.child(childIndex)
            if childItem.alreadySetupAttr[longName]:
                cmds.deleteAttr("%s.%s" % (childItem.shapeName, longName))
                childItem.removeChild(childItem.alreadySetupAttr[longName])
                childItem.alreadySetupAttr[longName] = None

    def addAttributeForEveryShape(self, rootItem, longName, value):
        for childIndex in range(rootItem.childCount()):
            childItem = rootItem.child(childIndex)
            if childItem.alreadySetupAttr[longName]:
                self.setAttributeItem(childItem.alreadySetupAttr[longName], value)
            else:
                self.addAttribute(childItem, longName, value)

    def addAttribute(self, item, attributeName, value = ""):
        if self.attributeList[attributeName].has_key("dataType"):
            cmds.addAttr(item.text(0), ln=self.attributeList[attributeName]["longName"], nn=self.attributeList[attributeName]["niceName"],
                         dt=self.attributeList[attributeName]["dataType"])
        elif self.attributeList[attributeName].has_key("attrType"):
            cmds.addAttr(item.text(0), ln=self.attributeList[attributeName]["longName"], nn=self.attributeList[attributeName]["niceName"],
                         at=self.attributeList[attributeName]["attrType"])

        defaultValue = self.attributeList[attributeName]["default"]
        if value:
            defaultValue = value
        attrItem = AttritubeItem(item, item.text(0), attributeName, defaultValue,
                      self.attributeList[attributeName], False)
        item.alreadySetupAttr[attributeName] = attrItem
        attrItem.attributeUpdate()

    def setAttributeItem(self, item, value = ""):
        item.attrValue = value
        item.attributeUpdate(reload=False)

def main():
    if cmds.window(__windowName__, exists = True):
        cmds.deleteUI(__windowName__)

    window = TxAttrTools()
    # app.setStyle(QtWidgets.QStyleFactory.create("plastique"))
    window.setObjectName(__windowName__)
    window.show()
