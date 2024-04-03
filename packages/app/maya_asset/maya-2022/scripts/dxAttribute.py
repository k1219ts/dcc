from PySide2 import QtWidgets
from PySide2 import QtCore
import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMayaUI as mui
import shiboken2 as shiboken
from dxAttributeUI import Ui_Form
import sys
import dxAssetTool

def getMayaWindow():
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    except:
        return None

class AttributeEdit(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self, getMayaWindow())
        self.setWindowFlags(QtCore.Qt.Window)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.getObjects()

        mayaWindow = getMayaWindow()
        self.move( mayaWindow.frameGeometry().center() - self.frameGeometry().center() )

        attr_list = [ "txBasePath", "txLayerName" ,"txmultiUV" ,"subdivision","set","M", "GRP"]

        materialList = ["bronze", "chrome", "gold", "metal", "silver", "fabric", "glass", "leather", "plastic", "rubber",
                        "feather",
                        "paint", "wood", "leaf", "ice", "ocean", "mineral", "rock", "snow", "skin", "fur", "eye", "layer","light","layerB"]

        for i in materialList:
            eval('self.ui.pushButton_%s.clicked.connect(self.button_clicked)' % i)

        for a in attr_list:
            eval('self.ui.pushButton_%s.clicked.connect(self.attr_clicked)' % a)


        self.ui.pushButton_assign.clicked.connect(self.sets_clicked)


    def getObjects(self):
        if cmds.pluginInfo('ZENNForMaya', q=True, l=True):
            self.objects = cmds.ls(sl=True, dag=True, type=['surfaceShape', 'nurbsCurve', 'ZN_Deform'], ni=True)
        else:
            self.objects = cmds.ls(sl=True, dag=True, type=['surfaceShape', 'nurbsCurve'], ni=True)


    def import_clicked(self):
        self.hyperShadeReset()
        materialImport()

    def button_clicked(self):
        self.getObjects()
        click_btn = self.sender()
        materialName = click_btn.text()
        sgName = materialName + "SG"
        for shape in self.objects:
            if sgName in cmds.ls("*"):
                cmds.sets(shape, forceElement=sgName)
                AddMaterialSetAttributeByShadingGroup()
            else:
                if not cmds.attributeQuery("MaterialSet", n=shape, exists=True):
                    cmds.addAttr(shape, ln="MaterialSet", nn="MaterialSet", dt="string")
                cmds.setAttr("%s.MaterialSet" % shape, materialName, type="string")

    def sets_clicked(self):
        # Add New Material
        self.getObjects()
        cmds.select(clear=True)

        if cmds.objExists('MaterialSet'):
            allSet = cmds.ls(type='objectSet')
            for set in allSet:
                if 'default' in set or 'initial' in set or 'ZN_ExportSet' in set:
                    pass
                else:
                    cmds.delete(set)

        if not cmds.objExists("MaterialSet"):
            cmds.sets(name="MaterialSet")
        # set Member
        for shape in self.objects:
            if not cmds.attributeQuery('MaterialSet', n=shape, exists=True):
                cmds.addAttr(shape, ln='MaterialSet', dt='string')
            getM = cmds.getAttr('%s.MaterialSet' % shape)
            if getM:
                if not cmds.objExists(getM):
                    newMaterialSet = cmds.sets(name=getM)
                    cmds.sets(newMaterialSet, add='MaterialSet')
                cmds.sets(shape, add=getM)

    def attr_clicked(self):
        self.getObjects()
        click_btn = self.sender()
        attrName = click_btn.text()

        if attrName == "GRP":
            value = self.reloadAssetName()
            print value

        comboText = self.ui.comboBox_scaleX.currentText()
        if comboText:
            S = comboText.split("T")[0] + comboText.split("T")[1]
            T = comboText.split("ST")[0] + "T" + comboText.split("scaleST")[-1]

        attributeList = {"MaterialSet": {"dataType": "string",
                                         "niceName": "MaterialSet",
                                         "longName": "MaterialSet"},
                         "subdivision": {"dataType": "string",
                                         "niceName": "subdivisionScheme",
                                         "longName": "USD_ATTR_subdivisionScheme"},
                         "txBasePath": {"dataType": "string",
                                        "niceName": "txBasePath",
                                        "longName": "txBasePath"},
                         "txAssetName": {"dataType": "string",
                                         "niceName": "txBasePath",
                                         "longName": "txBasePath"},
                         "txLayerName": {"dataType": "string",
                                         "niceName": "txLayerName",
                                         "longName": "txLayerName"},
                         "txmultiUV": {"attrType": "long",
                                       "niceName": "txmultiUV",
                                       "longName": "txmultiUV"},
                         S: {"attrType": "float",
                             "niceName": S,
                             "longName": S},

                         T: {"attrType": "float",
                             "niceName":  T,
                             "longName":  T}
                         }

        for node in self.objects:
            if attrName == "M":
                attrName = "MaterialSet"

            if attrName == "GRP":
                attrName = "txBasePath"

            if attrName == "set":
                value = {S: self.ui.lineEdit_valueX.text(),
                         T: self.ui.lineEdit_valueY.text()}

                if value[S] or value[T]:
                    for i in value:
                        if value[S] == "0" or value[T] == "0":
                            if cmds.attributeQuery(attributeList[i]["longName"], n=node, exists=True):
                                cmds.deleteAttr("%s.%s" % (node, attributeList[i]["longName"]))
                        else:
                            # if not comboText == "scaleST":
                            #     if not cmds.attributeQuery("txscaleS", n=node, exists=True):
                            #         print "You need to have base layer's Value"
                            if not cmds.attributeQuery(attributeList[i]["longName"], n=node, exists=True):
                                cmds.addAttr(node, ln=attributeList[i]["longName"], nn=attributeList[i]["niceName"],
                                         at=attributeList[i]["attrType"])
                            cmds.setAttr("%s.%s" % (node, attributeList[i]["longName"]), float(value[i]))
                else:
                    print "Please put number in scaleS or scaleT."


            elif attributeList[attrName].has_key("dataType"):


                if attrName == "subdivision":
                    value = self.ui.comboBox.currentText()
                else:
                    value = eval('self.ui.lineEdit_%s.text()' % attrName)

                #
                if value == "0" or value == "delete":
                    cmds.deleteAttr("%s.%s" % (node, attributeList[attrName]["longName"]))
                else:
                    if not cmds.attributeQuery(attributeList[attrName]["longName"], n=node, exists=True):
                        cmds.addAttr(node, ln=attributeList[attrName]["longName"], nn=attributeList[attrName]["niceName"],
                                     dt=attributeList[attrName]["dataType"])
                    cmds.setAttr("%s.%s" % (node, attributeList[attrName]["longName"]), value,
                                 type=attributeList[attrName]["dataType"])

            elif attrName == "txmultiUV":
                value = eval('self.ui.lineEdit_%s.text()' % attrName)
                if value == "0":
                    if cmds.attributeQuery(attributeList[attrName]["longName"], n=node, exists=True):
                        cmds.deleteAttr("%s.%s" % (node, attributeList[attrName]["longName"]))
                    else:
                        pass

                elif value == "1":
                    if not cmds.attributeQuery(attributeList[attrName]["longName"], n=node, exists=True):
                        cmds.addAttr(node, ln=attributeList[attrName]["longName"], nn=attributeList[attrName]["niceName"],
                                 at=attributeList[attrName]["attrType"])
                    cmds.setAttr("%s.%s" % (node, attributeList[attrName]["longName"]), float(value))
                else:
                    pass
            else:
                pass

    def assign_clicked(self):
        self.hyperShadeReset()
        materialImport()
        findMaterialSetAssign()

    def hyperShadeReset(self):
        objects = cmds.ls(dag=True, type='surfaceShape', ni=True)
        for s in objects:
            cmds.sets(s, forceElement='initialShadingGroup')
        mel.eval('hyperShadePanelMenuCommand("hyperShadePanel1", "deleteUnusedNodes");')

    def reloadAssetName(self):
        selected =cmds.ls(sl=1)[0]
        if selected:
            assetName = selected.split('_model_')[0]
            if '_' in assetName:
                splitName = assetName.split('_')
                basePath = 'asset/%s/branch/%s/texture' % (splitName[0], splitName[-1])
            else:
                basePath = 'asset/%s/texture' % assetName
            self.ui.lineEdit_txBasePath.setText(basePath)
            return basePath
        else:
            pass

#-------------------------------------------------------------------------------
#
# Add MaterialSet Attribute by ShadingGroup
#
#-------------------------------------------------------------------------------
def AddMaterialSetAttributeByShadingGroup():
    selected = cmds.ls(sl=True, dag=True, type='surfaceShape')
    if not selected:
        selected = cmds.ls(dag=True, type='surfaceShape')

    for shape in selected:
        sg = cmds.listConnections(shape, type='shadingEngine')
        if sg:
            if not 'initial' in sg[0]:
                mtlname = cmds.listConnections('%s.surfaceShader' % sg[0])
                if mtlname:
                    if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
                        cmds.addAttr(shape, ln='MaterialSet', dt='string')
                    cmds.setAttr('%s.MaterialSet' % shape, mtlname[0], type='string')

def findMaterialSetAssign():
    materialList = ["bronze", "chrome", "gold", "metal", "silver", "fabric", "glass", "leather", "plastic", "rubber",
                    "feather",
                    "paint", "wood", "leaf", "ice", "ocean", "mineral", "rock", "snow", "skin", "fur", "eye", "tile"]

    selected = cmds.ls(sl=True, dag=True, type='surfaceShape', ni=True)
    if selected:
        objects = selected
    else:
        objects = cmds.ls(dag=True, type='surfaceShape', ni=True)
    for s in objects:
        if cmds.attributeQuery('MaterialSet', n=s, ex=True):
            if cmds.getAttr('%s.MaterialSet'%s) in materialList:
                materialName = cmds.getAttr('%s.MaterialSet' % s)
                cmds.sets(s, forceElement=materialName + "SG")
            else:
                cmds.sets(s, forceElement="error" + "SG")
                # continue
        else:
            cmds.sets(s, forceElement="error" + "SG")

def materialImport():
    dir="/dexter/Cache_DATA/ASSET/1.pipeline/Katana/material"
    filename = os.path.join(dir,"mayaMaterial.ma")
    cmds.file(filename, i=True)

def open_pub():
    import maya.cmds as cmds
    import os
    import subprocess
    try:
        sceneFile = cmds.file(q=True, sn=True)
        show = sceneFile.split('/works')[0]
        assetFile = sceneFile.split('/')[-1]
        if 'groom' in assetFile:
            asset = assetFile.split('_groom')[0]
        elif 'model' in assetFile:
            asset = assetFile.split('_model')[0]
        else:
            pass

        pubpath = os.path.join(show, '_3d', 'asset', asset)
        if '_' in asset:
            splitName = asset.split('_')
            asset = splitName[0]
            branch = splitName[1]
            pubpath = os.path.join(show, '_3d', 'asset', asset, 'branch', branch)

        if os.path.exists(pubpath):
            subprocess.Popen(['xdg-open', str(pubpath)])
        else:
            print 'Failed: Pub directory does not exist. You need to publish this asset.'
    except:
        print 'Failed: You need to make a scene file.'


def main():
    if cmds.window('Form', exists=True, q=True):
        cmds.deleteUI( 'Form' )
    # app = QtWidgets.QApplication(sys.argv)
    mainVar = AttributeEdit()
    mainVar.show()
    # sys.exit(app.exec_())


if __name__ == "__main__":
    main()
