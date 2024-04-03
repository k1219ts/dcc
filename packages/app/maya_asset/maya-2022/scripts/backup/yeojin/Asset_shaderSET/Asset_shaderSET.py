import sys

from shaderSET3 import Ui_Form
import os
#
# from PyQt4 import QtGui
# PySide == PyQt4
# PySide2 == PyQt5
from pymodule.Qt  import QtWidgets
from pymodule.Qt  import QtCore

import maya.cmds as cmds
import sgComponent as sgc
import maya.mel as mel
import maya.OpenMayaUI as mui
import shiboken2 as shiboken


def getMayaWindow():
    try:
        ptr = mui.MQtUtil.mainWindow()
        return shiboken.wrapInstance(long(ptr), QtWidgets.QWidget)
    except:
        return None


class Asset_shaderSET(QtWidgets.QWidget):



    def __init__(self):
        QtWidgets.QWidget.__init__(self, getMayaWindow())
        print "init"

        self.setWindowFlags(QtCore.Qt.Window)

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.lineEdit.returnPressed.connect(self.shader_Name)
        self.ui.comboBox.currentIndexChanged.connect(self.currentIndexChanged)

        self.ui.checkBox.clicked.connect(self.create_diffC)
        self.ui.checkBox_2.clicked.connect(self.create_specG)
        self.ui.checkBox_3.clicked.connect(self.create_specR)
        self.ui.checkBox_4.clicked.connect(self.create_norm)
        self.ui.checkBox_5.clicked.connect(self.create_bump)
        self.ui.checkBox_6.clicked.connect(self.create_disF)
        self.ui.checkBox_7.clicked.connect(self.create_Alpha)
        self.ui.checkBox_8.clicked.connect(self.create_DxManifold2D)
        self.ui.checkBox_9.clicked.connect(self.check_DoubleSided)

        self.ui.pushButton.clicked.connect(self.create_shaderSET)
        self.ui.pushButton_3.clicked.connect(self.create_Node)


    def shader_Name(self):
        assetName = self.ui.lineEdit.text()
        shaderName = assetName + "_SHD"
        print shaderName

    def currentIndexChanged(self,text):
        print self.ui.comboBox.currentText()

    def create_diffC(self):
        print 'diffC checked'
    def create_specG(self):
        print 'specG checked'
    def create_specR(self):
        print 'specR checked'
    def create_norm(self):
        print 'norm checked'
    def create_bump(self):
        print 'bump checked'
    def create_disF(self):
        print 'disF checked'
    def create_Alpha(self):
        print 'Alpha checked'
    def create_DxManifold2D(self):
        print 'DxManifold2D checked'
    def check_DoubleSided(self):
        print 'DoubleSided checked'



    def create_shaderSET(self):
        assetName = self.ui.lineEdit.text()
        shaderName = assetName + "_SHD"
        print shaderName

        # create shader
        cmds.shadingNode('PxrSurface', asShader=True, n=shaderName)
        cmds.createNode('shadingEngine', n=shaderName + 'SG')
        cmds.connectAttr(shaderName + '.outColor', shaderName + 'SG.surfaceShader')
        if self.ui.checkBox_9.isChecked():
            cmds.setAttr('%s.diffuseDoubleSided' % shaderName, 1)

        #create manifold
        if self.ui.checkBox_8.isChecked():
            mnF = cmds.shadingNode('DxManifold2D', asTexture=True, n="DxManifold2D_" + assetName)


        #diffC
        if self.ui.checkBox.isChecked():
            txNodeDiffC = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_diffC")
            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeDiffC)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeDiffC)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeDiffC)

            cmds.select(txNodeDiffC)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeDiffC, "diffC"))

            #create Grade
            gradeDiffC = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_diffC")

            #connect node
            cmds.connectAttr(gradeDiffC + '.resultRGB', shaderName + '.diffuseColor')
            cmds.connectAttr(txNodeDiffC + '.resultRGB', gradeDiffC + '.inputRGB')




        #specG
        if self.ui.checkBox_2.isChecked():

            txNodeSpecG = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_specG")

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeSpecG)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeSpecG)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeSpecG)

            cmds.select(txNodeSpecG)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeSpecG, "specG"))

            # create Grade
            gradeSpecG = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_specG")

            # connect node
            cmds.connectAttr(gradeSpecG + '.resultRGB', shaderName + '.specularFaceColor')
            cmds.connectAttr(txNodeSpecG + '.resultRGB', gradeSpecG + '.inputRGB')

        #specR
        if self.ui.checkBox_3.isChecked():
            txNodeSpecR = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_specR")
            cmds.setAttr("%s.linearize" % txNodeSpecR, 0)

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeSpecR)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeSpecR)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeSpecR)

            cmds.select(txNodeSpecR)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeSpecR, "specR"))

            # create Grade
            gradeSpecR = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_specR")

            # connect node
            cmds.connectAttr(gradeSpecR + '.resultR', shaderName + '.specularRoughness')
            cmds.connectAttr(txNodeSpecR + '.resultRGB', gradeSpecR + '.inputRGB')


        #norm
        if self.ui.checkBox_4.isChecked():
            txNodeNorm = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_norm")
            cmds.setAttr("%s.linearize" % txNodeNorm, 0)


            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeNorm)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeNorm)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeNorm)

            cmds.select(txNodeNorm)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeNorm, "norm"))

            # create node
            normNode = cmds.shadingNode('PxrNormalMap', asTexture=True, n="PxrNormalMap_" + assetName)
            cmds.setAttr("%s.adjustAmount" % normNode, 1)

            # connect node
            cmds.connectAttr(txNodeNorm + '.resultRGB', normNode + '.inputRGB')
            cmds.connectAttr(normNode + '.resultN', shaderName + '.bumpNormal')


        #bump
        if self.ui.checkBox_5.isChecked():
            txNodeBump = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_bump")

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeBump)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeBump)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeBump)

            cmds.select(txNodeBump)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeBump, "bump"))

            # create node
            bumpNode = cmds.shadingNode('PxrBump', asTexture=True, n="PxrBump_" + assetName)
            cmds.setAttr("%s.adjustAmount" % bumpNode, 1)

            # connect node
            cmds.connectAttr('%s.resultR' % txNodeBump, '%s.inputBump' % bumpNode)
            cmds.connectAttr('%s.resultN' % bumpNode, '%s.bumpNormal' % shaderName)


        #disF
        if self.ui.checkBox_6.isChecked():
            displaceNode = cmds.shadingNode('PxrDisplace', asShader=True, n="PxrDisplace_" + assetName)
            dispTransform = cmds.shadingNode('PxrDispTransform', asTexture=True, n="PxrDispTransform_" + assetName)
            txNodedisF = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_disF")
            cmds.setAttr("%s.linearize" % txNodedisF, 0)

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodedisF)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodedisF)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodedisF)


            cmds.select(txNodedisF)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodedisF, "disF"))
            # connect node
            cmds.connectAttr('%s.resultF' % dispTransform, '%s.dispScalar' % displaceNode)
            cmds.connectAttr('%s.resultR' % txNodedisF, '%s.dispScalar' % dispTransform)
            cmds.connectAttr('%s.outColor' % displaceNode, '%s.displacementShader' % (assetName + '_SHDSG'))


        #Alpha
        if self.ui.checkBox_7.isChecked():
            txNodeAlpha = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_Alpha")
            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeAlpha)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeAlpha)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' %txNodeAlpha)

            cmds.select(txNodeAlpha)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeAlpha, "Alpha"))

            # connect node
            cmds.connectAttr(txNodeAlpha + '.resultRGBR', shaderName + '.presence')


    def create_Node(self):
        assetName = self.ui.lineEdit.text()
        #create manifold
        if self.ui.checkBox_8.isChecked():
            mnF = cmds.shadingNode('DxManifold2D', asTexture=True, n="DxManifold2D_" + assetName)

        #diffC
        if self.ui.checkBox.isChecked():
            txNodeDiffC = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_diffC")
            print txNodeDiffC
            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeDiffC)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeDiffC)

            if self.ui.checkBox_8.isChecked():
                print 'yes'
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeDiffC)

            cmds.select(txNodeDiffC)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeDiffC, "diffC"))
            #create Grade
            gradeDiffC = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_diffC")
            #connect node
            cmds.connectAttr(txNodeDiffC + '.resultRGB', gradeDiffC + '.inputRGB')



        #specG
        if self.ui.checkBox_2.isChecked():

            txNodeSpecG = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_specG")

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeSpecG)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeSpecG)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeSpecG)

            cmds.select(txNodeSpecG)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeSpecG, "specG"))

            # create Grade
            gradeSpecG = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_specG")

            # connect node

            cmds.connectAttr(txNodeSpecG + '.resultRGB', gradeSpecG + '.inputRGB')

        #specR
        if self.ui.checkBox_3.isChecked():
            txNodeSpecR = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_specR")
            cmds.setAttr("%s.linearize" % txNodeSpecR, 0)

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeSpecR)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeSpecR)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeSpecR)

            cmds.select(txNodeSpecR)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeSpecR, "specR"))

            # create Grade
            gradeSpecR = cmds.shadingNode('DxGrade', asTexture=True, n="DxGrade_" + assetName + "_specR")

            # connect node

            cmds.connectAttr(txNodeSpecR + '.resultRGB', gradeSpecR + '.inputRGB')


        #norm
        if self.ui.checkBox_4.isChecked():
            txNodeNorm = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_norm")
            cmds.setAttr("%s.linearize" % txNodeNorm, 0)


            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeNorm)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeNorm)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeNorm)

            cmds.select(txNodeNorm)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeNorm, "norm"))

            # create node
            normNode = cmds.shadingNode('PxrNormalMap', asTexture=True, n="PxrNormalMap_" + assetName)
            cmds.setAttr("%s.adjustAmount" % normNode, 1)

            # connect node
            cmds.connectAttr(txNodeNorm + '.resultRGB', normNode + '.inputRGB')



        #bump
        if self.ui.checkBox_5.isChecked():
            txNodeBump = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_bump")

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeBump)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeBump)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodeBump)

            cmds.select(txNodeBump)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeBump, "bump"))

            # create node
            bumpNode = cmds.shadingNode('PxrBump', asTexture=True, n="PxrBump_" + assetName)
            cmds.setAttr("%s.adjustAmount" % bumpNode, 1)

            # connect node
            cmds.connectAttr('%s.resultR' % txNodeBump, '%s.inputBump' % bumpNode)



        #disF
        if self.ui.checkBox_6.isChecked():
            displaceNode = cmds.shadingNode('PxrDisplace', asShader=True, n="PxrDisplace_" + assetName)
            dispTransform = cmds.shadingNode('PxrDispTransform', asTexture=True, n="PxrDispTransform_" + assetName)
            txNodedisF = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_disF")
            cmds.setAttr("%s.linearize" % txNodedisF, 0)

            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodedisF)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodedisF)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' % txNodedisF)


            cmds.select(txNodedisF)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodedisF, "disF"))
            # connect node
            cmds.connectAttr('%s.resultF' % dispTransform, '%s.dispScalar' % displaceNode)
            cmds.connectAttr('%s.resultR' % txNodedisF, '%s.dispScalar' % dispTransform)


        #Alpha
        if self.ui.checkBox_7.isChecked():
            txNodeAlpha = cmds.shadingNode('DxTexture', asTexture=True, n="DxTexture_" + assetName + "_Alpha")
            if self.ui.comboBox.currentText() == 'File':
                cmds.select(txNodeAlpha)
                mel.eval('rmanSetAttr("%s","txmode","0");' % txNodeAlpha)

            if self.ui.checkBox_8.isChecked():
                cmds.connectAttr('%s.result' % mnF, '%s.manifold' %txNodeAlpha)

            cmds.select(txNodeAlpha)
            mel.eval('setAttr -type "string" %s.txchannel "%s";' % (txNodeAlpha, "Alpha"))

            # connect node






def main():
    # app = QtWidgets.QApplication(sys.argv)
    mainVar = Asset_shaderSET()
    mainVar.show()
    # sys.exit(app.exec_())

if __name__ == "__main__":
    main()
