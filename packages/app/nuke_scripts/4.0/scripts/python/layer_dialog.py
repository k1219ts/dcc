# -*- coding: utf-8 -*-
import sys, os, nuke

from PySide2 import QtWidgets, QtCore

import  ui_layer_dialog
# reload(ui_layer_dialog)

class LayerDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = ui_layer_dialog.Ui_Dialog()
        self.ui.setupUi(self)
        self.resize(1100, 513)

        fullPath = nuke.value('root.name')
        if fullPath.startswith('/netapp/dexter'):
            fullPath = fullPath.replace('/netapp/dexter', '')

        steps = fullPath.split( os.path.sep )
        self.project = steps[2]
        print(self.project)

        self.scriptName = os.path.basename(nuke.root().name())
        self.startPath = nuke.root().name().split('/script/')[0]

        self.seq = self.scriptName.split('_')[0]
        self.shotName = '_'.join(self.scriptName.split('_')[:2])
        self.scriptVersion = self.scriptName.split('_')[3].split('.')[0]

        self.ui.pushButton.setVisible(False)
        #------------------------------------------------------------------------------
        self.ui.gridLayout.setSpacing(0)
        self.ui.treeWidget.setColumnCount(5)
        self.ui.treeWidget.headerItem().setText(0, 'Type')
        self.ui.treeWidget.headerItem().setText(1, 'Layer Number')
        self.ui.treeWidget.headerItem().setText(2, 'Layer Type')
        self.ui.treeWidget.headerItem().setText(3, 'Version')
        self.ui.treeWidget.headerItem().setText(4, 'File Path')
        self.ui.treeWidget.header().resizeSection(0, 130)
        self.ui.treeWidget.header().resizeSection(1, 120)
        #------------------------------------------------------------------------------
        self.ui.label_2.setStyleSheet("""
        QLabel { color : orange; font : 14pt;};
        """
        #QTreeView {color: rgb(180,180,180); background: rgb(50,50,50); border: none;}
        )
        #------------------------------------------------------------------------------
        self.widgetFont = self.font()
        self.widgetFont.setPointSize(11)
        self.setFont(self.widgetFont)

        if self.project == 'log':
            for i in range(3):
                item = WriteItem(self.ui.treeWidget, self.project, self.shotName, self.scriptVersion)
                item.typeComboBox.setCurrentIndex(i)
                item.setText(0, 'Layer Item')
                item.refreshPath()


        self.ui.spinBox.valueChanged.connect(self.createWriteItem)
        self.ui.pushButton_2.clicked.connect(self.createWriteNode)
        self.ui.spinBox.setValue(3)

    def createWriteNode(self):
        firstPos = None
        dot = nuke.createNode('Dot')
        dot.setYpos(dot.ypos() +  50)
        if self.project == 'mkk2':
            for i in range(self.ui.treeWidget.topLevelItemCount()):
                writeItem = self.ui.treeWidget.topLevelItem(i)
                fPath = self.startPath + '/' + writeItem.text(4)
                refNode = nuke.createNode('Reformat')
                refNode.setInput(0, dot)

                wNode = nuke.createNode('Write')
                wNode['channels'].setValue('rgba')
                wNode['file'].setValue(fPath)
                wNode['file_type'].setValue('exr')
                wNode['colorspace'].setValue('cineon2')
                wNode['compression'].setValue('none')

                wNode.setInput(0, refNode)
                if firstPos:
                    refNode.setXYpos(firstPos[0]+100, firstPos[1])
                    wNode.setXYpos(firstPos[0]+100, firstPos[1]+50)
                    firstPos[0] += 100

                else:
                    firstPos = [refNode.xpos(), refNode.ypos()]
                    wNode.setXYpos(firstPos[0], firstPos[1]+50)

        elif self.project == 'log':
            for i in range(self.ui.treeWidget.topLevelItemCount()):
                writeItem = self.ui.treeWidget.topLevelItem(i)
                fPath = self.startPath + '/' + writeItem.text(4)
                if writeItem.getTypeText() in ['plate', 'comp']:
                #if writeItem.getTypeText() == 'plate':
                    rem = nuke.createNode('Remove')
                    rem.setInput(0, dot)
                    rem['operation'].setValue('keep')
                    rem['channels'].setValue('rgb')

                    wNode = nuke.createNode('Write')
                    wNode['channels'].setValue('rgb')
                    wNode['file'].setValue(fPath)
                    wNode['file_type'].setValue('exr')
                    wNode['colorspace'].setValue('linear')
                    wNode['datatype'].setValue("16 bit half")
                    wNode['compression'].setValue("none")
                    wNode['metadata'].setValue("all metadata")
                    wNode['autocrop'].setValue(True)

                    wNode.setInput(0, rem)
                    if firstPos:
                        rem.setXYpos(firstPos[0]+100, firstPos[1])
                        wNode.setXYpos(firstPos[0]+100, firstPos[1]+50)
                        firstPos[0] += 100

                    else:
                        firstPos = [rem.xpos(), rem.ypos()]
                        wNode.setXYpos(firstPos[0], firstPos[1]+50)

                elif writeItem.getTypeText() == 'alphaFG':
                    sh = nuke.createNode('Shuffle')
                    sh['red'].setValue(4)
                    sh['green'].setValue(4)
                    sh['blue'].setValue(4)
                    sh['alpha'].setValue(4)
                    sh.setInput(0, dot)

                    rem = nuke.createNode('Remove')
                    rem.setInput(0, sh)
                    rem['operation'].setValue('keep')
                    rem['channels'].setValue('rgb')

                    wNode = nuke.createNode('Write')
                    wNode['channels'].setValue('rgb')
                    wNode['file'].setValue(fPath)
                    wNode['file_type'].setValue('exr')
                    wNode['colorspace'].setValue('linear')
                    wNode['datatype'].setValue("16 bit half")
                    wNode['compression'].setValue("none")
                    wNode['metadata'].setValue("all metadata")
                    wNode['autocrop'].setValue(True)

                    wNode.setInput(0, rem)
                    if firstPos:
                        sh.setXYpos(firstPos[0]+100, firstPos[1])
                        rem.setXYpos(firstPos[0]+100, firstPos[1]+50)
                        wNode.setXYpos(firstPos[0]+100, firstPos[1]+100)
                        firstPos[0] += 100

                    else:
                        firstPos = [sh.xpos(), sh.ypos()]
                        rem.setXYpos(firstPos[0]+100, firstPos[1]+50)
                        wNode.setXYpos(firstPos[0]+100, firstPos[1]+100)

                elif writeItem.getTypeText() == 'comp':
                    pass


        elif self.project == 'ssy':
            for i in range(self.ui.treeWidget.topLevelItemCount()):
                writeItem = self.ui.treeWidget.topLevelItem(i)
                fPath = self.startPath + '/' + writeItem.text(4)
                wNode = nuke.createNode('Write', inpanel=True)
                wNode.setInput(0, dot)
                wNode.knob('file').setValue(fPath)
                wNode.knob('file_type').setValue("exr")
                wNode.knob('channels').setValue('rgba')
                wNode.knob('colorspace').setValue("sRGB")
                wNode.knob('tile_color').setValue(4278190335) # set red tile
#                wNode.knob('file_type').setValue("dpx")
#                wNode.knob('datatype').setValue("10 bit")
#                wNode.knob('channels').setValue('rgba')
#                wNode.knob('colorspace').setValue("sRGB")
#                wNode.knob('tile_color').setValue(4278190335) # set red tile

                if nuke.root()['views'].toScript().startswith('main'):
                    pass
                else:
                    wNode.knob('views').setValue('left')

                if firstPos:
                    wNode.setXYpos(firstPos[0]+100, firstPos[1]+50)
                    firstPos[0] += 100

                else:
                    firstPos = [dot.xpos(), dot.ypos()]
                    wNode.setXYpos(firstPos[0], firstPos[1]+50)

        elif self.project == 'vgd':
            for i in range(self.ui.treeWidget.topLevelItemCount()):
                writeItem = self.ui.treeWidget.topLevelItem(i)
                fPath = self.startPath + '/' + writeItem.text(4)
                wNode = nuke.createNode('Write', inpanel=True)
                wNode.setInput(0, dot)
                wNode.knob('file').setValue(fPath)
                wNode.knob('file_type').setValue("exr")
                wNode.knob('channels').setValue('rgba')
                wNode.knob('colorspace').setValue("ACES/ACES - ACEScg")
                wNode.knob('tile_color').setValue(4278190335) # set red tile

                if nuke.root()['views'].toScript().startswith('main'):
                    pass
                else:
                    wNode.knob('views').setValue('left')

                if firstPos:
                    wNode.setXYpos(firstPos[0]+100, firstPos[1]+50)
                    firstPos[0] += 100

                else:
                    firstPos = [dot.xpos(), dot.ypos()]
                    wNode.setXYpos(firstPos[0], firstPos[1]+50)

        self.close()


    def createWriteItem(self, number):
        itemCount = self.ui.treeWidget.topLevelItemCount()

        if itemCount < number:
            for i in range(number-itemCount):
                item = WriteItem(self.ui.treeWidget, self.project, self.shotName, self.scriptVersion)
                print(self.scriptVersion)
                item.setText(0, 'Layer Item')
                item.refreshPath()

        elif itemCount >= number:

            for i in reversed(range(number, itemCount+1)):
                self.ui.treeWidget.takeTopLevelItem(i)


class WriteItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None, project=None, shotname=None, version=None):
        super(WriteItem, self).__init__(parent)

        self.project = project
        if self.project == 'mkk2':
            self.typeList=['bg','fg','fx','snow', 'depth']
        elif self.project == 'log':
            self.typeList=['plate','alphaFG','comp']

        else:
            self.typeList=['fg', 'bg','fx']

        self.shotname = shotname
        self.layerNumber = 1
        self.layerType = 'bg'

        #self.version = 'v01'
        self.version = version
        #self.scVersion = version[0].capitalize() + str(int(version[1:]))
        self.scVersion = version

        self.numberSpinBox = QtWidgets.QSpinBox()
        self.numberSpinBox.setMinimum(1)
        self.treeWidget().setItemWidget( self, 1, self.numberSpinBox)
        self.numberSpinBox.setValue((self.treeWidget().topLevelItemCount()))

        self.typeComboBox = QtWidgets.QComboBox()
        self.typeComboBox.addItems(self.typeList)
        self.treeWidget().setItemWidget( self, 2, self.typeComboBox)

        self.versionEdit = QtWidgets.QLineEdit()
        #self.versionEdit.setText(self.version)
        self.versionEdit.setText(self.scVersion)

        self.treeWidget().setItemWidget( self, 3, self.versionEdit)

        self.numberSpinBox.valueChanged.connect(self.refreshNumber)
        self.typeComboBox.currentIndexChanged.connect(self.refreshType)
        self.versionEdit.textEdited.connect(self.refreshVersion)

    def refreshVersion(self, version):
        self.version = version
        self.refreshPath()

    def refreshNumber(self, number):
        self.layerNumber = number
        self.refreshPath()

    def refreshType(self, index):
        self.layerType = self.typeComboBox.itemText(index)
        self.refreshPath()

    def refreshPath(self):
        if self.project == 'mkk2':
            fd = 'MK2_%s_%s_layer%d_%s_%s' % (self.shotname, self.scVersion,
                                              self.getLayerNumber(), str(self.getTypeText()),
                                              self.version
                                              )
            layerPath = 'layer/%s/%s.####.exr' % (fd,fd)

        elif self.project == 'log':
            fd = '%s_%s_%s' % (self.shotname, str(self.getTypeText()), self.version)
            layerPath = 'layer/%s/%s.####.exr' % (fd,fd)

        elif self.project == 'ssy':
            vnum = int(self.scVersion[1:])

            pfd = "%s_%s" % (self.shotname, str(vnum).zfill(3))
            fd = '%s_%s_%s' % (self.shotname, str(self.getTypeText()) + str(self.getLayerNumber()), 'v' + (str(vnum).zfill(2)))
            #layerPath = 'layer/%s/%s.####.dpx' % (fd,fd)
            layerPath = 'layer/%s/%s/%s.####.exr' % (pfd, fd,fd)
        else:
            fd = '%s_%s_%s' % (self.shotname, str(self.getTypeText()), self.version)
            #fd = '%s_%s_%s' % (self.shotname, str(self.getTypeText()), self.scVersion)
            layerPath = 'layer/%s/%s.####.exr' % (fd,fd)


        self.setText(4, layerPath)

    def getLayerNumber(self):
        return self.numberSpinBox.value()

    def getTypeText(self):
        return self.typeComboBox.currentText()
