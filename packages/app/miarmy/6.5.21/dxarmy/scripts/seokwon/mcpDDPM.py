# encoding:utf-8

import maya.cmds as cmds
import os, site
import utils
import dxUI
from HUD import HUDmodules;reload(HUDmodules)

from Qt import QtCore, QtGui, QtWidgets, load_ui

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
site.addsitedir('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "./ui/ribPub.ui")

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        dxUI.setup_ui(uiFile, self)
        self.connectSignal()
        self.mayafile_tmp = ''

    def connectSignal(self):
        self.pubBtn.clicked.connect(self.publish)
        self.scFileBtn.clicked.connect(self.mayafileImport)
        self.pubFileBtn.clicked.connect(self.mayafileExport)
        self.brws.clicked.connect(self.trct)

    def mayafileImport(self):
        multipleFilter = "Maya Files (*.mb *.ma);;Maya ASCII (*.ma);;Maya Binary (*.mb)"
        inputFile = str(cmds.fileDialog2(fileMode=1, fileFilter=multipleFilter, caption="Import Maya Scene File")[0])
        self.scFile.setText(inputFile)
        defaultPath = os.sep.join(inputFile.split(os.sep)[:-2]) + "/data/"
        self.pubFile.setText(defaultPath)

    def mayafileExport(self):
        if len(self.pubFile.text()) == 0:
            outPath = "/"
        else:
            outPath = self.pubFile.text()
        animDir = str(cmds.fileDialog2(fileMode=3, fileFilter="outputPath", dir=outPath, caption="Output Path")[0])
        self.pubFile.setText(animDir)

    def resetHUD(self):
        hudList = ["MCP_Char", "MCP_Duration", "CRD_Action", "CRD_Path", "CRD_Type", "CRD_Blend_Cycle", "CRD_Blend_Entry", "CRD_Blend_Exit"]
        for i in hudList:
            if cmds.headsUpDisplay(i, exists=1):
                cmds.headsUpDisplay(i, rem=1)

    def createOpVar(self):
        opName = ['CRD_action_name', 'CRD_path', 'CRD_type', 'CRD_blend_cycle', 'CRD_blend_entry', 'CRD_blend_exit']
        for i in opName:
            if not cmds.optionVar(ex=i):
                cmds.optionVar(sv=(i, ''))

    def getDuration(self):
        minT = cmds.playbackOptions(q=True, min=True)
        maxT = cmds.playbackOptions(q=True, max=True)
        absTime = int(maxT - minT + 1)
        return absTime

    def hudSet(self):
        charName = str(cmds.file(q=True, sn=True)).split(os.sep)[-1].split(".")[0]
        animFile = str(cmds.fileDialog2(fileMode=1, fileFilter="Anim Files (*.anim)", caption="Import Anim File")[0])


        cmds.optionVar(sv=('CRD_action_name', self.action_line))
        cmds.optionVar(sv=('CRD_path', self.lineEditC.text()))
        cmds.optionVar(sv=('CRD_type', self.type_line))
        cmds.optionVar(sv=('CRD_blend_cycle', self.action_cycle))
        cmds.optionVar(sv=('CRD_blend_entry', self.entry_line))
        cmds.optionVar(sv=('CRD_blend_exit', self.exit_line))

        if cmds.headsUpDisplay('CRD_Char', exists=1):
            cmds.headsUpDisplay('CRD_Char', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Char', l="Character  ", allowOverlap=1, dataFontSize="large", b=1, s=5,
                                lfs="large", bs="small", command=("'%s" % charName))

        if cmds.headsUpDisplay('CRD_Duration', exists=1):
            cmds.headsUpDisplay('CRD_Duration', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Duration', l="Duration  ", allowOverlap=1, dataFontSize="large", b=0, s=6,
                                lfs="large", bs="small", atr=True, command="'%s" % self.getDuration())

        if cmds.headsUpDisplay('CRD_Action', exists=1):
            cmds.headsUpDisplay('CRD_Action', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Action', l="Action Name  ", allowOverlap=1, dataFontSize="large", b=1, s=3,
                                lfs="large", bs="small", atr=True, command="'%s" % animFile)

        if cmds.headsUpDisplay('CRD_Path', exists=1):
            cmds.headsUpDisplay('CRD_Path', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Path', l="Path  ", allowOverlap=1, dataFontSize="large", b=4, s=1, lfs="large",
                                bs="small", atr=True, command="cmds.optionVar(q='CRD_path')")

        if cmds.headsUpDisplay('CRD_Type', exists=1):
            cmds.headsUpDisplay('CRD_Type', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Type', l="Type  ", allowOverlap=1, dataFontSize="large", b=5, s=1, lfs="large",
                                bs="small", atr=True, command="cmds.optionVar(q='CRD_type')")

        if cmds.headsUpDisplay('CRD_Blend_Cycle', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Cycle', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Cycle', l="Blend Cycle  ", allowOverlap=1, dataFontSize="large", b=6, s=1,
                                lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_cycle')")

        if cmds.headsUpDisplay('CRD_Blend_Entry', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Entry', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Entry', l="Blend Entry  ", allowOverlap=1, dataFontSize="large", b=7, s=1,
                                lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_entry')")

        if cmds.headsUpDisplay('CRD_Blend_Exit', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Exit', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Exit', l="Blend Exit  ", allowOverlap=1, dataFontSize="large", b=8, s=1,
                                lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_exit')")

    def getOptionsFromUi(self):
        options = dict()
        options['artist'] = str("Artist")
        options['scene'] = str()
        options['depart'] = str()
        options['status'] = str()
        options['progress'] = str(self.progress_comboBox.currentText())
        options['auto_hud'] = str(self.autoHud_checkBox.isChecked())
        options['codec'] = str(self.codec_comboBox.currentText())
        if self.prvSizeW_lineEdit.text():
            options['width'] = int(self.prvSizeW_lineEdit.text())
        else:
            options['width'] = 0
        if self.prvSizeH_lineEdit.text():
            options['height'] = int(self.prvSizeH_lineEdit.text())
        else:
            options['height'] = 0
        options['wh'] = [options['width'], options['height']]
        options['seqTime'] = self.camSeq_checkBox.isChecked()
        options['compRetime'] = self.addCompRetime_checkBox.isChecked()
        options['startTime'] = int(float(self.startFrame_lineEdit.text()))
        options['endTime'] = int(float(self.endFrame_lineEdit.text()))
        options['sizePreset'] = str(self.prvSizePreset_comboBox.currentText())
        options['project'] = str(self.projectList_comboBox.currentText())
        options['remove_sequence'] = self.rmSeq_checkBox.isChecked()
        options['offScreen'] = self.offScreen_checkBox.isChecked()
        options['camera_sequencer'] = self.camSeq_checkBox.isChecked()
        options['addstemp'] = self.addStamp_checkBox.isChecked()
        options['addCompRetime'] = self.addCompRetime_checkBox.isChecked()
        options['showImgeplane'] = self.showImageplane_checkBox.isChecked()
        options['outpath'] = str(self.output_lineEdit.text())
        try:
            options['seqStartTime'] = int(float(self.seqStartFrame_lineEdit.text()))
            options['seqEndTime'] = int(float(self.seqEndFrame_lineEdit.text()))
        except Exception as e:
            logger.debug('[Exception] : {0}'.format(e))
        return options


    def make(self, type):
        fromWindow = False
        addStamp = False
        options = self.getOptionsFromUi()
        options.update({'script_path': CURRENT_DIR})
        options.update({'mayafile': str(cmds.file(q=True, sn=True))})
        options.update({'fps': int(mel.eval('currentTimeUnitToFPS'))})
        options.update({'metadata': utils.createMetadataString(options['mayafile'], options['artist'])})
        if self.autoHud_checkBox.isChecked():
            self.createHud()
        if self.prvSizePreset_comboBox.currentText() == 'From Window':
            fromWindow = True
        if self.addStamp_checkBox.isChecked():
            addStamp = True

        movClass = mayaMov.MakeMovie()
        movClass.options = options
        movClass.playblast(
            parent=self,
            fromWindow=fromWindow,
            type=type,
            addstamp=addStamp
        )
        if self.autoHud_checkBox.isChecked():
            HUDmodules.mg_removeHUD()

    def createPath(self):
        filePath = self.pubFile.text()
        if not os.path.exists(os.path.dirname(filePath)):
            os.makedirs(os.path.dirname(filePath))

def main():
    global myWindow
    try:
        myWindow.close()
    except:
        pass
    myWindow = Window()
    myWindow.show()

if __name__ == '__main__':
    main()






