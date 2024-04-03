# encoding:utf-8
# !/usr/bin/env python

import maya.cmds as cmds
import shutil
import McdActionFunctions
import McdPlacementFunctions
import McdActionEditorGUI
import sys
import os

scrPath = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Scripts'
if not scrPath in sys.path:
    sys.path.append(scrPath)
import McdSaveActionF

from Qt import QtCore, QtGui, QtWidgets, load_ui

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/mcdAction.ui")

def hconv(text):
    return unicode(text, 'utf-8')

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent = None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.connectSignal()
        self.num = 1
        self.ui.movPath.setText("/dexter/Cache_DATA/CRD/Action/actionMov.mov")

    def connectSignal(self):

        self.ui.btnSet.clicked.connect(self.sceneOpen)  # 버튼 등록 및 실행
        self.ui.btnAct.clicked.connect(self.crtAct)
        self.ui.btnSav.clicked.connect(self.savAct)
        self.ui.btnTls.clicked.connect(self.timeSet)
        self.ui.btnAnim.clicked.connect(self.nextanFile)
        self.ui.btnPrv.clicked.connect(self.setPrv)
        self.ui.btnHud.clicked.connect(self.hudSet)
        self.ui.btnAce.clicked.connect(self.actionEdit)
        self.ui.btnCrw.clicked.connect(self.openCrw)
        self.ui.btnMov.clicked.connect(self.brwMov)
        self.ui.btnPb.clicked.connect(self.playBlst)
        self.ui.revBtn.clicked.connect(self.animReverse)

    def actionEdit(self):
        McdActionEditorGUI.McdActionEditorGUI()

    def timeSet(self):
        self.tlm = cmds.keyframe("Crw_Hips", q=True, lastSelected=True, timeChange=True)
        cmds.playbackOptions(minTime=0, maxTime=self.tlm[0])
        cmds.currentTime(0)
        cmds.select("Crw_Hips")
        if not cmds.toggleAxis(q=True, o=True) == 1:
            cmds.toggleAxis(o=True)

    def setKey(self):
        cmds.nameCommand('minLine', annotation='Set min', command='float $minT = `currentTime -q`;playbackOptions -minTime $minT -ast $minT;')
        cmds.hotkey(k='[', n='minLine')
        cmds.nameCommand('maxLine', annotation='Set max', command='float $maxT = `currentTime -q`;playbackOptions -maxTime $maxT -aet $maxT;')
        cmds.hotkey(k=']', n='maxLine')

    def resKey(self):
        cmds.nameCommand('undoVp', annotation='undoV', command='{  string $currentPanel = `getPanel -withFocus`;   string $hyperGraphEditor = $currentPanel + "HyperGraphEd";   if (`hyperGraph -ex $hyperGraphEditor`) {      HyperGraphPanelUndoViewChange;  } else { 	  ModelingPanelUndoViewChange;   }}')
        cmds.hotkey(k='[', n='undoVp')
        cmds.nameCommand('redoVp', annotation='redoV', command='{  string $currentPanel = `getPanel -withFocus`;   string $hyperGraphEditor = $currentPanel + "HyperGraphEd";   if (`hyperGraph -ex $hyperGraphEditor`) {      HyperGraphPanelRedoViewChange;  } else { 	 ModelingPanelRedoViewChange;  }}')
        cmds.hotkey(k=']', n='redoVp')

    def animReverse(self):
        jointList = list()
        for i in cmds.ls("Crw_*", type="joint"):
            if str(i).count("_loco") != 1:
                jointList.append(str(i))
            else:
                pass
        minTime = cmds.playbackOptions(q=True, min=True)
        maxTime = cmds.playbackOptions(q=True, max=True)
        rAtts = ["rotateX", "rotateY", "rotateZ"]
        jDct = dict()
        for j in jointList:
            jDct[j] = dict()
            for k in rAtts:
                jDct[j][k] = dict()
                for frm in range(int(minTime), int(maxTime + 1)):
                    jDct[j][k][frm] = cmds.getAttr(j + "." + k, t=frm)
        jDct["Crw_Hips"]["translateX"] = dict()
        for jtr in range(int(minTime), int(maxTime + 1)):
            jDct["Crw_Hips"]["translateX"][jtr] = cmds.getAttr("Crw_Hips.translateX", t=jtr)

        for z in jointList:
            for n in rAtts:
                for m in range(int(minTime), int(maxTime + 1)):
                    if z.count("RightHandSub") == 1:
                        cmds.setKeyframe(z.replace("RightHand", "Lefthand"), at=n, t=(m, m), v=jDct[z][n][m])
                    elif z.count("LefthandSub") == 1:
                        cmds.setKeyframe(z.replace("Lefthand", "RightHand"), at=n, t=(m, m), v=jDct[z][n][m])
                    elif z.count("Right") == 1:
                        cmds.setKeyframe(z.replace("Right", "Left"), at=n, t=(m, m), v=jDct[z][n][m])
                    elif z.count("Left") == 1:
                        cmds.setKeyframe(z.replace("Left", "Right"), at=n, t=(m, m), v=jDct[z][n][m])
                    else:
                        if n.count("rotateX") == 1 or n.count("rotateY") == 1:
                            cmds.setKeyframe(z, at=n, t=(m, m), v=-jDct[z][n][m])
                        else:
                            cmds.setKeyframe(z, at=n, t=(m, m), v=jDct[z][n][m])
        for jtrs in range(int(minTime), int(maxTime + 1)):
            if jtrs == int(minTime):
                pass
            else:
                trx = -jDct["Crw_Hips"]["translateX"][jtrs] - abs(jDct["Crw_Hips"]["translateX"][int(minTime)])
                cmds.setKeyframe("Crw_Hips", at="translateX", t=(jtrs, jtrs), v=trx)

    def nextanFile(self):

        self.resetSc()
        self.setKey()

        allAgentShapes = cmds.ls(type="McdAgent")
        if allAgentShapes != [] and allAgentShapes != None:
            McdPlacementFunctions.dePlacementAgent()

        cmds.select("Crw_Hips", hierarchy=True)
        cmds.selectKey(keyframe=True)
        cmds.currentTime(0)
        cmds.delete(all=True, c=True)
        cmds.select(cl=True)
        cmds.select("Crw_Hips")

        animPlug = '/usr/autodesk/maya2016.5/bin/plug-ins/animImportExport.so'

        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            cmds.loadPlugin(animPlug)
            cmds.pluginInfo(animPlug, edit=True, autoload=True)

        self.animDir = cmds.fileDialog2(fileMode=1, fileFilter="animImport (*.anim)", caption="Import Anim File")
        cmds.file(self.animDir[0], i=True)

        self.showAn = str(self.animDir[0].split(os.sep)[-1]).split(".")[0]
        self.ui.lineEditA.setText(self.showAn)

    def resetSc(self):

        if cmds.headsUpDisplay('CRD_Frame', exists=1):
            cmds.headsUpDisplay('CRD_Frame', rem=1)

        if cmds.headsUpDisplay('CRD_Duration', exists=1):
            cmds.headsUpDisplay('CRD_Duration', rem=1)

        if cmds.headsUpDisplay('CRD_Action', exists=1):
            cmds.headsUpDisplay('CRD_Action', rem=1)

        if cmds.headsUpDisplay('CRD_Path', exists=1):
            cmds.headsUpDisplay('CRD_Path', rem=1)

        if cmds.headsUpDisplay('CRD_Type', exists=1):
            cmds.headsUpDisplay('CRD_Type', rem=1)

        if cmds.headsUpDisplay('CRD_Blend_Cycle', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Cycle', rem=1)

        if cmds.headsUpDisplay('CRD_Blend_Entry', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Entry', rem=1)

        if cmds.headsUpDisplay('CRD_Blend_Exit', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Exit', rem=1)

    def sceneOpen(self):

        self.resetSc()

        if os.path.exists('/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/0HumanType01_actionMaking0.ma'):
            mayafile = '/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/0HumanType01_actionMaking0.ma'
            if os.path.exists('/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/HumanType01_Action.ma'):
                os.remove('/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/HumanType01_Action.ma')
            else:
                copyFile = '/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/HumanType01_Action.ma'
                shutil.copy2(mayafile, copyFile)
                cmds.file(copyFile, iv=True, f=True, o=True)
        else:
            cmds.warning("Please check your scene file location for making action.")

    def crtAct(self):
        McdActionFunctions.McdCreateActionCmd()

    def savAct(self):
        stF = int(cmds.playbackOptions(q=True, minTime=True))
        etF = int(cmds.playbackOptions(q=True, maxTime=True))

        fcdAction = str(cmds.ls(sl=True)[0])
        cmds.setAttr(fcdAction + ".matchName", True)
        self.animDir = cmds.fileDialog2(fileMode=0, caption="Save McdAction File", fileFilter="Maya Ascii (*.ma)")
        McdSaveActionF.McdSaveAction(self.animDir[0])  # Save Action
        sactPath = os.sep.join(self.animDir[0].split(os.sep)[:-1])
        self.addFr = "_" + str(stF) + "_" + str(etF)
        sactFile = self.animDir[0].split(os.sep)[-1].split(".")[0] + self.addFr + ".mov"
        nName = sactPath + "/" + sactFile
        self.ui.movPath.setText(nName)
        cmds.select(fcdAction)

    def openCrw(self):

        pbDir = os.sep.join(self.ui.movPath.text().split(os.sep)[:-1])
        os.system('/usr/bin/nautilus %s &' % pbDir)


    def setPrv(self):

        self.resKey()

        self.sel = cmds.ls(long=True, selection=True, type='dagNode')

        if cmds.nodeType(self.sel) == "McdAction":
            miTm = cmds.playbackOptions(q=1, minTime=True)
            maTm = cmds.playbackOptions(q=1, maxTime=True)
            self.acTime = maTm - miTm + 2

            cdAction = str(cmds.ls(sl=1)[0]).split("_action_")[0]

            loN = "*_decision_" + str(cmds.getAttr("McdGlobal1.activeAgentName"))

            if not cmds.ls(loN, dag=True, type="McdDecision"):
                cmds.warning("Please create default Logic")
            else:
                defNode = cmds.ls(loN, dag=True, type="McdDecision")
                if not (cmds.getAttr(str(defNode[0]) + ".default") == True):
                    cmds.setAttr(str(defNode[0]) + ".default", True)
                cmds.setAttr(str(defNode[0]) + ".defaultAction", cdAction, type="string")
                cmds.setAttr('McdGlobal1.smpTrans', True)
            cmds.setAttr("McdBrain1.startTime", 1)
            cmds.playbackOptions(minTime=1, maxTime=self.acTime)

            allAgentShapes = cmds.ls(type="McdAgent")
            if allAgentShapes != [] and allAgentShapes != None:
                McdPlacementFunctions.dePlacementAgent()
            else:
                McdPlacementFunctions.placementAgent()

            cmds.select(self.sel[0])

            self.opName = ['CRD_action_name', 'CRD_path', 'CRD_type', 'CRD_blend_cycle',
                           'CRD_blend_entry', 'CRD_blend_exit']
            self.typeDic = {(0, 0, 0, 0, 0, 0): 'Static', (0, 0, 1, 0, 0, 0): 'Locomotion(Z+)',
                            (1, 0, 1, 0, 0, 0): 'Locomotion', (0, 1, 1, 0, 0, 0): 'Climb',
                            (1, 0, 1, 0, 1, 0): 'Turning', (1, 1, 1, 1, 0, 0): 'Ramp'}
            self.createOpVar()
            self.miarmy = self.sel[0]

            dLength = cmds.getAttr(self.miarmy + ".length")
            nodeName = self.miarmy.split('|')[-1]
            nodeName = '_'.join(nodeName.split('_')[:-2])

            statList = [0] * 6
            statList[0] = cmds.getAttr(self.miarmy + ".txState")
            statList[1] = cmds.getAttr(self.miarmy + ".tyState")
            statList[2] = cmds.getAttr(self.miarmy + ".tzState")
            statList[3] = cmds.getAttr(self.miarmy + ".rxState")
            statList[4] = cmds.getAttr(self.miarmy + ".ryState")
            statList[5] = cmds.getAttr(self.miarmy + ".rzState")

            dCycleFilter = cmds.getAttr(self.miarmy + ".cycleFilter")
            dTransIn = cmds.getAttr(self.miarmy + ".transIn")
            dTransOut = cmds.getAttr(self.miarmy + ".transOut")

            self.action_line = str(nodeName)

            if self.typeDic.has_key(tuple(statList)):
                self.type_line = self.typeDic[tuple(statList)]

            self.action_cycle = '%.1f ' % round(dCycleFilter * 100) + '% (' + str(int(dLength * dCycleFilter)) + 'f)'
            self.entry_line = '%.1f' % round(dTransIn * 100) + '% (' + str(int(dLength * dTransIn)) + 'f)'
            self.exit_line = '%.1f' % round(dTransOut * 100) + '% (' + str(int(dLength - dLength * dTransOut)) + 'f)'

        else:
            cmds.warning("Select an any action node.")

    def hudSet(self):

        cmds.optionVar(sv=('CRD_action_name', self.action_line))
        cmds.optionVar(sv=('CRD_path', self.ui.lineEditC.text()))
        cmds.optionVar(sv=('CRD_type', self.type_line))
        cmds.optionVar(sv=('CRD_blend_cycle', self.action_cycle))
        cmds.optionVar(sv=('CRD_blend_entry', self.entry_line))
        cmds.optionVar(sv=('CRD_blend_exit', self.exit_line))

        if cmds.headsUpDisplay('CRD_Frame', exists=1):
            cmds.headsUpDisplay('CRD_Frame', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Frame', l="Current Frame  ", allowOverlap=1, dataFontSize="large", b=1, s=1, lfs="large", bs="small", preset='currentFrame')

        if cmds.headsUpDisplay('CRD_Duration', exists=1):
            cmds.headsUpDisplay('CRD_Duration', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Duration', l="Duration  ", allowOverlap=1, dataFontSize="large", b=2, s=1, lfs="large", bs="small", atr=True, command='cmds.getAttr("%s.length")' % self.miarmy)

        if cmds.headsUpDisplay('CRD_Action', exists=1):
            cmds.headsUpDisplay('CRD_Action', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Action', l="Action Name  ", allowOverlap=1, dataFontSize="large", b=3, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_action_name')")

        if cmds.headsUpDisplay('CRD_Path', exists=1):
            cmds.headsUpDisplay('CRD_Path', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Path', l="Path  ", allowOverlap=1, dataFontSize="large", b=4, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_path')")

        if cmds.headsUpDisplay('CRD_Type', exists=1):
            cmds.headsUpDisplay('CRD_Type', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Type', l="Type  ", allowOverlap=1, dataFontSize="large", b=5, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_type')")

        if cmds.headsUpDisplay('CRD_Blend_Cycle', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Cycle', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Cycle', l="Blend Cycle  ", allowOverlap=1, dataFontSize="large", b=6, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_cycle')")

        if cmds.headsUpDisplay('CRD_Blend_Entry', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Entry', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Entry', l="Blend Entry  ", allowOverlap=1, dataFontSize="large", b=7, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_entry')")

        if cmds.headsUpDisplay('CRD_Blend_Exit', exists=1):
            cmds.headsUpDisplay('CRD_Blend_Exit', rem=1)
        else:
            cmds.headsUpDisplay('CRD_Blend_Exit', l="Blend Exit  ", allowOverlap=1, dataFontSize="large", b=8, s=1, lfs="large", bs="small", atr=True, command="cmds.optionVar(q='CRD_blend_exit')")

    def createOpVar(self):
        for i in self.opName:
            if not cmds.optionVar(ex=i):
                cmds.optionVar(sv=(i, ''))

    def playBlst(self):
        pbDir = os.sep.join(self.ui.movPath.text().split(os.sep)[:-1])
        pbFile = self.ui.movPath.text().split(os.sep)[-1].split(".")[0]
        pbPath = pbDir + "/" + pbFile

        if not os.path.isdir(pbDir):
            os.mkdir(pbDir)

        stF = int(cmds.playbackOptions(q=True, minTime=True))
        etF = int(cmds.playbackOptions(q=True, maxTime=True))

        if str(cmds.file(q=True, l=True)[0]) == "/dexter/Cache_DATA/CRD/Asset/humanType01/humanType01_Script/HumanType01_Action.ma":
            cmds.select("McdAgent2")

        cmds.playblast(f=pbPath, fo=True, fmt="qt", st=stF, et=etF, v=False, orn=True, os=0, fp=4, p=100, wh=[1280, 720], sqt=False)
        '''
        fps = int(mel.eval('currentTimeUnitToFPS'))
        currentDir = "/netapp/backstage/pub/apps/maya/global/DDPM/DDPM"
        fileName = self.movPath.text()
        sceneFile = cmds.file(q=True, l=True)[0]
        ARTISTNAME = getpass.getuser()
        movMetaData = self.getMovMetadata(sceneFile, ARTISTNAME)

        cmd = '%s/scripts/imgToMov.sh %d %d %d %s %s %s %s "sequence" %s' % (currentDir, stF, etF, fps, 'H.264 LT', pbPath, fileName, pbFile, movMetaData)
        os.system(cmd)
        '''
    def brwMov(self):
        fileName = cmds.fileDialog2(fileMode=0, caption="Save Mov File", fileFilter="Mov Files (*.mov)")[0]
        if len(fileName):
            self.ui.movPath.setText(fileName)

    def getMovMetadata(self, MayaFileFullPath, artistName):
        movMetadata = '\'{"mayaFilePath":"%s","artist":"%s"}\'' % (MayaFileFullPath, artistName)
        return movMetadata



def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()


if __name__ == '__main__':
    main()