# encoding:utf-8
# !/usr/bin/env python

import maya.cmds as cmds
import maya.mel as mel
import shutil
import os
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools

currentpath = os.path.abspath(__file__)
UIFILE = os.path.join(os.path.dirname(currentpath), "animExport.ui")

MCP_PREVIEW_FILE = '/stdrepo/ANI/Library/Mocap_Library/01_Asset/02_Char/00_Human/Human_v01/humanType01_skinning.ma'
MCP_PREVIEW_FILE_ROOT_JOINT_NAME = "Crw_Hips"

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        uiFile = QtCore.QFile(UIFILE)
        uiFile.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        ui = loader.load(uiFile)
        setup_ui(ui, self)
        self.checkPlugIn()
        self.connectSignal()

    def checkPlugIn(self):
        animPlug = '/usr/autodesk/maya2018/bin/plug-ins/animImportExport.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            cmds.loadPlugin(animPlug)
            cmds.pluginInfo(animPlug, edit=True, autoload=True)

    def closeEvent(self, event):
        self.resetHUD()
        try:
            global myWindow
            myWindow.close()
        except:
            pass

    def connectSignal(self):
        # Anim Export Buttons
        self.BTN_import_fbx.clicked.connect(self.importFBX)
        self.BTN_root_joint_select.clicked.connect(self.selectRootJoint)
        self.BTN_Open_Anim_Folder.clicked.connect(self.openAnimFolder)
        self.BTN_export_anim.clicked.connect(self.exportAnim)
        # Preview Export Buttons
        self.BTN_Preview_File_Open.clicked.connect(self.prvOpen)
        self.BTN_timeline_set.clicked.connect(self.timelineSet)
        self.BTN_import_anim.clicked.connect(self.importAnim)
        self.BTN_Hud.clicked.connect(self.toggleHUD)
        self.BTN_Playblast.clicked.connect(self.doitPlayblast)
        self.BTN_Mov_Path_Modify.clicked.connect(self.movPathModify)
        self.BTN_Open_Mov_Folder.clicked.connect(self.openMovFolder)
        # Motion Builder Export Buttons
        self.BTN_Export_MB.clicked.connect(self.exportMotionBuilder)

    def selectRootJoint(self):
        if cmds.ls("Crw_Hips", type="joint"):
            tarJnt = "Crw_Hips"
        else:
            tarJnt = str(cmds.ls(sl=True)[0])
        self.LB_RootJointName.setText(tarJnt)

    def openAnimFolder(self):
        pbDir = os.sep.join(self.animDir.split(os.sep)[:-2]) + "/05_Anim/"
        os.system('/usr/bin/nautilus %s &' % pbDir)

    def importFBX(self):
        self.resetHUD()
        cmds.file(f=True, new=True)
        impFbxFile = str(cmds.fileDialog2(fileMode=1, fileFilter="FBX Files (*.fbx)", caption="Import FBX File")[0])
        if impFbxFile:
            cmds.file(impFbxFile, i=True, iv=True, typ="FBX", ra=True, uns=False, pr=True, mnc=False, op="fbx")
            self.animDir = impFbxFile
            del impFbxFile
            # Delete NameSpace
            nsList = [str(i) for i in cmds.namespaceInfo(lon=True, r=True) if str(i) not in ["UI", "shared"]]
            if len(nsList) != 0:
                for i in nsList:
                    cmds.namespace(rm=i, mnr=True)
        else:
            return

    def exportAnim(self):
        getPath = self.animDir
        tarJnt = self.LB_RootJointName.text()
        cmds.select(tarJnt, hierarchy=True)
        tarHjnt = [str(i) for i in cmds.ls(sl=True)]
        cmds.select(tarJnt, d=True)
        tarRjnt = [str(j) for j in cmds.ls(sl=True)]
        cmds.select(tarJnt)
        
        if len(tarJnt) == 0:
            cmds.error("Select Root Joint")
            return
        else:
            pass
        animPath = os.sep.join(getPath.split(os.sep)[:-2]) + "/05_Anim/"
        if not os.path.exists(animPath):
            os.mkdir(animPath)
        mdFile = getPath.split(os.sep)[-1].replace(".fbx", ".anim")
        if mdFile.count("maya_") == 1:
            animFile = animPath + mdFile.replace("maya_", "")
        else:
            animFile = animPath + mdFile
        pnum = int(self.LW_animList.count()) + 1
        outTex = ("%02d" % pnum) + " : " + os.sep.join(animFile.split(os.sep)[-3:]) + "\n"
        tm = int(max(cmds.keyframe(tarJnt, q=True)))
        cmds.playbackOptions(minTime=0, maxTime=tm)
        cmds.currentTime(0)
        etAt = [".sx", ".sy", ".sz", ".v", ".radi", ".liw"]
        trAt = [".tx", ".ty", ".tz"]
        rtAt = [".rx", ".ry", ".rz"]
        alAt = trAt + rtAt + etAt
        mel.eval("channelBoxCommand -break; ")
        for i in range(len(tarHjnt)):
            for j in range(len(etAt)):
                temA = tarHjnt[i] + etAt[j]
                mel.eval("CBdeleteConnection %s;" % temA)
                cmds.setAttr(temA, lock=True)
        for k in range(len(tarRjnt)):
            for l in range(len(trAt)):
                temB = tarRjnt[k] + trAt[l]
                mel.eval("CBdeleteConnection %s;" % temB)
                cmds.setAttr(temB, lock=True)
        for m in range(len(tarHjnt)):
            for n in range(len(rtAt)):
                temC = tarHjnt[m] + rtAt[n]
                cmds.setAttr(temC, 0)
        cmds.bakeResults(tarHjnt, hi="below", sm=False, t=(0, tm), sb=1.0, dic=True, pok=True, mr=True)
        for o in range(len(tarHjnt)):
            for p in range(len(alAt)):
                temD = tarHjnt[o] + alAt[p]
                mel.eval('CBunlockAttr %s;' % temD)
        cmds.selectKey(tarHjnt, keyframe=True)
        cmds.filterCurve(f="euler")
        cmds.delete(tarHjnt, sc=True, uac=False, hi="below", cp=False, s=False)
        ofsin = int(self.LE_Action_Start_Frame.text())
        if ofsin > 1:
            cmds.cutKey(tarHjnt, o="keys", t=(1, ofsin - 1), cl=True)
            cmds.keyTangent(tarHjnt, time=(0, ofsin), itt="auto", ott="auto")
            cmds.keyframe(tarHjnt, e=True, iub=True, r=True, o="over", t=("%d:" % ofsin,), tc=-(ofsin-1))
            cmds.selectKey(tarHjnt, clear=True)
            cmds.selectKey(tarHjnt, keyframe=True)
            cmds.file(animFile, force=True,
                      options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1",
                      typ="animExport", eas=True)
            cmds.playbackOptions(minTime=0, aet=tm-ofsin+1)
            if os.path.isfile(animFile):
                self.LW_animList.addItem(outTex)
                self.BTN_Open_Anim_Folder.setEnabled(True)
        elif ofsin == 1:
            cmds.selectKey(tarHjnt, clear=True)
            cmds.selectKey(tarHjnt, keyframe=True)
            cmds.file(animFile, force=True,
                      options="precision=8;nodeNames=1;verboseUnits=0;whichRange=1;options=keys;hierarchy=below;controlPoints=0;shapes=1;useChannelBox=0;copyKeyCmd=-animation objects -option keys -hierarchy below -controlPoints 0 -shape 1",
                      typ="animExport", eas=True)
            if os.path.isfile(animFile):
                self.LW_animList.addItem(outTex)
                self.BTN_Open_Anim_Folder.setEnabled(True)
        else:
            cmds.error("Input start frame with positive integer number")

    def resetHUD(self):
        hudList = ["MCP_project", "MCP_filename", "MCP_charname", "MCP_Duration", "MCP_Frame"]
        for i in hudList:
            if cmds.headsUpDisplay(i, exists=1):
                cmds.headsUpDisplay(i, rem=1)

    def prvOpen(self):
        self.resetHUD()
        cmds.file(f=True, new=True)
        if os.path.exists(MCP_PREVIEW_FILE):
            prv_path = os.sep.join(MCP_PREVIEW_FILE.split(os.sep)[:-1]) + "/temp/"
            prv_file = prv_path + "humanType01.ma"
            if not os.path.exists(prv_path):
                os.mkdir(prv_path)
            if os.path.exists(prv_file):
                os.remove(prv_file)
                shutil.copy2(MCP_PREVIEW_FILE, prv_file)
                cmds.file(prv_file, iv=True, f=True, o=True)
            else:
                shutil.copy2(MCP_PREVIEW_FILE, prv_file)
                cmds.file(prv_file, iv=True, f=True, o=True)
        else:
            cmds.error("Please check your scene file location to make preview.")
            return

    def timelineSet(self):
        tarJnt = MCP_PREVIEW_FILE_ROOT_JOINT_NAME
        tlm = int(max(cmds.keyframe(tarJnt, q=True)))
        cmds.playbackOptions(minTime=0, maxTime=tlm)
        cmds.currentTime(0)
        cmds.select(tarJnt)

    def importAnim(self):
        self.resetHUD()
        tarJnt = MCP_PREVIEW_FILE_ROOT_JOINT_NAME
        cmds.select(tarJnt, hierarchy=True)
        cmds.selectKey(keyframe=True)
        cmds.currentTime(0)
        cmds.delete(all=True, c=True)
        cmds.select(cl=True)
        cmds.select(tarJnt)
        ''' StartPath Set'''
        animPath = str(cmds.fileDialog2(fileMode=1, fileFilter="Anim Files (*.anim)", caption="Import Anim File")[0])
        cmds.file(animPath, i=True, iv=True, ra=True, uns=False, pr=True, mnc=False)
        showAn = str(animPath.split(os.sep)[-1]).split(".")[0]
        self.LE_File_Name.setText(showAn)
        prvDir = os.sep.join(animPath.split(os.sep)[:-2]) + "/06_Preview/"
        if not os.path.exists(prvDir):
            os.mkdir(prvDir)
        prvPath = prvDir + animPath.split(os.sep)[-1].replace(".anim", ".mov")
        self.LE_Export_Mov_Path.setText(prvPath)

    def getDuration(self):
        minT = int(cmds.playbackOptions(q=True, min=True))
        maxT = int(cmds.playbackOptions(q=True, max=True))
        dur = maxT - minT + 1
        return dur

    def checkViewJoint(self):
        currentPanel = [str(i) for i in cmds.getPanel(vis=True) if str(i).count("modelPanel") == 1][0]
        cmds.modelEditor(currentPanel, e=True, j=False)

    def toggleHUD(self):
        self.checkViewJoint()
        charName = self.LE_Char_Name.text()
        animFileName = self.LE_File_Name.text()
        projectName = self.LE_Project_Name.text()
        duration = self.getDuration()

        if cmds.headsUpDisplay('MCP_project', exists=1):
            cmds.headsUpDisplay('MCP_project', rem=1)
        else:
            cmds.headsUpDisplay('MCP_project', l="Project  ", ao=True, dfs="large", b=1, s=1, atr=True, lfs="large", bs="small", c="'%s'" % projectName)

        if cmds.headsUpDisplay('MCP_filename', exists=1):
            cmds.headsUpDisplay('MCP_filename', rem=1)
        else:
            cmds.headsUpDisplay('MCP_filename', l="File Name  ", ao=True, dfs="large", b=1, s=3, atr=True, lfs="large", bs="small", c="'%s'" % animFileName)

        if cmds.headsUpDisplay('MCP_charname', exists=1):
            cmds.headsUpDisplay('MCP_charname', rem=1)
        else:
            cmds.headsUpDisplay('MCP_charname', l="Character Name  ", ao=True, dfs="large", b=7, s=4, atr=True, lfs="large", bs="small", c="'%s'" % charName)

        if cmds.headsUpDisplay('MCP_project', exists=1):
            if cmds.optionVar(q="viewAxisVisibility") == 0:
                mel.eval("ToggleViewAxis;")
            else:
                pass
        else:
            if cmds.optionVar(q="viewAxisVisibility") == 1:
                mel.eval("ToggleViewAxis;")
            else:
                pass

        if cmds.headsUpDisplay('MCP_Duration', exists=1):
            cmds.headsUpDisplay('MCP_Duration', rem=1)
        else:
            cmds.headsUpDisplay('MCP_Duration', l="Duration  ", ao=True, dfs="large", b=1, s=6, atr=True, lfs="large", bs="small", c="'%s'" % duration)

        if cmds.headsUpDisplay('MCP_Frame', exists=1):
            cmds.headsUpDisplay('MCP_Frame', rem=1)
        else:
            cmds.headsUpDisplay('MCP_Frame', l="Frame  ", ao=True, dfs="large", b=1, s=8, lfs="large", bs="small", preset="currentFrame")

    def doitPlayblast(self):
        if not self.LE_Export_Mov_Path.text():
            cmds.error("Input MOV file Path to export.")
            return
        else:
            pass
        movPath = self.LE_Export_Mov_Path.text()
        movDir = os.sep.join(movPath.split(os.sep)[:-1])
        if not os.path.isdir(movDir):
            os.mkdir(movDir)
        minT = int(cmds.playbackOptions(q=True, minTime=True))
        maxT = int(cmds.playbackOptions(q=True, maxTime=True))
        cmds.playblast(f=movPath, fo=True, fmt="qt", st=minT, et=maxT, v=False, orn=True, os=0, fp=4, p=100, wh=[1280, 720], sqt=False)

    def checkMovFolder(self):
        movExPath = self.LE_Export_Mov_Path.text()
        if movExPath:
            getPath = os.sep.join(movExPath.split(os.sep)[:-1])
        else:
            getPath = "/stdrepo/ANI"
        return getPath

    def movPathModify(self):
        startingPath = self.checkMovFolder()
        filePath = cmds.fileDialog2(dir=startingPath ,fileMode=0, caption="Save Mov File", fileFilter="Mov Files (*.mov)")[0]
        if filePath:
            self.LE_Export_Mov_Path.setText(filePath)

    def openMovFolder(self):
        pbDir = self.checkMovFolder()
        os.system('/usr/bin/nautilus %s &' % pbDir)

    def exportMotionBuilder(self):
        scl = ["scaleX", "scaleY", "scaleZ"]
        minT = cmds.playbackOptions(q=True, min=True)
        maxT = cmds.playbackOptions(q=True, max=True)
        cmds.currentTime(minT)
        sel = str(cmds.ls(sl=True)[0])
        env = str(cmds.ls(sl=True)[1])
        nsChar = sel.split(":")[0]
        cmds.bakeResults(["*:*_Skin_*_JNT", "*:weapon_JNT"], simulation=True, t=(minT, maxT), sampleBy=1, dic=True,
                         pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)
        mel.eval("channelBoxCommand -break; ")
        if cmds.keyframe(nsChar + ":place_CON.globalScale", q=True):
            mel.eval("CBdeleteConnection %s;" % (nsChar + ":place_CON.globalScale"))
        getV = cmds.getAttr(nsChar + ":place_CON.globalScale")
        cmds.setAttr(nsChar + ":place_CON.globalScale", getV * 10)
        cmds.xform(nsChar + ":place_NUL", cp=True)
        for w in scl:
            cmds.setAttr(nsChar + ":place_NUL." + w, 10)
        pvPoint = cmds.xform(sel, q=True, ws=True, t=True)
        cmds.move(pvPoint[0], pvPoint[1], pvPoint[2], env + ".scalePivot", env + ".rotatePivot", a=True)
        for n in scl:
            cmds.setAttr(env + "." + n, 10)

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
