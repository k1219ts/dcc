# encoding:utf-8
import maya.cmds as cmds
import maya.mel as mel
import sys
import os
import site
import time
import shutil
import json
import getpass
import McdPlacementFunctions
import McdMeshDriveSetupPub
import random
if sys.platform.find('win') > -1:
    TractorRoot = 'N:/backstage/pub/apps/tractor/win64/Tractor-2.0'
    site.addsitedir('%s/lib/python2.7/Lib/site-packages' % TractorRoot)
else:
    TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2'
    site.addsitedir('%s/lib/python2.7/site-packages' % TractorRoot)
import tractor.api.author as author
from McdGeneral import *
from McdSimpleCmd import *
from McdRender import *
ScriptRoot = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Crowd_RnD/script'
renderType = ['Tractor', 'Local']
pubType = ['Alembic', 'Rib']
from Qt import QtCore, QtGui, QtWidgets, load_ui
currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/crwExportF.ui")

def hconv(text):
    return unicode(text, 'utf-8')

class abcCal():

    def __init__(self):
        self.animBoundData = {}

    def abcMa(self, name, frame, bounds):
        if not name in self.animBoundData:
            self.animBoundData[name] = {}
        self.animBoundData[name][frame] = bounds

    def bOx(self):
        if self.animBoundData:
            for fn in self.animBoundData:
                f = open(fn, 'w')
                json.dump(self.animBoundData[fn], f, indent=4, sort_keys=True)
                f.close()

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.ui = load_ui(uiFile)
        self.connectSignal()
        self.num = 1
        self.attCon = ['.translateX', '.translateY', '.translateZ', '.rotateX', '.rotateY', '.rotateZ']
        self.jntOri = ['.jointOrientX', '.jointOrientY', '.jointOrientZ']
        # tractor
        self.mayafile = cmds.file(q=True, sn=True)
        self.mayafile_tmp = ''
        self.temp = str(int(time.time()))[5:]
        self.globalNode = mel.eval("McdSimpleCommand -execute 2")
        startframe = cmds.playbackOptions(q=1, minTime=1)
        endframe = cmds.playbackOptions(q=1, maxTime=1)
        if endframe - startframe < 0:
            print "Please check your render frame."
            return

    def connectSignal(self):
        self.ui.almBtn.clicked.connect(self.btnExc)
        self.ui.comboBox_render.addItems(renderType)
        self.ui.comboBox_render.setEnabled(True)
        self.ui.comboBox_render.setCurrentIndex(0)
        self.ui.pubTyp.addItems(pubType)
        self.ui.pubTyp.setEnabled(True)
        self.ui.pubTyp.setCurrentIndex(0)
        self.ui.layNum.setEnabled(True)
        self.ui.connect(self.ui.comboBox_render, QtCore.SIGNAL("currentIndexChanged(int)"), self.checkChange)
        self.ui.connect(self.ui.pubTyp, QtCore.SIGNAL("currentIndexChanged(int)"), self.pubChange)
        self.ui.mdChk.stateChanged.connect(self.btnCheck)

    def pubChange(self, rendtp):
        if rendtp == 0:
            if str(self.ui.comboBox_render.currentText()) == 'Local':
                if len(self.ui.layNum.text()) == 0:
                    self.ui.layNum.setEnabled(False)
                else:
                    self.ui.layNum.clear()
                    self.ui.layNum.setEnabled(False)
                self.ui.checkBox.setEnabled(True)
            else:
                self.ui.layNum.setEnabled(True)
                self.ui.checkBox.setEnabled(False)
        else:
            self.ui.checkBox.setEnabled(False)
            if len(self.ui.layNum.text()) == 0:
                self.ui.layNum.setEnabled(False)
            else:
                self.ui.layNum.clear()
                self.ui.layNum.setEnabled(False)

    def checkChange(self, rendtp):
        if rendtp == 0: # Tractor
            if str(self.ui.pubTyp.currentText()) == 'Alembic':
                self.ui.checkBox.setEnabled(False)
                self.ui.layNum.setEnabled(True)
            else:
                if len(self.ui.layNum.text()) == 0:
                    self.ui.layNum.setEnabled(False)
                else:
                    self.ui.layNum.clear()
                    self.ui.layNum.setEnabled(False)
        else:  # Local
            if len(self.ui.layNum.text()) == 0:
                self.ui.layNum.setEnabled(False)
            else:
                self.ui.layNum.clear()
                self.ui.layNum.setEnabled(False)
            if str(self.ui.pubTyp.currentText()) == 'Alembic':
                self.ui.checkBox.setEnabled(True)

    def texAttr(self):
        sgMap = dict()
        sel = cmds.ls("MDGGrp_*", dag=True, type='surfaceShape', ni=True)
        selTr = cmds.ls("MDGGrp_*", dag=True, type='transform', ni=True)
        disTex = []
        for i in selTr:
            agId = cmds.getAttr('%s.agentId' % i)
            for j in cmds.listRelatives(i, c=1):
                if not cmds.attributeQuery('rman__riattr__user_Agent__Index', n=j, ex=True):
                    cmds.addAttr(j, ln='rman__riattr__user_Agent__Index', nn='Agent Id', at='long')
                cmds.setAttr(j + '.rman__riattr__user_Agent__Index', agId)
        for i in cmds.listConnections(sel, type='shadingEngine'):
            for j in cmds.listConnections('%s.surfaceShader' % i):
                if cmds.listConnections('%s.color' % j):
                    for k in cmds.listConnections('%s.color' % j):
                        if (cmds.nodeType(k) == 'file'):
                            sgMap[i] = cmds.getAttr('%s.fileTextureName' % k)
                        elif (cmds.nodeType(k) == 'layeredTexture'):
                            arw = cmds.listConnections(k, type='file')[0]
                            texF = cmds.getAttr('%s.fileTextureName' % arw).split(".")
                            if (len(texF) == 3):
                                sgMap[i] = texF[0] + "." + texF[2]
                            elif (len(texF) == 2):
                                sgMap[i] = texF[0] + "." + texF[1]
                else:
                    sgMap[i] = "Null"
                    disTex.append(j)
        for e in sel:
            if cmds.listConnections(e, type='shadingEngine'):
                for r in cmds.listConnections(e, type='shadingEngine'):
                    if (sgMap[r].count("/crowd/asset") == 1):  # 군중 경로로 정리된 텍스쳐를 사용한 경우
                        if not cmds.attributeQuery('rman__riattr__user_mapname', n=e, ex=True):
                            cmds.addAttr(e, ln='rman__riattr__user_mapname', nn='Mcd_tex', dt='string')
                            # if (str(self.randCheck.checkState()) == "PySide.QtCore.Qt.CheckState.Checked"):  Check Box
                        temDir = "%s/tex/" % os.sep.join(sgMap[r].split(os.sep)[:-2])
                        if not cmds.attributeQuery('rman__riattr__user_txVarNum', n=e, ex=True):
                            cmds.addAttr(e, ln='rman__riattr__user_txVarNum', nn='txVarNum', at='long')
                        if sgMap[r].split(os.sep)[-1].count("_diffC_"):
                            txVarN = int(sgMap[r].rsplit("_diffC_")[-1].split(".")[0])
                            temFile = sgMap[r].rsplit("_diffC_")[0].split(os.sep)[-1]
                            temTex = temDir + temFile
                            cmds.setAttr(e + '.rman__riattr__user_txVarNum', txVarN)
                            cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')
                        elif sgMap[r].split(os.sep)[-1].count("_diffC."):
                            temFile = sgMap[r].split(os.sep)[-1].split("_diffC")[0]
                            temTex = temDir + temFile
                            cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')
                        if cmds.attributeQuery('rman__riattr__user_txAssetName', n=e, ex=True):
                            cmds.deleteAttr('%s.rman__riattr__user_txAssetName' % e)
                            cmds.deleteAttr('%s.rman__riattr__user_txLayerName' % e)
                    elif (sgMap[r].count("/texture/pub/") == 1):
                        if not cmds.attributeQuery('rman__riattr__user_txVarNum', n=e, ex=True):
                            cmds.addAttr(e, ln='rman__riattr__user_txVarNum', nn='txVarNum', at='long')
                        if sgMap[r].split(os.sep)[-1].count("_diffC_"):
                            txVarN = int(sgMap[r].rsplit("_diffC_")[-1].split(".")[0])
                            cmds.setAttr(e + '.rman__riattr__user_txVarNum', txVarN)
                        # cmds.setAttr(e + '.rman__riattr__user_txVarNum', random.randint(1, 9))
                '''
                elif (sgMap[r].count("/texture/pub/") == 1):        # 텍스쳐팀 경로를 사용한 경우
                    texVer = sgMap[r].split(os.sep)[:-1][-1]
                    texD = "%s/tex/" % os.sep.join(sgMap[r].split(os.sep)[:-2])
                    temDir = texD + texVer + "/"
                    if sgMap[r].split(os.sep)[-1].count(".tif"):
                        temFile = sgMap[r].split(os.sep)[-1].replace("_diffC.tif","")
                    elif sgMap[r].split(os.sep)[-1].count(".jpg"):
                        temFile = sgMap[r].split(os.sep)[-1].replace("_diffC.jpg","")
                    temTex = temDir + temFile
                    cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')
                '''
        print disTex

    def pathSet(self):
        selFile = str(os.path.basename(cmds.file(q=1, sn=1)))
        fPat = str(os.path.dirname(cmds.file(q=1, sn=1)))
        setPath = str(os.sep.join(fPat.split(os.sep)[0:-1]))
        if not os.path.exists(setPath + "/cache/alembic/"):
            if not os.path.exists(setPath + "/cache/"):
                os.mkdir(setPath + "/cache/")
            os.mkdir(setPath + "/cache/alembic/")
        if len(cmds.ls("agline_*", type="displayLayer")) != 0:
            shtFd = str(selFile).split(".")[0]
            self.shtPth = str(setPath + "/cache/alembic/%s/" % shtFd)
            if not os.path.exists(self.shtPth):
                os.mkdir(self.shtPth)
            else:
                pass
        else:
            pass

    def createMayaTempFile(self):
        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        exr = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[-1]
        temp_name = name
        self.mayafile_tmp = os.path.join(os.path.dirname(str(cmds.file(q=1, sn=1))), 'renderScenes', temp_name + exr)
        if not os.path.exists(os.path.dirname(self.mayafile_tmp)):
            os.makedirs(os.path.dirname(self.mayafile_tmp))
        cmds.file(save=True)
        shutil.copy2( str(cmds.file(q=True, sn=True)), self.mayafile_tmp )

    def createOutDir(self):
        outdir = cmds.getAttr('McdGlobal1.outputFolder')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    def postScript(self, Parent=None):
        Parent.addCleanup(author.Command(argv=['/bin/rm -f', self.mayafile_tmp], service=''))

    ##### Alembic Export Metadata #####
    def crw_jobscript(self, name, ofile, subProc):
        job = author.Job()
        job.title = name + str(os.path.basename(str(cmds.file(q=True, sn=True))).split('.')[0])
        job.comment = ''
        job.metadata = ''
        job.envkey = ['cache2016.5']
        job.service = 'Miarmy'
        job.maxactive = 10
        job.tier = 'cache'
        job.projects = ['export']
        job.tags = ['GPU']
        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1
        if subProc == 1:            # Rib Tractor
            ProcPrimTask = author.Task(title='crwTask')
            command = ['mayapy', '%%D(%s/%s.py)' % (ScriptRoot, ofile), self.mayafile_tmp]
            ProcPrimTask.addCommand(author.Command(service='', envkey=['cache2016.5'], tags=['py'], argv=command))
            JobTask.addChild(ProcPrimTask)
            RibTask = author.Task(title='RibExport')
            RibTask.serialsubtasks = 0
            JobTask.addChild(RibTask)
            self.ribFrameTask(RibTask)
        elif subProc == 0:          # Alembic Tractor
            AlembicTask = author.Task(title='AlembicExport')
            AlembicTask.serialsubtasks = 0
            JobTask.addChild(AlembicTask)
            firstJobFrame = cmds.playbackOptions(q=1, minTime=1)
            lastJobFrame = cmds.playbackOptions(q=1, maxTime=1)
            AlmFrameTask = author.Task(title='crwExport %s, %s' % (firstJobFrame, lastJobFrame))
            command = ['mayapy', '%%D(%s/%s.py)' % (ScriptRoot, ofile), self.mayafile_tmp]
            AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2016.5'], tags=['py'], argv=command))
            AlembicTask.addChild(AlmFrameTask)
        elif subProc == 2:          # Alembic divided Tractor
            AlembicTask = author.Task(title='AlembicExport')
            AlembicTask.serialsubtasks = 0
            JobTask.addChild(AlembicTask)
            stframe = cmds.playbackOptions(q=1, minTime=1)
            enframe = cmds.playbackOptions(q=1, maxTime=1)
            dpList = cmds.ls("agline_*", type="displayLayer")
            divFileNumb = 1
            for e in range(len(dpList)):
                AlmFrameTask = author.Task(title='crwExport %s, %s' % (str(stframe), str(enframe)))
                command = ['mayapy', '%%D(%s/%s.py)' % (ScriptRoot, ofile), self.mayafile_tmp, stframe, enframe, e, str(divFileNumb)]
                AlmFrameTask.addCommand(author.Command(service='', envkey=['cache2016.5'], tags=['py'], argv=command))
                AlembicTask.addChild(AlmFrameTask)
                divFileNumb += 1
        self.postScript(Parent=JobTask)
        job.addChild(JobTask)
        return job

    def ribFrameTask(self, Parent=None):
        framePerTask = 10
        stframe = cmds.getAttr('%s.startFrame' % self.globalNode)
        enframe = cmds.getAttr('%s.endFrame' % self.globalNode)
        for i in range(stframe, enframe + 1, framePerTask):
            firstJobFrame = i
            if i + framePerTask > enframe:
                lastJobFrame = enframe
            else:
                lastJobFrame = i + framePerTask - 1
            RibFrameTask = author.Task(title='RibExport %s, %s' % (firstJobFrame, lastJobFrame))
            command = ['mayapy', '%%D(%s/crwRibExport.py)' % ScriptRoot, self.mayafile_tmp, firstJobFrame, lastJobFrame]
            RibFrameTask.addCommand(author.Command(service='Miarmy', envkey=['cache2016.5'], tags=['py'], argv=command))
            Parent.addChild(RibFrameTask)

        ##################################
        # Main Job Command
        ##################################

    def rib_trc(self):
        self.createMayaTempFile()
        self.createOutDir()
        pubType = self.crw_jobscript('(Crw-Rib)', 'crwProcPrimExport', 1)
        self.tracshot(pubType)

    def trc(self):
        self.plainShowName = str(cmds.file(q=1, sn=1))
        self.texAttr()
        self.pathSet()
        self.createMayaTempFile()
        selB = cmds.ls("MDGGrp_*")
        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        cchPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/cache/alembic/"
        if len(cmds.ls("agline_*", type="displayLayer")) == 0:
            pubType = self.crw_jobscript('(Crw-Alm) ', 'AlembicExport_rmantd', 0)
            bbxPath = cchPath + name + '_' + self.temp + ".bbox"  # 파일명_생성시간
            cchFile = cchPath + name + '_' + self.temp + ".abc"
            qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
            qPat = bbxPath + "\n\n" + "Scene File \n"
            qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Alembic Cache \n"
            self.ui.plainTextEdit.insertPlainText(qNum)
            self.ui.plainTextEdit.insertPlainText(qFile)
            self.ui.plainTextEdit.insertPlainText(qPat)
            self.ui.plainTextEdit.insertPlainText(self.plainShowName)
        else:
            ageNum = 0
            dpList = cmds.ls("agline_*", type="displayLayer")
            chNum = len(dpList)
            for i in dpList:
                ageNum += len(cmds.listConnections(str(i) + ".drawInfo", type="transform"))
            qNum = "Total " + str(ageNum) + " Agents. " + str(chNum) + " Cache Files. \n\n" + "Alembic Cache \n"
            bbxPath = self.shtPth
            cchFile = self.shtPth
            qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
            qPat = bbxPath + "\n\n" + "Scene File \n"
            self.ui.plainTextEdit.insertPlainText(qNum)
            self.ui.plainTextEdit.insertPlainText(qFile)
            self.ui.plainTextEdit.insertPlainText(qPat)
            self.ui.plainTextEdit.insertPlainText(self.plainShowName)
            pubType = self.crw_jobscript('(Crw-Alm) ', 'AlembicExport_rmantd', 2)
        self.tracshot(pubType)

    def tracshot(self, job):
        # spool
        job.priority = 1000.0
        author.setEngineClientParam(
            hostname='10.0.0.25', port=80,
            user=getpass.getuser(), debug=True)
        job.spool()
        author.closeEngineClient()

    def rib_locl(self):
        McdRenderBegin(1, 1)

    def locl(self):
        self.plainShowName = str(cmds.file(q=1, sn=1))
        self.texAttr()
        self.pathSet()
        self.createMayaTempFile()
        minT = cmds.playbackOptions(q=1, minTime=1)
        maxT = cmds.playbackOptions(q=1, maxTime=1)
        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        cchPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/cache/alembic/"
        bbxPath = cchPath + name + '_' + self.temp + ".bbox"  # 파일명_생성시간
        cchFile = cchPath + name + '_' + self.temp + ".abc"
        if self.ui.checkBox.isChecked() == True:
            selB = cmds.ls("MDGGrp_*", l=1)
        else:
            selB = cmds.ls(sl=1)
        bs = ""
        for i in selB:
            bs += " -rt " + str(i)
        jobCmd = '''-pythonPerFrameCallback "crwData.abcMa(name='%s',frame=#FRAME#,bounds=#BOUNDSARRAY#)"''' % bbxPath
        jobCmd += " -pythonPostJobCallback crwData.bOx()"
        jobCmd += " -fr %f %f -atp rman -wuvs -ws -wv -ef -df ogawa %s -f %s" % (minT, maxT, bs, cchFile)
        animPlug = '/usr/autodesk/maya2016.5/bin/plug-ins/AbcExport.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            cmds.loadPlugin(animPlug)
            cmds.pluginInfo(animPlug, edit=True, autoload=True)
        cmds.AbcExport(v=1, j=jobCmd)
        qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
        qPat = bbxPath + "\n\n" + "Scene File \n"
        qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Alembic Cache \n"
        self.ui.plainTextEdit.insertPlainText(qNum)
        self.ui.plainTextEdit.insertPlainText(qFile)
        self.ui.plainTextEdit.insertPlainText(qPat)
        self.ui.plainTextEdit.insertPlainText(self.plainShowName)

    def MDriveOn(self):
        McdMeshDriveSetupPub.McdExportMD2Cache()
        McdPlacementFunctions.dePlacementAgent()
        McdMeshDriveSetupPub.MDDuplicate()
        McdMeshDriveSetupPub.McdCreateMeshDriveIMNode(1, 1)

    def btnCheck(self):
        tx = self.ui.mdChk.checkState()
        if str(tx) == "PySide.QtCore.Qt.CheckState.Unchecked":
            self.ui.almBtn.setText("Pub")
            self.ui.pubTyp.setEnabled(True)
            self.ui.comboBox_render.setEnabled(True)
            self.ui.layNum.setEnabled(False)

        elif str(tx) == "PySide.QtCore.Qt.CheckState.PartiallyChecked":
            self.ui.almBtn.setText("MD On")
            self.ui.pubTyp.setEnabled(False)
            self.ui.comboBox_render.setEnabled(False)
            self.ui.layNum.setEnabled(False)

        elif str(tx) == "PySide.QtCore.Qt.CheckState.Checked":
            self.ui.almBtn.setText("MD Pub")
            self.ui.pubTyp.setEnabled(True)
            self.ui.comboBox_render.setEnabled(True)
            self.ui.layNum.setEnabled(True)

    def btnExc(self):
        if cmds.getAttr("McdGlobal1.boolMaster[10]") == False:
            tx = self.ui.mdChk.checkState()
            if str(tx) == "PySide.QtCore.Qt.CheckState.Unchecked":
                self.doIt()
            elif str(tx) == "PySide.QtCore.Qt.CheckState.PartiallyChecked":
                self.MDriveOn()
            elif str(tx) == "PySide.QtCore.Qt.CheckState.Checked":
                self.MDriveOn()
                getNum = self.ui.layNum.text()
                if len(getNum) != 0:
                    if str(getNum).isdigit():
                        self.setLayers(getNum)
                    else:
                        cmds.error("Input type error")
                else:
                    pass
                self.doIt()
        else:
            cmds.confirmDialog(t="Warning", m="Turn off Real Time Display")

    def setLayers(self, lay):
        selTr = cmds.ls("MDGGrp_*", dag=True, type='transform', ni=True)
        lp = divmod(len(selTr), int(lay))
        if lp[1] == 0:
            for i in range(int(lay)):
                layerNameNum = str(i + 1)
                cmds.createDisplayLayer(name="agline_" + layerNameNum, number=1, nr=True)
                for j in range(lp[0]):
                    nm = lp[0] * i + j
                    cmds.editDisplayLayerMembers("agline_" + layerNameNum, "MDGGrp_" + str(nm), noRecurse=True)
        else:
            for i in range(int(lay)):
                layerNameNum = str(i + 1)
                cmds.createDisplayLayer(name="agline_" + layerNameNum, number=1, nr=True)
                for j in range(lp[0]):
                    nm = lp[0] * i + j
                    cmds.editDisplayLayerMembers("agline_" + layerNameNum, "MDGGrp_" + str(nm), noRecurse=True)
            for e in range(lp[1]):
                if len(cmds.ls("agline_" + str(lay))) == 1:
                    cmds.editDisplayLayerMembers("agline_" + str(lay), selTr[-(e + 1)], noRecurse=True)
                else:
                    cmds.error("agline_%s Layer is not exist." % str(lay))

    def doIt(self):
        self.ui.plainTextEdit.clear()
        if (self.ui.pubTyp.currentText() == 'Alembic'):
            if (self.ui.comboBox_render.currentText() == 'Tractor'):
                self.trc()
            elif (self.ui.comboBox_render.currentText() == 'Local'):
                self.locl()
        elif (self.ui.pubTyp.currentText() == 'Rib'):
            if (self.ui.comboBox_render.currentText() == 'Tractor'):
                self.rib_trc()
            elif (self.ui.comboBox_render.currentText() == 'Local'):
                self.rib_locl()

def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()

if __name__ == '__main__':
    main()