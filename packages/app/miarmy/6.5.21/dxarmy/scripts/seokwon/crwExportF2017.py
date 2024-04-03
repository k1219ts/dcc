# encoding:utf-8
# Miarmy Alembic/Rib Publish Tool for Maya 2017

# It's not work. Choi_SeokWon deleted ../Choi_SeokWon/Crowd_RnD/script folder

import maya.cmds as cmds
import maya.mel as mel
import os, site, datetime, shutil, json, getpass, webbrowser, sys, configobj
import McdPlacementFunctions
import McdModifiedModules.McdMeshDriveSetupPub as McdMeshDriveSetupPub
reload(McdMeshDriveSetupPub)
import dxArmy
from pymodule.Qt import QtCore, QtGui, QtWidgets, QtCompat
from McdGeneral import *
from McdSimpleCmd import *
from McdRender import *

config_fn = '{PATH}/config/rfm.config'.format(PATH=os.getenv('BACKSTAGE_PATH'))
if not os.path.exists(config_fn):
    config_fn = '/netapp/backstage/pub/config/rfm.config'
getConfig = configobj.ConfigObj(config_fn)
try:
    tractorAPI = getConfig['TractorAPI']
    if not tractorAPI in sys.path:
        sys.path.append(tractorAPI)
    import tractor.api.author as author
except:
    pass

import RibFilter.rif_process as rif_process

# ScriptRoot = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Crowd_RnD/script'
ScriptRoot = os.path.dirname(os.path.abspath(__file__))

renderType = ['Tractor', 'Local']
pubType = ['Rib', 'Alembic']
uiFile = os.path.join(ScriptRoot, 'ui', 'crwExportF.ui')

class abcCal():
    def __init__(self):
        self.animBoundData = dict()

    def abcMa(self, name, frame, bounds):
        if not name in self.animBoundData:
            self.animBoundData[name] = dict()
        self.animBoundData[name][frame] = bounds

    def bOx(self):
        if self.animBoundData:
            for fn in self.animBoundData:
                f = open(fn, 'w')
                json.dump(self.animBoundData[fn], f, indent=4, sort_keys=True)
                f.close()

def setup_ui(ui, base_instance=None):
    for member in dir(ui):
        if not member.startswith('__') and member is not 'staticMetaObject':
            setattr(base_instance, member, getattr(ui, member))

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        ui = QtCompat.load_ui(uiFile)
        setup_ui(ui, self)
        self.connectSignal()
        self.mayafile_tmp = ''

    def connectSignal(self):
        self.almBtn.clicked.connect(self.btnExc)
        self.comboBox_render.addItems(renderType)
        self.comboBox_render.setEnabled(True)
        self.comboBox_render.setCurrentIndex(0)
        self.pubTyp.addItems(pubType)
        self.pubTyp.setEnabled(True)
        self.pubTyp.setCurrentIndex(0)
        self.layNum.setEnabled(True)
        self.comboBox_render.currentIndexChanged.connect(self.checkChange)
        self.pubTyp.currentIndexChanged.connect(self.pubChange)
        self.mdChk.stateChanged.connect(self.btnCheck)
        self.brws.clicked.connect(self.trct)

    def btnExc(self):
        for i in cmds.ls(type="camera"):
            if str(i) == "perspShape":
                cmds.setAttr(str(i) + ".renderable", 1)
            else:
                cmds.setAttr(str(i) + ".renderable", 0)
        tx = self.mdChk.checkState()
        if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            self.doIt()
        elif str(tx) == "PySide2.QtCore.Qt.CheckState.PartiallyChecked":
            self.MDriveOn()
        elif str(tx) == "PySide2.QtCore.Qt.CheckState.Checked":
            self.MDriveOn()
            getNum = self.layNum.text()
            if len(getNum) != 0:
                if str(getNum).isdigit():
                    self.setLayers(getNum)
                else:
                    cmds.error("Input type error")
            else:
                pass
            self.doIt()

    def rangeEnb(self, bln):
        self.LB_range1.setEnabled(bln)
        self.LB_range2.setEnabled(bln)
        self.LE_mintime.setEnabled(bln)
        self.LE_maxtime.setEnabled(bln)

    def dpEnb(self, bln):
        self.LB_dispatch_frames.setEnabled(bln)
        self.SB_dispatch_frames.setEnabled(bln)

    def btnCheck(self):
        tx = self.mdChk.checkState()
        if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
            self.almBtn.setText("Pub")
            self.pubTyp.setEnabled(True)
            self.comboBox_render.setEnabled(True)
            self.layNum.setEnabled(False)
            if (self.pubTyp.currentText() == 'Alembic'):
                self.rangeEnb(False)
                self.dpEnb(False)
            else:
                self.rangeEnb(True)
                if str(self.comboBox_render.currentText()) == 'Local':
                    self.dpEnb(False)
                else:
                    self.dpEnb(True)

        elif str(tx) == "PySide2.QtCore.Qt.CheckState.PartiallyChecked":
            self.almBtn.setText("MD On")
            self.pubTyp.setEnabled(False)
            self.comboBox_render.setEnabled(False)
            self.layNum.setEnabled(False)
            self.rangeEnb(False)
            self.dpEnb(False)

        elif str(tx) == "PySide2.QtCore.Qt.CheckState.Checked":
            self.almBtn.setText("MD Pub")
            self.pubTyp.setEnabled(True)
            self.comboBox_render.setEnabled(True)
            self.layNum.setEnabled(True)
            self.rangeEnb(False)
            self.dpEnb(False)

    def MDriveOn(self):
        McdMeshDriveSetupPub.McdExportMD2Cache()        # MD Cache Export
        McdPlacementFunctions.dePlacementAgent()        # De-placement
        McdMeshDriveSetupPub.MDDuplicate()              # Duplicate Meshes and Randomize Texture
        McdMeshDriveSetupPub.McdCreateMeshDriveIMNode(1, 1) # Enable Mesh Drive

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

    def confirmPublishing(self):
        cam = str([str(i) for i in cmds.ls(type="camera") if cmds.getAttr(str(i) + ".renderable") == 1][0])
        mayaFile = str(cmds.file(q=1, sn=1))
        outPath = os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/" + mayaFile.split(os.sep)[-1].split(".")[0] + "/"
        outRib = str(cmds.getAttr("McdGlobal1.outputRibs"))
        outPic = str(cmds.getAttr("McdGlobal1.outputPics"))
        mms = "Renderable Cam : " + cam + "\nPub Path : " + outPath + "\nOut Ribs : " + outRib + "\nOut Pics : " + outPic
        result = str(cmds.confirmDialog(t='Confirm Publishing', m=mms, b=['Yes', 'No'], db='Yes', cb='No', ds='No'))
        if result == "Yes":
            return True
        else:
            return False

    def doIt(self):
        self.plainTextEdit.clear()
        if (self.pubTyp.currentText() == 'Alembic'):
            if (self.comboBox_render.currentText() == 'Tractor'):
                self.trc()
            elif (self.comboBox_render.currentText() == 'Local'):
                self.locl()
        elif (self.pubTyp.currentText() == 'Rib'):
            if (self.comboBox_render.currentText() == 'Tractor'):
                if self.confirmPublishing():
                    self.rib_trc()
                else:
                    return
            elif (self.comboBox_render.currentText() == 'Local'):
                if self.confirmPublishing():
                    self.rib_locl()
                else:
                    return

    def trct(self):
        webbrowser.register('firefox', None)
        webbrowser.Mozilla('firefox').open('http://10.0.0.25/tv')

    def exportTime(self):   # Date_Time Stamp
        expTime = datetime.datetime.today()
        dateSTM = str(expTime.month).zfill(2) + str(expTime.day).zfill(2)
        timeSTM = str(expTime.hour).zfill(2) + str(expTime.minute).zfill(2)
        timeText = dateSTM + timeSTM
        del expTime, dateSTM, timeSTM
        return timeText

    def pubChange(self):
        if str(self.pubTyp.currentText()) == 'Rib':
            tx = self.mdChk.checkState()
            if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
                self.rangeEnb(True)
            else:
                self.rangeEnb(False)
            self.checkBox.setEnabled(False)
            if len(self.layNum.text()) == 0:
                self.layNum.setEnabled(False)
            else:
                self.layNum.clear()
                self.layNum.setEnabled(False)
            if str(self.comboBox_render.currentText()) == 'Local':
                self.dpEnb(False)
            else:
                if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
                    self.dpEnb(True)
                else:
                    self.dpEnb(False)
        else:
            self.rangeEnb(False)
            self.dpEnb(False)
            if str(self.comboBox_render.currentText()) == 'Local':
                if len(self.layNum.text()) == 0:
                    self.layNum.setEnabled(False)
                else:
                    self.layNum.clear()
                    self.layNum.setEnabled(False)
                self.checkBox.setEnabled(True)
            else:
                self.layNum.setEnabled(True)
                self.checkBox.setEnabled(False)

    def checkChange(self):
        if str(self.comboBox_render.currentText()) == 'Tractor':
            if str(self.pubTyp.currentText()) == 'Alembic':
                self.checkBox.setEnabled(False)
                self.layNum.setEnabled(True)
                self.dpEnb(False)
            else:
                tx = self.mdChk.checkState()
                if str(tx) == "PySide2.QtCore.Qt.CheckState.Unchecked":
                    self.dpEnb(True)
                else:
                    self.dpEnb(False)
                if len(self.layNum.text()) == 0:
                    self.layNum.setEnabled(False)
                else:
                    self.layNum.clear()
                    self.layNum.setEnabled(False)
        else:
            self.dpEnb(False)
            if len(self.layNum.text()) == 0:
                self.layNum.setEnabled(False)
            else:
                self.layNum.clear()
                self.layNum.setEnabled(False)
            if str(self.pubTyp.currentText()) == 'Alembic':
                self.checkBox.setEnabled(True)

    def texAttr(self):
        '''
        모든 Mesh Drive 캐릭터에 에이전트 ID, 텍스쳐 경로 Attribute 추가
        agline_03 : xxxx_xx_xx_03.abc 파일로 출력    (local Export 불가)
        agtx_5 : txVarNum = 5번으로 출력     (local Export 가능)
        :return: None
        '''
        sgMap = dict()
        sel = cmds.ls("MDGGrp_*", dag=True, type='surfaceShape', ni=True)
        selTr = cmds.ls("MDGGrp_*", dag=True, type='transform', ni=True)
        disTex = list()
        for i in selTr:
            agId = cmds.getAttr('%s.agentId' % i)
            if not cmds.attributeQuery('rman__riattr__user_Agent__Index', n=i, ex=True):
                cmds.addAttr(i, ln='rman__riattr__user_Agent__Index', nn='Agent Id', at='long')
            cmds.setAttr(i + '.rman__riattr__user_Agent__Index', agId)
            for j in cmds.listRelatives(i, c=1):
                if not cmds.attributeQuery('rman__riattr__user_Agent__Index', n=j, ex=True):
                    cmds.addAttr(j, ln='rman__riattr__user_Agent__Index', nn='Agent Id', at='long')
                cmds.setAttr(j + '.rman__riattr__user_Agent__Index', agId)
        shdEngList = list()
        for shpNode in sel:
            for shdEng in cmds.listConnections(str(shpNode), type='shadingEngine'):
                if str(shdEng) in shdEngList:
                    pass
                else:
                    shdEngList.append(str(shdEng))
        for i in shdEngList:
            for j in cmds.listConnections('%s.surfaceShader' % i):
                if cmds.listConnections('%s.color' % j):
                    for k in cmds.listConnections('%s.color' % j):
                        if cmds.nodeType(k) == 'file':
                            sgMap[i] = cmds.getAttr('%s.fileTextureName' % k)
                        elif cmds.nodeType(k) == 'layeredTexture':
                            arw = cmds.listConnections(k, type='file')[0]
                            texF = str(cmds.getAttr('%s.fileTextureName' % arw)).split(".")
                            sgMap[i] = texF[0] + "." + texF[-1]
                        else:
                            cmds.warning(str(k) + " isn't linked any node.")
                else:
                    sgMap[i] = "Null"
                    disTex.append(j)
        for e in sel:
            for r in cmds.listConnections(str(e), type='shadingEngine'):
                if sgMap[r].count("/crowd/asset") == 1:  # 군중 경로로 정리된 텍스쳐를 사용한 경우
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
                    # cmds.setAttr(e + '.rman__riattr__user_txVarNum', rand.randint(1, 9))
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
        if cmds.ls("agtx_*", type="displayLayer"):
            for w in cmds.ls("agtx_*", type="displayLayer"):
                varNumb = str(w).split("_")[-1]
                print varNumb
                for i in cmds.listConnections(str(w) + ".drawInfo", type="transform"):
                    for k in cmds.listRelatives(str(i), c=True, type="shape"):
                        cmds.setAttr(k + '.rman__riattr__user_txVarNum', int(varNumb))
        if len(disTex) != 0:
            print(disTex)
        else:
            pass

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

    def createMayaTempFile(self):   # Save Scene File for Backup
        name = os.path.splitext(str(os.path.basename(cmds.file(q=1, sn=1))))[0]
        exr = os.path.splitext(str(os.path.basename(cmds.file(q=1, sn=1))))[-1]
        temp_name = name
        self.mayafile_tmp = os.path.join(os.path.dirname(str(cmds.file(q=1, sn=1))), 'renderScenes', temp_name + exr)
        if not os.path.exists(os.path.dirname(self.mayafile_tmp)):
            os.makedirs(os.path.dirname(self.mayafile_tmp))
        cmds.file(save=True)
        shutil.copy2( str(cmds.file(q=True, sn=True)), self.mayafile_tmp )

    def postScript(self, Parent=None):
        Parent.addCleanup(author.Command(argv=['/bin/rm -f', self.mayafile_tmp], service=''))

    ##### Alembic Export Metadata #####
    def crw_jobscript(self, name, ofile, subProc):
        job = author.Job()
        job.title = name + str(os.path.basename(str(cmds.file(q=True, sn=True))).split('.')[0])
        job.comment = ''
        job.metadata = ''
        job.envkey = ['miarmy-2017-6.2.18']
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
            ProcPrimTask.addCommand(author.Command(service='', envkey=['miarmy-2017-6.2.18'], tags=['py'], argv=command))
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
            AlmFrameTask.addCommand(author.Command(service='', envkey=['miarmy-2017-6.2.18'], tags=['py'], argv=command))
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
                AlmFrameTask.addCommand(author.Command(service='', envkey=['miarmy-2017-6.2.18'], tags=['py'], argv=command))
                AlembicTask.addChild(AlmFrameTask)
                divFileNumb += 1
        self.postScript(Parent=JobTask)
        job.addChild(JobTask)
        return job
        # return job.asTcl()

    def ribFrameTask(self, Parent=None):
        framePerTask = 10
        stframe = cmds.getAttr('McdGlobal1.startFrame')
        enframe = cmds.getAttr('McdGlobal1.endFrame')
        for i in range(stframe, enframe + 1, framePerTask):
            firstJobFrame = i
            if i + framePerTask > enframe:
                lastJobFrame = enframe
            else:
                lastJobFrame = i + framePerTask - 1
            RibFrameTask = author.Task(title='RibExport %s, %s' % (firstJobFrame, lastJobFrame))
            command = ['mayapy', '%%D(%s/crwRibExport.py)' % ScriptRoot, self.mayafile_tmp, firstJobFrame, lastJobFrame]
            RibFrameTask.addCommand(author.Command(service='Cache', envkey=['miarmy-2017-6.2.18'], tags=['py'], argv=command))
            Parent.addChild(RibFrameTask)

        ##################################
        # Main Job Command
        ##################################

    def rib_trc(self):
        cmds.setAttr("McdGlobal1.boolMaster[10]", False)  # Real Time Display Off
        if cmds.ls("Cam_Transition_Map"):
            cmds.delete("Cam_Transition_Map")
        self.createMayaTempFile()
        chunks = int(self.SB_dispatch_frames.value())
        minTime = int(self.LE_mintime.text())
        maxTime = int(self.LE_maxtime.text())
        mayaFile = str(cmds.file(q=1, sn=1))
        pubFile = os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/" + mayaFile.split(os.sep)[-1].split(".")[0] + "/"
        if not os.path.exists(os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/"):
            os.mkdir(os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/")
        if not os.path.exists(pubFile):
            os.mkdir(pubFile)
        if cmds.ls("MDGGrp_*", type="transform"):
            McdMeshDriveSetupPub.McdMeshDrive2Clear()
        # ===>
        camName = "persp"
        self.makeRePath(pubFile + "rib/" + camName)
        self.makeRePath(pubFile + "ProcPrimAssets")
        ribPath = pubFile + "rib/" + camName + "_rif/"
        self.makeRePath(pubFile + "rib/" + camName + "_rif")
        selB = cmds.ls("McdAgent*", type="transform")
        pointsParticlePath = str(cmds.file(q=1, sn=1).split(os.sep)[-1].split(".")[0])
        self.makeRePath(pubFile + pointsParticlePath + ".abc")
        sceneFilePath = str(cmds.file(q=1, sn=1))
        options = {'m_chunk': chunks, 'm_mayafile': mayaFile, 'm_outdir': pubFile, 'm_start': minTime, 'm_end': maxTime}
        dxArmy.ExportRibSpool(options)
        # Notice Pub Path Information
        qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Rib Cache Path\n"
        qFile = ribPath + "\n\n" + "Particle Point Alembic Path \n"
        qPat = pubFile + pointsParticlePath + ".abc" + "\n\n" + "Scene File \n"
        self.plainTextEdit.insertPlainText(qNum)
        self.plainTextEdit.insertPlainText(qFile)
        self.plainTextEdit.insertPlainText(qPat)
        self.plainTextEdit.insertPlainText(sceneFilePath)

    def makeRePath(self, getPath):
        if os.path.exists(getPath):
            if os.path.splitext(getPath)[-1]:
                newPath = os.sep.join(getPath.split(os.sep)[:-1]) + "/" + os.path.splitext(getPath.split(os.sep)[-1])[0] + "_rev" + os.path.splitext(getPath.split(os.sep)[-1])[-1]
                if os.path.exists(newPath):
                    self.makeRePath(newPath)
                else:
                    os.rename(getPath, newPath)
            else:
                newPath = os.sep.join(getPath.split(os.sep)[:-1]) + "/" + getPath.split(os.sep)[-1] + "_rev"
                if os.path.exists(newPath):
                    self.makeRePath(newPath)
                else:
                    os.rename(getPath, newPath)

    def trc(self):      # Alembic Publish by Tractor
        plainShowName = str(cmds.file(q=1, sn=1))
        self.texAttr()
        self.pathSet()
        self.createMayaTempFile()
        selB = cmds.ls("MDGGrp_*")
        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        cchPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/cache/alembic/"
        if len(cmds.ls("agline_*", type="displayLayer")) == 0:
            pubType = self.crw_jobscript('(Crw-Alm) ', 'AlembicExport_rmantd', 0)
            bbxPath = cchPath + name + ".bbox"  # 파일명_생성시간
            cchFile = cchPath + name + ".abc"
            if len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
                agcamFile = cchFile.replace(".abc", "_CAM.abc")
                agcharFile = cchFile.replace(".abc", "_CHR.abc")
                qFile = cchFile + "\n\n" + "Joint Char \n" + agcharFile + "\n\n" + "Camera \n" + agcamFile + "\n\n" + "Bound Box ( Json ) \n"
            elif len(cmds.ls("agchar_01", type="displayLayer")) == 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
                agcamFile = cchFile.replace(".abc", "_CAM.abc")
                qFile = cchFile + "\n\n" + "Camera \n" + agcamFile + "\n\n" + "Bound Box ( Json ) \n"
            elif len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) == 0:
                agcharFile = cchFile.replace(".abc", "_CHR.abc")
                qFile = cchFile + "\n\n" + "Joint Char \n" + agcharFile + "\n\n" + "Bound Box ( Json ) \n"
            else:
                qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
            qPat = bbxPath + "\n\n" + "Scene File \n"
            qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Alembic Cache \n"
            self.plainTextEdit.insertPlainText(qNum)
            self.plainTextEdit.insertPlainText(qFile)
            self.plainTextEdit.insertPlainText(qPat)
            self.plainTextEdit.insertPlainText(plainShowName)
        else:
            ageNum = 0
            dpList = cmds.ls("agline_*", type="displayLayer")
            chNum = len(dpList)
            for i in dpList:
                ageNum += len(cmds.listConnections(str(i) + ".drawInfo", type="transform"))
            qNum = "Total " + str(ageNum) + " Agents. " + str(chNum) + " Cache Files. \n\n" + "Alembic Cache \n"
            bbxPath = self.shtPth
            cchFile = self.shtPth
            if len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
                qFile = cchFile + "\n\n" + "Joint Char \n" + cchFile + "\n\n" + "Camera \n" + cchFile + "\n\n" + "Bound Box ( Json ) \n"
            elif len(cmds.ls("agchar_01", type="displayLayer")) == 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
                qFile = cchFile + "\n\n" + "Camera \n" + cchFile + "\n\n" + "Bound Box ( Json ) \n"
            elif len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) == 0:
                qFile = cchFile + "\n\n" + "Joint Char \n" + cchFile + "\n\n" + "Bound Box ( Json ) \n"
            else:
                qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
            qPat = bbxPath + "\n\n" + "Scene File \n"
            self.plainTextEdit.insertPlainText(qNum)
            self.plainTextEdit.insertPlainText(qFile)
            self.plainTextEdit.insertPlainText(qPat)
            self.plainTextEdit.insertPlainText(plainShowName)
            pubType = self.crw_jobscript('(Crw-Alm) ', 'AlembicExport_rmantd', 2)
        self.tracshot(pubType)

    def tracshot(self, job):
        job.priority = 1000.0
        author.setEngineClientParam(hostname='10.0.0.25', port=80, user=getpass.getuser(), debug=True)
        job.spool()
        author.closeEngineClient()

    # def tracshot(self, pubTypeProc):
    #     tclscript = pubTypeProc
    #     tclfile = self.mayafile_tmp + '.alf'
    #     f = open(tclfile, 'w')
    #     f.write(tclscript)
    #     f.close()
    #     if sys.platform.find('win') > -1:
    #         return
    #     tractorSpool = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/bin/tractor-spool'
    #     engine = '10.0.0.30'
    #     port = '80'
    #     priority = '1000'
    #     cmd = '%s --engine=%s:%s --priority %s %s' % (tractorSpool, engine, port, priority, tclfile)
    #     p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    #     output, errors = p.communicate()
    #     if output:
    #         print '# job spool result : ' + output
    #     if errors:
    #         print '# job spool errors : ' + errors
    #         os.remove(self.mayafile_tmp)
    #     # clear tclscript
    #     os.remove(tclfile)

    def rib_locl(self):
        globalNode = str(mel.eval('McdSimpleCommand -execute 2'))
        cmds.setAttr(globalNode + ".boolMaster[10]", False)  # Real Time Display Off
        if cmds.ls("Cam_Transition_Map"):
            cmds.delete("Cam_Transition_Map")
        self.createMayaTempFile()
        cmds.setAttr('%s.runProPath' % globalNode, '/netapp/backstage/pub/apps/miarmy2/applications/linux/multiRender_DSO', type='string')
        minT = int(self.LE_mintime.text())
        maxT = int(self.LE_maxtime.text())
        mayaFile = str(cmds.file(q=1, sn=1))
        outPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/data/" + mayaFile.split(os.sep)[-1].split(".")[0] + "/"
        if not os.path.exists(os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/"):
            os.mkdir(os.sep.join(mayaFile.split(os.sep)[:-2]) + "/data/")
        if not os.path.exists(outPath):
            os.mkdir(outPath)
        camName = "persp"
        self.makeRePath(outPath + "rib/" + camName)
        self.makeRePath(outPath + "ProcPrimAssets")
        self.makeRePath(outPath + "rib/" + camName + "_rif")
        ribPath = outPath + "rib/" + camName + "_rif/"
        selB = cmds.ls("McdAgent*", type="transform")
        pointsParticlePath = str(cmds.file(q=1, sn=1).split(os.sep)[-1].split(".")[0])
        self.makeRePath(outPath + pointsParticlePath + ".abc")
        sceneFilePath = str(cmds.file(q=1, sn=1))
        dxArmy.RibExport.ExportRib(0, minT, maxT, outPath)
        dxArmy.RibExport.ExportRib(3, minT, maxT, outPath)
        self.rib_filter(outPath, camName)
        qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Rib Cache Path\n"
        qFile = ribPath + "\n\n" + "Particle Point Alembic Path \n"
        qPat = outPath + pointsParticlePath + ".abc" + "\n\n" + "Scene File \n"
        self.plainTextEdit.insertPlainText(qNum)
        self.plainTextEdit.insertPlainText(qFile)
        self.plainTextEdit.insertPlainText(qPat)
        self.plainTextEdit.insertPlainText(sceneFilePath)

    def locl(self): # Alembic Local Pub
        plainShowName = str(cmds.file(q=1, sn=1))
        self.texAttr()
        self.pathSet()
        self.createMayaTempFile()
        animPlug = '/usr/autodesk/maya2017/bin/plug-ins/AbcExport.so'
        if cmds.pluginInfo(animPlug, q=True, l=True) == False:
            cmds.loadPlugin(animPlug)
            cmds.pluginInfo(animPlug, edit=True, autoload=True)
        minT = cmds.playbackOptions(q=1, minTime=1)
        maxT = cmds.playbackOptions(q=1, maxTime=1)
        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        cchPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/cache/alembic/"
        bbxPath = cchPath + name + ".bbox"  # 파일명_생성시간
        cchFile = cchPath + name + ".abc"
        if len(cmds.ls("agchar_01", type="displayLayer")) == 0:
            pass
        else:
            dr = ""
            for sh in cmds.listConnections("agchar_01.drawInfo", type="transform"):
                dr += " -rt " + str(sh)
            agcharFile = cchPath + name + "_CHR.abc"
            chkCmd = "-fr %f %f -atp rman -wuvs -ws -wv -ef -df ogawa %s -f %s" % (minT, maxT, dr, agcharFile)
            cmds.AbcExport(v=1, j=chkCmd)
        if len(cmds.ls("agcam_01", type="displayLayer")) == 0:
            pass
        else:
            wr = ""
            for sh in cmds.listConnections("agcam_01.drawInfo", type="transform"):
                wr += " -rt " + str(sh)
            agcamFile = cchPath + name + "_CAM.abc"
            camCmd = "-fr %f %f -atp rman -wuvs -ws -wv -ef -df ogawa %s -f %s" % (minT, maxT, wr, agcamFile)
            cmds.AbcExport(v=1, j=camCmd)
        if self.checkBox.isChecked() == True:
            selB = cmds.ls("MDGGrp_*", l=1)
        else:
            selB = cmds.ls(sl=1)
        bs = ""
        for i in selB:
            bs += " -rt " + str(i)
        jobCmd = '''-pythonPerFrameCallback "crwData.abcMa(name='%s',frame=#FRAME#,bounds=#BOUNDSARRAY#)"''' % bbxPath
        jobCmd += " -pythonPostJobCallback crwData.bOx()"
        jobCmd += " -fr %f %f -atp rman -wuvs -ws -wv -ef -df ogawa %s -f %s" % (minT, maxT, bs, cchFile)
        cmds.AbcExport(v=1, j=jobCmd)
        if len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
            qFile = cchFile + "\n\n" + "Joint Char \n" + agcharFile + "\n\n" + "Camera \n" + agcamFile + "\n\n" + "Bound Box ( Json ) \n"
        elif len(cmds.ls("agchar_01", type="displayLayer")) == 0 and len(cmds.ls("agcam_01", type="displayLayer")) != 0:
            qFile = cchFile + "\n\n" + "Camera \n" + agcamFile + "\n\n" + "Bound Box ( Json ) \n"
        elif len(cmds.ls("agchar_01", type="displayLayer")) != 0 and len(cmds.ls("agcam_01", type="displayLayer")) == 0:
            qFile = cchFile + "\n\n" + "Joint Char \n" + agcharFile + "\n\n" + "Bound Box ( Json ) \n"
        else:
            qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
        qPat = bbxPath + "\n\n" + "Scene File \n"
        qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Alembic Cache \n"
        self.plainTextEdit.insertPlainText(qNum)
        self.plainTextEdit.insertPlainText(qFile)
        self.plainTextEdit.insertPlainText(qPat)
        self.plainTextEdit.insertPlainText(plainShowName)

    def rib_filter(self, outPath, camName):
        ribFiles = list()
        if os.path.exists(outPath + "rib/" + camName + "/"):
            ribPath = outPath + "rib/" + camName + "/"
            for i in os.listdir(ribPath):
                if os.path.splitext(i)[-1] == '.rib':
                    ribFiles.append(os.path.join(ribPath + i))
        ribFiles.sort()
        self.mkpath(ribFiles)

    def mkpath(self, files ):
        self.files  = files
        # create output dir
        filepath = os.path.dirname( self.files[0] )
        dirname  = os.path.basename( filepath )
        self.outdir = os.path.join( os.path.dirname(filepath), '%s_rif' % dirname )
        if not os.path.exists(self.outdir):
            os.makedirs( self.outdir )
        self.run()

    def run( self ):
        for i in range( len(self.files)+1 ):
            if i != 0:
                label = self.files[i-1]
                # rif main process
                outFile = os.path.join( self.outdir, os.path.basename(label) )
                rif_process.rifUiDoIt(label, outFile)
        cmds.confirmDialog(title="Notice", message="Local Publishing Completed.")

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
