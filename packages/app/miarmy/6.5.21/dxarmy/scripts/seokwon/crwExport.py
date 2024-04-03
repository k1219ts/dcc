# encoding:utf-8

import maya.cmds as cmds

import sys
import os
import site
import time
import shutil
import subprocess
import json

if sys.platform.find('win') > -1:
    TractorRoot = 'N:/backstage/pub/apps/tractor/win64/Tractor-2.0'
    site.addsitedir('%s/lib/python2.7/Lib/site-packages' % TractorRoot)
else:
    TractorRoot = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2'
    site.addsitedir('%s/lib/python2.7/site-packages' % TractorRoot)

import tractor.api.author as author

ScriptRoot = '/dexter/Cache_DATA/animation/A0_Artist/Choi_SeokWon/Crowd_RnD/script'
renderType = ['Tractor', 'Local']

from Qt import QtCore, QtGui, QtWidgets, load_ui

currentpath = os.path.abspath(__file__)
uiFile = os.path.join(os.path.dirname(currentpath), "../ui/crwExportB.ui")

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

        startframe = cmds.playbackOptions(q=1, minTime=1)
        endframe = cmds.playbackOptions(q=1, maxTime=1)

        if endframe - startframe < 0:
            print "Please check your render frame."
            return

    def connectSignal(self):
        self.ui.almBtn.clicked.connect(self.doIt)
        self.ui.comboBox_render.addItems(renderType)
        self.ui.comboBox_render.setEnabled(True)
        self.ui.comboBox_render.setCurrentIndex(0)
        self.ui.connect(self.ui.comboBox_render, QtCore.SIGNAL("currentIndexChanged(int)"), self.checkChange)

    def checkChange(self, rendtp):

        if rendtp == 0:
            self.ui.checkBox.setEnabled(False)
        else:
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
                    cmds.addAttr(j, ln='rman__riattr__user_Agent__Index', nn='Agent Id', at='float')
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

                            #                if (str(self.randCheck.checkState()) == "PySide.QtCore.Qt.CheckState.Checked"):     # Check Box
                        temDir = "%s/tex/" % os.sep.join(sgMap[r].split(os.sep)[:-2])

                        if not cmds.attributeQuery('rman__riattr__user_txVarNum', n=e, ex=True):
                            cmds.addAttr(e, ln='rman__riattr__user_txVarNum', nn='txVarNum', at='long')

                            # .../indianSoldierA_lowA_diffC_1.tif   _diffC_ 다음에 Randomize Texture가 일의 자리인 경우
                        if (sgMap[r].split(".")[0][-2] == "_"):
                            txVarN = int(sgMap[r].split(os.sep)[-1][-5])
                            temFile = sgMap[r].split(os.sep)[-1][:-12]
                            temTex = temDir + temFile
                            cmds.setAttr(e + '.rman__riattr__user_txVarNum', txVarN)
                            cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')
                            # .../indianSoldierA_lowA_diffC_12.tif   _diffC_ 다음에 Randomize Texture가 십의 자리인 경우
                        elif (sgMap[r].split(".")[0][-3] == "_"):
                            txVarN = int(sgMap[r].split(os.sep)[-1][-6:-4])
                            temFile = sgMap[r].split(os.sep)[-1][:-13]
                            temTex = temDir + temFile
                            cmds.setAttr(e + '.rman__riattr__user_txVarNum', txVarN)
                            cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')
                        elif (sgMap[r].split(".")[0][-1] == "C"):
                            temFile = sgMap[r].split(os.sep)[-1][:-10]
                            temTex = temDir + temFile
                            cmds.setAttr('%s.rman__riattr__user_mapname' % e, temTex, type='string')

                        if cmds.attributeQuery('rman__riattr__user_txAssetName', n=e, ex=True):
                            cmds.deleteAttr('%s.rman__riattr__user_txAssetName' % e)
                            cmds.deleteAttr('%s.rman__riattr__user_txLayerName' % e)

                '''
                    elif (sgMap[r].count("/texture/pub/") == 1):
                        if not cmds.attributeQuery('rman__riattr__user_txVarNum', n=e, ex=True):
                            cmds.addAttr(e, ln='rman__riattr__user_txVarNum', nn='txVarNum', at='long')
                        cmds.setAttr(e + '.rman__riattr__user_txVarNum', rand.randint(0, 2))


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

        self.selFile = os.path.basename(cmds.file(q=1, sn=1))
        self.setPath = os.path.dirname(cmds.file(q=1, sn=1))

        if not os.path.exists(self.setPath + "/../cache/alembic/"):
            if not os.path.exists(self.setPath + "/../cache/"):
                os.mkdir(self.setPath + "/../cache/")
            os.mkdir(self.setPath + "/../cache/alembic/")

    def createMayaTempFile(self):

        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        exr = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[-1]
        temp_name = name + '_' + self.temp  # 파일명_생성시간

        self.mayafile_tmp = os.path.join(os.path.dirname(cmds.file(q=1, sn=1)), 'renderScenes', temp_name + exr)
        print self.mayafile_tmp

        # mkdir
        if not os.path.exists(os.path.dirname(self.mayafile_tmp)):
            os.makedirs(os.path.dirname(self.mayafile_tmp))
        # save as
        cmds.file(save=True)

        shutil.copy2(self.mayafile, self.mayafile_tmp)

    def postScript(self, Parent=None):
        Parent.addCleanup(author.Command(argv=['/bin/rm -f', self.mayafile_tmp], service='Saga0'))

    ##### Alembic Export Metadata #####

    def crwAlm_jobscript(self):
        job = author.Job()
        job.title = '(Crw-Alm) ' + str(os.path.basename(self.mayafile).split('.')[0])
        job.comment = ''
        job.metadata = ''
        job.envkey = ['cache2016.5']
        job.service = 'Saga0'
        job.maxactive = 10
        job.tier = 'cache'
        job.projects = ['crw']
        job.tags = ['cache']

        job.newDirMap(src='X:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')
        job.newDirMap(src='R:/', dst='/dexter/', zone='NFS')

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1

        crwAlembicTask = author.Task(title='crwAlembicTask')
        command = ['mayapy', '%%D(%s/AlembicExport_rmantd.py)' % ScriptRoot, self.mayafile_tmp]
        crwAlembicTask.addCommand(author.Command(service='Saga0', envkey=['cache2016.5'], tags=['py'], argv=command))
        JobTask.addChild(crwAlembicTask)

        self.postScript(Parent=JobTask)

        job.addChild(JobTask)
        return job.asTcl()

        ##################################
        # Main Job Command
        ##################################

    def trc(self):
        self.plainShowName = str(cmds.file(q=1, sn=1))
        self.texAttr()
        self.pathSet()
        self.createMayaTempFile()
        selB = cmds.ls("MDGGrp_*")

        name = os.path.splitext(os.path.basename(cmds.file(q=1, sn=1)))[0]
        cchPath = os.sep.join(str(cmds.file(q=1, sn=1)).split(os.sep)[:-2]) + "/cache/alembic/"

        bbxPath = cchPath + name + '_' + self.temp + ".bbox"  # 파일명_생성시간
        cchFile = cchPath + name + '_' + self.temp + ".abc"

        qFile = cchFile + "\n\n" + "Bound Box ( Json ) \n"
        qPat = bbxPath + "\n\n" + "Scene File \n"
        qNum = "Total " + str(len(selB)) + " Agents \n\n" + "Alembic Cache \n"

        self.ui.plainTextEdit.insertPlainText(qNum)
        self.ui.plainTextEdit.insertPlainText(qFile)
        self.ui.plainTextEdit.insertPlainText(qPat)
        self.ui.plainTextEdit.insertPlainText(self.plainShowName)


        tclscript = self.crwAlm_jobscript()

        tclfile = self.mayafile_tmp + '.alf'
        f = open(tclfile, 'w')
        f.write(tclscript)
        f.close()

        if sys.platform.find('win') > -1:
            return

        tractorSpool = '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/bin/tractor-spool'
        engine = '10.0.0.30'
        port = '80'
        priority = '1000'
        cmd = '%s --engine=%s:%s --priority %s %s' % (tractorSpool, engine, port, priority, tclfile)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

        output, errors = p.communicate()
        if output:
            print '# job spool result : ' + output
        if errors:
            print '# job spool errors : ' + errors
            os.remove(self.mayafile_tmp)

        # clear tclscript
        os.remove(tclfile)

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

    def doIt(self):


        if (self.ui.comboBox_render.currentText() == 'Tractor'):
            self.trc()
        elif (self.ui.comboBox_render.currentText() == 'Local'):
            self.locl()
        else:
            pass

def main():
    global myWindow
    myWindow = Window()
    myWindow.ui.show()


if __name__ == '__main__':
    main()