
# !/usr/bin/python2.7
from __future__ import print_function

import os
import subprocess

import DXUSD.Message as msg
import PUtils as putl
import TUtils as tutl
from DXUSD.Structures import Arguments
import DXUSD.Vars as var
from pxr import Usd, Sdf

import tractor.api.author as author
import getpass

SERVICE_KEY = "Cache"
MAX_ACTIVE = 10
PROJECTS = ["export"]
TIER = "cache"
TAGS = ["GPU"]
ENVIROMNET_KEY = ""
TRACTOR_IP = '10.0.0.25'
PORT = 80

class convertPack():
    def __init__(self, **kwargs):
        self.dst = kwargs['dst']
        self.show = kwargs['show']
        self.list = kwargs['list']
        self.lyrTypes = kwargs['lyrTypes']
        self.dataType = kwargs['dataType']
        self.textureType = kwargs['textureType']
        self.machineType = kwargs['machineType']
        self.user = getpass.getuser()

        if self.machineType == 'TRACTOR':
            self.Spool()
        else:
            self.doit(self.list)

    def Spool(self):
        #jobtile = ','.join(self.list) if len(self.list) >1 else self.list[0]
        jobtile = self.list[0]
        # jobtile = '(USD2MAYA) [%s] %s' % ('test', 'test')
        jobtile = '(USD2MAYA) [%s] %s' % (str(self.show), str(jobtile))
        comment = 'out directory : %s' % str(self.dst)
        metadata = 'lyrTypes : %s' % self.lyrTypes

        job = author.Job()
        job.title = jobtile
        job.comment = comment
        job.metadata = metadata
        job.service = SERVICE_KEY
        job.maxactive = MAX_ACTIVE
        job.tier = TIER
        job.tags = TAGS
        job.projects = PROJECTS

        self.rootJob = author.Task(title='root')
        job.addChild(self.rootJob)

        self.doit(self.list)
        self.SpoolJob(job, self.user)

    def MakeCommand(self, cmd):
        return author.Command(argv=cmd, service=SERVICE_KEY)

    def SpoolJob(self, job, user):
        job.priority = 100
        author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user= user, debug=True)
        job.spool()
        author.closeEngineClient()


    def doit(self, pathlist):
        for asset in pathlist:
            task = 'asset'
            dstdir = os.path.join(self.dst, '_3d', 'asset')
            if '_' in asset:
                task = 'shot'
                dstdir = os.path.join(self.dst, '_3d', 'shot')

            usdpath = self.getUsdPath(self.show, asset, task)
            # print('usdpath:',usdpath)

            if usdpath:
                if task == 'asset':
                    if self.IsTask(usdpath,'branch'):
                        branchlist = putl.GetBranchList(usdpath)
                        if branchlist:
                            if not self.dataType == 'Locator':
                                self.branch_doit(branchlist,dstdir)

                    if self.dataType == 'Locator':
                        self.ReferenceDoit(usdpath)

                    if self.IsTask(usdpath, 'model') or self.IsTask(usdpath, 'rig'):
                        print('usdpath:', usdpath)
                        self.makeSubTask(usdpath, dstdir, self.dataType)

                    if self.IsTask(usdpath, 'agent'):
                        self.BatchETC(usdpath, dstdir, self.dataType)
                        print ('test')

                elif task == 'shot':
                    self.shot_doit(usdpath, dstdir, asset)


    def branch_doit(self, pathlist, dstdir):
        for index, usdpath in enumerate(pathlist):
            self.makeSubTask(usdpath, dstdir, self.dataType)


    def shot_doit(self,usdpath, dstdir, asset):
        shot = asset
        pathList, sceneFiles = putl.GetAniReferenceList(usdpath)
        print('>>>pathList:',pathList)

        if 'asset' in os.listdir(os.path.dirname(usdpath)):
            assetdir = os.path.join(os.path.dirname(usdpath), 'asset')
            print('assetdir:',assetdir)
            if os.path.exists(assetdir):
                for asset in os.listdir(assetdir):
                    assetpath = os.path.join(assetdir, asset, '%s.usd'%asset )
                    if not assetpath in pathList:
                        pathList.append(assetpath)

        # if pathList:
        #     for path in pathList:
        #         if 'wrecker' in path:
        #             self.Batch(path, dstdir, self.dataType)

        if sceneFiles:
            seq = shot.split('_')[0]
            dstdir = os.path.join(self.dst, '_3d', 'shot', seq, shot, 'scenes')
            putl.MakeDir(dstdir)
            for sceneFile in sceneFiles:
                putl.CopyFile(sceneFile, dstdir)

        layoutusdpath = putl.GetLayoutMaster(usdpath)
        if layoutusdpath:
            print('layoutusdpath:',layoutusdpath)
            # reflist = self.GetReferenceList(layoutusdpath)
            self.ReferenceDoit(layoutusdpath)

            dstdir = os.path.join(self.dst, '_3d', 'asset')
            self.makeSubTask(layoutusdpath, dstdir, dataType = self.dataType)


    def BatchETC(self, usdpath, dstdir, dataType = 'Mesh'):
        if self.machineType == 'TRACTOR':
            title = usdpath.split('/')[-1].split('.')[0]
            JobTask = author.Task(title=title)
            self.rootJob.addChild(JobTask)
            self.JobTask = JobTask

        cmd = self.BatchCommand(dstdir, usdpath, "Model", dataType=dataType, machineType=self.machineType)
        agentTask = author.Task(title='Agent Task')
        agentTask.addCommand(self.MakeCommand(cmd))
        self.JobTask.addChild(agentTask)


    def makeSubTask(self,usdpath, dstdir, dataType = 'Mesh'):
        if self.machineType == 'TRACTOR':
            title = usdpath.split('/')[-1].split('.')[0]
            JobTask = author.Task(title=title)
            self.rootJob.addChild(JobTask)
            self.JobTask = JobTask

        if self.IsTask(usdpath, 'groom'):
            self.lyrTypes.append('Groom')

        taskList =[]
        lyrTypes = self.lyrTypes
        if dataType == 'Reference':
            lyrTypes= ['Texture', 'Model']

        print('lyrTypes:',lyrTypes)
        # lyrTypes = []
        for lyrtype in lyrTypes :
            cmd = self.BatchCommand(dstdir, usdpath, lyrtype, dataType=dataType, machineType=self.machineType)
            if cmd:
                data = {}
                data[lyrtype] = cmd
                taskList.append(data)

        if taskList:
            for data in taskList:
                for k, v in data.items():
                    if k == 'Texture':
                        texTask = author.Task(title='%s Task' %k)
                        texTask.addCommand(self.MakeCommand(v))
                    if k == 'Model':
                        modelTask = author.Task(title='%s Task' %k)
                        modelTask.addCommand(self.MakeCommand(v))
                    if k == 'Rig':

                        rigTask = author.Task(title='%s Task' %k)
                        rigTask.addCommand(self.MakeCommand(v))
                    if k == 'Groom':
                        groomTask = author.Task(title='%s Task' %k)
                        groomTask.addCommand(self.MakeCommand(v))

            if 'Model' in lyrTypes and 'Rig' in lyrTypes:
                if 'Texture' in lyrTypes:
                    modelTask.addChild(texTask)
                rigTask.addChild(modelTask)
                self.MakeGroomTask(rigTask, cmd, lyrTypes)

            elif 'Model' in lyrTypes:
                if 'Texture' in lyrTypes:
                    modelTask.addChild(texTask)
                self.JobTask.addChild(modelTask)

            elif 'Rig' in lyrTypes:
                if 'Texture' in lyrTypes:
                    rigTask.addChild(texTask)
                self.MakeGroomTask(rigTask, cmd, lyrTypes)

            else:
                self.JobTask.addChild(texTask)


    def MakeGroomTask(self, rigTask, cmd, lyrTypes):
        if 'Groom' in lyrTypes:
            groomTask = author.Task(title='Groom Task')
            groomTask.addCommand(self.MakeCommand(cmd))
            groomTask.addChild(rigTask)
            self.JobTask.addChild(groomTask)
        else:
            self.JobTask.addChild(rigTask)


    def ReferenceDoit(self, usdpath):
        # if not self.dataType == 'Locator':
        #     return
        self.reflist = self.GetReferenceList(usdpath)
        if not self.reflist:
            return

        arg = Arguments()
        arg.D.SetDecode(usdpath)
        dstdir =''
        asset = usdpath.split('/')[-1].split('.')[0]

        #get dstdir
        if arg.has_key('asset'):
            dstdir = os.path.join(self.dst, '_3d', 'asset', asset)

        if arg.has_key('shot'):
            shotname = '%s_%s' % (arg.seq, arg.shot)
            dstdir = os.path.join(self.dst, '_3d', 'asset', shotname)

        # doit
        for index, refpath in enumerate(self.reflist):
            self.makeSubTask(refpath, dstdir , dataType='Reference')



    def BatchCommand(self, dstdir, usdpath, lyrtype, dataType='mesh', task='asset', machineType='LOCAL'):

        batchcmd = '%s/Batch.py' % os.path.dirname(__file__)
        dcc = '/backstage/dcc/DCC'

        if machineType == 'LOCAL':
            cmd = dcc
            cmd += ' maya --zelos --terminal'
            cmd += ' mayapy %s ' % batchcmd
            cmd += ' --src %s' % usdpath
            cmd += ' --dst %s' % dstdir
            cmd += ' --lyrtype %s' % lyrtype
            cmd += ' --dataType %s' % dataType
            cmd += ' --task %s' % task
            # print(cmd)
            subprocess.Popen(cmd, shell=True).wait()
            return

        elif machineType == 'TRACTOR':
            cmd = [dcc, 'rez-env']
            cmd += ['maya-2018' ,'prmantoolkit']
            cmd += ['dxrulebook-1.0.0', 'python-2.7.16', 'dxusd-2.0.0', 'dxusd_maya-1.0.0', 'usd_maya-19.11']
            cmd += ['alembic-1.7.1', 'zelos',  'baselib-2.5', 'maya_rigging','renderman-23.5']
            cmd += ['--', 'mayapy', batchcmd]
            cmd += ['--dst', dstdir]
            cmd += ['--src', usdpath]
            cmd += ['--task', task]
            cmd += ['--lyrtype', lyrtype]
            cmd += ['--dataType', dataType]
            return cmd


    def GetReferenceList(self, usdpath):
        reflist = []
        stage = Usd.Stage.Open(usdpath)
        dprim = stage.GetDefaultPrim()
        tutl.walk(stage, dprim, reflist, usdpath)
        return reflist


    def IsTask(self, usdpath, task=''):
        from pxr import Usd
        isTask = False
        stage = Usd.Stage.Open(usdpath)
        dPrim = stage.GetDefaultPrim()
        if task in dPrim.GetVariantSet('task').GetVariantNames():
            isTask = True
        return isTask


    def getUsdPath(self ,show ,asset ,task):
        usdpath =''

        if task == 'asset':
            usdpath = '/show/{SHOW}/_3d/asset/{ASSET}/{ASSET}.usd'.format(SHOW=show,
                                                                          ASSET=asset)

            if not os.path.exists(usdpath):
                usdpath = '/assetlib/_3d/asset/{ASSET}/{ASSET}.usd'.format(ASSET=asset)



        if task == 'shot':
            shot = asset
            seq = shot.split('_')[0]
            usdpath = '/show/{SHOW}/_3d/shot/{SEQ}/{SHOT}/{SHOT}.usd'.format(SHOW=show,
                                                                             SHOT=shot,
                                                                             SEQ=seq)

        if not os.path.exists(usdpath):
            # self.msgBox('Error: Please check AssetName')
            pass
        else:
            return usdpath

    def msgBox(self, message = ''):
        from PySide2 import QtWidgets
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(message)
        msg.exec_()





# def CopyUsdfile(dstdir, path, type=''):
#     arg = Arguments()
#     arg.D.SetDecode(path)
#     scdir = arg.D.ASSET
#
#     if type == 'branch':
#         dstdir = os.path.join(dstdir, arg.asset)
#
#     putl.MakeDir(dstdir)
#     putl.CopyFile(scdir, dstdir)
