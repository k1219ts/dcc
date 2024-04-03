#coding:utf-8

# !/usr/bin/python2.7
from __future__ import print_function

import os
import re
import subprocess

import DXUSD.Message as msg
from usd2maya import PUtils as putl
from usd2maya import TUtils as tutl
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

DEV = re.match("/works/", __file__)
PACKAGE_FOLDER = "/backstage/dcc/packages/app/dxpackager/1.0/scripts"


class AssetConvertPack():
    def __init__(self, **kwargs):
        self.dst = kwargs['dst']
        self.show = kwargs['show']
        self.assetlist = kwargs['list']
        self.packtasks = kwargs['tasklist']
        self.tasklist = []
        self.envlist = kwargs['envlist']
        self.texFmt = kwargs['texFmt']
        self.machineType = kwargs['machineType']
        self.user = getpass.getuser()
        self.dataformat = ''

        if self.machineType == 'Tractor':
            self.spool()
        else:
            self.doit()

    def doit(self):
        task = 'asset'
        dstdir = os.path.join(self.dst, '_3d', 'asset')

        for asset in self.assetlist:
            usdpath = self.getusdpath(self.show, asset, task)
            if usdpath:
                if task == 'asset':
                    if putl.istask(usdpath,'branch'):
                        branchlist = putl.getbranchlist(usdpath)
                        if branchlist:
                            for branchpath in branchlist:
                                self.convert(dstdir, branchpath)

                    if putl.istask(usdpath, 'model') or putl.istask(usdpath, 'rig') or putl.istask(usdpath, 'lidar'):
                        self.convert(dstdir, usdpath)

        if self.envlist:
            for asset in self.envlist:
                usdpath = self.getusdpath(self.show, asset, task)
                self.convert_reference(usdpath)
                self.convert(dstdir, usdpath, dataType='Locator')


    def spool(self):
        try: jobtile = self.assetlist[0]
        except: jobtile = 'unknown'
        jobtile = '(USD2MAYA) [%s] %s' % (str(self.show), str(jobtile))
        comment = 'out directory : %s' % str(self.dst)
        metadata = 'tasklist : %s' % self.packtasks

        job = author.Job()
        job.title = jobtile
        job.comment = comment
        job.metadata = metadata
        job.service = SERVICE_KEY
        job.maxactive = MAX_ACTIVE
        job.tier = TIER
        job.tags = TAGS
        job.projects = PROJECTS
        self.Root= author.Task(title='root')
        job.addChild(self.Root)
        self.doit()
        self.spooljob(job, self.user)

    def spooljob(self, job, user):
        job.priority = 100
        author.setEngineClientParam(hostname=TRACTOR_IP, port=PORT, user= user, debug=True)
        job.spool()
        author.closeEngineClient()

    def convert(self, dstdir, usdpath, dataType='Mesh'):
        self.tasklist = []
        for task in self.packtasks:
            if putl.istask(usdpath, task):
                if 'model' == task:
                    self.tasklist += ['texture']
                self.tasklist += [task]

            else:
                print('>>>>> %s TASK : False' %task )

        if self.machineType == 'Tractor':
            title = usdpath.split('/')[-1].split('.')[0]
            jobtask = author.Task(title=title)
            self.Root.addChild(jobtask)
            self.command_spool(jobtask, dstdir, usdpath, dataType='Mesh', texFmt=self.texFmt)

        else:
            for task in self.tasklist:
                lyrtype = task
                self.command_local(dstdir, usdpath, lyrtype, dataType=dataType, texFmt=self.texFmt)
        print('>>>>>Finished Asset Convert')

    def convert_reference(self, usdpath):
        self.reflist = putl.GetReferenceList(usdpath)
        if not self.reflist:
            return

        arg = Arguments()
        arg.D.SetDecode(usdpath)
        asset = usdpath.split('/')[-1].split('.')[0]
        refdir = ''
        if arg.has_key('asset'):
            refdir = os.path.join(self.dst, '_3d', 'asset', asset)
        # if arg.has_key('shot'):
        #     shotname = '%s_%s' % (arg.seq, arg.shot)
        #     refdir = os.path.join(self.dst, '_3d', 'asset', shotname)

        # doit
        for refpath in self.reflist:
            self.convert(refdir, refpath)


    def command_spool(self,jobtask, dstdir, usdpath, dataType='Mesh', texFmt='texture'):
        for task in self.tasklist:
            command = self.command_local(dstdir, usdpath, task, dataType=dataType, texFmt=texFmt)
            if task == 'model':
                modeltask = author.Task(title=str('model'))
                modeltask.addCommand(author.Command(argv=command))
                # prevcommand = self.command_preview(usdpath, dstdir+'/preivew')
                # modeltask.addCommand(author.Command(argv=prevcommand))
                jobtask.addChild(modeltask)

                if 'texture' in self.tasklist:
                    textask = author.Task(title=str('texture'))
                    modeltask.addChild(textask)
                    texcommand = self.command_local(dstdir, usdpath, 'texture', dataType=dataType, texFmt=texFmt)
                    textask.addCommand(author.Command(argv=texcommand))

            elif task == 'rig':
                rigtask = author.Task(title=str('rig'))
                rigtask.addCommand(author.Command(argv=command))
                # prevcommand = self.command_preview(usdpath, dstdir+'/preivew')
                # rigtask.addCommand(author.Command(argv=prevcommand))
                jobtask.addChild(rigtask)

            elif task == 'texture':
                continue

            # elif task == 'prevtex':
            #     prevtask = author.Task(title=str('prevtex'))
            #     modeltask.addChild(prevtask)
            #     prevcommand = self.command_local(dstdir, usdpath, 'prevtex', dataType=dataType, texFmt=texFmt)
            #     prevtask.addCommand(author.Command(argv=prevcommand))

            else:
                etctask = author.Task(title=str(task))
                etctask.addCommand(author.Command(argv=command))
                jobtask.addChild(etctask)

    # def command_preview(self, sceneFile, previewDir):
    #     batchfile = '%s/assetpreview.py' % os.path.dirname(__file__)
    #     cmd = ['/backstage/dcc/DCC', 'rez-env']
    #     cmd += ['maya-2018' ,'prmantoolkit']
    #     cmd += ['dxrulebook-1.0.0', 'python-2.7.16', 'dxusd-2.0.0', 'dxusd_maya-1.0.0', 'usd_maya-19.11']
    #     cmd += ['alembic-1.7.1', 'zelos',  'baselib-2.5', 'maya_rigging','renderman-23.5']
    #     cmd += ['--', 'mayapy', batchfile]
    #     cmd += [sceneFile]
    #     cmd += [previewDir]

    #     if self.machineType == 'Local':
    #         subprocess.Popen(' '.join(cmd), shell=True).wait()

    #     return cmd

    def command_local(self,dstdir, usdpath, lyrtype, dataType='Mesh', categoryTask='asset', texFmt='texture'):
        folder = os.path.join(PACKAGE_FOLDER, "usdpackager") if DEV else os.path.dirname(__file__)
        batchfile = os.path.join(folder, "usd2maya/mayaBatch.py")
        if self.machineType == 'Local':
            cmd = '/backstage/dcc/DCC'
            cmd += ' maya --zelos --terminal'
            if lyrtype == 'texture':# or lyrtype == 'prevtex':
                cmd += ' python %s ' % batchfile
            else:
                cmd += ' mayapy %s ' % batchfile
            cmd += ' --src %s' % usdpath
            cmd += ' --dst %s' % dstdir
            cmd += ' --lyrtype %s' % lyrtype
            cmd += ' --dataType %s' % dataType
            cmd += ' --task %s' % categoryTask
            cmd += ' --texFmt %s' % texFmt
            subprocess.Popen(cmd, shell=True).wait()
            return cmd

        elif self.machineType == 'Tractor':
            cmd = ['/backstage/dcc/DCC', 'rez-env']
            cmd += ['maya-2018' ,'prmantoolkit']
            cmd += ['dxrulebook-1.0.0', 'python-2.7.16', 'dxusd-2.0.0', 'dxusd_maya-1.0.0', 'usd_maya-19.11']
            cmd += ['alembic-1.7.1', 'zelos',  'baselib-2.5', 'maya_rigging','renderman-23.5']
            if lyrtype == 'texture':# or lyrtype == 'prevtex':
                cmd += ['--', 'python', batchfile]
            else:
                cmd += ['--', 'mayapy', batchfile]
            cmd += ['--dst', dstdir]
            cmd += ['--src', usdpath]
            cmd += ['--task', categoryTask ]
            cmd += ['--lyrtype', lyrtype]
            cmd += ['--dataType', dataType]
            cmd += ['--texFmt', texFmt]
            return cmd


    def getusdpath(self ,show ,asset ,task):
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
        if os.path.exists(usdpath):
            return usdpath
