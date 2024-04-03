#coding:utf-8
from __future__ import print_function

import os, sys, re, json, subprocess, getpass

scriptsDir = os.path.dirname(__file__)

import tractor.api.author as author
from tractor.TractorEngine import TractorEngine

import DXRulebook.Interface as rb


class VendorCheck:
    def __init__(self, mayaFileList, vendorScript='', spoolCache=False, overwrite=False, user=''):
        self.TRACTOR_IP = '10.0.0.25'
        self.PORT = 80
        self.mayaFileList = mayaFileList.split(',')
        self.spoolCache = spoolCache
        self.overwrite = overwrite
        self.user = getpass.getuser()
        if user != '':
            self.user = user
        self.vendorScript = scriptsDir+'/vendorMaya.py'
        if vendorScript != '':
            self.vendorScript = vendorScript

    def subTask(self, parent):
        for mf in self.mayaFileList:
            prevtask = author.Task(title=str(os.path.basename(mf)))
            parent.addChild(prevtask)

            snPath = mf
            fileName = os.path.basename(snPath)
            fileNameNoext = os.path.splitext(fileName)[0]

            showName = fileName.split('_')[0]
            showRulebookFile = os.path.join('/show', showName, '_config', 'DXRulebook.yaml')
            if os.path.exists(showRulebookFile):
                os.environ['DXRULEBOOKFILE'] = showRulebookFile
                rb.Reload()

            try:
                coder = rb.Coder()
                tmpv = coder.F.MAYA.vendor.Decode(fileName)
                tmpv.departs = 'ANI'
                showCode = tmpv.show

                aniWorks = coder.D.ANI.WORKS.Encode(**tmpv)
                aniFile = coder.F.MAYA.BASE.Encode(**tmpv)
                snPathAs = os.path.join(aniWorks, aniFile)
            except:
                print(u'파일 이름이 올바르지 않습니다.')
                return
                # os._exit(1)

            command = ['/backstage/dcc/DCC', 'rez-env',
                'pylibs-2.7', 'assetbrowser', 'baselib-2.5', 'maya-2018',
                'maya_animation', 'maya_asset', 'maya_layout', 'maya_matchmove', 'maya_rigging', 'maya_toolkit',
                'dxrulebook-1.0.0', 'python-2', 'dxusd-2.0.0', 'dxusd_maya-1.0.0', 'usd_maya-19.11',
                '--', 'mayapy',
                scriptsDir+'/vendorPreview.py',
                mf,
                snPathAs
            ]

            print(command)
            prevtask.addCommand(
                author.Command(argv=command)
            )

            task = author.Task(title=str(os.path.basename(mf)))
            prevtask.addChild(task)

            command = ['/backstage/dcc/DCC', 'rez-env',
                'pylibs-2.7', 'assetbrowser', 'baselib-2.5', 'maya-2018',
                'maya_animation', 'maya_asset', 'maya_layout', 'maya_matchmove', 'maya_rigging', 'maya_toolkit',
                'dxrulebook-1.0.0', 'python-2', 'dxusd-2.0.0', 'dxusd_maya-1.0.0', 'usd_maya-19.11',
                '--', 'mayapy',
                self.vendorScript,
                mf,
                self.user
            ]

            # if self.spoolCache:
            #     command.append('cache')

            if self.overwrite:
                command.append('overwrite')

            command.append('user='+self.user)

            print(command)
            task.addCommand(
                author.Command(argv=command)
            )

    def spool(self):
        title = '(VENDOR CHECK) '+str(os.path.dirname(self.mayaFileList[0]))
        job = author.Job(
            title=title,
            tier='cache',
            priority=100,
            # paused=True,
            projects=['export'],
            service='Cache',
            comment='',
            metadata=''
        )

        jobMsgCmd = ['/backstage/dcc/DCC', 'rez-env', 'rocketchattoolkit', '--', 'TrBotMsg']
        # Error
        job.newPostscript(argv=jobMsgCmd + ['-b', 'BadBot'], when='error')
        # Done
        job.newPostscript(argv=jobMsgCmd + ['-b', 'GoodBot'], when='done')

        JobTask = author.Task(title='job')
        # JobTask.serialsubtasks = 1
        job.addChild(JobTask)
        self.subTask(JobTask)
        print(job)
        # return
        engine = TractorEngine(hostname=self.TRACTOR_IP, port=self.PORT, user=self.user, debug=True)
        state, msg = engine.spool(job)
        print('# STATE')
        print('\t', state)
        print('# MESSAGE')
        print('\t', msg)

        return state, msg

if __name__ == '__main__':
    if len(sys.argv) < 3:
        exit(1)

    if sys.argv[1] == 'VendorCheck':
        vendorScript = ''
        spoolCache=False
        overwrite=False
        user=getpass.getuser()

        for arg in sys.argv:
            if arg.startswith('vendorScript='):
                vendorScript = arg.split('=')[-1]
            # elif arg.startswith('spoolCache='):
            #     spoolCache = bool(eval(arg.split('=')[-1]))
            elif arg.startswith('overwrite='):
                overwrite = bool(eval(arg.split('=')[-1]))
            elif arg.startswith('user='):
                user = arg.split('=')[-1]

        vndChk = VendorCheck(sys.argv[2], vendorScript=vendorScript, spoolCache=spoolCache, overwrite=overwrite, user=user)
        vndChk.spool()
