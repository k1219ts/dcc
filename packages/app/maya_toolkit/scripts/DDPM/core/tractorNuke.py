#encoding=utf-8
#!/usr/bin/env python

import os
import site
import dxConfig
TractorRoot = dxConfig.getConf('TRACTOR_API')
site.addsitedir(TractorRoot)
import tractor.api.author as author

backStageMayaPath = os.getenv("BACKSTAGE_MAYA_PATH")
DDPMPATH = "{0}/global/DDPM".format(backStageMayaPath)

class JobScript:
    def __init__(self, platePath, m_shotName,
                 m_project, userName, startFrame, endFrame, isRetime, sound):
        """
        platePath  : ex. /show/kfyg/shot/CHS/CHS_0465/ani/dev/preview/CHS_0465_ani_v01_w05
        m_shotName : CHS_0465_ani_v01_w05
        m_project  : CHS_0465
        userName   : gyeongheon.jeong
        """

        self.platePath = platePath
        self.m_shotName = m_shotName
        self.m_projectName = m_project
        self.userName = userName
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isRetime = isRetime
        self.soundFile = sound

    def jobScript(self):
        job = author.Job()
        job.title = '(Preview) ' + str(self.m_shotName)
        job.comment = ''
        job.metadata = ''

        #maya_version = os.getenv('MAYA_VER')
        maya_version = "maya2017"
        job.envkey = ['cache%s' % maya_version]
        job.service = 'Cache'
        job.maxactive = 10
        job.tier = 'cache'
        job.projects = ['preview']

        # directory mapping
        job.newDirMap(src='S:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')
        job.newDirMap(src='R:/', dst='/dexter/', zone='NFS')

        JobTask = author.Task(title='Job')
        JobTask.serialsubtasks = 1

        prvTask = author.Task(title='preview')

        taskCmd =  ["/usr/local/Nuke9.0v5/Nuke9.0", "-t"]
        taskCmd += ["{0}/core/nukeMov.py".format( DDPMPATH )]
        taskCmd += [ self.platePath, str(self.startFrame), str(self.endFrame),
                     self.m_shotName, self.m_projectName, self.userName, self.isRetime, self.soundFile ]

        prvTask.addCommand( author.Command(argv=taskCmd, service='HW', tags=['mayapy']) )
        JobTask.addChild(prvTask)

        job.addChild(JobTask)
        job.priority = 1000
        author.setEngineClientParam(hostname='10.0.0.25', port=80,
                                    user=self.userName, debug=True)
        job.spool()
        author.closeEngineClient()
        return job.asTcl()

    def spool(self):
        tclscript = self.jobScript()
