'''
'    @author    : daeseok.chae
'    @date      : 2017.02.10
'    @brief     : Tractor use script
'''
from MessageBox import MessageBox
import sys
sys.path.append('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.0/lib/python2.7/site-packages')
import getpass
import tractor.api.author as author

import dxConfig

class TractorPublish():
    def __init__(self):
        self.svc = "Cache"
        self.hostName = dxConfig.getConf("TRACTOR_CACHE_IP")
        self.port = 80
        
        
    def createJob(self, jobName):
        jobTitle = '(Animbrowser) %s' % str(jobName)
        self.tractorJob = author.Job(title=jobTitle,
                                     priority=999,
                                     tier='batch',
                                     projects=['export'],
                                     service=self.svc)
        
    def createRootTask(self, rootTaskName = "Empty for Serial"):
        self.rootTask = author.Task(title = rootTaskName)
        self.rootTask.serialsubtasks = True
        self.tractorJob.addChild(self.rootTask)
        
    def addTask(self, parentTask, title, command):
        task = author.Task(title = title,
                           argv = command,
                           service = self.svc)
        parentTask.addChild(task)
        
    def sendJobSpool(self, debug = False):
        author.setEngineClientParam(hostname = self.hostName,
                                    port = 80,
                                    user = getpass.getuser(),
                                    debug = debug)
        
        self.tractorJob.spool()
        author.closeEngineClient()
        
        MessageBox("Success tractor Job")
