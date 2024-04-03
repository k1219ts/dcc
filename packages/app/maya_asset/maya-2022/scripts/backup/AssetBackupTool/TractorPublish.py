'''
'    @author    : daeseok.chae
'    @date      : 2017.03.30
'    @brief     : Tractor use script
'''
from MessageBox import MessageBox
import sys
import dxConfig
sys.path.append(dxConfig.getConf("TRACTOR_API"))
import getpass
import tractor.api.author as author

class TractorPublish():
    def __init__(self, hostName = dxConfig.getConf("TRACTOR_CACHE_IP"), port = dxConfig.getConf("TRACTOR_PORT")):
        '''
        :param hostName: tractor hostIP : 10.0.0.30 (by deafult)
        :param port: tractor hostPort : 80 (by default)
        '''
        self.svc = "Cache||USER"
        self.hostName = hostName
        self.port = port
        self.tractorJob = None
        
    def createJob(self, jobName, priority = 10):
        jobTitle = jobName
        self.tractorJob = author.Job(title=jobTitle,
                                     priority=priority,
                                     tier='batch',
                                     projects=['export'],
                                     service=self.svc)
        self.tractorJob.tags = ["GPU"]
        
    def createRootTask(self, rootTaskName = "Empty for Serial"):
        print "createRootTask"
        self.rootTask = author.Task(title = rootTaskName)
        self.rootTask.serialsubtasks = True

        if self.tractorJob is None:
            self.createJob(jobName = "None Job")

        self.tractorJob.addChild(self.rootTask)
        
    def addTask(self, parentTask, title, command):
        print "addTask"
        task = author.Task(title = title,
                           argv = command,
                           service = self.svc)
        parentTask.addChild(task)

        return task
        
    def sendJobSpool(self, debug = False):
        print "sendJobSpool"
        author.setEngineClientParam(hostname = self.hostName,
                                    port = self.port,
                                    user = getpass.getuser(),
                                    debug = debug)
        
        self.tractorJob.spool()
        author.closeEngineClient()
        
        MessageBox("Success tractor Job")