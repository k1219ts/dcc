"""
Submitter ROP node is responsible of crawling the ROP network
and finding nodes that are relevant to the submission

LAST RELEASE:
- 2017.07.09 $1 : submitter hipfile convention change
- 2017.07.13 $2 : job projects, tier, tags convention change
                  RopNode( merge ) bug fix
                  connectNodeTask bug fix
- 2017.08.25    : Support Redshift_ROP
- 2017.09.13 $3 : submitter hipfile path change
                  saveHipFile change to cleanup
                  possible to render any ROP
- 2017.09.21 $4 : Fetch network traversal process update
- 2017.10.14 $5 : readshift limit tag change
"""

import hou
from config import *
from TractorEngine import TractorEngine
import ROP


class SubmitterROP(object):
    def __init__(self, submitterNode):
        self.redshift = False
        self.sub = submitterNode

        # Data storage
        self.nodesToProcess = list()
        self.hipFile = hou.hipFile.path()

        # Let's go
        self.houdiniParms()
        self.saveHipFile()
        #
        self.collectNodes()
        self.createJob()


    def saveHipFile(self):
        """Back up hip file and open .hdas when submitting on the farm"""
        try:
            oldFile = self.hipFile

            hipDir = hou.hscriptExpandString("$HIP")
            hipName = hou.hscriptExpandString("$HIPNAME")
            extension = hou.hscriptExpandString("$HIPFILE").split(".")[-1]

            now = datetime.datetime.now()
            dateStamp = now.strftime('%Y-%m-%d-%H:%M:%S')
            userName = os.getlogin()

            filename = '{HIPNAME}-{OS}-{USER}-{STAMP}.{EXT}'.format(
                HIPNAME=hipName, OS=self.sub.name(), USER=userName, STAMP=dateStamp, EXT=extension
            )
            self.hipFile = os.path.join(hipDir, filename)

            hou.hipFile.save(file_name=self.hipFile, save_to_recent_files=False)

            hou.hipFile.setName(oldFile)
            hou.hipFile.save()

            print "# Debug : Render Hip submission: {FILE}".format(FILE=self.hipFile)

        except hou.OperationFailed as e:
            print "# Error : Saving failed", e


    def houdiniParms(self):
        """Populate from Houdini node UI"""
        self.title = self.sub.parm("title").eval()
        self.service = "Houdini"
        self.tier = 'FX'
        self.maxActive = self.sub.parm("maxActive").eval()
        self.priority = self.sub.parm("priority").eval()
        self.user = os.getlogin()
        self.comment = self.sub.parm("comment").eval()
        self.metadata = self.sub.parm("metadata").eval()
        self.cleanup = self.sub.parm('cleanup').eval()
        self.tractorIP = self.sub.parm("engine").evalAsString()
        self.makeMov = 0

    def collectNodes(self):
        """Collects the nodes to be processed"""
        nodes = self.sub.inputAncestors()
        self.nodesToProcess = list(nodes)

        submitter = (self.sub, (hou.frame(),))

        if submitter in self.nodesToProcess:
            self.nodesToProcess.remove(submitter)


    def createJob(self):
        rednodes = list()
        for n in self.nodesToProcess:
            if n.type().name() == 'Redshift_ROP':
                rednodes.append(n)
        if rednodes:
            self.nodesToProcess = list(rednodes)
            self.redshiftJobScript()
        else:
            self.hfsJobScript()


    def redshiftJobScript(self):
        self.redshift = True
        self.job = author.Job(
            title=self.title,
            priority=self.priority,
            comment=self.comment,
            metadata=self.metadata,
            service='GPUFARM',
            tier='FX-GPU',
            #envkey=self.envkey
        )

        self.root = author.Task(title=str(self.sub.name()))
        # self.root.envkey = self.envkey
        self.root.serialsubtasks = 0

        # cleanup submitted render hipfile
        if self.cleanup:
            self.root.addCleanup(
                author.Command(argv='/bin/rm -f %s' % self.hipFile)
            )

        for n in self.nodesToProcess:
            self.connectROP(n, self.root)

        self.job.addChild(self.root)
        # self.job.paused = True
        # print self.job.asTcl()



    def hfsJobScript( self ):
        self.job = author.Job(
            title=self.title,
            priority=self.priority,
            comment=self.comment,
            metadata=self.metadata,
            service='Houdini',
            tier='FX',
            maxactive=self.maxActive,
            #envkey=self.envkey
        )

        self.m_taskMap = dict()
        self.m_ropMap = dict()
        self.m_collector = list()
        self.rootTask = author.Task(title=str(self.sub.name()))
        # self.rootTask.envkey = self.envkey
        tree = FetchTree(self.sub, self.nodesToProcess)
        inputs = tree.getInputs()
        for n in inputs:
            self.TaskWalkUp(n, self.rootTask)

        if self.cleanup:
            self.rootTask.addCleanup(
                author.Command(argv='/bin/rm -f %s' % self.hipFile)
            )

        self.job.addChild(self.rootTask)
        # self.job.paused = True


    #---------------------------------------------------------------------------
    #
    # Task Connection
    #
    #---------------------------------------------------------------------------
    def TaskWalkUp(self, node, parentTask):
        tree = FetchTree(node, self.nodesToProcess)
        icollectors = tree.getICollectors()
        inputs = tree.getInputs()

        ifever = True
        if icollectors:
            collector = icollectors[0]
            c_tree = FetchTree(collector, self.nodesToProcess)
            ocollectors = c_tree.getOCollectors()
            if not ocollectors:
                # Jump to collector
                self.m_collector.append(collector)
                self.Create_FetchTask(collector, parentTask)
                self.TaskWalkUp(collector, parentTask)
                ifever = False

        if ifever:
            task = self.Create_FetchTask(node, parentTask)
            for n in tree.traversal():
                self.TaskWalkUp(n, task)


    def Create_Task(self, taskname, subtasks, parentTask, tasktype):
        if not self.m_taskMap.has_key(taskname):
            task = author.Task(title=taskname)
            # task.envkey = self.envkey
            task.serialsubtasks = subtasks
            parentTask.addChild(task)
            self.m_taskMap[taskname] = task

            # cmd make mov
            if self.makeMov:
                makeMovDirCommand = ['mkdir','-p', self.movDir]
                command = author.Command(argv=makeMovDirCommand)
                task.addCommand(command)
                makeMovCommand = ['/backstage/bin/DCC', 'rez-env', 'ffmpeg-4.2.1', '--', 'ffmpeg', '-y',
                                  '-framerate', str(self.fps),
                                  '-start_number', str(int(self.startFrame)),
                                  '-i', self.outputjpgPath,
                                  '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                                  '-b:v', '10000k',
                                  '-pix_fmt', 'yuv420p',
                                  self.outputMov]
                command = author.Command(argv=makeMovCommand)
                task.addCommand(command)

    def Create_FetchTask(self, node, parentTask):
        task_name = str(node.name())
        task_type = node.type().name()
        if task_type == "DXK_mantra_preview" and not node.isBypassed():
            self.trange = node.parm("trange").eval()
            self.startFrame = node.parm('f1').eval()
            self.fps = node.parm("fps").eval()
            self.makeMov = node.parm("makemov").eval()
            self.outputMov = node.parm("vm_mov").eval()
            self.outputJpg = node.parm("vm_picture").eval()

            # mov dir
            movDir = self.outputMov
            movDirSplit = movDir.split('/')[:-1]
            self.movDir = '/'.join(movDirSplit)

            # jpg path for mov
            if not self.trange == 0:
                outputJpgSplit = self.outputJpg.split('.')
                outputJpgSplit[-2] = '%04d'
                self.outputjpgPath = '.'.join(outputJpgSplit)
            else:
                self.outputjpgPath = self.outputJpg

        tree = FetchTree(node, self.nodesToProcess)
        # inputs  = tree.getInputs()
        inputs  = tree.traversal()
        outputs = tree.getOutputs()
        icollectors = tree.getICollectors()

        # 1. Create FetchTask, serialsubtasks=1, addChild out node task
        self.Create_Task(task_name, 1, parentTask, task_type)
        task = self.m_taskMap[task_name]

        if icollectors and not icollectors[0] in self.m_collector:
            # 2. Jump to Collector
            self.m_collector.append(icollectors[0])
            self.Create_FetchTask(icollectors[0], task)

        if len(inputs) == 1:
            # 2. Add Input Task
            self.Create_FetchTask(inputs[0], task)
        elif len(inputs) > 1:
            # 2. Add ICOLLECT Task
            self.Create_ICollectTask(node, task)

        # 3. Add ROP Task
        if not self.m_ropMap.has_key(task_name):
            self.connectROP(node, task)
            self.m_ropMap[task_name] = True

        # 4. Add OCOLLECT Task
        if len(outputs) > 1:
            self.Create_OCollectTask(node, task)

        return task


    def Create_ICollectTask(self, node, parentTask):
        task_name = str(node.name() + '_ICOLLECT')
        task_type = node.type().name()

        tree = FetchTree(node, self.nodesToProcess)
        inputs = tree.traversal()

        self.Create_Task(task_name, 0, parentTask, task_type)
        task = self.m_taskMap[task_name]

        for n in inputs:
            self.Create_FetchTask(n, task)


    def Create_OCollectTask(self, node, parentTask):
        tree = FetchTree(node, self.nodesToProcess)
        tails = tree.getTails()
        if not tails:
            return

        task_name = str(node.name() + '_OCOLLECT')
        task_type = node.type().name()
        self.Create_Task(task_name, 0, parentTask, task_type)
        task = self.m_taskMap[task_name]

        for n in tails:
            self.Create_FetchTask(n, task)


    def connectROP( self, fetchNode, parentTask ):
        if not fetchNode.isBypassed():
            rop = ROP.ROP( fetchNode, parentTask, hipFile=self.hipFile )


    #---------------------------------------------------------------------------
    def submit( self ):
        if self.redshift:
            engine = TractorEngine('redshift')

            self.job.projects = ['redshift']
            self.job.tags = ['GPU']
            self.job.priority = 90

            engine.spool(self.job)

        else:
            engine = TractorEngine(self.tractorIP)

            self.job.projects = []
            self.job.tags = ['3d']

            project = 'fx_other'
            src = self.hipFile.split('/')
            if 'show' in src:
                file_project = src[src.index('show')+1]
                if file_project in engine.cfg_projects:
                    project = 'fx_%s' % file_project
            self.job.projects = [str(project)]

            engine.spool(self.job)


    def __repr__(self):
        print "*"*80
        print "-------------- SUBMISSION START --------------"
        print "----------------------------------------------"
        if self.job:
            return self.job.asTcl()
        else:
            return "Job is not defined"
        print "*"*80
        print "-------------- SUBMISSION END --------------"
        print "\n"



class FetchTree(object):
    def __init__(self, node, source=list()):
        self.node = node
        self.source = source

        self.icollectors = list()
        self.ocollectors = list()
        self.tails = list()


    def getInputs(self, *argv):
        node = self.node
        if argv: node = argv[0]

        nodes = list()
        for c in node.inputConnections():
            n = c.inputNode()
            if nodes:
                r_depth = len(nodes[0].inputAncestors())
                c_depth = len(n.inputAncestors())
                if c_depth >= r_depth:
                    nodes.insert(0, n)
                else:
                    nodes.append(n)
            else:
                nodes.append(n)

        return nodes


    def getOutputs(self, *argv):
        node = self.node
        if argv: node = argv[0]

        delnodes = ['DXC_config', 'DXC_submitter']
        nodes = list()
        for c in node.outputConnections():
            n = c.outputNode()
            name = n.type().name()
            if not name in delnodes:
                if self.source:
                    if n in self.source:
                        nodes.append(n)
                else:
                    nodes.append(n)

        return nodes


    def walkup(self, node):
        # find collector
        outputs = self.getOutputs(node)
        if len(outputs) > 1:
            if self.node.name() != node.name() and not node in self.icollectors:
                self.icollectors.append(node)

        for n in self.getInputs(node):
            self.walkup(n)

    def getICollectors(self):
        self.walkup(self.node)
        return self.icollectors


    def walkdown(self, node):
        # find tails
        outputs = self.getOutputs(node)
        if not outputs and node in self.source:
            if not node in self.tails:
                self.tails.append(node)

        # find collector
        inputs = self.getInputs(node)
        if len(inputs) > 1 and outputs:
            if self.node.name() != node.name() and not node in self.ocollectors:
                if self.ocollectors:
                    connected = node.inputAncestors()
                    for o in self.ocollectors:
                        if not o in connected:
                            self.ocollectors.append(node)
                else:
                    self.ocollectors.append(node)

        for n in self.getOutputs(node):
            self.walkdown(n)

    def getOCollectors(self):
        self.walkdown(self.node)
        return self.ocollectors


    def getTails(self):
        self.walkdown(self.node)
        nodes = list()

        # exclude tails by ocollectors
        if self.ocollectors:
            for t in self.tails:
                connected = t.inputAncestors()
                for o in self.ocollectors:
                    if o in connected:
                        nodes.append(o)
                    else:
                        nodes.append(t)
        else:
            nodes = self.tails

        return nodes


    def traversal(self, *argv):
        node = self.node
        if argv: node = argv[0]

        nodes = list()
        for c in node.inputConnections():
            n = c.inputNode()
            outputs = self.getOutputs(n)
            if len(outputs) == 1:
                nodes.append(n)

        return nodes
