"""
General ROP node implementation
Called by the Submitter ROP
Reponsible for generating chunk and tasks for itself and return it under a node Task

LAST RELEASE:
- 2017.07.09 $1 : ROPTask addCleanup
- 2017.07.13 $2 : limit tags convention change
- 2017.07.xx    : render script change hrender.py call renderScript.py
- 2017.07.xx    : cleanup task remove. hrender.py include cleanup process
- 2017.08.26 $3 : fx_redshift tag add
- 2017.09.11 $4 : hrender.py remove, direct execute renderScript.py, environment setup by tractor-houdinihandler
- 2017.09.13 $5 : Save HipFile to cache export path by FileI/O option (saveHipFile)
"""
import hou

from config import *

_current_file = os.path.abspath( __file__ )
_current_path = os.path.dirname( _current_file )

class ROP(object):
    def __init__(self, currentNode, parentTask, hipFile=""):
        self.currentNode = currentNode
        self.parentTask  = parentTask
        self.hipFile = hipFile
        self.useSlot = 1    # default slot
        self.makeMov = False

        self.rendernode = None

        self.doIt()

    def houdiniParms(self):
        self.rendernode = ROP.convertFetchtoSOP(self.currentNode)
        if not self.rendernode:
            return False

        self.range = self.rendernode.parm('trange').eval()
        self.startFrame = self.rendernode.parm('f1').eval()
        self.endFrame   = self.rendernode.parm('f2').eval()
        self.chunkSize  = 1
        self.isSimulation = 0

        node = ROP.convertSOPtoFetch(self.currentNode)
        self.findConfigNodes(node)

        # limt tags control
        self.limitTags = ['houdini%s' % self.useSlot]
        process_tag = 'fx_sim'
        if 'DXK_mantra' in node.type().name():
            process_tag = 'fx_mta'
        elif node.type().name() == 'Redshift_ROP':
            process_tag = 'fx_redshift'
        self.limitTags.append(process_tag)

        return True

    @staticmethod
    def convertSOPtoFetch(node):
        if node.type().category().name() == "Sop": # find out if it's a Fetch node
            dependents = node.dependents()         # Get all SOP dependents
            for dep in dependents:
                if dep.type().category().name() == "Driver": # Find the Fetch dependent
                    return dep
        else:
            return node
    @staticmethod
    def convertFetchtoSOP(node):
        if node.type().name() == "fetch":
            SOPPath = node.parm("source").eval()
            return hou.node(SOPPath)
        else:
            return node

    def findConfigNodes(self, node):
        configCount = 0
        for o in node.outputs():
            if configCount > 1:
                print "Ambiguous config structure"
                return
            else:
                if o.type().description() == "DXC Config" and not o.isBypassed():
                    configCount += 1
                    self.chunkSize = o.parm("chunkSize").eval()
                    self.isSimulation = o.parm("isSimulation").eval()
                    self.useSlot = int( o.parm("slot").eval() )

    def floatRange(self, start, stop, step):
        """Helper function for non-integer frame ranges"""
        assert step > 0
        total = start
        compo = 0.0
        while total < stop:
            yield total
            y = step - compo
            temp = total + y
            compo = (temp - total) - y
            total = temp

    def nodeTypeToIgnore(self, node):
        nodeTypesToIgnore = ["Null", "Merge"]
        if node.type().description() in nodeTypesToIgnore:
            return True

    def makeNodeTask(self):
        ifever = self.houdiniParms()
        if not ifever:
            return
        if self.isSimulation:
            chunkTask = self.makeChunkTask(self.startFrame, self.endFrame)
            self.parentTask.addChild( chunkTask )
        else:
            if self.range:
                chunkGroupTask = author.Task(title=str(self.currentNode.name() + ' ROP'))
                self.parentTask.addChild( chunkGroupTask )
                for chunkStartFrame in self.floatRange(self.startFrame, self.endFrame+self.chunkSize, self.chunkSize):
                    chunkEndFrame = chunkStartFrame+self.chunkSize-1
                    if chunkStartFrame > self.endFrame:
                        break
                    if chunkEndFrame >= self.endFrame:
                        chunkEndFrame = self.endFrame

                    chunkTask = self.makeChunkTask(chunkStartFrame, chunkEndFrame)
                    chunkGroupTask.addChild( chunkTask )
            else:
                chunkTask = self.makeChunkTask(hou.frame(), hou.frame())
                self.parentTask.addChild( chunkTask )

    def makeChunkTask(self, startFrame, endFrame):
        identifier = self.currentNode.name() + '-' + str(startFrame)
        name = self.currentNode.path() + " - Frame range: {startframe}-{endframe}".format(startframe=startFrame, endframe=endFrame)
        chunkTask = author.Task(title=name)

        renderCommand = ['DCC']
        renderCommand += ['rez-env']
        renderCommand += ['houBundle-'+os.getenv('BUNDLE_NAME')]
        renderCommand += ['--', 'renderCmd']
        renderCommand += ['-userName', os.getlogin()]
        renderCommand += ['-startFrame', startFrame, '-endFrame', endFrame]
        renderCommand += ['-ROPnode', self.currentNode.path()]
        renderCommand += ['-hipFile', self.hipFile]
        command = author.Command(
            argv=renderCommand,
            # atleast=self.useSlot,
            # atmost=self.useSlot,
            # refersto=identifier,
            tags=self.limitTags
        )
        chunkTask.addCommand(command)

        return chunkTask


    def doIt(self):
        if not self.nodeTypeToIgnore(self.currentNode):
            self.makeNodeTask()
            # save hipfile by DXC_FileIO
            if self.rendernode:
                if self.rendernode.parent().type().name() == 'DXC_FileIO':
                    import DXC_FileIO
                    ioClass = DXC_FileIO.FileIO(self.rendernode.parent())
                    ioClass.hipFile = self.hipFile
                    ioClass.saveHipFile()



    def __repr__(self):
        return self.parentTask.asTcl()
