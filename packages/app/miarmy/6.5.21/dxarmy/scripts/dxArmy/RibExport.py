import maya.cmds as cmds
import maya.mel as mel
import os, sys
from McdGeneral import *
import McdPlacementFunctions
scrPath = "/netapp/backstage/pub/apps/renderman2/MiarmyRiFilter"
if not scrPath in sys.path:
    sys.path.append(scrPath)
import rif_process


def ExportRib(Mode, Start, End, Outdir):
    # place agent out
    #cmd = 'McdPlacementCmd -am 3 -ign 0;'
    #mel.eval(cmd)
    #McdAfterPlaceFunction()
    McdPlacementFunctions.placementAgent()

    allAgtGrp = cmds.ls(type='McdAgentGroup')
    try:
        cmds.hide(allAgtGrp)
    except:
        pass

    # turn off mesh drive and agent cache
    allGlb = cmds.ls(type='McdGlobal')
    if McdIsBlank(allGlb):
        raise Exception('No found McdGlobal Node.')
        return

    filename = cmds.file(q=True, sn=True)
    basename = os.path.splitext(os.path.basename(filename))[0]
    for i in allGlb:
        cmds.setAttr('%s.enableMeshDrv' % i, 0)
        cmds.setAttr('%s.selectionCallback' % i, 1)
        cmds.setAttr('%s.outputFolder' % i, Outdir, type='string')
        outputRibs = cmds.getAttr('%s.outputRibs' % i)
        if not outputRibs:
            cmds.setAttr('%s.outputRibs' % i, basename, type='string')
        cmds.setAttr('%s.runProPath' % i, '/netapp/backstage/pub/apps/miarmy2/applications/linux/multiRender_DSO/', type='string')
        cmds.setAttr('%s.procPrim' % i, True)
        cmds.setAttr('%s.comprib' % i, True)
    exp = ArmyRender(Mode, Start, End)
    exp.doIt()

class ArmyRender:
    '''
    Miarmy rib export
    param:
        -mode : 0 normal, 1 prim asset only, 2 rib only, 3 points only
    '''
    def __init__(self, mode, start, end):
        self.mode  = mode
        self.start = start
        self.end   = end


    def doIt(self):
        self.renderPreCheck()

        # prim asset
        if cmds.getAttr('%s.procPrim' % self.globalNode):
            if self.mode == 0 or self.mode == 1:
                self.exportPrimAssets()
        # rib
        if self.mode == 2 or self.mode == 0:
            self.exportRibs()
        # points
        if self.mode == 3:
            self.exportPoints()


    def renderPreCheck(self):
        '''
        result:
            - self.Camera
            - self.globalNode
            - self.outputFolder
        '''
        renderCamList = list()
        for c in cmds.ls(type='camera'):
            if cmds.getAttr('%s.renderable' % c):
                renderCamList.append(c)
        self.Camera = "persp"

        self.globalNode   = mel.eval('McdSimpleCommand -execute 2')
        self.outputFolder = cmds.getAttr(self.globalNode + '.outputFolder')
        self.outputRibs   = cmds.getAttr(self.globalNode + '.outputRibs')


    def exportPoints(self):
        '''
        Agents Alembic Points Export
        '''
        brainNode = mel.eval('McdSimpleCommand -execute 3')
        solverFrame = cmds.getAttr('%s.startTime' % brainNode)
        cmds.currentTime(solverFrame-1)

        import PointsExport
        abcfile = os.path.join(self.outputFolder, '%s.abc' % self.outputRibs)
        PointsExport.AgentPointsExport(abcfile, self.start, self.end)


    def exportPrimAssets(self):
        outdir = os.path.join(self.outputFolder, 'ProcPrimAssets')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        allAgentGrpNodes = cmds.ls(type='McdAgentGroup')
        print allAgentGrpNodes

        for i in range(len(allAgentGrpNodes)):
            typePath = os.path.join(outdir, 'McdAgentType' + str(i))
            if not os.path.exists(typePath):
                os.makedirs(typePath)
            geoPath = os.path.join(typePath, 'McdGeoFiles')
            if not os.path.exists(geoPath):
                os.makedirs(geoPath)

        mel.eval('McdRMPPExportCmd;')


    def exportRibs(self):
        outdir = os.path.join(self.outputFolder, 'rib', self.Camera)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        cmds.setAttr('%s.renderMode' % self.globalNode, 1)
        cmds.setAttr('%s.archiveMode' % self.globalNode, 1)
        cmds.setAttr('%s.framebuf' % self.globalNode, 0)
        cmds.setAttr('%s.rmAttrList[5]' % self.globalNode, 1)
        cmds.setAttr('%s.camera' % self.globalNode, self.Camera, type='string')

        if self.start:
            cmds.setAttr('%s.startFrame' % self.globalNode, self.start)
        if self.end:
            cmds.setAttr('%s.endFrame' % self.globalNode, self.end)

        startFrame = cmds.getAttr('%s.startFrame' % self.globalNode)
        endFrame   = cmds.getAttr('%s.endFrame' % self.globalNode)
        isEnableCache = cmds.getAttr('%s.enableCache' % self.globalNode)
        if endFrame - startFrame < 0:
            raise Exception('Please check your render frame')

        brainNode = mel.eval('McdSimpleCommand -execute 3')
        solverFrame = cmds.getAttr('%s.startTime' % brainNode)
        solverFrame -= 1
        if solverFrame > startFrame:
            solverFrame = startFrame

        if (solverFrame > endFrame):
            return

        totalFrame = int(endFrame - solverFrame + 1)
        frameList = list()
        for i in range(totalFrame):
            if (i > 2):
                if isEnableCache and solverFrame < startFrame:
                    solverFrame += 1
                    continue
                else:
                    cmds.currentTime(int(solverFrame))
            else:
                cmds.currentTime(int(solverFrame))

            if solverFrame >= startFrame:
                mel.eval('McdRenderCmd')
                frameList.append(str(int(solverFrame)))
            solverFrame += 1
        oAgentResume()
        ribList = list()
        for e in os.listdir(outdir):
            for j in frameList:
                if str(e).split(".")[-2].count(j):
                    ribList.append(str(outdir) + "/" + str(e))
        ribList.sort()
        self.mkPath(ribList)

    def mkPath(self, files):
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



