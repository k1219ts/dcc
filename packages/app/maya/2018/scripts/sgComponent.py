#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	Component SceneGraph Nodes procedural
#
#	2017.02.25	$3
#-------------------------------------------------------------------------------

import os, sys
import string

import maya.cmds as cmds
import maya.mel as mel

import sgCommon


#-------------------------------------------------------------------------------
def get_currentAbcDisplay( filename ):
    mode = 1
    if filename.find('_mid') > -1:
        mode = 2
    elif filename.find('_low') > -1:
        mode = 3
    elif filename.find('_sim') > -1:
        mode = 4
    return mode

# display -> bbox:0, render:1, mid:2, low:3, sim:4
def get_reloadFileName( filename, display ):
    displayMap = {0: '', 1: '', 2: 'mid', 3: 'low', 4: 'sim'}

    filePath = os.path.dirname( filename )
    baseName = os.path.basename( filename )
    src = os.path.splitext( baseName )
    baseName  = src[0]
    extension = src[-1]

    result = filename

    source = baseName.split('_')
    if 'mid' in source:
        source.remove( 'mid' )
    if 'low' in source:
        source.remove( 'low' )
    if 'sim' in source:
        source.remove( 'sim' )

    if len(source) == 1:
        temp = list(source)
        istr = displayMap[display]
        if istr:
            temp.insert(1, istr )
        fn = os.path.join( filePath, string.join(temp, '_') + extension )
        if os.path.exists( fn ):
            result = fn
    else:
        for i in range(1,len(source)+1):
            temp = list(source)
            istr = displayMap[display]
            if istr:
                temp.insert( i, istr )
            fn = os.path.join( filePath, string.join(temp, '_') + extension )
            if os.path.exists( fn ):
                result = fn

    return result



def createZGpuMesh( File=None, Name=None, Parent=None ):
    name = '%s_Creator' % Name
    if not Name:
        name = 'ZGpuMeshCreator1'
    creator = cmds.createNode( 'ZGpuMeshCreator', n=name )

    name = Name
    if not Name:
        name = 'ZGpuMeshShape1'
    if Parent:
        shape = cmds.createNode( 'ZGpuMeshShape', n=Name, p=Parent )
    else:
        shape = cmds.createNode( 'ZGpuMeshShape', n=Name )

    cmds.setAttr( '%s.animation' % creator, True )
    cmds.setAttr( '%s.file' % creator, File, type='string' )

    cmds.sets( shape, e=True, forceElement='initialShadingGroup' )

    cmds.connectAttr( 'time1.outTime', '%s.time' % creator )
    cmds.connectAttr( '%s.output' % creator, '%s.input' % shape )

    return shape




#-------------------------------------------------------------------------------
#
#	Archive - ZGpuMeshShape
#
#-------------------------------------------------------------------------------
class Archive:
    def __init__( self, node ):
        # plug-in setup
        sgCommon.pluginSetup( ['AbcImport', 'ZMayaTools'] )

        # ZGpuMesh Config
        self.m_creator = None
        self.m_animation = True

        self.m_baked = None
        self.m_startFrame = 0
        self.m_endFrame = 0
        self.m_fitTime = False

        self.m_curNode = node.split('.')[0]
        childs = cmds.listRelatives( self.m_curNode, c=True, f=True )
        if childs:
            child = None
            for c in childs:
                if c.split('|')[-1] == '%sArc' % self.m_curNode:
                    child = c
            if not child:
                child = childs[-1]
            self.m_arcNode = child
        else:
            self.m_arcNode = '%sArc' % self.m_curNode

        self.m_arcNodeName = self.m_arcNode.split('|')[-1]
        
        self.m_currentFile = cmds.getAttr( '%s.abcFileName' % self.m_curNode )
        self.m_worldFile   = cmds.getAttr( '%s.worldFileName' % self.m_curNode )

        # mesh:0, gpu: 1
        self.m_mode		   = cmds.getAttr( '%s.mode' % self.m_curNode )
        # bbox:0, render:1, mid:2, low:3, sim:4
        self.m_display	   = cmds.getAttr( '%s.display' % self.m_curNode )

        self.m_currentDisp = get_currentAbcDisplay( self.m_currentFile )
        self.m_currentMode = 0

        self.m_newAbcFile = get_reloadFileName( self.m_currentFile, self.m_display )


    def doIt( self ):
        if cmds.objExists( self.m_arcNode ):
            if cmds.nodeType( self.m_arcNode ) == 'ZGpuMeshShape':
                self.m_currentMode = 1
            self.reloadAbc()
        else:
            self.importAbc()
            self.importWorld()

        # renderFile
        renderFile = cmds.getAttr( '%s.renderFile' % self.m_curNode )
        if renderFile:
			if renderFile == self.m_newAbcFile:
				cmds.setAttr( '%s.renderFile' % self.m_curNode, '', type='string' )
        else:
            setFileName = ''
            rfn = get_reloadFileName( self.m_newAbcFile, 1 )
            if self.m_newAbcFile != rfn:
                setFileName = rfn
            cmds.setAttr( '%s.renderFile' % self.m_curNode, setFileName, type='string' )

        cmds.select( self.m_curNode )


    def importWorld( self ):
        if not self.m_worldFile:
            return
        if not os.path.exists( self.m_worldFile ):
            mel.eval( 'print "# Error : File not found!\\n"' )
            return

        sgCommon.initTransform( self.m_curNode )

        # old-style world file
        if os.path.splitext(self.m_worldFile)[-1] == '.world':
            wClass = sgCommon.WorldAnimation( self.m_curNode, self.m_worldFile )
            wClass.m_baked = self.m_baked
            wClass.doIt()
        # new_style world file
        else:
            self.m_curNode = sgCommon.import_worldAlembic( self.m_curNode, self.m_baked, self.m_worldFile )


    def setZGpuMeshAttributes( self ):
        cmds.setAttr( '%s.abcFileName' % self.m_curNode, self.m_newAbcFile, type='string' )
        cmds.setAttr( '%s.file' % self.m_creator, self.m_newAbcFile, type='string' )
        print '### newABCFile: ', self.m_newAbcFile
        # for renderman
        sgCommon.setPostTransformScript( self.m_curNode )
        sgCommon.setInVisAttribute( self.m_arcNode, 1 ) # value: prune

    def getZGpuMeshFile( self ):
        creator = cmds.listConnections( self.m_arcNode, type='ZGpuMeshCreator', s=True, d=False )
        if creator:
            return cmds.getAttr( '%s.file' % creator[0] )

    # if not found file
    def errorSetup( self ):
        cmds.setAttr( '%s.display' % self.m_curNode, self.m_currentDisp )
        cmds.setAttr( '%s.mode' % self.m_curNode, self.m_currentMode )


    def reloadAbc( self ):
        # mode : gpu
        if self.m_mode == 1:
            if self.m_currentMode == 1:
                self.m_creator = cmds.listConnections( self.m_arcNode, type='ZGpuMeshCreator', s=True, d=False )[0]
                currentFile    = cmds.getAttr( '%s.file' % self.m_creator )
                print '### alembic file changed to ', self.m_newAbcFile
                if currentFile != self.m_newAbcFile:
                    if os.path.exists( self.m_newAbcFile ):
                        self.setZGpuMeshAttributes()
                    else:
                        self.errorSetup()
                else:
                    self.errorSetup()
            else:
                if os.path.exists( self.m_newAbcFile ):
                    cmds.delete( self.m_arcNode )
                    self.createZGpuMesh()
                else:
                    self.errorSetup()
        # mode : mesh
        else:
            print '### alembic file changed to ', self.m_newAbcFile
            if os.path.exists( self.m_newAbcFile ):
                cmds.delete( self.m_arcNode )
                self.createMesh( '-d -m import' )
            else:
                self.errorSetup()


    def importAbc( self ):
        if not os.path.exists( self.m_newAbcFile ):
            self.m_newAbcFile = get_reloadFileName( self.m_newAbcFile, 1 )
            cmds.setAttr( '%s.display' % self.m_curNode, 1 )
        else:
            if self.m_currentFile == self.m_newAbcFile:
                cmds.setAttr( '%s.display' % self.m_curNode, self.m_currentDisp )

        # GPU
        if self.m_mode == 1:
            self.createZGpuMesh()
        # MESH
        else:
            opts = '-d -m import'
            if self.m_fitTime:
                opts += ' -ftr'
            self.createMesh( opts )


    def createZGpuMesh( self ):
        if not self.m_creator:
            self.m_creator = cmds.createNode( 'ZGpuMeshCreator', n='%s_Creator' % self.m_arcNodeName )

        self.m_arcNode = cmds.createNode( 'ZGpuMeshShape', n=self.m_arcNodeName, p=self.m_curNode )

        cmds.sets( self.m_arcNode, e=True, forceElement='initialShadingGroup' )
        cmds.connectAttr( '%s.output' % self.m_creator, '%s.input' % self.m_arcNode )
        if self.m_animation:
            cmds.connectAttr( 'time1.outTime', '%s.time' % self.m_creator )
        self.setZGpuMeshAttributes()


    def createMesh( self, option ):
        cmds.setAttr( '%s.abcFileName' % self.m_curNode, self.m_newAbcFile, type='string' )
        self.m_arcNode = cmds.group( n=self.m_arcNodeName, p=self.m_curNode, em=True )
        mel.eval( 'AbcImport %s -rpr "%s" "%s"' % (option, self.m_arcNode, self.m_newAbcFile) )
        mel.eval( 'print "# Result : AbcMesh Import <%s>\\n"' % self.m_newAbcFile )




#-------------------------------------------------------------------------------
#
#	Archive - ZAbcViewer
#
#-------------------------------------------------------------------------------
class ZAbcViewerArchive:
    def __init__( self, node ):
        # plug-in setup
        sgCommon.pluginSetup( ['AbcImport', 'ZeomForMaya'] )

        self.m_baked = None
        self.m_startFrame = 0
        self.m_endFrame = 0
        self.m_fitTime = False

        self.m_curNode = node.split('.')[0]
        childs = cmds.listRelatives( self.m_curNode, c=True, f=True )
        if childs:
            child = None
            for c in childs:
                if c.split('|')[-1] == '%sArc' % self.m_curNode:
                    child = c
            if not child:
                child = childs[-1]
            self.m_arcNode = child
        else:
            self.m_arcNode = '%sArc' % self.m_curNode

        self.m_arcNodeName = self.m_arcNode.split('|')[-1]
        
        self.m_currentFile = cmds.getAttr( '%s.abcFileName' % self.m_curNode )
        self.m_worldFile   = cmds.getAttr( '%s.worldFileName' % self.m_curNode )

        # mesh:0, gpu: 1
        self.m_mode		   = cmds.getAttr( '%s.mode' % self.m_curNode )
        # bbox:0, render:1, mid:2, low:3, sim:4
        self.m_display	   = cmds.getAttr( '%s.display' % self.m_curNode )

        self.m_currentDisp = get_currentAbcDisplay( self.m_currentFile )
        self.m_currentMode = 0

        self.m_newAbcFile = get_reloadFileName( self.m_currentFile, self.m_display )

    def doIt( self ):
        if cmds.objExists( self.m_arcNode ):
            if cmds.nodeType( self.m_arcNode ) == 'ZAbcViewer':
                self.m_currentMode = 1
                if cmds.getAttr( '%s.displayMode' % self.m_arcNode ) == 1:
                    self.m_currentDisp = 0
            self.reloadAbc()
        else:
            self.importAbc()

        # renderFile
        renderFile = cmds.getAttr( '%s.renderFile' % self.m_curNode )
        if renderFile:
            pass
#			if renderFile != self.m_newAbcFile:
#				print 'sibong'
        else:
            setFileName = ''
            rfn = get_reloadFileName( self.m_newAbcFile, 1 )
            if self.m_newAbcFile != rfn:
                setFileName = rfn
            cmds.setAttr( '%s.renderFile' % self.m_curNode, setFileName, type='string' )

        self.importWorld()
        cmds.select( self.m_curNode )

    def importWorld( self ):
        if not self.m_worldFile:
            return
        if not os.path.exists( self.m_worldFile ):
            mel.eval( 'print "# Error : File not found!\\n"' )
            return

        sgCommon.initTransform( self.m_curNode )

        # old-style world file
        if os.path.splitext(self.m_worldFile)[-1] == '.world':
            wClass = sgCommon.WorldAnimation( self.m_curNode, self.m_worldFile )
            wClass.m_baked = self.m_baked
            wClass.doIt()
        # new_style world file
        else:
            sgCommon.import_worldAlembic( self.m_curNode, self.m_baked, self.m_worldFile )


    def setGpuCacheAttributes( self ):
        cmds.setAttr( '%s.abcFileName' % self.m_curNode, self.m_newAbcFile, type='string' )
        cmds.setAttr( '%s.file' % self.m_arcNode, self.m_newAbcFile, type='string' )
        # display
        if self.m_display == 0:
            cmds.setAttr( '%s.displayMode' % self.m_arcNode, 1 )
        else:
            cmds.setAttr( '%s.displayMode' % self.m_arcNode, 2 )

        # set-color
        cmds.setAttr( '%s.surfaceColor' % self.m_arcNode, 1.0, .8, .6 )

        # for renderman
        sgCommon.setPostTransformScript( self.m_curNode )
        sgCommon.setInVisAttribute( self.m_arcNode, 1 )	# value: prune

    # if not found file
    def errorSetup( self ):
        cmds.setAttr( '%s.display' % self.m_curNode, self.m_currentDisp )
        cmds.setAttr( '%s.mode' % self.m_curNode, self.m_currentMode )

    def reloadAbc( self ):
        # mode : gpu
        if self.m_mode == 1:
            if self.m_currentMode == 1:
                currentGpuFile = cmds.getAttr( '%s.file' % self.m_arcNode )
                if self.m_display != self.m_currentDisp or currentGpuFile != self.m_newAbcFile:
                    if os.path.exists( self.m_newAbcFile ):
                        self.setGpuCacheAttributes()
                    else:
                        self.errorSetup()
            else:
                if os.path.exists( self.m_newAbcFile ):
                    cmds.delete( self.m_arcNode )
                    self.createGPU()
                else:
                    self.errorSetup()
        # mode : mesh
        else:
            if self.m_display != self.m_currentDisp or self.m_mode != self.m_currentMode:
                if os.path.exists( self.m_newAbcFile ):
                    cmds.delete( self.m_arcNode )
                    self.createMesh( '-d -m import' )
                else:
                    self.errorSetup()

    def importAbc( self ):
        if not os.path.exists( self.m_newAbcFile ):
            self.m_newAbcFile = get_reloadFileName( self.m_newAbcFile, 1 )
            cmds.setAttr( '%s.display' % self.m_curNode, 1 )

        if self.m_mode == 1:
            self.createGPU()
        else:
            opts = '-d -m import'
            if self.m_fitTime:
                opts += ' -ftr'
            self.createMesh( opts )

    def createGPU( self ):
        self.m_arcNode = cmds.createNode( 'ZAbcViewer', n=self.m_arcNodeName, p=self.m_curNode )
        self.setGpuCacheAttributes()

    def createMesh( self, option ):
        cmds.setAttr( '%s.abcFileName' % self.m_curNode, self.m_newAbcFile, type='string' )
        self.m_arcNode = cmds.group( n=self.m_arcNodeName, p=self.m_curNode, em=True )
        mel.eval( 'AbcImport %s -rpr "%s" "%s"' % (option, self.m_arcNode, self.m_newAbcFile) )
        mel.eval( 'print "# Result : AbcMesh Import <%s>\\n"' % self.m_newAbcFile )




#-------------------------------------------------------------------------------
def componentReload( attr ):
    cpClass = Archive( attr )
    cpClass.doIt()


def componentImport( attr, fileName ):
    curNode = attr.split('.')[0]
    arcNode = '%sArc' % curNode

    cmds.setAttr( '%s.abcFileName' % curNode, fileName, type='string' )
    cmds.setAttr( '%s.renderFile' % curNode, '', type='string' )
    cmds.setAttr( '%s.worldFileName' % curNode, '', type='string' )

    wopt  = cmds.optionVar( q='dxCompoWorldAnim' )
    baked = True
    if wopt != 'None':
        if wopt == 'Separate':
            baked = False
        abcfile = get_reloadFileName( fileName, 1 )
        worldFile = None
        # old-style world file
        wf = abcfile.replace( '.abc', '.world' )
        if os.path.exists( wf ):
            worldFile = wf
        # new-style alembic world file
        wf = abcfile.replace( '.abc', '.wrd' )
        if os.path.exists( wf ):
            worldFile = wf
        if worldFile:
            cmds.setAttr( '%s.worldFileName' % curNode, worldFile, type='string' )

    cpClass = Archive( curNode )
    cpClass.m_baked = baked
    cpClass.doIt()

    if cmds.optionVar( q='dxCompoFitTime' ):
        if cpClass.m_startFrame > 0 and cpClass.m_endFrame > 0:
            cmds.playbackOptions( minTime=cpClass.m_startFrame )
            cmds.playbackOptions( maxTime=cpClass.m_endFrame )
            cmds.playbackOptions( animationStartTime=cpClass.m_startFrame )
            cmds.playbackOptions( animationEndTime=cpClass.m_endFrame )


