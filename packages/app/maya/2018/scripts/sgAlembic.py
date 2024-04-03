#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#       Sanghun Kim, rman.td@gmail.com
#
#	for Alembic export & import
#
#	2017.03.04 $3
#-------------------------------------------------------------------------------
#
#   WorldConMuteCtrl
#   - setMute
#   - setUnMute
#
#   CacheExport
#   - doIt
#       - getNodes
#       - getFrameRange
#           - findBindPosKey
#               - get_rigNodeObjects
#       - exportWorld
#       - updateAttributes
#       - exportMesh
#           - setMute
#           - jobCommand : python callback - abcCallBack
#               - getCacheName
#               - getObjects
#                   - get_attrObjects
#                      - get_rigNodeObjects
#               - frameRangeOptions
#                   - get_rigNodeObjects
#           - bbox write
#           - setUnMute
#
#-------------------------------------------------------------------------------

import os, sys
import string
import getpass
import time
import json

# for alembic python
from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from alembic.Util import *
kWrapExisting = WrapExistingFlag.kWrapExisting

import maya.api.OpenMaya as OpenMaya

import maya.cmds as cmds
import maya.mel as mel

import dplCommon
import sgCommon


CON_MAP = {
            'dexter':   {'nodes': ['place_CON', 'direction_CON', 'move_CON'],
                         'attrs': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']},
            'toneplus': {'nodes': ['world_ctrl', 'global_ctrl', 'COG_ctrl'],
                         'attrs': ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']}
        }


CallBackProc = 'sgAlembic.abcCallBack'
try:
    if cmds.about( batch=True ):
        CallBackProc = 'geoCache.sgAlembic.abcCallBack'
except:
    pass
#-------------------------------------------------------------------------------
#
#	alembic export callback
#
#-------------------------------------------------------------------------------
animBoundsData = dict()

def abcCallBack( name, frame, bounds ):
    if not name in animBoundsData:
        animBoundsData[name] = dict()
    animBoundsData[name][frame] = bounds


#-------------------------------------------------------------------------------
#
#	common
#
#-------------------------------------------------------------------------------
def get_rigConObjects( root, nodes ):
    ns_name, node_name = sgCommon.getNameSpace( root )
    result = list()
    for n in nodes:
        node = '%s:%s' % (ns_name, n)
        if cmds.objExists( node ):
            result.append( node )
    if len(nodes) == len(result):
        return result



#-------------------------------------------------------------------------------
#
#	world con controler
#
#-------------------------------------------------------------------------------
class WorldConMuteCtrl:
    def __init__( self, rigNode, nodeType ):
        self.m_node = rigNode
        self.m_type = nodeType
        self.m_data = dict()

    def getRigConObjects( self ):
        vender = 'dexter'
        if self.m_type == 'attrRig':
            rigAttrs = cmds.listAttr( self.m_node, st='*Rig' )
            vender   = rigAttrs[0].split('_')[0]
        if not CON_MAP.has_key( vender ):
            return
        ns_name, node_name = sgCommon.getNameSpace( self.m_node )
        targetObjects = list()
        sourceObjects = CON_MAP[vender]['nodes']
        for n in sourceObjects:
            if ns_name:
                node = '%s:%s' % ( ns_name, n )
            else:
                node = n
            if cmds.objExists( node ):
                targetObjects.append( node )
        attrs = CON_MAP[vender]['attrs']
        if len(targetObjects) == len(sourceObjects):
            return targetObjects, attrs

    def getComponentConObjects( self ):
        targetObjects = list()
        fullpath = cmds.ls( self.m_node, l=True )[0]
        src = fullpath.split('|')
        for i in range( 1, len(src)-1 ):
            node = string.join( src[:i+1], '|' )
            targetObjects.append( node )
        attrs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']
        return targetObjects, attrs

    def getConObjects( self ):
        if self.m_type == 'dxRig' or self.m_type == 'attrRig':
            return self.getRigConObjects()
        else:
            return self.getComponentConObjects()

    def getVal( self ):
        conList, attrs = self.getConObjects()
        if not conList:
            return
        for c in conList:
            self.m_data[ c ] = dict()
            for ln in attrs:
                if not cmds.getAttr( '%s.%s' % (c, ln), l=True ):
                    self.m_data[c][ln] = cmds.getAttr( '%s.%s' % (c, ln) )
            # initScale
            if cmds.attributeQuery( 'initScale', n=c, ex=True ):
                self.m_data[c]['initScale'] = cmds.getAttr( '%s.initScale' % c )


    # mute control ----------------------------------------------------------------
    def setMute( self ):
        if not self.m_data:
            self.getVal()
        for c in self.m_data:
            for at in self.m_data[c]:
                if at=='sx' or at=='sy' or at=='sz' or at=='initScale':
                    try:
                        cmds.setAttr( '%s.%s' % (c, at), 1 )
                    except:
                        pass
                else:
                    cmds.setAttr( '%s.%s' % (c, at), 0 )
                cmds.mute( '%s.%s' % (c, at), d=False, f=True )

    def setUnMute( self ):
        if not self.m_data:
            return
        for c in self.m_data:
            for at in self.m_data[c]:
                try:
                    cmds.setAttr( '%s.%s' % (c, at), self.m_data[c][at] )
                except:
                    pass
                cmds.mute( '%s.%s' % (c, at), d=True, f=True )


#-------------------------------------------------------------------------------
#
#	cache export
#
#-------------------------------------------------------------------------------
class CacheExport:
    def __init__( self,
                FilePath=None, Nodes=list(),
                Start=None, End=None, Step=1, Just=False, Worldspace=False
            ):
        # plug-in setup
        if not cmds.pluginInfo( 'AbcExport', q=True, l=True ):
            cmds.loadPlugin( 'AbcExport' )

        self.m_username		= None
        if not self.m_username:
            self.m_username = getpass.getuser()

        self.m_filePath		= FilePath
        self.m_nodes		= Nodes	# select export rig nodes
        self.m_start		= Start
        self.m_end			= End
        self.m_step			= Step
        self.m_just			= Just
        self.m_worldspace   = Worldspace
        self.m_meshTypes	= ['render', 'mid', 'low', 'sim']
        self.m_logDict		= {'render':list(), 'mid':list(), 'low':list(),
                               'sim':list(), 'world':list(), 'mesh':list() }

        self.exp_dxRigNodes = list()	# dxRig export nodes
        self.exp_dxArcNodes = list()	# dxComponent export nodes
        self.exp_attrNodes	= list()	# group rig nodes ( outsourcing data )


        self.data_world		= dict()
        self.data_bakedWorld= dict()
        self.data_initScale = dict()
        self.data_conList   = dict()
        self.data_copyFiles = list()
        self.data_start		= self.m_start
        self.data_end		= self.m_end


    def doIt( self ):
        startTime = time.time()

        self.getNodes()

        if not self.exp_dxRigNodes and not self.exp_dxArcNodes and not self.exp_attrNodes:
            mel.eval( 'print "# Error : Not found export objects\\n"' )
            return

        if self.exp_dxRigNodes:
            mel.eval( 'print "# Debug : export dxRig nodes -> %s\\n"' % string.join( self.exp_dxRigNodes, ' ' ) )
        if self.exp_dxArcNodes:
            mel.eval( 'print "# Debug : export dxArc nodes -> %s\\n"' % string.join( self.exp_dxArcNodes, ' ' ) )
        if self.exp_attrNodes:
            mel.eval( 'print "# Debug : export attr nodes -> %s\\n"' % string.join( self.exp_attrNodes, ' ' ) )

        # make dir
        if not os.path.exists( self.m_filePath ):
            os.makedirs( self.m_filePath )

        # get frame duration
        self.getFrameRange()

        # export world animation
        if self.m_worldspace == False:
            self.exportWorld()

        # import lookdev attributes
        self.updateAttributes()

        # export mesh
        self.exportMesh()

        endTime = time.time()
        mel.eval( 'print "# Result : time %.2f sec\\n"' % (endTime-startTime) )
        # debug
        for i in self.m_logDict.keys():
            if self.m_logDict[i]:
                for f in self.m_logDict[i]:
                    mel.eval( 'print "# Result : AbcExport <%s>\\n"' % f )


    #---------------------------------------------------------------------------
    # export root rig nodes
    def getNodes( self ):
        objs = list()
        if self.m_nodes:
            objs = list( self.m_nodes )
        else:
            # dexter pipe-line nodes
            objs += cmds.ls( type=['dxRig', 'dxComponent', 'dxAbcArchive'] )
            # vendor
            for i in cmds.ls( '|*', r=True ):
                if cmds.listAttr( i, st='*Rig' ):	# vendor_charRig, vendor_propRig
                    objs.append( i )

        for o in objs:
            ntype = cmds.nodeType( o )
            if ntype == 'dxRig':
                if cmds.getAttr( '%s.action' % o ) == 1:
                    self.exp_dxRigNodes.append( o )

            elif ntype == 'dxComponent' or ntype == 'dxAbcArchive':
                # action -> 1: cache export, 2: layout export, 3: cache copy
                # mode -> 0: mesh, 1: gpu
                root = cmds.ls(o,l=1)[0].split('|')[1]
                if not cmds.nodeType(root) == 'dxAssembly':
                    action = cmds.getAttr( '%s.action' % o )
                    mode   = cmds.getAttr( '%s.mode' % o )
                    if action == 1 and mode == 0:
                        self.exp_dxArcNodes.append( o )
                    else:
                        abcFile = cmds.getAttr( '%s.abcFileName' % o )
                        wrdFile = cmds.getAttr( '%s.worldFileName' % o )
                        if action == 3 and abcFile:
                            pass	# copy cache

            else:
                self.exp_attrNodes.append( o )


    def getCacheName( self, nodeName ):
        cacheName = nodeName
        ns_name = ""
        if cmds.attributeQuery( 'assetName', n=nodeName, ex=True ):
            asset_name = cmds.getAttr( '%s.assetName' % nodeName )
            ns_name, node_name = sgCommon.getNameSpace( nodeName )
            if asset_name and len(asset_name.split('/')) == 1:
                cacheName = asset_name
                # if ns_name:
                #     cacheName = '%s_%s_rig_GRP' % ( ns_name, asset_name )
                # else:
                cacheName = '%s_rig_GRP' % asset_name
        else:
            ns_name, cacheName = sgCommon.getNameSpace(nodeName)
        return ns_name, cacheName

    # frame range
    def getFrameRange( self ):
        if not self.m_start:
            self.m_start = int( cmds.playbackOptions(q=True, min=True) )
        if not self.m_end:
            self.m_end   = int( cmds.playbackOptions(q=True, max=True) )
        self.data_start  = self.m_start
        self.data_end    = self.m_end
        if self.m_start != self.m_end:
            self.data_start -= 1
            self.data_end   += 1
            
        if not self.m_just:
            self.data_start = int ( self.m_start - 50 )
            self.findBindPosKey()

    # find bind pos frame
    def findBindPosKey( self ):
        
        for i in self.exp_dxRigNodes+self.exp_attrNodes:
            controlers = None
            if cmds.attributeQuery( 'controlers', n=i, ex=True ):
                if cmds.attributeQuery( 'rigType', n=i, ex=True ):
                    if cmds.getAttr( '%s.rigType' % i ) == 0:
                        controlers = cmds.getAttr( '%s.controlers' % i )
                else:
                    controlers = cmds.getAttr( '%s.controlers' % i )
            if controlers:
                startKeys = list()
                for c in controlers:
                    for n in self.get_rigNodeObjects( i, c ):
                        keyAttrs = cmds.listAttr( n, k=True )
                        if keyAttrs:
                            for a in keyAttrs:
                                keys = cmds.keyframe( '%s.%s' % (n, a), q=True )
                                if keys:
                                    startKeys += keys
                    else:
                        mel.eval( 'print "# Error Skip : find rig control -> %s, %s\\n"' % (i, c) )
                if startKeys:
                    startKeys = list( set(startKeys) )
                    startKeys.sort()
                    if int(startKeys[0]) < self.data_start:
                        self.data_start = int(startKeys[0])
                        if self.data_start < 950:
                            self.data_start = 950


    #---------------------------------------------------------------------------
    #
    # World Animation - for alembic format
    #
    #---------------------------------------------------------------------------
    def exportWorld( self ):
        #	dxRigs
        for i in self.exp_dxRigNodes:
            attrList = CON_MAP['dexter']['attrs']	# ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
            conList  = CON_MAP['dexter']['nodes']	# ['place_CON', 'direction_CON', 'move_CON']

            rigConList = get_rigConObjects( i, conList )
            if rigConList:
                nameSpace, baseName   = self.getCacheName( i )
                wfile	   = os.path.join( self.m_filePath, nameSpace, '%s.wrd' % baseName )
                wfile = wfile.replace(':', '_')
                # make dir
                if not os.path.exists(os.path.dirname(wfile)):
                    os.makedirs(os.path.dirname(wfile))
                self.m_logDict['world'].append(wfile)
                sgCommon.export_worldAlembic( rigConList, rigConList[0],
                                              self.data_start, self.data_end, self.m_step,
                                              wfile )

        #	dxArcNodes
        for i in self.exp_dxArcNodes:
            parentPath = cmds.listRelatives( i, p=True, f=True )
            if parentPath:
                conList = parentPath[0].split('|')[1:]

                nameSpace, baseName = self.getCacheName( i )
                wfile    = os.path.join( self.m_filePath, nameSpace, '%s.wrd' % baseName )
                wfile = wfile.replace(':', '_')
                # make dir
                if not os.path.exists(os.path.dirname(wfile)):
                    os.makedirs(os.path.dirname(wfile))
                self.m_logDict['world'].append(wfile)
                sgCommon.export_worldAlembic( conList, conList[-1],
                                              self.data_start, self.data_end, self.m_step,
                                              wfile )

        #	attrNodes
        for i in self.exp_attrNodes:
            rigAttrs = cmds.listAttr( i, st='*Rig' )
            vender   = rigAttrs[0].split('_')[0]
            if CON_MAP.has_key( vender ):
                attrList = CON_MAP[vender]['attrs']
                conList  = CON_MAP[vender]['nodes']

                rigConList = get_rigConObjects( i, conList )
                if rigConList:
                    nameSpace, baseName   = self.getCacheName( i )
                    wfile	   = os.path.join( self.m_filePath, nameSpace, '%s.wrd' % baseName )
                    wfile = wfile.replace(':', '_')
                    # make dir
                    if not os.path.exists(os.path.dirname(wfile)):
                        os.makedirs(os.path.dirname(wfile))
                    self.m_logDict['world'].append(wfile)
                    sgCommon.export_worldAlembic( rigConList, rigConList[-1],
                                                  self.data_start, self.data_end, self.m_step,
                                                  wfile, vender )

    # set Mute
    def setMute( self ):
        self.m_muteData = dict()
        # dxRigs
        for i in self.exp_dxRigNodes:
            muteClass = WorldConMuteCtrl( i, 'dxRig' )
            muteClass.setMute()
            self.m_muteData[ i ] = muteClass
        # attrNodes
        for i in self.exp_attrNodes:
            muteClass = WorldConMuteCtrl( i, 'attrRig' )
            muteClass.setMute()
            self.m_muteData[ i ] = muteClass
        # archiveNodes
        for i in self.exp_dxArcNodes:
            muteClass = WorldConMuteCtrl( i, 'dxComponent' )
            muteClass.setMute()
            self.m_muteData[ i ] = muteClass

    def setUnMute( self ):
        for i in self.m_muteData:
            muteClass = self.m_muteData[ i ]
            muteClass.setUnMute()

    #---------------------------------------------------------------------------
    #
    # Export
    #
    #---------------------------------------------------------------------------
    def exportMesh( self ):
        if self.m_worldspace == False:
            self.setMute()

        jobCmdList = list()
        # mesh
        opts = '-uv -wv -wuvs -ef -a ObjectSet -a ObjectName -atp rman -df ogawa -sn'
        for i in self.exp_dxRigNodes:
            jobCmdList += self.jobCommand( i, '-ws '+opts )
        for i in self.exp_attrNodes:
            jobCmdList += self.jobCommand( i, '-ws '+opts )
        for i in self.exp_dxArcNodes:
            jobCmdList += self.jobCommand( i, '-ws '+opts )

        if jobCmdList:
            cmds.AbcExport( v=True, j=jobCmdList )

        # bbox write
        if animBoundsData:
            for fn in animBoundsData:
                f = open( fn, 'w' )
                json.dump( animBoundsData[fn], f, indent=4 )
                f.close()
                
        if self.m_worldspace == False:
            self.setUnMute()


    def jobCommand( self, rigNode, options ):
        result   = list()
        nameSpace, baseName = self.getCacheName( rigNode )
        meshes   = self.getObjects( rigNode )
        for t in meshes.keys():
            objs = meshes[t]
            if objs:
                jobCmd = ''
                if t == 'mid' or t == 'low' or t == 'sim':
                    abcfile = os.path.join( self.m_filePath, nameSpace, '%s_%s.abc' % (baseName, t) )
                    abcfile = abcfile.replace(':', '_')
                else:
                    abcfile = os.path.join( self.m_filePath, nameSpace, '%s.abc' % baseName )
                    abcfile = abcfile.replace(':', '_')
                    # callback
                    bboxfile = abcfile.replace( '.abc', '.bbox' )
                    jobCmd += '-pythonPerFrameCallback %s(name="%s",frame=#FRAME#,bounds=#BOUNDSARRAY#) ' % ( CallBackProc, bboxfile )
                    # ..ToDo
                if not os.path.exists(os.path.dirname(abcfile)):
                    os.makedirs(os.path.dirname(abcfile))
                    
                jobCmd += options
                jobCmd += self.frameRangeOptions( rigNode )
                for i in objs:
                    jobCmd += ' -rt %s' % i
                jobCmd += ' -file %s' % abcfile
                result.append( jobCmd )
                # wirte log
                self.m_logDict[t].append( abcfile )
        return result

    def frameRangeOptions( self, rigNode ):
        ifever = 0
        if cmds.attributeQuery( 'controlers', n=rigNode, ex=True ):
            for c in cmds.getAttr( '%s.controlers' % rigNode ):
                ctrl = self.get_rigNodeObjects( rigNode, c )
                connect = cmds.listConnections( ctrl, type='animCurve' )
                if connect:
                    ifever += 1
        if rigNode in self.exp_dxArcNodes:
            ifever += 1
        #if ifever > 0:
        #    return ' -fr %s %s -s %s' % ( self.data_start, self.data_end, self.m_step )
        #else:
        #    return ' -fr 1 1'
        return ' -fr %s %s -s %s' % ( self.data_start, self.data_end, self.m_step )

    #---------------------------------------------------------------------------
    #
    # Objects
    #
    #---------------------------------------------------------------------------
    def getObjects( self, rigNode ):
        result = { 'render': list(), 'mid': list(), 'low': list(), 'sim': list() }
        # dxRigs or attrNodes
        if rigNode in self.exp_dxRigNodes or rigNode in self.exp_attrNodes:
            for t in result.keys():
                if t in self.m_meshTypes:
                    result[t] = self.get_attrObjects( rigNode, '%sMeshes' % t )
        # dxArcNodes
        if rigNode in self.exp_dxArcNodes:
            for shape in cmds.ls(rigNode, dag=True, type=['surfaceShape','nurbsCurve'], ni=True):
                result['render'] += cmds.listRelatives( shape, p=True, f=True )
        return result

    def get_attrObjects( self, rigNode, attrName ):
        result = list()
        if cmds.attributeQuery( attrName, n=rigNode, ex=True ):
            for o in cmds.getAttr( '%s.%s' % (rigNode, attrName) ):
                result += self.get_rigNodeObjects( rigNode, o )
        return result

    def get_rigNodeObjects( self, rootName, objName ):
        result = list()
        for o in cmds.ls( objName, r=True, l=True ):
            src = o.split('|')
            if rootName in src:
                result.append( o )
                # Deformed Objects
                shape = cmds.ls( o, dag=True, type='surfaceShape', ni=True )
                if shape:
                    if shape[0].find('Deformed') > -1:
                        print 'ToDo: attribute copy'
        return result

    #---------------------------------------------------------------------------
    #
    # Look-dev Attributes
    #
    #---------------------------------------------------------------------------
    def updateAttributes( self ):
        current = cmds.file( q=True, sn=True )
        src = current.split('/')
        showName = src[ src.index('show')+1 ]

        print "test :", self.exp_dxRigNodes + self.exp_dxArcNodes + self.exp_attrNodes
        
        for i in self.exp_dxRigNodes + self.exp_attrNodes + self.exp_dxArcNodes:
            assetName = ''
            if cmds.attributeQuery( 'assetName', n=i, ex=True ):
                assetName = cmds.getAttr( '%s.assetName' % i )
            if not assetName:
                assetName = i.split(':')[-1].replace('_rig_GRP', '')
                
            atfiles = dplCommon.lkdv_getAttrFile( showName, assetName )
            print '### atfile: ', atfiles
            if atfiles:
                for atfile in atfiles:
                    mel.eval( 'print "# Debug : import attributes : %s -> %s\\n"' % (assetName, atfile) )
                    body = json.loads( open(atfile, 'r').read() )
                    if body.has_key( 'Attributes' ):
                        dplCommon.lkdv_importAssetAttrs( i, body['Attributes'] )
                        #print '### attributes: ', body['Attributes']


#-------------------------------------------------------------------------------
#
#	alembic merge import
#
#-------------------------------------------------------------------------------
class CacheMerge:
    def __init__( self, fileName ):
        if not cmds.pluginInfo( 'AbcImport', q=True, l=True ):
            cmds.loadPlugin( 'AbcImport' )

        self.m_data = dict()
        self.m_file = fileName

        self.readAbc()

    #---------------------------------------------------------------------------
    # common
    def dataUpdate( self, dtype, dvalue ):
        if not self.m_data.has_key( dtype ):
            self.m_data[dtype] = list()
        self.m_data[dtype].append( dvalue )

    def getObjects( self, objName, plug=None ):
        result = cmds.ls( objName.split(':')[-1], r=True )
        for i in result:
            connects = cmds.listConnections( i, s=True, d=False, p=True )
            if connects:
                for c in connects:
                    sources = cmds.connectionInfo( c, dfs=True )
                    if sources:
                        for s in sources:
                            tmp = s.split('.')
                            if plug:
                                if i == tmp[0] and plug == tmp[-1]:
                                    cmds.disconnectAttr( c, s )
                            else:
                                if i == tmp[0]:
                                    cmds.disconnectAttr( c, s )
        return result

    def doConnect( self ):
        basename = os.path.splitext( os.path.basename(self.m_file) )[0]

        abcNode = cmds.createNode( 'AlembicNode', name='%s_AlembicNode' % basename )
        cmds.setAttr( '%s.abc_File' % abcNode, self.m_file, type='string' )

        timeNode = cmds.ls( type='time' )[0]
        cmds.connectAttr( '%s.outTime' % timeNode, '%s.time' % abcNode, f=True )
        #print self.m_data
        for plug in self.m_data:
            for i in range(len(self.m_data[plug])):
                data     = self.m_data[plug][i]
                source   = data.split('.')
                dst_plug = source[-1]

                for o in self.getObjects( source[0], dst_plug ):
                    # transform operation - rotate
                    if dst_plug.find('rotate') > -1:
                        converNode = cmds.createNode( 'unitConversion' )
                        cmds.setAttr( '%s.conversionFactor' % converNode, 0.0174532925199 )
                        cmds.connectAttr( '%s.%s[%s]' % (abcNode, plug, i),
                                          '%s.input' % converNode, f=True )
                        cmds.connectAttr( '%s.output' % converNode,
                                          '%s.%s' % (o, dst_plug), f=True )
                    # deformed mesh
                    # transform operation - translate, scale
                    else:
                        cmds.connectAttr( '%s.%s[%s]' % (abcNode, plug, i),
                                          '%s.%s' % (o, dst_plug), f=True )

        cmds.dgeval( abcNode )
        # playback
        start = cmds.getAttr( '%s.startFrame' % abcNode )
        end   = cmds.getAttr( '%s.endFrame' % abcNode )
        cmds.playbackOptions( ast=start, min=start, aet=end, max=end )


    #---------------------------------------------------------------------------
    # file read
    def readAbc( self ):
        iarch = IArchive( str(self.m_file) )
        root  = iarch.getTop()
        self.visitObject( root )

        # connect attrs
        if not self.m_data:
            cmds.warning( 'CacheMerge Set Values -> %s' % self.m_file )
            return
        self.doConnect()
        cmds.select( cl=True )
        cmds.warning( 'CacheMerge Import & Set Values -> %s' % self.m_file )

    def visitObject( self, iobj ):
        for obj in iobj.children:
            self.m_objects = self.getObjects( obj.getName() )

            ohead = obj.getHeader()
            if IXform.matches( ohead ):
                self.visitXform( obj )
            if IPolyMesh.matches( ohead ):
                self.visitPolyMesh( obj )
            if ISubD.matches( ohead ):
                cmds.warning( 'Not support SubdivMesh' )
            if INuPatch.matches( ohead ):
                cmds.warning( 'Not support NurbsSurface' )
            if ICurves.matches( ohead ):
                cmds.warning("Skip NurbsCurve")
                # self.visitCurves( obj )

            self.visitObject( obj )


    # IXform -----------------------------------------------------------------
    def M44dToList( self, value ):
        mtx = list()
        for x in range(4):
            for y in range(4):
                mtx.append( value[x][y] )
        return mtx

    def visitXform( self, iobj ):
        ixform = IXform( iobj, kWrapExisting )
        schema = ixform.getSchema()
        if schema.isConstant():
            #print '# deubg : constant %s' % iobj.getName()
            mtx = schema.getValue().getMatrix()
            for o in self.m_objects:
                cmds.xform( o, m=self.M44dToList(mtx), ws=True )
        else:
            op_data = list()
            Samp    = schema.getValue()
            numOps  = Samp.getNumOps()
            isComplex    = False
            numTransOp   = 0;   valTrans   = V3d( 0, 0, 0 );
            numRotateXOp = 0;   valRotateX = 0;
            numRotateYOp = 0;   valRotateY = 0;
            numRotateZOp = 0;   valRotateZ = 0;
            numScaleOp   = 0;   valScale   = V3d( 1, 1, 1 );
            for i in range( numOps ):
                op = Samp[i]
                # Translate
                if op.isTranslateOp():
                    numTransOp += 1
                    if op.isXAnimated():
                        op_data.append( 'translateX' )
                    if op.isYAnimated():
                        op_data.append( 'translateY' )
                    if op.isZAnimated():
                        op_data.append( 'translateZ' )
                    valTrans = op.getTranslate()
                # Rotate
                if op.isRotateXOp():
                    numRotateXOp += 1
                    if op.isAngleAnimated():
                        op_data.append( 'rotateX' )
                    valRotateX = op.getXRotation()
                if op.isRotateYOp():
                    numRotateYOp += 1
                    if op.isAngleAnimated():
                        op_data.append( 'rotateY' )
                    valRotateY = op.getYRotation()
                if op.isRotateZOp():
                    numRotateZOp += 1
                    if op.isAngleAnimated():
                        op_data.append( 'rotateZ' )
                    valRotateZ = op.getZRotation()
                # Scale
                if op.isScaleOp():
                    numScaleOp += 1
                    if op.isXAnimated():
                        op_data.append( 'scaleX' )
                    if op.isYAnimated():
                        op_data.append( 'scaleY' )
                    if op.isZAnimated():
                        op_data.append( 'scaleZ' )
                    valScale = op.getScale()

            if numTransOp > 1 or numRotateXOp > 1 or numRotateYOp > 1 or numRotateZOp > 1 or numScaleOp > 1:
                isComplex = True

            if isComplex:
                #print '# debug : isComplex'
                op_data  = list()
                op_data += ['translateX', 'translateY', 'translateZ']
                op_data += ['rotatePivotTranslateX', 'rotatePivotTranslateY', 'rotatePivotTranslateZ']
                op_data += ['rotatePivotX', 'rotatePivotY', 'rotatePivotZ']
                op_data += ['rotateX', 'rotateY', 'rotateZ']
                op_data += ['rotateAxisX', 'rotateAxisY', 'rotateAxisZ']
                op_data += ['scalePivotTranslateX', 'scalePivotTranslateY', 'scalePivotTranslateZ']
                op_data += ['scalePivotX', 'scalePivotY', 'scalePivotZ']
                op_data += ['shearXY', 'shearXZ', 'shearYZ']
                op_data += ['scaleX', 'scaleY', 'scaleZ']

            for d in op_data:
                self.dataUpdate( 'transOp', '%s.%s' % (iobj.getName(), d) )

            # set value
            for o in self.m_objects:
                cmds.setAttr( '%s.translate' % o, *valTrans )
                cmds.setAttr( '%s.rotate' % o, valRotateX, valRotateY, valRotateZ )
                cmds.setAttr( '%s.scale' % o, *valScale )


    def visibleProperty( self, iobj ):
        iCompoundProp   = iobj.getProperties()
        compoundHeaders = iCompoundProp.propertyheaders
        for header in compoundHeaders:
            if header.getName() == 'visible':
                prop = iCompoundProp.getProperty( header.getName() )
                if prop.isConstant():
                    vis = prop.getValue()
                    for o in self.m_objects:
                        cmds.setAttr( '%s.visibility' % o, abs(vis) )
                else:
                    self.dataUpdate( 'prop', '%s.visibility' % iobj.getName() )
            else:
                for o in self.m_objects:
                    cmds.setAttr( '%s.visibility' % o, 1 )


    # IPolyMesh --------------------------------------------------------------
    def visitPolyMesh( self, iobj ):
        meshTopProp = iobj.getProperties()
        meshProp	= meshTopProp.getProperty( 0 )

        pointsProp	= meshProp.getProperty( 'P' )
        if pointsProp.isConstant():
            # set points position
            points = pointsProp.samples[0]
            for o in self.m_objects:
                self.SetPolyMesh( o, points )
        else:
            self.dataUpdate( 'outPolyMesh', '%s.inMesh' % iobj.getName() )

    def SetPolyMesh( self, objName, points ):
        selection = OpenMaya.MSelectionList()
        selection.add( objName )
        dagPath = selection.getDagPath( 0 )

        mFnMesh = OpenMaya.MFnMesh( dagPath )
        if mFnMesh.numVertices != len(points):
            cmds.error( 'Not match topology :%s' % objName )
            return

        newPointArray = OpenMaya.MPointArray()
        for i in range( len(points) ):
            newPoint = OpenMaya.MPoint( points[i][0], points[i][1], points[i][2] )
            newPointArray.append( newPoint )
        mFnMesh.setPoints( newPointArray )


    # ICurves ----------------------------------------------------------------
    def visitCurves( self, iobj ):
        curveTopProp = iobj.getProperties()
        curveProp	 = curveTopProp.getProperty( 0 )

        pointsProp	 = curveProp.getProperty( 'P' )
        if pointsProp.isConstant():
            # set points position
            points = pointsProp.samples[0]
            for o in self.m_objects:
                self.SetNurbsCurve( o, points )
        else:
            self.dataUpdate( 'outNCurveGrp', '%s.create' % iobj.getName() )

    def SetNurbsCurve( self, objName, points ):
        selection = OpenMaya.MSelectionList()
        selection.add( objName )
        dagPath = selection.getDagPath( 0 )

        mFnCurve = OpenMaya.MFnNurbsCurve( dagPath )
        if mFnCurve.numCVs != len( points ):
            cmds.error( 'Not match topology : %s' % objName )
            return

        newPointArray = OpenMaya.MPointArray()
        for i in range(len(points)):
            newPoint = OpenMaya.MPoint( points[i][0], points[i][1], points[i][2] )
            newPointArray.append( newPoint )
        mFnCurve.setCVs( newPointArray )
        mFnCurve.updateCurve()

