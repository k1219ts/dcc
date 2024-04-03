#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	SceneGraph Zenn procedural
#
#	2017.02.10	$2
#-------------------------------------------------------------------------------
#
#	Common
#	- setWorldMute, nodeVisibility, checkUp_zennBodyMesh, getZennOutNodes,
#	- zennExport, zennInitialize
#	- zennBodyMeshUpdate
#	Zenn Export
#	- export_by_maya
#	Cache Viewer
#	- checkup_zennConstant, create_zennStrandsViewer, zennStrandsViewer
#
#-------------------------------------------------------------------------------

import os, sys
import string
import glob
import re
import json
import time

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

import sgCommon
import sgAlembic

# 2017.05.15 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
import pymongo
from pymongo import MongoClient
import datetime
import getpass
import dxConfig

DBIP = dxConfig.getConf('DB_IP')
DBNAME = "PIPE_PUB"

def getPubVersion(show, task, data_type, asset_name):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print(show, task, data_type, asset_name)
    recentDoc = coll.find_one({'show': show,
                               'task': task,
                               'data_type': data_type,
                               'asset_name': asset_name},
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1

#-------------------------------------------------------------------------------
#
#	Common
#
#-------------------------------------------------------------------------------
def setWorldMute( rootNode ):
    #transformAttrMap = {'tx':0, 'ty':0, 'tz':0, 'rx':0, 'ry':0, 'rz':0, 'sx':1, 'sy':1, 'sz':1 }
    transformAttrMap = {'tx':0, 'ty':0, 'tz':0, 'rx':0, 'ry':0, 'rz':0 }
    conList = ['move_CON', 'direction_CON', 'place_CON']
    for c in conList:
        con = c
        ns_name, node_name = sgCommon.getNameSpace( rootNode )
        if ns_name:
            con = '%s:%s' % ( ns_name, c )

        # transform
        for ln in transformAttrMap:
            if not cmds.getAttr( '%s.%s' % (con, ln), l=True ):
                cmds.setAttr( '%s.%s' % (con, ln), transformAttrMap[ln] )
                cmds.mute( '%s.%s' % (con, ln), d=False, f=True )

        # initScale
        if cmds.attributeQuery( 'initScale', n=con, ex=True ):
            cmds.setAttr( '%s.initScale' % con, 1 )
            cmds.mute( '%s.initScale' % con, d=False, f=True )


def nodeVisibility( nodeName ):
    vizConnect = cmds.listConnections( '%s.visibility' % nodeName )
    if vizConnect:
        return True
    else:
        return cmds.getAttr( '%s.visibility' % nodeName )


def checkUp_zennBodyMesh( zennNode ):
    isConstant = True
    bodyMesh = None
    for i in cmds.listHistory( zennNode ):
        if cmds.nodeType(i) == 'mesh':
            bodyMesh = i
        if cmds.nodeType(i) == 'AlembicNode':
            isConstant = False
    if not bodyMesh:
        return isConstant, False

    meshTrans = cmds.listRelatives( bodyMesh, p=True, f=True )[0]
    if nodeVisibility( meshTrans ):
        return isConstant, True
    else:
        return isConstant, False


def getZennOutNodes( selectedNode=None ):
    typeMap = {'ZN_StrandsViewer': 'inStrands', 'ZN_FeatherSetViewer': 'inFeatherSet'}

    outNodes = list()
    isConstant = True

    nodes = list()
    if selectedNode:
        nodes = cmds.ls( selectedNode.split(',') )
    else:
        nodes = cmds.ls( type=typeMap.keys() )

    for n in nodes:
        ntype = cmds.nodeType( n )
        if ntype in typeMap.keys():
            trNode = cmds.listRelatives( n, p=True, f=True )[0]
            if nodeVisibility( trNode ):
                connected = cmds.listConnections( '%s.%s' % (n, typeMap[ntype]) )
                if connected:
                    const, enable = checkUp_zennBodyMesh( connected[0] )
                    if enable:
                        outNodes.append( connected[0] )
                        if not const:
                            isConstant = const
        else:
            for nt in typeMap.keys():
                viewerConnect = cmds.listConnections( n, type=nt )
                if viewerConnect:
                    if nodeVisibility( viewerConnect[0] ):
                        const, enable = checkUp_zennBodyMesh( n )
                        if enable:
                            outNodes.append( n )
                            if not const:
                                isConstant = const
    return isConstant, outNodes


def zennExport( outPath, outNodes, start, end, step ):
    if not os.path.exists( outPath ):
        os.makedirs( outPath )

    mel.eval( 'print "# Debug : ZENN-Cache Export\\n"' )
    print(outNodes)
    cmds.ZN_CacheGenCmd( startFrame=start-1, endFrame=end, step=step,
                         cachePath=outPath,
                         nodeNames=string.join(outNodes, ' ') )
    mel.eval( 'print "# Result : Frame Range %s - %s\\n"' % (start, end) )
    mel.eval( 'print "# Result : ZENN-Cache Path -> %s\\n"' % outPath )


def zennInitialize( Nodes, RestFrame ):
    if Nodes:
        nodes = Nodes
    else:
        nodes = cmds.ls( type=['ZN_StrandsViewer', 'ZN_FeatherSetViewer'] )
    if not nodes:
        return

    mel.eval( 'print "#---------------------------------------------------------------#\\n"' )
    mel.eval( 'print "#\\n"' )
    mel.eval( 'print "#		Zenn Init\\n"' )
    mel.eval( 'print "#\\n"' )
    mel.eval( 'print "#---------------------------------------------------------------#\\n"' )
    for n in nodes:
        connected = cmds.listConnections( n, s=True, d=False )
        for h in cmds.listHistory( n ):
            if cmds.nodeType(h) == 'ZN_Import':
                if RestFrame:
                    cmds.currentTime( RestFrame )
                else:
                    restTime = cmds.getAttr( '%s.restTime' % h )
                    cmds.currentTime( restTime )
                cmds.setAttr( '%s.updateMesh' % h, 1 )
        if connected:
            for i in connected:
                cmds.dgeval( i )



def zennBodyMeshUpdate():
    newMesh = cmds.ls(sl=True, dag=True, type='surfaceShape', ni=True, l=True)
    if not newMesh:
        mel.eval( 'print "# Error : select bodymesh!\n"')
        return
    znNodes = cmds.ls(sl=True, type=['ZN_Import', 'ZN_PartialMeshGen'])
    if not znNodes:
        mel.eval( 'print "# Error : select ZN_Import nodes!\n"' )

    for z in znNodes:
        if cmds.nodeType(z) == 'ZN_Import':
            cmds.connectAttr( '%s.w' % newMesh[0], '%s.inBodyMesh' % z, f=True )
            gz = cmds.listConnections( z, d=False, type='ZN_Global' )
            #cmds.setAttr( '%s.bodyMeshName' % gz[0], newMesh[0], type='string' )
        if cmds.nodeType(z) == 'ZN_PartialMeshGen':
            cmds.connectAttr( '%s.w' % newMesh[0], '%s.inMesh' % z, f=True )
            gz = cmds.listConnections( z, d=False, type='ZN_Global' )




#-------------------------------------------------------------------------------
#
#	Maya Zenn Export
#
#-------------------------------------------------------------------------------
def export_by_maya( zennOutDir, Start=1, End=1, Step=1, restFrame=950 ):
# zenn init

    cmds.currentTime( 1 )
    const, outNodes = getZennOutNodes()
    if Start + End > 2:
        const = False
    zennInitialize( outNodes, restFrame )

    zennLog = dict()
    for z in outNodes:
        zniNode = None

        historyNodes = cmds.listHistory( z )
        for h in historyNodes:
            if cmds.nodeType(h) == 'ZN_Import':
                zniNode = h

        meshes = cmds.listConnections( zniNode, type='surfaceShape' )
        if meshes:
            meshPath = cmds.listRelatives( meshes[0], p=True, f=True )[0]
            rootNode = meshPath.split('|')[1]
            outdir   = os.path.join( zennOutDir, rootNode.replace('_rig_GRP', '') )
            if zennLog.has_key( outdir ):
                zennLog[outdir]['nodes'].append( z )
            else:
                tmp = { 'nodes': [z], 'cache': rootNode, 'file': '' }
                zennLog[outdir] = tmp
            if const:
                zennLog[outdir]['constant'] = 1

    start = Start; end = End; step = Step;

    if zennLog:
        for o in zennLog:
            try:
                setWorldMute( zennLog[o]['cache'] )
            except:
                print('no world CON error')
            zennExport( o, zennLog[o]['nodes'], start, end, step )
        return zennLog, start, end




#-------------------------------------------------------------------------------
#
#   Zenn Cache Export by Alembic Cache
#
#-------------------------------------------------------------------------------
def export_by_alembic( zennFile, abcFile, subCache, fps, outPath,
                       Start, End, Step ):
    cmds.file( new=True, f=True )

    # time unit setup
    if fps:
        cmds.currentUnit( time=fps )

    # import zenn template
    cmds.file( zennFile, i=True, type='mayaBinary', iv=True, mnc=False,
               rpr=os.path.splitext( os.path.basename(zennFile) )[0],
               options='v=0;', pr=True )

    mel.eval( 'print "#---------------------------------------------------------------#\\n"' )
    mel.eval( 'print "#\\n"' )
    mel.eval( 'print "#		By Alembic Cache\\n"' )
    mel.eval( 'print "#  zenn file -> %s\\n"' % zennFile )
    mel.eval( 'print "#  abc file  -> %s\\n"' % abcFile )
    if subCache:
        mel.eval( 'print "#  zenn cached -> %s\\n"' % subCache )
    mel.eval( 'print "#\\n"' )

    cmds.currentTime( 1 )
    zennInitialize( None, None )

    # alembic cache merge
    mgClass = sgAlembic.CacheMerge( abcFile )


    const, outNodes = getZennOutNodes()
    if not outNodes:
        cmds.error( 'Zenn outNodes not found' )
        return
    # sub cache
    if subCache:
        outNodes = list(set(outNodes) - set(subCache))

    mel.eval( 'print "# Zenn outNodes -> %s %s\\n"' % (const, string.join(outNodes, ', ')) )

    zennOutDir = os.path.join( outPath, os.path.basename(os.path.dirname(abcFile)) )

    zennLog = { zennOutDir: {'nodes':outNodes, 'cache': abcFile, 'file': zennFile} }

    start = Start; end = End;
    if const:
        zennLog[zennOutDir]['constant'] = 1
        start = 1; end = 1

    render_width = str(cmds.getAttr("defaultResolution.width"))
    render_height = str(cmds.getAttr("defaultResolution.height"))

    zennExport( zennOutDir, outNodes, start, end, Step )
    return zennLog, render_width, render_height






#-------------------------------------------------------------------------------
#
#	Zenn Cache Viewer
#
#-------------------------------------------------------------------------------
# checkup constant
def checkup_zennConstant( nodePath ):
    files = glob.glob( '%s/bbox_*' % nodePath )
    if len(files) <=3:
        files.sort()
        frame = re.compile( r'\d+' ).findall( os.path.basename(files[-1]) )
        if frame:
            return frame[0]
        else:
            return False
    return False


# create node
def create_zennStrandsViewer( isConstant=None ):
    if not cmds.pluginInfo( 'ZENNForMaya', q=True, l=True ):
        cmds.loadPlugin( 'ZENNForMaya' )

    zn_load   = cmds.createNode( 'ZN_Load' )
    zn_viewer = cmds.createNode( 'ZN_StrandsViewer' )
    cmds.setAttr( '%s.batchModeDraw' % zn_viewer, 0 )

    if isConstant:
        timeNode = 'ztime_%s' % isConstant
        if not cmds.objExists( timeNode ):
            timeNode = cmds.createNode( 'time', n=timeNode )
        cmds.setAttr( '%s.outTime' % timeNode, int(isConstant) )
    else:
        timeNode = 'time1'
    cmds.connectAttr( '%s.outTime' % timeNode, '%s.inTime' % zn_load )
    cmds.connectAttr( '%s.outStrands' % zn_load, '%s.inStrands' % zn_viewer )
    return zn_load, zn_viewer


def zennStrandsViewer( cachePath ):
    nodes = dict()
    for i in os.listdir( cachePath ):
        nodePath = os.path.join( cachePath, i )
        if os.path.isdir( nodePath ):
            info_file = os.path.join( nodePath, 'info' )
            if os.path.exists( info_file ):
                ratio = 0.1
                fread = open( info_file, 'r' ).read()
                if fread:
                    d = json.loads( fread )
                    if d.has_key( 'StrandsCount' ):
                        snum = float( d['StrandsCount'] )
                        ratio = snum / (snum*snum/100.0)
                        if ratio > 1.0:
                            ratio = 1.0
                nodes[i] = ( checkup_zennConstant(nodePath), ratio )
    if not nodes:
        return

    trans = list()
    for i in nodes:
        isConst, displayRatio = nodes[i]
        load, viewer = create_zennStrandsViewer( isConst )

        # viewer
        cmds.setAttr( '%s.colorMode' % viewer, 2 )
        cmds.setAttr( '%s.displayRatio' % viewer, displayRatio )
        cmds.setAttr( '%s.hideBackface' % viewer, 0 )
        # load
        cmds.setAttr( '%s.cachePath' % load, cachePath, type='string' )
        cmds.setAttr( '%s.cacheName' % load, i, type='string' )



        transform = cmds.listRelatives( viewer, p=True )[0]
        transform = cmds.rename( transform, i )
        trans.append( transform )

    groupName = '%s_zennGrp' % os.path.basename( cachePath )

    if cmds.objExists( groupName ):
        cmds.parent( trans, groupName )

        for t in trans:
            cmds.setAttr( '%s.t' % t, 0, 0, 0 )
            cmds.setAttr( '%s.r' % t, 0, 0, 0 )
            cmds.setAttr( '%s.s' % t, 1, 1, 1 )

    else:
        groupName = cmds.group( trans, n=groupName )

    cmds.setAttr('%s.rotatePivot' % groupName, 0, 0, 0 )
    cmds.setAttr('%s.scalePivot' % groupName, 0, 0, 0 )

    cmds.select( cl=True )
    mel.eval( 'print "# Result : zennStransViewer <%s>\\n"' % cachePath )
    return groupName




#-------------------------------------------------------------------------------
#
#   Alembic Points Cache Export by Zenn
#
#-------------------------------------------------------------------------------
class XZennPointsExport:
    def __init__( self, fileName, nodeList ):
        self.m_fileName = fileName
        self.m_nodeList = nodeList

        self.m_start = 1
        self.m_end   = 1
        self.m_step  = 1.0

        #self.doIt()

    def getTimeSample( self ):
        # time sample
        tunit = cmds.currentUnit( t=True, q=True )
        fps   = sgCommon._timeUnitMap[ tunit ]
        self.timeSample = TimeSampling( 1.0/fps*self.m_step, self.m_start/fps )

    def doIt( self ):
        startTime = time.time()

        if os.path.exists( self.m_fileName ):
            os.remove( self.m_fileName )

        self.getTimeSample()
        oarch = OArchive( str(self.m_fileName), asOgawa=True )
        pointerMap = dict()
        for n in self.m_nodeList:
            root = oarch.getTop()
            obj  = OPoints( root, str(n), self.timeSample )
            schema = obj.getSchema()
            arbParam   = schema.getArbGeomParams()
            scaleProp  = OFloatGeomParam( arbParam, 'scale', False, GeometryScope.kVaryingScope, 3, self.timeSample )
            orientProp = OQuatfGeomParam( arbParam, 'orient', False, GeometryScope.kVaryingScope, 1, self.timeSample )
            pointerMap[n] = [schema, arbParam, scaleProp, orientProp]

            self.setConstantData( arbParam, n )

        for f in xrange( self.m_start, self.m_end+1 ):
            for n in self.m_nodeList:
                self.setSampleData( pointerMap[n][0], pointerMap[n][2], pointerMap[n][3], f, n )

        endTime   = time.time()
        # debug
        cmds.warning( 'export file -> %s : %s' % (self.m_fileName, endTime-startTime) )


    def setConstantData( self, GeomParam, Node ):
        # rtp
        rtpVals = cmds.ZN_FollicleInfoCmd( nodeName=Node, attribute='rtp' )

        n_size = len(rtpVals)
        rtps = IntArray( n_size )

        for i in xrange( n_size ):
            rtps[i] = rtpVals[i]

        rtpSamp = OInt32GeomParamSample()
        rtpSamp.setScope( GeometryScope.kVertexScope )
        rtpSamp.setVals( rtps )
        rtpProp = OInt32GeomParam( GeomParam, 'rtp', False, GeometryScope.kVaryingScope, 1 )
        rtpProp.set( rtpSamp )

        # archivePath
        arcSource = list()
        inFeathers = cmds.listConnections( '%s.inFeather' % Node )
        if inFeathers:
            for zn in inFeathers:
                fn = cmds.getAttr( '%s.archiveFilePath' % zn )
                if fn:
                    arcSource.append( fn )
                else:
                    shapeName = cmds.listConnections( zn, s=True, d=False )
                    if shapeName:
                        dirpath = '/show/real/shot/HGA/HGA_0290/hair/pub/data/asb/source'
                        fn = os.path.join( dirpath, '%s.abc' % shapeName[0] )
                        arcSource.append( fn )

        arcSamp = OStringGeomParamSample()
        arcSamp.setScope( GeometryScope.kConstantScope )
        arcPath = StringArray( len(arcSource) )
        for i in range( len(arcSource) ):
            arcPath[i] = str(arcSource[i])
        arcSamp.setVals( arcPath )
        arcProp = OStringGeomParam( GeomParam, 'archivePath', False, GeometryScope.kConstantScope, 1 )
        arcProp.set( arcSamp )


    def setSampleData( self, PointSchema, ScaleGeomParam, OrientGeomParam, Frame, Node ):
        # current frame
        cmds.currentTime( Frame )

        zn_mtx  = cmds.ZN_FollicleInfoCmd( nodeName=Node, attribute='tm' )
        zn_size = len(zn_mtx) / 16

        positions  = V3fArray( zn_size )
        velocities = V3fArray( zn_size )
        ids        = IntArray( zn_size )
        widths     = FloatArray( zn_size )
        scales     = FloatArray( zn_size*3 )
        orients    = QuatfArray( zn_size )
        rtps       = IntArray( zn_size )

        for i in xrange( zn_size ):
            velocities[i]   = V3f( 0, 0, 0 )
            ids[i]          = i
            widths[i]       = 1

            mtx      = OpenMaya.MMatrix( zn_mtx[i*16:i*16+16] )
            transMtx = OpenMaya.MTransformationMatrix( mtx )
            # position
            tr = transMtx.translation( OpenMaya.MSpace.kWorld )
            positions[i] = V3f( tr.x, tr.y, tr.z )
            # scale
            sc = transMtx.scale( OpenMaya.MSpace.kWorld )
            scales[i*3]   = sc[0]
            scales[i*3+1] = sc[1]
            scales[i*3+2] = sc[2]
            # rotate
            ro = transMtx.rotation( asQuaternion=True )
            orients[i] = Quatf( ro.x, ro.y, ro.z, ro.w )

        widthSamp = OFloatGeomParamSample()
        widthSamp.setScope( GeometryScope.kVertexScope )
        widthSamp.setVals( widths )

        psamp = OPointsSchemaSample()
        psamp.setPositions( positions )
        psamp.setIds( ids )
        psamp.setVelocities( velocities )
        psamp.setWidths( widthSamp )
        PointSchema.set( psamp )

        # arbGeomParams
        scaleSamp = OFloatGeomParamSample()
        scaleSamp.setScope( GeometryScope.kVertexScope )
        scaleSamp.setVals( scales )
        ScaleGeomParam.set( scaleSamp )
        orientSamp = OQuatfGeomParamSample()
        orientSamp.setScope( GeometryScope.kVertexScope )
        orientSamp.setVals( orients )
        OrientGeomParam.set( orientSamp )


class ZennPointsExport:
    def __init__( self, outDir, nodeList ):
        self.m_outDir   = outDir
        self.m_nodeList = nodeList

        self.m_start = 1
        self.m_end   = 1
        self.m_step  = 1.0

    def getTimeSample( self ):
        tunit = cmds.currentUnit( t=True, q=True )
        fps   = sgCommon._timeUnitMap[ tunit ]
        self.timeSample = TimeSampling( 1.0/fps*self.m_step, self.m_start/fps )

    def doIt( self ):
        startTime = time.time()

        self.getTimeSample()
        self.m_frames = sgCommon.get_frames( self.m_start, self.m_end, self.m_step )
        DATA = dict()
        for n in self.m_nodeList:
            fn    = os.path.join( self.m_outDir, '%s.asb' % n )
            if os.path.exists( fn ):
                os.remove( fn )
            oarch = OArchive( str(fn), asOgawa=True )
            root  = oarch.getTop()
            obj   = OPoints( root, str(n), self.timeSample )
            schema= obj.getSchema()
            arbParam  = schema.getArbGeomParams()
            scaleProp = OFloatGeomParam( arbParam, 'scale', False, GeometryScope.kVaryingScope, 3, self.timeSample )
            orientProp= OQuatfGeomParam( arbParam, 'orient',False, GeometryScope.kVaryingScope, 1, self.timeSample )
            DATA[n]   = [ schema, arbParam, scaleProp, orientProp ]

            self.setConstantData( arbParam, n )

        for f in self.m_frames:
            for n in self.m_nodeList:
                self.setSampleData( DATA[n][0], DATA[n][2], DATA[n][3], f, n )

        endTime = time.time()
        # debug
        cmds.warning( 'export dir   -> %s' % self.m_outDir )
        cmds.warning( 'export nodes -> %s' % string.join(self.m_nodeList, ',') )

        # insert db
        ppRulebook = rulebook.Coder()
        try:
            # write log in db [ 2017.04.19 by daeseok.chae ]
            ppRulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_publish.yaml")

            decodingRule = ppRulebook.asset.zenn_assembly.decode(self.m_outDir,
                                                                product_name="assembly_path")

            ppRulebook.asset.flag["ASSET"] = decodingRule["ASSET"]
            ppRulebook.asset.flag["TYPE"] = decodingRule["TYPE"]
            ppRulebook.flag["PROJECT"] = decodingRule["PROJECT"]
            ppRulebook.flag["VER"] = decodingRule["VER"]
            task = "asset"
            dataType = "zenn_assembly"

            typeTaskCoder = ppRulebook._child[task]._child[dataType]

            files = {}
            for product in typeTaskCoder.myProducts:
                if product == "decode_path":
                    continue
                try:
                    files[product] = [typeTaskCoder.product[product]]
                except:
                    files[product] = [""]

            files["node_list"] = []
            for i in self.m_nodeList:
                files["node_list"].append(i)

            record = {"show": str(ppRulebook.flag["PROJECT"]),
                      "asset_type": str(ppRulebook.asset.flag["TYPE"]),
                      "data_type": dataType,
                      "asset_name": str(ppRulebook.asset.flag["ASSET"]),
                      "tags": tag_parser.run(self.m_outDir),
                      "task": task,
                      "artist": getpass.getuser(),
                      "enabled": True,
                      "version": getPubVersion(show=str(ppRulebook.flag["PROJECT"]),
                                               task=task,
                                               data_type=dataType,
                                               asset_name=str(ppRulebook.asset.flag["ASSET"])),
                      "time": datetime.datetime.now().isoformat(),
                      "task_publish": {},
                      "files": files,
                      }

            COLLNAME = str(ppRulebook.flag["PROJECT"])
            client = MongoClient(DBIP)
            database = client[DBNAME]
            dbColl = database[COLLNAME]

            dbColl.insert_one(record)

            print("success db write", record)
        except Exception as e:
            COLLNAME = "puberror"
            dbName = "test"
            client = MongoClient(DBIP)
            database = client[dbName]
            dbColl = database[COLLNAME]

            record = {"user": getpass.getuser(),
                      "errorMsg": str(e),
                      "time": datetime.datetime.now().isoformat(),
                      "project": str(ppRulebook.flag["PROJECT"]),
                      "asset": str(ppRulebook.asset.flag["ASSET"]),
                      "type": "zenn_assembly",
                      "outputpath": self.m_outDir,
                      "maya_version": "maya2_2017"}

            dbColl.insert_one(record)


    def setConstantData( self, GeomParam, Node ):
        # rtp
        rtpVals = cmds.ZN_FollicleInfoCmd( nodeName=Node, attribute='rtp' )

        n_size = len(rtpVals)
        rtps = IntArray( n_size )

        for i in xrange( n_size ):
            rtps[i] = rtpVals[i]

        rtpSamp = OInt32GeomParamSample()
        rtpSamp.setScope( GeometryScope.kVertexScope )
        rtpSamp.setVals( rtps )
        rtpProp = OInt32GeomParam( GeomParam, 'rtp', False, GeometryScope.kVaryingScope, 1 )
        rtpProp.set( rtpSamp )

        # archivePath
        arcSource = list()
        inFeathers = cmds.listConnections( '%s.inFeather' % Node )
        if inFeathers:
            for zn in inFeathers:
                fn = cmds.getAttr( '%s.archiveFilePath' % zn )
                if fn:
                    arcSource.append( fn )
                else:
                    shapeName = cmds.listConnections( zn, s=True, d=False )
                    if shapeName:
                        dirpath = '/show/real/shot/HGA/HGA_0290/hair/pub/data/asb/source'
                        fn = os.path.join( dirpath, '%s.abc' % shapeName[0] )
                        arcSource.append( fn )

        arcSamp = OStringGeomParamSample()
        arcSamp.setScope( GeometryScope.kConstantScope )
        arcPath = StringArray( len(arcSource) )
        for i in range( len(arcSource) ):
            arcPath[i] = str(arcSource[i])
        arcSamp.setVals( arcPath )
        arcProp = OStringGeomParam( GeomParam, 'archivePath', False, GeometryScope.kConstantScope, 1 )
        arcProp.set( arcSamp )


    def setSampleData( self, PointSchema, ScaleGeomParam, OrientGeomParam, Frame, Node ):
        # current frame
        cmds.currentTime( Frame )

        zn_mtx  = cmds.ZN_FollicleInfoCmd( nodeName=Node, attribute='tm' )
        zn_size = len(zn_mtx) / 16

        positions  = V3fArray( zn_size )
        velocities = V3fArray( zn_size )
        ids        = IntArray( zn_size )
        widths     = FloatArray( zn_size )
        scales     = FloatArray( zn_size*3 )
        orients    = QuatfArray( zn_size )
        rtps       = IntArray( zn_size )

        for i in xrange( zn_size ):
            velocities[i]   = V3f( 0, 0, 0 )
            ids[i]          = i
            widths[i]       = 1

            mtx      = OpenMaya.MMatrix( zn_mtx[i*16:i*16+16] )
            transMtx = OpenMaya.MTransformationMatrix( mtx )
            # position
            tr = transMtx.translation( OpenMaya.MSpace.kWorld )
            positions[i] = V3f( tr.x, tr.y, tr.z )
            # scale
            sc = transMtx.scale( OpenMaya.MSpace.kWorld )
            scales[i*3]   = sc[0]
            scales[i*3+1] = sc[1]
            scales[i*3+2] = sc[2]
            # rotate
            ro = transMtx.rotation( asQuaternion=True )
            orients[i] = Quatf( ro.x, ro.y, ro.z, ro.w )

        widthSamp = OFloatGeomParamSample()
        widthSamp.setScope( GeometryScope.kVertexScope )
        widthSamp.setVals( widths )

        psamp = OPointsSchemaSample()
        psamp.setPositions( positions )
        psamp.setIds( ids )
        psamp.setVelocities( velocities )
        psamp.setWidths( widthSamp )
        PointSchema.set( psamp )

        # arbGeomParams
        scaleSamp = OFloatGeomParamSample()
        scaleSamp.setScope( GeometryScope.kVertexScope )
        scaleSamp.setVals( scales )
        ScaleGeomParam.set( scaleSamp )
        orientSamp = OQuatfGeomParamSample()
        orientSamp.setScope( GeometryScope.kVertexScope )
        orientSamp.setVals( orients )
        OrientGeomParam.set( orientSamp )




def exportAbc( outDir, Start, End, Step ):
    nodes = cmds.ls( type='ZN_FeatherInstance' )
    zclass = ZennPointsExport( outDir, nodes )
    zclass.m_start = Start
    zclass.m_end   = End
    zclass.m_step  = Step
    zclass.doIt()
