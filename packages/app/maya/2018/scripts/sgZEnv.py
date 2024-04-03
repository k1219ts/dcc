#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	SceneGraph Zenv procedural
#
#	2017.03.21	$2
#-------------------------------------------------------------------------------

import os, sys
import string
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

# 2017.05.12 by daeseok.chae
from dxname import rulebook
from dxname import tag_parser
import pymongo
from pymongo import MongoClient
import datetime
import getpass
import dxConfig

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "PIPE_PUB"

def getPubVersion(show, task, data_type, asset_name):
    client = MongoClient(DBIP)
    db = client[DBNAME]
    coll = db[show]

    print show, task, data_type, asset_name
    recentDoc = coll.find_one({'show': show,
                               'task': task,
                               'data_type': data_type,
                               'asset_name': asset_name},
                              sort=[('version', pymongo.DESCENDING)])
    if recentDoc:
        return recentDoc['version'] + 1
    else:
        return 1

class ZEnvPointsExport:
    def __init__( self, fileName, nodeList ):
        self.m_fileName = fileName
        self.m_nodeList = nodeList

        #self.doIt()

    def doIt( self ):
        startTime = time.time()
        sourceFiles = None

        if os.path.exists( self.m_fileName ):
            os.remove( self.m_fileName )
        oarch = OArchive( str(self.m_fileName), asOgawa=True )
        for n in self.m_nodeList:
            root = oarch.getTop()
            obj  = OPoints( root, str(n) )
            schema = obj.getSchema()
            arb	   = schema.getArbGeomParams()

            self.setSampleData( schema, arb, n )
            sourceFiles = self.setConstantData( arb, n )

        endTime = time.time()
        # debug
        cmds.warning( 'export file -> %s : %s' % (self.m_fileName, endTime-startTime) )

        # insert db
        ppRulebook = rulebook.Coder()
        try:
            # write log in db [ 2017.04.19 by daeseok.chae ]
            ppRulebook.load_rulebook("/netapp/backstage/pub/lib/python_lib/dxname/name_for_publish.yaml")

            outputPath = self.m_fileName

            if outputPath.startswith('/netapp/dexter/show'):
                outputPath = outputPath.replace('/netapp/dexter/show', '/show')

            outputPath = os.path.dirname(outputPath)

            decodingRule = ppRulebook.asset.env_assembly.decode(outputPath,
                                                                product_name="root")

            ppRulebook.asset.flag["ASSET"] = decodingRule["ASSET"]
            ppRulebook.asset.flag["TYPE"] = decodingRule["TYPE"]
            ppRulebook.flag["PROJECT"] = decodingRule["PROJECT"]
            ppRulebook.flag["VER"] = decodingRule["VER"]
            task = "asset"
            dataType = "assembly"

            files = {}

            files['assembly'] = [outputPath]

            files["source_path"] = []
            for i in sourceFiles:
                files["source_path"].append(i)

            record = {"show": str(ppRulebook.flag["PROJECT"]),
                      "asset_type": str(ppRulebook.asset.flag["TYPE"]),
                      "data_type": dataType,
                      "asset_name": str(ppRulebook.asset.flag["ASSET"]),
                      "tags": tag_parser.run(outputPath),
                      "task": task,
                      "artist": getpass.getuser(),
                      "enabled": True,
                      "version": getPubVersion(show=str(ppRulebook.flag["PROJECT"]),
                                               task = task,
                                               data_type = dataType,
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

            print "success db write", record
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
                      "type": "assembly",
                      "outputpath": self.m_fileName,
                      "maya_version": "maya2_2017"}

            dbColl.insert_one(record)


    def setConstantData( self, GeomParam, Node ):
        files, paths, bounds = self.getSourceList( Node )

        d_size = len(files)
        # bound
        boundSamp = OFloatGeomParamSample()
        boundSamp.setScope( GeometryScope.kConstantScope )
        boundVal  = FloatArray( d_size*6 )
        # arcfiles
        fileSamp  = OStringGeomParamSample()
        fileSamp.setScope( GeometryScope.kConstantScope )
        fileVal   = StringArray( d_size )
        # arcpath
        if paths:
            pathSamp  = OStringGeomParamSample()
            pathSamp.setScope( GeometryScope.kConstantScope )
            pathVal   = StringArray( d_size )

        # set values
        for i in range( d_size ):
            fileVal[i] = str( files[i] )
            if paths:
                pathVal[i] = str( paths[i] )
            for x in range(6):
                boundVal[i*6+x] = bounds[i*6+x]

        boundSamp.setVals( boundVal )
        fileSamp.setVals( fileVal )
        if paths:
            pathSamp.setVals( pathVal )

        boundProp = OFloatGeomParam( GeomParam, 'bound', False, GeometryScope.kConstantScope, 6 )
        boundProp.set( boundSamp )
        filesProp = OStringGeomParam( GeomParam, 'arcfiles', False, GeometryScope.kConstantScope, 1 )
        filesProp.set( fileSamp )
        if paths:
            pathProp  = OStringGeomParam( GeomParam, 'arcpath', False, GeometryScope.kConstantScope, 1 )
            pathProp.set( pathSamp )

        return files


    def setSampleData( self, PointSchema, GeomParam, Node ):
        count = cmds.getAttr( '%s.count' % Node )
        positions = V3fArray( count )
        ids       = IntArray( count )
        scales    = FloatArray( count*3 )
        orients   = QuatfArray( count )
        rtps      = IntArray( count )

        # source index
        sids    = cmds.ZEnvPointInfoCmd( nodeName=Node, attribute='sid' )
        # get 4x4 matrix
        matrixs = cmds.ZEnvPointInfoCmd( nodeName=Node, attribute='tm' )

        for i in xrange( count ):
            ids[i]  = i
            rtps[i] = int( sids[i] )
            mtx     = OpenMaya.MMatrix( matrixs[i*16:i*16+16] )
            transMtx= OpenMaya.MTransformationMatrix( mtx )
            # translate
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

        # time sample
        ts = TimeSampling( 1.0/24.0, 1/24.0 )

        psamp = OPointsSchemaSample()
        psamp.setPositions( positions )
        psamp.setIds( ids )
        PointSchema.set( psamp )

        # arbGeomParams
        #   scale
        scaleSamp = OFloatGeomParamSample()
        scaleSamp.setScope( GeometryScope.kVertexScope )
        scaleSamp.setVals( scales )
        ScaleGeomParam = OFloatGeomParam( GeomParam, 'scale', False, GeometryScope.kVaryingScope, 3, ts )
        ScaleGeomParam.set( scaleSamp )
        #   orient
        orientSamp = OQuatfGeomParamSample()
        orientSamp.setScope( GeometryScope.kVertexScope )
        orientSamp.setVals( orients )
        OrientGeomParam = OQuatfGeomParam( GeomParam, 'orient', False, GeometryScope.kVaryingScope, 1, ts )
        OrientGeomParam.set( orientSamp )
        #   rtp
        rtpsSamp = OInt32GeomParamSample()
        rtpsSamp.setScope( GeometryScope.kVertexScope )
        rtpsSamp.setVals( rtps )
        rtpsGeomParam = OInt32GeomParam( GeomParam, 'rtp', False, GeometryScope.kVaryingScope, 1 )
        rtpsGeomParam.set( rtpsSamp )


    def getSourceList( self, node ):
        files  = list()
        paths  = list()
        bounds = list()
        objectName_valid = 0
        sourceSets = cmds.listConnections( '%s.inSourceSets' % node, sh=True )
        for s in sourceSets:
            sourceShape = cmds.listConnections( s, s=True, d=False, sh=True )
            if sourceShape:
                sourceShape = sourceShape[0]
                fileFormat  = cmds.getAttr( '%s.fileFormat' % sourceShape ) # 0: .abc, 1: .rib
                objectName  = cmds.getAttr( '%s.objectName' % sourceShape )
                assetPath   = cmds.getAttr( '%s.assetPath' % sourceShape )
                version     = cmds.getAttr( '%s.version' % sourceShape )
                if fileFormat == 0:
                    fn = os.path.join( assetPath, 'model', '%s_model_%s.abc' % \
                            (os.path.basename(assetPath), version) )
                else:
                    fn = assetPath
                if os.path.exists( fn ):
                    files.append( fn )
                    paths.append( objectName.replace('_low', '') )
                    if objectName:
                        objectName_valid += 1
                    bounds += self.getBound(sourceShape)
        if objectName_valid == 0:
            paths = None
        return files, paths, bounds

    def getBound( self, node ):
        # node = shapeNode
        xmin, ymin, zmin = cmds.getAttr( '%s.boundingBoxMin' % node )[0]
        xmax, ymax, zmax = cmds.getAttr( '%s.boundingBoxMax' % node )[0]
        return [xmin, xmax, ymin, ymax, zmin, zmax]



def exportAbc( outDir ):
    groupNodes = cmds.ls( sl=True, type='ZEnvGroup' )
    if not groupNodes:
        groupNodes = cmds.ls( type='ZEnvGroup' )
    for i in range( len(groupNodes) ):
        nodes = cmds.ls( groupNodes[i], dag=True, type='ZEnvPointSet', l=True )
        if len( nodes ) > 1:
            mel.eval( 'print "## debug : convention check!\\n"' )
        else:
            fn = os.path.join( outDir, '%s.asb' % groupNodes[i] )
            zvClass = ZEnvPointsExport( fn, [nodes[0]] )
            zvClass.doIt()