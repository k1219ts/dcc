#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	SceneGraph Common procedural
#
#	2017.03.05	$5
#-------------------------------------------------------------------------------
#
#	Common
#   - pluginSetup
#	- getNameSpace, getImportReferenceFiles, unlockAttrs
#	- initTransform
#   - setPostTransformScript, setPreShapeScript, setInVisAttribute
#
#	Animation Key
#	- offsetKey, offsetKeyAttr,
#	- coreKeyDump, attributesKeyDump, coreKeyLoad, attributesKeyLoad
#	- WorldAnimation
#
#	Alembic Data
#	- get_frames,
#	- export_worldAlembic, import_worldAlembic
#   - AbcXformImportKey
#   - getMtx, setMtx
#
#-------------------------------------------------------------------------------

import os, sys
import string
import math
import re
import json
import numpy

import maya.api.OpenMaya as OpenMaya
import maya.api.OpenMayaAnim as OpenMayaAnim

# for alembic python
from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from alembic.Util import *
kWrapExisting = WrapExistingFlag.kWrapExisting

import maya.cmds as cmds
import maya.mel as mel


#-------------------------------------------------------------------------------
#
#	Common
#
#-------------------------------------------------------------------------------
def pluginSetup( plugins ):
    for p in plugins:
        if not cmds.pluginInfo( p, q=True, l=True ):
            cmds.loadPlugin( p )


def getNameSpace( nodeName ):
    ns_name = ""
    src = nodeName.split(':')
    if len(src) > 1:
        ns_name = string.join( src[:-1], ':' )
    node_name = src[-1]
    return ns_name, node_name


def getImportReferenceFiles():
    refs = cmds.file( q=True, r=True )
    for f in refs:
        if cmds.referenceQuery( f, isLoaded=True ):
            cmds.file( f, importReference=True )
    return refs


def unlockAttrs( nodeName=None, Attrs=list() ):
    checkPoint = True
    while checkPoint:
        tmp_check = 0
        for ln in Attrs:
            plug = cmds.connectionInfo( '%s.%s' % (nodeName, ln),
                                        getLockedAncestor=True )
            if plug:
                cmds.setAttr( plug, lock=False )
                tmp_check += 1
        if tmp_check == 0:
            checkPoint = False


def initTransform( node ):
    cmds.setAttr( '%s.t' % node, 0, 0, 0 )
    cmds.setAttr( '%s.r' % node, 0, 0, 0 )
    cmds.setAttr( '%s.s' % node, 1, 1, 1 )


def setPostTransformScript( node ):
    at = 'rman__torattr___postTransformScript'
    if not cmds.attributeQuery( at, n=node, ex=True ):
        cmds.addAttr( node, ln=at, dt='string' )
    cmds.setAttr( '%s.%s' % (node, at), 'dxarc', type='string' )

def setPreShapeScript( node ):
    at = 'rman__torattr___preShapeScript'
    if not cmds.attributeQuery( at, n=node, ex=True ):
        cmds.addAttr( node, ln=at, dt='string' )
    cmds.setAttr( '%s.%s' % (node, at), 'dxarc', type='string' )

def setInVisAttribute( node, value ):
    at = 'rman__torattr___invis'
    if not cmds.attributeQuery( at, n=node, ex=True ):
        cmds.addAttr( node, ln=at, at='long' )
    cmds.setAttr( '%s.%s' % (node, at), value )

#-------------------------------------------------------------------------------
#
#	Animation Key
#
#-------------------------------------------------------------------------------
def selectedOffsetKey():
    offsetKey(cmds.ls(sl=True), 1)

def offsetKey( node, offset ):
    objs = cmds.ls( node, dag=True, ni=True )
    connections = cmds.listConnections( objs, type='animCurve' )
    if not connections:
        return
    for i in connections:
        ln   = cmds.listConnections( i, p=True )
        src = ln[0].split('.')
        offsetKeyAttr( src[0], src[-1], offset )

def offsetKeyAttr( node, attr, offset ):
    ln = '%s.%s' % ( node, attr )
    frames = cmds.keyframe( ln, q=True, a=True )
    # end frame
    if frames:
        end_value = cmds.getAttr( ln, t=frames[-1] )
        tmp_value = cmds.getAttr( ln, t=frames[-1]-offset )
        set_value = end_value - tmp_value
        cmds.setKeyframe( ln, itt='spline', ott='spline', t=frames[-1]+offset,
                          at=attr, v=end_value+set_value )
        # start frame
        start_value = cmds.getAttr( ln, t=frames[0] )
        tmp_value	= cmds.getAttr( ln, t=frames[0]+offset )
        set_value	= tmp_value - start_value
        cmds.setKeyframe( ln, itt='spline', ott='spline', t=frames[0]-offset,
                          at=attr, v=start_value-set_value )
    else:
        # end frame
        end_value = int( cmds.playbackOptions( q=True, min=True ) ) + 1
        cmds.setKeyframe( ln, itt='spline', ott='spline', t=end_value,
                          at=attr, v=end_value )
        # start frame
        start_value = int( cmds.playbackOptions( q=True, max=True ) ) + 1
        cmds.setKeyframe( ln, itt='spline', ott='spline', t=start_value,
                          at=attr, v=start_value )


def coreKeyDump( node, attr ):
    connections = cmds.listConnections( '%s.%s' % (node, attr), type='animCurve', s=True, d=False ) # add destination, source options
    if connections:
        result = dict()
        result['frame'] = cmds.keyframe( node, at=attr, q=True )
        result['value'] = cmds.keyframe( node, at=attr, q=True, vc=True )
        result['angle'] = cmds.keyTangent( node, at=attr, q=True, ia=True, oa=True )
        if cmds.keyTangent( node, at=attr, q=True, wt=True )[0]:
            result['weight'] = cmds.keyTangent( node, at=attr, q=True, iw=True, ow=True )
        result['infinity'] = cmds.setInfinity( node, at=attr, q=True, pri=True, poi=True )
        return result
    else:
        gv = cmds.getAttr( '%s.%s' % (node, attr) )
        gt = cmds.getAttr( '%s.%s' % (node, attr), type=True )
        return {'value':gv, 'type':gt}

def attributesKeyDump( node, attrs ):	# attrs=list()
    result = dict()
    for ln in attrs:
        result[ln] = coreKeyDump( node, ln )
    return result


def coreKeyLoad( node, attr, keyData ):
    cmds.cutKey( node, at=attr )

    for i in keyData['frame']:
        cmds.setKeyframe( node, at=attr, time=i,
                          value=keyData['value'][keyData['frame'].index(i)] )

    if keyData.has_key( 'infinity' ):
        value = keyData['infinity']
        cmds.setInfinity( node, at=attr, pri=value[0], poi=value[1] )

    if keyData.has_key( 'weight' ):
        cmds.keyTangent( node, at=attr, weightedTangents=True )

    for i in keyData['frame']:
        if keyData.has_key( 'angle' ):
            cmds.keyTangent( node, at=attr, e=True, time=(i,i),
                             inAngle =keyData['angle'][keyData['frame'].index(i)*2],
                             outAngle=keyData['angle'][keyData['frame'].index(i)*2+1] )
        if keyData.has_key( 'weight' ):
            cmds.keyTangent( node, at=attr, e=True, time=(i,i),
                             inWeight =keyData['weight'][keyData['frame'].index(i)*2],
                             outWeight=keyData['weight'][keyData['frame'].index(i)*2+1], )

def attributesKeyLoad( node, keyData ):
    for ln in keyData:
        if type(keyData[ln]).__name__ == 'dict':
            if 'frame' in keyData[ln].keys():
                coreKeyLoad( node, ln, keyData[ln] )
            else:
                gv = keyData[ln]['value']
                gt = keyData[ln]['type']
                if gt == 'string':
                    cmds.setAttr( '%s.%s' % (node, ln), gv, type='string' )
                else:
                    cmds.setAttr( '%s.%s' % (node, ln), gv )
        else:
            cmds.setAttr( '%s.%s' % (node, ln), keyData[ln] )


# read custom world format
class WorldAnimation:
    def __init__( self, node, filename ):
        self.m_baked	= None
        self.m_node		= node
        self.m_filename = filename

        self.m_nameSpace = None
        src = node.split(':')
        if len(src) > 1:
            self.m_nameSpace = string.join( src[:-1], ':' )

    # edit 2016.09.12 $1 - for current world anim node update
    def createCtrl( self ):
        parentPath = cmds.listRelatives( self.m_node, p=True, f=True )
        if self.m_baked:
            if parentPath and parentPath[0].find('world_CON') > -1:
                self.m_wcon = parentPath[0]
                connected = cmds.listConnections(self.m_wcon, type='animCurve')
                cmds.delete( connected )
            else:
                cName = 'world_CON'
                if self.m_nameSpace:
                    cName = self.m_nameSpace + ':' + cName
                cNode = cmds.group( n=cName, em=True )
                self.m_wcon = cNode
                cmds.parent( self.m_node, cNode )
        else:
            if parentPath and parentPath[0].find('move_CON') > -1:
                src = parentPath[0].split('|')
                for i in [2, 3, 4]:
                    connected = cmds.listConnections( string.join(src[:i], '|'), type='animCurve' )
                    if connected:
                        cmds.delete( connected )
            else:
                mName = 'move_CON'
                dName = 'direction_CON'
                pName = 'place_CON'
                if self.m_nameSpace:
                    mName = self.m_nameSpace + ':' + mName
                    dName = self.m_nameSpace + ':' + dName
                    pName = self.m_nameSpace + ':' + pName
                mNode = cmds.group( n=mName, em=True )
                dNode = cmds.group( mNode, n=dName )
                pNode = cmds.group( dNode, n=pName )

                nodes = cmds.ls( pNode, dag=True )
                cmds.parent( self.m_node, nodes[-1] )

    def doIt( self ):
        # check world animation type
        parentPath = cmds.listRelatives( self.m_node, p=True, f=True )
        if parentPath:
            if len( parentPath[0].split('|') ) > 2:
                self.m_baked = False
            else:
                self.m_baked = True

        if os.path.splitext( self.m_filename )[-1] == '.world':
            # desc : create world transform node
            self.createCtrl()

            f = open( self.m_filename, 'r' )
            body = json.load( f )
            f.close()

            if self.m_baked:
                if body.has_key('WorldBaked'):
                    self.setBakedKeys( body['WorldBaked'] )
                else:
                    self.setKeys( body['World'] )
            else:
                if body.has_key('World'):
                    self.setKeys( body['World'] )

            # desc : set initScale
            if body.has_key('InitScale') and body['InitScale']:
                keyData = body['InitScale']['initScale']
                for ln in ['sx', 'sy', 'sz']:
                    if keyData.has_key('frame'):
                        coreKeyLoad( self.m_node, ln, keyData )
                    else:
                        cmds.setAttr( '%s.%s' % (self.m_node, ln), keyData['value'] )


    def setBakedKeys( self, keyData ):
        parentPath = cmds.listRelatives( self.m_node, p=True, f=True )[0]

        dgmod = OpenMaya.MDGModifier()

        TL_list = ['translateX', 'translateY', 'translateZ', 'scaleX', 'scaleY', 'scaleZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        objList = list()
        for i in TL_list:
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, '%s_%s' % (parentPath, i) )
            objList.append( obj )
        for i in TA_list:
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, '%s_%s' % (parentPath, i) )
            objList.append( obj )

        dgmod.doIt()

        keyObjList = list()
        for o in objList:
            obj = OpenMayaAnim.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )

        frameList = keyData.keys()
        frameList.sort()

        for f in frameList:
            mtx = OpenMaya.MMatrix( keyData[f] )
            transMtx = OpenMaya.MTransformationMatrix( mtx )

            tr = transMtx.translation( OpenMaya.MSpace.kWorld )
            for i in range(3):
                keyObjList[i].addKey( OpenMaya.MTime(int(f), OpenMaya.MTime.uiUnit()), tr[i] )

            sc = transMtx.scale( OpenMaya.MSpace.kWorld )
            for i in range(3):
                keyObjList[i+3].addKey( OpenMaya.MTime(int(f), OpenMaya.MTime.uiUnit()), sc[i] )

            ro = transMtx.rotation()
            ro = OpenMaya.MEulerRotation( ro )
            for i in range(3):
                keyObjList[i+6].addKey( OpenMaya.MTime(int(f), OpenMaya.MTime.uiUnit()), ro[i] )

        curveNames = list()
        for i in TL_list:
            index = TL_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (parentPath, i) )
        for i in TA_list:
            index = TA_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index+6] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (parentPath, i) )
        cmds.filterCurve( curveNames )


    # new world anim-key
    def setKeys( self, keyData ):
        parentPath = cmds.listRelatives( self.m_node, p=True, f=True )[0]
        src = parentPath.split('|')
        # move_CON
        node = string.join( src[:4], '|' )
        attributesKeyLoad( node, keyData['move_CON'] )
        # direction_CON
        node = string.join( src[:3], '|' )
        attributesKeyLoad( node, keyData['direction_CON'] )
        # place_CON
        node = string.join( src[:2], '|' )
        attributesKeyLoad( node, keyData['place_CON'] )





#-------------------------------------------------------------------------------
#
#	Alembic Data
#
#-------------------------------------------------------------------------------
_timeUnitMap = {'game':15.0, 'film':24.0, 'pal':25.0,
                'ntsc':30.0, 'show':48.0, 'palf':50.0, 'ntscf':60.0}

def get_frames( start, end, step ):
    result = list()
    for f in numpy.arange( start, end+1, step ):
        if f <= end:
            result.append( f )
    if result:
        if result[-1] != end:
            result.append( end )
    return result


def export_worldAlembic( Nodes, scaleNode, Start, End, Step, fileName, vender=None ):
    _ifInitScale = False
    initScaleAttribute = 'initScale'
    if vender:
        if vender == 'toneplus':
            initScaleAttribute = 'sx'

    if scaleNode and cmds.attributeQuery( initScaleAttribute, n=scaleNode, ex=True ):
        _ifInitScale = True

    oarch = OArchive( str(fileName), asOgawa=True )

    # time sample
    tunit = cmds.currentUnit( t=True, q=True )
    ts    = TimeSampling( 1.0/_timeUnitMap[tunit] * Step, Start/_timeUnitMap[tunit] )

    # baked world con
    world_nam = string.join( Nodes[-1].split(':')[:-1]+['world_CON'], ':' )
    world_con = OXform( oarch.getTop(), str(world_nam), ts )
    world_mtx, frs = getMtx( Nodes[-1], Start, End, Step )
    #	arbGeomParam
    if _ifInitScale:
        world_arg   = world_con.getSchema().getArbGeomParams()
        world_param = OFloatGeomParam( world_arg, 'initScale', False, GeometryScope.kConstantScope, 1, ts )
    # separate world con
    if len(Nodes) > 1:
        place_con          = OXform( oarch.getTop(), str(Nodes[0]), ts )
        place_mtx, frs     = getMtx( Nodes[0], Start, End, Step, 'matrix' )
        direction_con      = OXform( place_con, str(Nodes[1]), ts )
        direction_mtx, frs = getMtx( Nodes[1], Start, End, Step, 'matrix' )
        move_con	       = OXform( direction_con, str(Nodes[2]), ts )
        move_mtx, frs      = getMtx( Nodes[2], Start, End, Step, 'matrix' )
        #	arbGeomParam
        if _ifInitScale:
            move_arg	  = move_con.getSchema().getArbGeomParams()
            move_param	  = OFloatGeomParam( move_arg, 'initScale', False, GeometryScope.kConstantScope, 1, ts )

    frames = get_frames( Start, End, Step )
    for i in range(len(world_mtx)):
        w_samp = XformSample()
        w_mtx  = M44d()
        w_mtx.makeIdentity()
        if len(Nodes) > 1:
            p_samp = XformSample()
            p_mtx  = M44d()
            p_mtx.makeIdentity()
            d_samp = XformSample()
            d_mtx  = M44d()
            d_mtx.makeIdentity()
            m_samp = XformSample()
            m_mtx  = M44d()
            m_mtx.makeIdentity()
        for x in range(4):
            for y in range(4):
                w_mtx[x][y] = world_mtx[i][x*4+y]
                if len(Nodes) > 1:
                    p_mtx[x][y] = place_mtx[i][x*4+y]
                    d_mtx[x][y] = direction_mtx[i][x*4+y]
                    m_mtx[x][y] = move_mtx[i][x*4+y]
        w_samp.setMatrix( w_mtx )
        world_con.getSchema().set( w_samp )
        if len(Nodes) > 1:
            p_samp.setMatrix( p_mtx )
            place_con.getSchema().set( p_samp )
            d_samp.setMatrix( d_mtx )
            direction_con.getSchema().set( d_samp )
            m_samp.setMatrix( m_mtx )
            move_con.getSchema().set( m_samp )

        # initScale
        if _ifInitScale:
            iscale    = FloatArray( 1 )
            getValue  = cmds.getAttr( '%s.%s' % (scaleNode, initScaleAttribute), t=frames[i] )
            iscale[0] = getValue
            if vender:
                if vender == 'toneplus':
                    iscale[0] = 1.0 / getValue

            iw_samp   = OFloatGeomParamSample()
            iw_samp.setVals( iscale )
            world_param.set( iw_samp )
            if len(Nodes) > 1:
                im_samp   = OFloatGeomParamSample()
                im_samp.setVals( iscale )
                move_param.set( im_samp )
    # debug
    mel.eval( 'print "# Result : \\"World\\" write <%s>\\n"' % fileName )


def import_worldAlembic( curNode, baked, fileName ):
    baseName = os.path.splitext( os.path.basename(fileName) )[0]
    ns_name  = string.join( baseName.split(':')[:-1], ':' )

    if ns_name:
        try:
            cmds.namespace( add=ns_name )
        except:
            pass
    cmds.namespace( set=':' )

    root = cmds.group( n='%s:import_world' % ns_name, em=True )
    if baked:
        cmds.AbcImport( fileName, ft='world_CON', rpr=root, debug = True )
        try:
            cmds.rename('world_CON', os.path.basename(os.path.dirname(fileName)))
        except:
            pass
    else:
        xfClass = AbcXformImportKey( File=fileName,
                                     ExcludeFilterObjects='world_CON',
                                     Reparent=root )

    children = cmds.listRelatives( root, ad=True )
    curNode = cmds.parent( curNode, children[0] )[0]
    initTransform( curNode )
    if cmds.attributeQuery( 'initScale', n=children[0], ex=True ):
        for i in ['sx', 'sy', 'sz']:
            cmds.connectAttr( '%s.initScale' % children[0], '%s.%s' % (curNode, i) )

    cmds.parent( children[-1], w=True )
    cmds.delete( root )

    return curNode


class AbcXformImportKey:
    def __init__( self, File='', FilterObjects=None, ExcludeFilterObjects=None, Reparent=None ):
        self.m_fileName = File
        self.m_ft  = FilterObjects
        self.m_eft = ExcludeFilterObjects
        self.m_rpr = Reparent

        timeUnit   = cmds.currentUnit( t=True, q=True )
        self.m_fps = _timeUnitMap[timeUnit]

        # for debug data
        self.data = dict()

        self.readAbc()

    def readAbc( self ):
        iarch = IArchive( str(self.m_fileName) )
        root  = iarch.getTop()
        self.visitObject( root )

    def visitObject( self, iobj ):
        for obj in iobj.children:
            ohead = obj.getHeader()
            if IXform.matches( ohead ):
                name = obj.getName()
                if self.m_ft:
                    if name.find(self.m_ft) > -1:
                        self.visitXform( obj )
                elif self.m_eft:
                    if name.find(self.m_eft) == -1:
                        self.visitXform( obj )
                else:
                    self.visitXform( obj )
            self.visitObject( obj )


    def visitXform( self, iobj ):
        name  = iobj.getName()
        node  = name
        if self.m_rpr:
            pname = self.m_rpr + str(iobj.getParent()).replace('/' ,'|')
        else:
            pname = str(iobj.getParent()).replace('/', '|')

        # create node
        if not cmds.objExists( name ):
            if len(pname) > 1:
                node = cmds.group( n=name, em=True, p=pname )
            else:
                node = cmds.group( n=name, em=True )

        ixform = IXform( iobj, kWrapExisting )

        schema   = ixform.getSchema()
        timesamp = schema.getTimeSampling()
        numsamps = schema.getNumSamples()
        self.data[name] = {'iSchema': schema, 'iTime': timesamp, 'numSamps': numsamps}

        # arbGeomParams
        arbMap = dict()
        arbs   = schema.getArbGeomParams()
        if arbs:
            for i in range(arbs.getNumProperties()):
                prop = arbs.getProperty( i )
                atn  = prop.getName()
                gval = prop.getValue()
                if type(gval).__name__ == 'FloatArray':
                    # add attributes
                    if not cmds.attributeQuery( atn, n=node, ex=True ):
                        cmds.addAttr( node, ln=atn, k=True )
                    if prop.isConstant():
                        cmds.setAttr( '%s.%s' % (node, atn), prop.getValue()[0] )
                    else:
                        arbMap[atn] = prop.getValue()

        frames  = list()
        matrixs = list()
        start_frame = int(timesamp.getSampleTime(0) * self.m_fps)
        end_frame   = int(timesamp.getSampleTime(numsamps-1) * self.m_fps)
        # print '# start : %s, end : %s' % (start_frame, end_frame)

        if schema.isConstant():
            mtx = GetM44dToList(GetXformSchemaMatrix(start_frame, schema, timesamp, numsamps))
            if len(pname) > 1:
                cmds.xform(node, m=mtx, os=True)
            else:
                cmds.xform(node, m=mtx, ws=True)
        else:
            # create animcurves
            mObjects, mfnAnimCurves = self.createAnimCurvesOfNode(node)
            for f in xrange(start_frame, end_frame+1):
                ftime = f / self.m_fps
                m = GetM44dToList(GetXformSchemaMatrix(ftime, schema, timesamp, numsamps))
                self.addMatrixKey(f, m, OpenMaya.MSpace.kWorld, mfnAnimCurves)

            curveNames = list()
            for i in self.TL_list:
                index = self.TL_list.index(i)
                mfn = OpenMaya.MFnDependencyNode(mObjects[index])
                name = mfn.name()
                curveNames.append(name)
                cmds.connectAttr('%s.output' % name, '%s.%s' % (node, i), f=True)
            for i in self.TA_list:
                index = self.TA_list.index(i)
                mfn = OpenMaya.MFnDependencyNode(mObjects[index+6])
                name = mfn.name()
                curveNames.append(name)
                cmds.connectAttr('%s.output' % name, '%s.%s' % (node, i), f=True)
            cmds.filterCurve(curveNames)



    def createAnimCurvesOfNode(self, node):
        '''
        Create AnimCurve Node
        return:
        - mObjects
        - mfnAnimCurves
        '''
        self.TL_list = ['translateX', 'translateY', 'translateZ', 'scaleX', 'scaleY', 'scaleZ']
        self.TA_list = ['rotateX', 'rotateY', 'rotateZ']

        dgmod = OpenMaya.MDGModifier()

        mObjects = list()
        for i in self.TL_list:
            # delete current node
            node_name = '%s_%s' % (node, i)
            if cmds.objExists(node_name):
                cmds.delete(node_name)
            obj = dgmod.createNode('animCurveTL')
            dgmod.renameNode(obj, node_name)
            mObjects.append(obj)
        for i in self.TA_list:
            # delete current node
            node_name = '%s_%s' % (node, i)
            if cmds.objExists(node_name):
                cmds.delete(node_name)
            obj = dgmod.createNode('animCurveTA')
            dgmod.renameNode(obj, node_name)
            mObjects.append(obj)
        dgmod.doIt()

        mfnAnimCurves = list()
        for o in mObjects:
            obj = OpenMayaAnim.MFnAnimCurve()
            obj.setObject(o)
            mfnAnimCurves.append(obj)
        return mObjects, mfnAnimCurves

    def addMatrixKey(self, frame, matrix, space, mfnAnimCurves):
        '''
        Set Key by 4x4 Matrix
        '''
        mtx = OpenMaya.MMatrix(matrix)
        tmtx = OpenMaya.MTransformationMatrix(mtx)
        mtime = OpenMaya.MTime(frame, OpenMaya.MTime.uiUnit())

        tr = tmtx.translation(space)
        for x in range(3):
            mfnAnimCurves[x].addKey(mtime, tr[x])

        sc = tmtx.scale(space)
        for x in range(3):
            mfnAnimCurves[x+3].addKey(mtime, sc[x])

        ro = tmtx.rotation()
        ro.reorderIt(0) # rotateOrder
        for x in range(3):
            mfnAnimCurves[x+6].addKey(mtime, ro[x])



def GetWeightAndIndex(iFrame, iTime, numSamps):
    '''
    param:
    - iFrame : frame time (frame / fps)
    - iTime  : AbcCoreAbstract.TimeSampling
    - numSamps : time sample count
    return:
    - alpha
    - oIndex -> oTime
    - oCeilIndex -> oCeilTime
    '''
    if numSamps == 0:
        numSamps = 1

    floorIndex = iTime.getFloorIndex(iFrame, numSamps)
    floorTime  = iTime.getSampleTime(floorIndex)

    oIndex = floorIndex
    oTime  = floorTime
    oCeilIndex = oIndex

    if math.fabs(iFrame - floorTime) < 0.0001:
        return 0.0, oTime, oTime

    ceilIndex = iTime.getCeilIndex(iFrame, numSamps)

    if oIndex == ceilIndex:
        return 0.0, oTime, oTime

    ceilTime  = iTime.getSampleTime(ceilIndex)

    oCeilIndex = ceilIndex
    oCeilTime  = ceilTime

    alpha = (iFrame - floorTime) / (ceilTime - floorTime)

    if math.fabs(1.0 - alpha) < 0.0001:
        oIndex = oCeilIndex
        return 0.0, oTime, oCeilTime

    return alpha, oTime, oCeilTime


def GetXformSchemaMatrix(iFrame, iSchema, iTime, numSamps):
    '''
    Get Matrix
    param:
    - iFrame : frame time (frame / fps)
    - iSchema : IXform Schema
    - iTime : TimeSampling
    - numSamps
    return:
    - M44d Matrix
    '''
    alpha, itime, iceiltime = GetWeightAndIndex(iFrame, iTime, numSamps)
    if alpha != 0.0 and itime != iceiltime:
        mlo = iSchema.getValue(ISampleSelector(itime)).getMatrix()
        mhi = iSchema.getValue(ISampleSelector(iceiltime)).getMatrix()
        m = ((1 - alpha) * mlo) + (alpha * mhi)
        return m
    else:
        m = iSchema.getValue(ISampleSelector(itime)).getMatrix()
        return m


def GetM44dToList(value):
    '''
    M44d Matrix to Python List
    '''
    mtx = list()
    for x in range(4):
        for y in range(4):
            mtx.append(value[x][y])
    return mtx

class AbcXformImportKey_Old:
    def __init__( self, File='', FilterObjects=None, ExcludeFilterObjects=None, Reparent=None ):
        self.m_fileName = File
        self.m_ft  = FilterObjects
        self.m_eft = ExcludeFilterObjects
        self.m_rpr = Reparent

        timeUnit   = cmds.currentUnit( t=True, q=True )
        self.m_fps = _timeUnitMap[timeUnit]

        self.readAbc()

    def readAbc( self ):
        iarch = IArchive( str(self.m_fileName) )
        root  = iarch.getTop()
        self.visitObject( root )

    def visitObject( self, iobj ):
        for obj in iobj.children:
            ohead = obj.getHeader()
            if IXform.matches( ohead ):
                name = obj.getName()
                if self.m_ft:
                    if name.find(self.m_ft) > -1:
                        self.visitXform( obj )
                elif self.m_eft:
                    if name.find(self.m_eft) == -1:
                        self.visitXform( obj )
                else:
                    self.visitXform( obj )
            self.visitObject( obj )

    def visitXform( self, iobj ):
        name  = iobj.getName()
        node  = name
        if self.m_rpr:
            pname = self.m_rpr + str(iobj.getParent()).replace('/' ,'|')
        else:
            pname = str(iobj.getParent()).replace('/', '|')

        # create node
        if not cmds.objExists( name ):
            if len(pname) > 1:
                node = cmds.group( n=name, em=True, p=pname )
            else:
                node = cmds.group( n=name, em=True )

        ixform = IXform( iobj, kWrapExisting )

        schema   = ixform.getSchema()
        timesamp = schema.getTimeSampling()
        numsamps = schema.getNumSamples()

        # arbGeomParams
        arbMap = dict()
        arbs   = schema.getArbGeomParams()
        if arbs:
            for i in range(arbs.getNumProperties()):
                prop = arbs.getProperty( i )
                atn  = prop.getName()
                gval = prop.getValue()
                if type(gval).__name__ == 'FloatArray':
                    # add attributes
                    if not cmds.attributeQuery( atn, n=node, ex=True ):
                        cmds.addAttr( node, ln=atn, k=True )
                    if prop.isConstant():
                        cmds.setAttr( '%s.%s' % (node, atn), prop.getValue()[0] )
                    else:
                        arbMap[atn] = prop.getValue()

        frames  = list()
        matrixs = list()
        if schema.isConstant():
            frames.append(1)
            matrixs.append( self.M44dToList(schema.getValue().getMatrix()) )
        else:
            for i in xrange(numsamps):
                ctime  = timesamp.getSampleTime( i )
                cframe = ctime * self.m_fps
                frames.append( cframe )
                sampleSelector = ISampleSelector( ctime )
                matrixs.append( self.M44dToList(schema.getValue(sampleSelector).getMatrix()) )

        if iobj.getParent().getFullName() == '/':
            Space = OpenMaya.MSpace.kWorld
        else:
            Space = OpenMaya.MSpace.kObject
        setMtx( node, matrixs, frames, space=Space )
        #setMtx( node, matrixs, frames, space=OpenMaya.MSpace.kObject )


    def M44dToList( self, value ):
        mtx = list()
        for x in range(4):
            for y in range(4):
                mtx.append( value[x][y] )
        return mtx


class AbcXformApplyKey:
    def __init__(self, filepath, nodename):
        self.m_fileName = filepath
        self.m_node = nodename

        timeUnit = cmds.currentUnit(t=True, q=True)
        self.m_fps = _timeUnitMap[timeUnit]

        self.readAbc()

    def readAbc(self):
        iarch = IArchive(str(self.m_fileName))
        root = iarch.getTop()
        self.visitObject(root)

    def visitObject(self, iobj):
        for obj in iobj.children:
            ohead = obj.getHeader()
            if IXform.matches(ohead):
                name = obj.getName()
                self.visitXform(obj)
            self.visitObject(obj)

    def visitXform(self, iobj):
        name = iobj.getName()
        #node = name
        node = self.m_node

        pname = str(iobj.getParent()).replace('/', '|')
        ixform = IXform(iobj, kWrapExisting)

        schema = ixform.getSchema()
        timesamp = schema.getTimeSampling()
        numsamps = schema.getNumSamples()

        # arbGeomParams
        arbMap = dict()
        arbs = schema.getArbGeomParams()
        if arbs:
            for i in range(arbs.getNumProperties()):
                prop = arbs.getProperty(i)
                atn = prop.getName()
                gval = prop.getValue()
                if type(gval).__name__ == 'FloatArray':
                    # add attributes
                    if not cmds.attributeQuery(atn, n=node, ex=True):
                        cmds.addAttr(node, ln=atn, k=True)
                    if prop.isConstant():
                        cmds.setAttr('%s.%s' % (node, atn), prop.getValue()[0])
                    else:
                        arbMap[atn] = prop.getValue()

        frames = list()
        matrixs = list()
        if schema.isConstant():
            frames.append(1)
            matrixs.append(self.M44dToList(schema.getValue().getMatrix()))
        else:
            for i in xrange(numsamps):
                ctime = timesamp.getSampleTime(i)
                cframe = ctime * self.m_fps
                frames.append(cframe)
                sampleSelector = ISampleSelector(ctime)
                matrixs.append(self.M44dToList(schema.getValue(sampleSelector).getMatrix()))
        setMtx(node, matrixs, frames, space=OpenMaya.MSpace.kWorld)

    def M44dToList(self, value):
        mtx = list()
        for x in range(4):
            for y in range(4):
                mtx.append( value[x][y] )
        return mtx


#def import_worldAlembic_byKey( fileName ):
#    xfClass = AbcXformImportKey( File=fileName, ExcludeFilterObjects='world_CON' )

#-------------------------------------------------------------------------------
#
#	Matrix
#
#-------------------------------------------------------------------------------
# result = ( matrix list, frame list )
def getMtx( node, start, end, step, mtxSpace='worldMatrix' ):
    selection = OpenMaya.MSelectionList()
    selection.add( node )

    mobj = selection.getDependNode( 0 )
    mfn  = OpenMaya.MFnDependencyNode( mobj )

    mtxAttr = mfn.attribute( mtxSpace )
    mtxPlug = OpenMaya.MPlug( mobj, mtxAttr )
    if mtxSpace == 'worldMatrix':
        mtxPlug = mtxPlug.elementByLogicalIndex( 0 )

    # time wrap
    timewrap = cmds.listConnections( 'time1', d=False, s=True )

    mtxlist = list()
    frmlist = list()
    for f in get_frames( start, end, step ):
        cf = f
        if timewrap:
            cf = cmds.getAttr( '%s.output' % timewrap[0], time=f )

        frameCtx = OpenMaya.MDGContext( OpenMaya.MTime(cf, OpenMaya.MTime.uiUnit()) )
        mtxObj   = mtxPlug.asMObject( frameCtx )
        mtxData  = OpenMaya.MFnMatrixData( mtxObj )
        mtxValue = mtxData.matrix()

        mtxlist.append( list(mtxValue) )
        frmlist.append( cf )

    return mtxlist, frmlist


#   set matrix data
def setMtx( node, matrixs, frames, space=OpenMaya.MSpace.kWorld ):
    dsize = len( frames )
    if dsize == 1:
        if space == 2:  # kObject
            cmds.xform( node, m=matrixs[0], os=True )
        else:
            cmds.xform( node, m=matrixs[0], ws=True )
    else:
        # key data check
        mtxvals = list()
        for i in xrange(dsize):
            mval = matrixs[i]
            if not mval in mtxvals:
                mtxvals.append( mval )
        if len(mtxvals) == 1:
            if space == 2:  # kObject
                cmds.xform( node, m=mtxvals[0], os=True )
            else:
                cmds.xform( node, m=mtxvals[0], ws=True )

        dgmod   = OpenMaya.MDGModifier()
        TL_list = ['translateX', 'translateY', 'translateZ', 'scaleX', 'scaleY', 'scaleZ']
        TA_list = ['rotateX', 'rotateY', 'rotateZ']

        objList = list()
        for i in TL_list:
            # delete current node
            node_name = '%s_%s' % (node, i)
            if cmds.objExists( node_name ):
                cmds.delete( node_name )
            obj = dgmod.createNode( 'animCurveTL' )
            dgmod.renameNode( obj, node_name )
            objList.append( obj )
        for i in TA_list:
            # delete current node
            node_name = '%s_%s' % (node, i)
            if cmds.objExists( node_name ):
                cmds.delete( node_name )
            obj = dgmod.createNode( 'animCurveTA' )
            dgmod.renameNode( obj, node_name )
            objList.append( obj )
        dgmod.doIt()

        keyObjList = list()
        for o in objList:
            obj = OpenMayaAnim.MFnAnimCurve()
            obj.setObject( o )
            keyObjList.append( obj )

        rotateOrder = cmds.getAttr( '%s.rotateOrder' % node )

        for i in xrange(dsize):
            mtx  = OpenMaya.MMatrix( matrixs[i] )
            tmtx = OpenMaya.MTransformationMatrix( mtx )
            mtime= OpenMaya.MTime( frames[i], OpenMaya.MTime.uiUnit() )

            tr = tmtx.translation( space )
            for x in range(3):
                keyObjList[x].addKey( mtime, tr[x] )

            sc = tmtx.scale( space )
            for x in range(3):
                keyObjList[x+3].addKey( mtime, sc[x] )

            ro = tmtx.rotation()
            ro.reorderIt( rotateOrder )
            for x in range(3):
                keyObjList[x+6].addKey( mtime, ro[x] )

        curveNames = list()
        for i in TL_list:
            index = TL_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (node, i), f=True )
        for i in TA_list:
            index = TA_list.index(i)
            mfn   = OpenMaya.MFnDependencyNode( objList[index+6] )
            name  = mfn.name()
            curveNames.append( name )
            cmds.connectAttr( '%s.output' % name, '%s.%s' % (node, i), f=True )
        # cmds.filterCurve( curveNames )
