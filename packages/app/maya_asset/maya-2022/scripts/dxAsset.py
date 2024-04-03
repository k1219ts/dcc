#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   RenderMan TD
#
#       Sanghun Kim, rman.td@gmail.com
#
#       2015.06.20 $2
#
#-------------------------------------------------------------------------------

import os, sys
import math
import fnmatch
import re
import string

import maya.OpenMaya as OpenMaya
import maya.cmds as cmds
import maya.mel  as mel


#-------------------------------------------------------------------------------

def checkRenderStats( shape ):
    onlist = [
                    'castsShadows', 'receiveShadows', 'motionBlur', 'primaryVisibility', 'smoothShading',
                    'visibleInReflections', 'visibleInRefractions', 'doubleSided'
                    ]
    # set -> on
    for at in onlist:
        setAttr( '%s.%s' % (shape, at), 1 )
    # set -> off
    setAttr( '%s.opposite' % shape, 0 )


#-------------------------------------------------------------------------------
#
# Attributes control
#
#-------------------------------------------------------------------------------
class AssetAttribute():
    def __init__( self ):
        self.m_logt = {}
        self.m_selectionMode = 1        # scene all
        self.m_currentSelection = cmds.ls(sl=True)
        self.getObjects()

        self.m_assetname = ''
        self.m_layerMap = {}
        self.m_multitileUV = []
        self.m_multiUV  = []
        self.materialSet_noneList = []

    def getObjects( self ):
        selected = cmds.ls( sl=True, dag=True, type='surfaceShape', ni=True )
        if selected:
            self.m_objects = selected
            self.m_selectionMode = 0
        else:
            self.m_objects = cmds.ls( dag=True, type='surfaceShape', ni=True )

    # check display layer
    def checkDisplayLayer( self ):
        # clear
        self.m_layerMap.clear()
        del self.m_multitileUV[:]

        layers = cmds.ls( type='displayLayer' )
        layers.remove( 'defaultLayer' )
        for lyr in layers:
            layername = lyr.split('_LYR')[0]
            members = cmds.editDisplayLayerMembers( lyr, q=True )
            if members:
                uvmeshes = []
                shapeMembers = cmds.ls( members, dag=True, type='surfaceShape', ni=True )
                if self.m_selectionMode == 1:
                    for s in shapeMembers:
                        # layer map
                        self.m_layerMap[s] = layername
                        # uv selector
                        uvmeshes.append( '%s.map[:]' % s )
                else:
                    for s in self.m_objects:
                        if s in shapeMembers:
                            # layer map
                            self.m_layerMap[s] = layername
                            # uv selector
                            uvmeshes.append( '%s.map[:]' % s )
                if uvmeshes:
                    getuvs = cmds.polyEditUV( uvmeshes, q=True )
                    if getuvs:
                        u = []; v =[];
                        for i in range(0, len(getuvs), 2):
                            u.append( getuvs[i] )
                            v.append( getuvs[i+1] )
                        u.sort(); v.sort();
                        #print u[0], u[-1], v[0], v[-1]
# multi uv checking
                        if int( math.floor(u[0]) ) != int( math.floor(u[-1]) ):
                            vs = u[-1] - int(u[-1])
                            if vs > 0.1:
                                self.m_multitileUV.append( layername )

    # texture attributes
    def texture_proc( self, shape ):
        if self.m_layerMap.has_key( shape ):
            # asset name
            orgAtn = "rman__riattr__user_txAssetName"
            atn = 'txBasePath'
            if cmds.attributeQuery( orgAtn, n=shape, ex = True):
                cmds.deleteAttr(shape, at = orgAtn)
            if not cmds.attributeQuery( atn, n=shape, ex=True ):
                cmds.addAttr( shape, ln=atn, dt='string' )
            cmds.setAttr( '%s.%s' % (shape, atn), self.m_assetname, type='string' )

            # layer name
            layername = self.m_layerMap[shape]
            orgAtn = "rman__riattr__user_txLayerName"
            atn = 'txLayerName'
            if cmds.attributeQuery( orgAtn, n=shape, ex = True):
                cmds.deleteAttr(shape, at = orgAtn)
            if not cmds.attributeQuery( atn, n=shape, ex=True ):
                cmds.addAttr( shape, ln=atn, dt='string' )
            cmds.setAttr( '%s.%s' % (shape, atn), layername, type='string' )
        else:
            return shape

    def multiuv_proc(self, shape):
        if self.m_layerMap.has_key(shape):
            layername = self.m_layerMap[shape]
            # multitile uv
            orgAtn = 'rman__riattr__user_txmultiUV'
            atn = 'txmultiUV'
            if layername in self.m_multitileUV:
                if cmds.attributeQuery(orgAtn, n=shape, ex=True):
                    cmds.deleteAttr('%s.%s' % (shape, orgAtn))
                if not cmds.attributeQuery(atn, n=shape, ex=True):
                    cmds.addAttr(shape, ln=atn, at='long')
                cmds.setAttr('%s.%s' % (shape, atn), 1)
            else:
                if cmds.attributeQuery(orgAtn, n=shape, ex=True):
                    cmds.deleteAttr('%s.%s' % (shape, orgAtn))
                if cmds.attributeQuery(atn, n=shape, ex=True):
                    cmds.deleteAttr('%s.%s' % (shape, atn))
        else:
            return shape




        #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # object name
    def objectname_proc( self, shape ):
        atn = 'objectName'
        attr = '%s.%s' % (shape, atn)
        if not cmds.attributeQuery( atn, n=shape, ex=True ):
            cmds.addAttr( shape, ln=atn, dt='string' )
        cmds.setAttr( '%s.%s' % (shape, atn), shape, type='string' )
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # unique name
    def uniquename_proc( self, shape ):
        if fnmatch.fnmatch( shape, '*|*' ):
            return shape
        else:
            trname = cmds.listRelatives( shape, p=True )
            if trname:
                src = re.compile( r'(.+?)(\d+)(.+?)?' ).findall( trname[0] )
                if src:
                    if src[-1][-1]:
                        name = trname[0] + 'Shape'
                    else:
                        tmp = list(src[-1])
                        tmp.insert( -2, 'Shape' )
                        name = trname[0].replace( string.join(src[-1][:-1],''), string.join(tmp, '') )
                else:
                    name = trname[0] + 'Shape'
                if shape != name:
                    if not name in self.m_objects:
                        cmds.rename( shape, name )
                        self.m_objects[self.m_objects.index(shape)] = name
                    else:
                        return shape
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # check renderstats
    def renderstats_proc( self, shape ):
        onlist = [
                        'castsShadows', 'receiveShadows', 'motionBlur',
                        'primaryVisibility', 'smoothShading',
                        'visibleInReflections', 'visibleInRefractions', 'doubleSided'
                        ]
        # set -> on
        for at in onlist:
            cmds.setAttr( '%s.%s' % (shape, at), 1 )
        # set -> off
        cmds.setAttr( '%s.opposite' % shape, 0 )

        # subdivision
        attrname = 'rman__torattr___subdivScheme'
        if cmds.attributeQuery( attrname, n=shape, ex=True ):
            gt = cmds.getAttr( shape + '.' + attrname, type=True )
            gv = cmds.getAttr( shape + '.' + attrname )
            if gt != 'long':
                cmds.deleteAttr( shape + '.' + attrname )
                cmds.addAttr( shape, ln=attrname, at='long' )
                cmds.setAttr( shape + '.' + attrname, gv )

    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # MaterialSet
    def materialSet_proc( self, shape ):
        if not cmds.attributeQuery('MaterialSet', n=shape, ex=True):
            return shape

    #--------------------------------------------------------------------------

    def add( self, processes=None ):
        if not processes:
            return
        result = {}
        if 'uniquename' in processes:
            processes.remove( 'uniquename' )
            logObjects = []
            for i in self.m_objects:
                log = self.uniquename_proc( i )
                if log:
                    logObjects.append( log.split('|')[-1] )
            if logObjects:
                result['uniquename'] = list(set(logObjects))

        if 'texture' in processes:
            self.checkDisplayLayer()

        for i in self.m_objects:
            #print 'add attrs >>> %s' % i
            for p in processes:
                log = eval( 'self.%s_proc(i)' % p )
                if log:
                    if result.has_key(p):
                        result[p].append( log )
                    else:
                        result[p] = [log]
        return result


#-------------------------------------------------------------------------------
#
# Instance Object - API Style
#
#-------------------------------------------------------------------------------
def getInstances():
    instances = []
    iterDag = OpenMaya.MItDag( OpenMaya.MItDag.kBreadthFirst )
    while not iterDag.isDone():
        instanced = OpenMaya.MItDag.isInstanced( iterDag )
        if instanced:
            instances.append( iterDag.partialPathName() )
        iterDag.next()
    return instances

def unInstance():
    insts = getInstances()
    for i in insts:
        tr = cmds.listRelatives( i, parent=True, fullPath=True )[0]
        dp = cmds.duplicate( tr, renameChildren=True )
        cmds.delete( tr )



#-------------------------------------------------------------------------------
#
# Geometry CheckUp Main - API Style
#
#-------------------------------------------------------------------------------
def cross( A, B ):
    result = OpenMaya.MVector()
    result.x = A.y*B.z-A.z*B.y
    result.y = A.z*B.x-A.x*B.z
    result.z = A.x*B.y-A.y*B.x
    #print result.x, result.y, result.z
    return result

def dot( A, B ):
    result = A.x*B.x + A.y*B.y + A.z*B.z
    #print result
    return result

#-------------------------------------------------------------------------------
# Class
class CheckMesh():
    def __init__( self ):
        self.objName = None
        # temp data
        self.vsmesh    = []
        self.vsmeshVtx = []
        self.vsmeshPt  = []

        # options
        self.overlapmesh = 0
        self.multiuvset  = 1
        self.outsideuv   = 1
        self.threeside   = 1
        self.nside       = 1
        self.concave     = 1
        self.holed       = 1
        self.zeroface    = 1
        self.zerouv      = 1
        self.twisted     = 1
        self.lamina      = 1
        self.overlapface = 1
        self.zeroedge    = 1
        self.nonmanifold = 1
        self.twoedgevtx  = 1

        # result
        self.overlapmeshSel = OpenMaya.MSelectionList()
        self.multiuvsetSel  = OpenMaya.MSelectionList()
        self.outsideuvSel   = OpenMaya.MSelectionList()
        self.threesideSel   = OpenMaya.MSelectionList()
        self.nsideSel       = OpenMaya.MSelectionList()
        self.concaveSel     = OpenMaya.MSelectionList()
        self.holedSel       = OpenMaya.MSelectionList()
        self.zeroareaSel    = OpenMaya.MSelectionList()
        self.zerouvSel      = OpenMaya.MSelectionList()
        self.twistedSel     = OpenMaya.MSelectionList()
        self.laminaSel      = OpenMaya.MSelectionList()
        self.overlapfaceSel = OpenMaya.MSelectionList()
        self.zeroedgeSel    = OpenMaya.MSelectionList()
        self.nonmanifoldSel = OpenMaya.MSelectionList()
        self.twoedgevtxSel  = OpenMaya.MSelectionList()

        # log
        self.m_result = {
                        'overlapmesh': [],
                        'multiuvset' : [],
                        'outsideuv'  : [],
                        'threeside'  : [],
                        'nside'      : [],
                        'concave'    : [],
                        'holed'      : [],
                        'zeroface'   : [],
                        'zerouv'     : [],
                        'twisted'    : [],
                        'lamina'     : [],
                        'overlapface': [],
                        'zeroedge'   : [],
                        'nonmanifold': [],
                        'twoedgevtx' : [],
                        }

    # twisted face - just for concave
    def twistedFace( self, points ):
        v10 = OpenMaya.MVector( points[1] - points[0] )
        v30 = OpenMaya.MVector( points[3] - points[0] )
        ag0 = math.degrees( math.atan2( cross(v10,v30).length(), dot(v10,v30) ) )
        if ag0 > 90.0: ag0 = 180.0 - ag0 + 180.0

        v21 = OpenMaya.MVector( points[2] - points[1] )
        v01 = OpenMaya.MVector( points[0] - points[1] )
        ag1 = math.degrees( math.atan2( cross(v21,v01).length(), dot(v21,v01) ) )
        if ag1 > 90.0: ag1 = 180.0 - ag1 + 180.0

        v12 = OpenMaya.MVector( points[1] - points[2] )
        v32 = OpenMaya.MVector( points[3] - points[2] )
        ag2 = math.degrees( math.atan2( cross(v12,v32).length(), dot(v12,v32) ) )
        if ag2 > 90.0: ag2 = 180.0 - ag2 + 180.0

        v23 = OpenMaya.MVector( points[2] - points[3] )
        v03 = OpenMaya.MVector( points[0] - points[3] )
        ag3 = math.degrees( math.atan2( cross(v23,v03).length(), dot(v23,v03) ) )
        if ag3 > 90.0: ag3 = 180.0 - ag3 + 180.0

        if (ag0+ag1+ag2+ag3) != 360.0:
            return True

    # overlap mesh
    def overlapMesh( self ):
        if not self.overlapmesh:
            return
        _ifever = 0
        numVtx   = self.meshFn.numVertices()
        numEdges = self.meshFn.numEdges()
        numFace  = self.meshFn.numPolygons()
        #identity = '%s %s %s' % ( numVtx, numEdges, numFace )
        identity = str( [numVtx, numEdges, numFace] )

        vtxCount = OpenMaya.MIntArray()
        vtxList  = OpenMaya.MIntArray()
        self.meshFn.getVertices( vtxCount, vtxList )

        # 1st
        if identity in self.vsmesh:
            _ifever = 1
        else:
            self.vsmesh.append( identity )
            self.vsmeshVtx.append( str(vtxList) )
            _ifever = 0
        # 2st
        if _ifever == 1:
            if str(vtxList) in self.vsmeshVtx:
                _ifever = 1
            else:
                self.vsmeshVtx.append( str(vtxList) )
                _ifever = 0
        # check main
        if _ifever == 1:
            self.overlapmeshSel.merge( self.dagPath )

    # UV : multi-uvset, outside uv
    def uvmap( self ):
        stringArray = []
        self.meshFn.getUVSetNames( stringArray )
        if self.multiuvset:
            if len(stringArray) > 1:
                self.multiuvsetSel.merge( self.dagPath )
        if self.outsideuv:
            for i in stringArray:
                uvnum = self.meshFn.numUVs( i )
                if uvnum > 0:
                    _u = OpenMaya.MFloatArray()
                    _v = OpenMaya.MFloatArray()
                    self.meshFn.getUVs( _u, _v, i )
                    _u = list( _u )
                    _v = list( _v )
                    _u.sort()
                    _v.sort()
                    # add 20150620
                    u_min = _u[0]
                    if u_min == int(u_min):
                        u_min += 0.01
                    u_max = _u[-1]
                    if u_max == int(u_max):
                        u_max -= 0.01
                    v_min = _v[0]
                    if v_min == int(v_min):
                        v_min += 0.01
                    v_max = _v[-1]
                    if v_max == int(v_max):
                        v_max -= 0.01
                    # uv outside checking
                    if int( math.floor(u_min) ) != int( math.floor(u_max) ) or int( math.floor(v_min) ) != int( math.floor(v_max) ):
                        self.outsideuvSel.merge( self.dagPath )

    # face iterator
    def faceIterator( self ):
        # pointer
        util = OpenMaya.MScriptUtil()
        doublePtr = util.asDoublePtr()

        itFace = OpenMaya.MItMeshPolygon( self.dagPath )
        try:
            itFace.reset()
            while not itFace.isDone():

                index = itFace.index()
                vtxs  = OpenMaya.MIntArray()
                itFace.getVertices( vtxs )

                # 3 sides
                if self.threeside and len(vtxs) == 3:
                    self.threesideSel.merge( self.dagPath, itFace.currentItem() )

                # n sides
                if self.nside and len(vtxs) > 4:
                    self.nsideSel.merge( self.dagPath, itFace.currentItem() )

                # concave
                if self.concave and not itFace.isConvex():
                    self.concaveSel.merge( self.dagPath, itFace.currentItem() )

                # holed
                if self.holed and itFace.isHoled():
                    self.holedSel.merge( self.dagPath, itFace.currentItem() )

                # zero area
                if self.zeroface:
#                                       if itFace.zeroArea():
#                                               self.zeroareaSel.merge( self.dagPath, itFace.currentItem() )
                    itFace.getArea( doublePtr )
                    area = OpenMaya.MScriptUtil.getDouble( doublePtr )
                    if area == 0.0:
                        self.zeroareaSel.merge( self.dagPath, itFace.currentItem() )

                # zero uv
                if self.zerouv:
                    if not itFace.hasUVs():
                        self.zerouvSel.merge( self.dagPath, itFace.currentItem() )
#                                       itFace.getUVArea( doublePtr )
#                                       uvarea = OpenMaya.MScriptUtil.getDouble( doublePtr )
#                                       if uvarea == 0.0:
#                                               self.zerouvSel.merge( self.dagPath, itFace.currentItem() )

                # twisted
                if self.twisted and not itFace.isStarlike():
                    points = OpenMaya.MPointArray()
                    itFace.getPoints( points )
                    if self.twistedFace( points ):
                        self.twistedSel.merge( self.dagPath, itFace.currentItem() )

                # lamina & overlapface
                if itFace.isLamina():
                    if self.lamina:
                        strVtx = str( vtxs )
                        if not strVtx in self.laminaTemp:
                            self.laminaTemp.append( strVtx )
                            self.laminaSel.merge( self.dagPath, itFace.currentItem() )
                else:
                    if self.overlapface:
                        center = itFace.center( OpenMaya.MSpace.kWorld )
                        str_center = str( [center.x, center.y, center.z] )
                        if str_center in self.overlapfaceTemp:
                            self.overlapfaceSel.merge( self.dagPath, itFace.currentItem() )
                        else:
                            self.overlapfaceTemp.append( str_center )

                itFace.next()
        except: pass

    # edge iterator
    def edgeIterator( self ):
        # pointer
        util = OpenMaya.MScriptUtil()
        doublePtr = util.asDoublePtr()

        itEdge = OpenMaya.MItMeshEdge( self.dagPath )
        try:
            itEdge.reset()
            while not itEdge.isDone():

                index = itEdge.index()

                # edge length
                if self.zeroedge:
                    itEdge.getLength( doublePtr )
                    elen = OpenMaya.MScriptUtil.getDouble( doublePtr )
                    if elen < 0.00001:
                        self.zeroedgeSel.merge( self.dagPath, itEdge.currentItem() )

                # non-manifold
                if self.nonmanifold:
                    cface = OpenMaya.MIntArray()
                    itEdge.getConnectedFaces( cface )
                    if len(cface) > 2:
                        self.nonmanifoldSel.merge( self.dagPath, itEdge.currentItem() )

                itEdge.next()
        except: pass

    # vertex iterator
    def vertexIterator( self ):
        itVtx = OpenMaya.MItMeshVertex( self.dagPath )
        try:
            itVtx.reset()
            while not itVtx.isDone():

                index = itVtx.index()
                if self.twoedgevtx:
                    if not itVtx.onBoundary():
                        cedges = OpenMaya.MIntArray()
                        itVtx.getConnectedEdges( cedges )
                        if cedges.length() == 2:
                            self.twoedgevtxSel.merge( self.dagPath, itVtx.currentItem() )

                itVtx.next()
        except: pass

    def DoIt( self ):
        util = OpenMaya.MScriptUtil()
        doublePtr = util.asDoublePtr()

        selection = OpenMaya.MSelectionList()
        OpenMaya.MGlobal.getActiveSelectionList( selection )
        iterSel = OpenMaya.MItSelectionList( selection )

        while not iterSel.isDone():

            # temp
            self.laminaTemp = []
            self.overlapfaceTemp = []

            self.dagPath = OpenMaya.MDagPath()
            iterSel.getDagPath( self.dagPath )
            self.objName = self.dagPath.partialPathName()

            # for kMesh
            if self.dagPath.childCount() and self.dagPath.child(0).apiType() == OpenMaya.MFn.kMesh:
                self.meshFn = OpenMaya.MFnMesh( self.dagPath )

                # overlap mesh
                self.overlapMesh()

                # uvmap
                self.uvmap()

                # face iterator
                self.faceIterator()

                # edge iterator
                self.edgeIterator()

                # vertex iterator
                self.vertexIterator()

            iterSel.next()

        # result
        if self.overlapmeshSel.length() > 0:
            self.overlapmeshSel.getSelectionStrings( self.m_result['overlapmesh'] )
        if self.multiuvsetSel.length() > 0:
            self.multiuvsetSel.getSelectionStrings( self.m_result['multiuvset'] )
        if self.outsideuvSel.length() > 0:
            self.outsideuvSel.getSelectionStrings( self.m_result['outsideuv'] )
        if self.threesideSel.length() > 0:
            self.threesideSel.getSelectionStrings( self.m_result['threeside'] )
        if self.nsideSel.length() > 0:
            self.nsideSel.getSelectionStrings( self.m_result['nside'] )
        if self.concaveSel.length() > 0:
            self.concaveSel.getSelectionStrings( self.m_result['concave'] )
        if self.holedSel.length() > 0:
            self.holedSel.getSelectionStrings( self.m_result['holed'] )
        if self.zeroareaSel.length() > 0:
            self.zeroareaSel.getSelectionStrings( self.m_result['zeroface'] )
        if self.zerouvSel.length() > 0:
            self.zerouvSel.getSelectionStrings( self.m_result['zerouv'] )
        if self.twistedSel.length() > 0:
            self.twistedSel.getSelectionStrings( self.m_result['twisted'] )
        if self.laminaSel.length() > 0:
            self.laminaSel.getSelectionStrings( self.m_result['lamina'] )
        if self.overlapfaceSel.length() > 0:
            self.overlapfaceSel.getSelectionStrings( self.m_result['overlapface'] )
        if self.zeroedgeSel.length() > 0:
            self.zeroedgeSel.getSelectionStrings( self.m_result['zeroedge'] )
        if self.nonmanifoldSel.length() > 0:
            self.nonmanifoldSel.getSelectionStrings( self.m_result['nonmanifold'] )
        if self.twoedgevtxSel.length() > 0:
            self.twoedgevtxSel.getSelectionStrings( self.m_result['twoedgevtx'] )

    def printLog( self ):
        print self.m_result




#-------------------------------------------------------------------------------
#
# Geometry Checkup
#
#-------------------------------------------------------------------------------
class AssetGeometry:
    def __init__( self ):
        self.m_log = {}
        self.m_selectionMode = 1        # scene all
        self.m_currentSelection = cmds.ls(sl=True)
        self.m_tobjects = []
        self.getObjects()

    def getObjects( self ):
        selected = cmds.ls(sl=True, dag=True, type='surfaceShape', ni=True)
        if selected:
            self.m_objects = selected
            self.m_selectionMode = 0
        else:
            self.m_objects = cmds.ls(dag=True, type='surfaceShape', ni=True)
        if self.m_objects:
            self.m_tobjects = []
            for i in self.m_objects:
                self.m_tobjects += cmds.listRelatives(i, p=True, f=True)

    #--------------------------------------------------------------------------
    # geometry checkup
    def check( self, options ):
        if not options:
            return
        cmds.select( self.m_tobjects )

        geo = CheckMesh()
        for i in options:
            exec( 'geo.%s = %s' % (i, options[i]) )
        geo.DoIt()
    
        cmds.select( cl=True )
        if self.m_currentSelection:
            cmds.select( self.m_currentSelection )
        return geo.m_result
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    # geometry setup
    #       zero poly vertex
    def zeromovevertex_proc( self, trnode ):
        cmds.polyMoveVertex( trnode, lt=(0,0,0), ch=False )
        cmds.select( cl=True )

    #       freeze transform
    def freezetransform_proc( self, trnode ):
        cmds.makeIdentity( trnode, a=True, t=True, r=True, s=True, n=False )

    #       zero pivot transform
    def zeropivot_proc( self, trnode ):
        cmds.makeIdentity( trnode, a=False, t=True, r=True, s=True, n=False )

    #       delete constructionHistory
    def deletehistory_proc( self, trnode ):
        cmds.delete( trnode, ch=True )

    #       conform normal
    def conformnormal_proc( self, trnode ):
        cmds.polyNormal( trnode, nm=2, ch=0 )

    #       vertex normal unlock
    def unlocknormal_proc( self, trnode ):
        cmds.polyNormalPerVertex( '%s.vtx[:]' % trnode, ufn=True )

    #       initialize shader
    def initialshader_proc( self, trnode ):
        cmds.sets( trnode, e=True, fe='initialShadingGroup' )

    #       un-instance
    def uninstance_proc( self ):
        insts = []
        iterDag = OpenMaya.MItDag( OpenMaya.MItDag.kBreadthFirst )
        while not iterDag.isDone():
            if OpenMaya.MItDag.isInstanced( iterDag ):
                insts.append( iterDag.partialPathName() )
            iterDag.next()
        for i in insts:
            tr = cmds.listRelatives( i, parent=True, fullPath=True )[0]
            dp = cmds.duplicate( tr, renameChildren=True )
            self.m_objects += cmds.ls(dp, dag=True, type='surfaceShape', ni=True)
            cmds.delete( tr )
            if i in self.m_objects:
                self.m_objects.remove(i)

    def normaifoldgeometry_proc(self):
        # maya clean up
        # polyCleanupArgList 4 {"0", "1", "1", "0", "0", "0", "0", "0", "0", "1e-05", "0", "1e-05", "0", "1e-05", "0", "1", "0", "0"};

        cleanUpCommand = "polyCleanupArgList 4 "
        cleanUpCommand += '{"0", '  # apply to objects(all{1} / select{0})
        cleanUpCommand += '"1", '  # (select{2} / cleanup{1}) matching polygons
        cleanUpCommand += '"1", '  # Keep construction history (on {1} / off {0})

        # clean up option
        # for i in self.m_mayaCleanupUi:
        #     getValue = int(eval('self.ui.%s_checkBox.isChecked()' % i))
        #     if getValue:
        #         cleanUpCommand += '"1", '
        #     else:
        cleanUpCommand += '"0", '

        cleanUpCommand += '"0", "1e-05", '  # Faces with zero geometry area
        cleanUpCommand += '"0", "1e-05", '  # edge with zero length
        cleanUpCommand += '"0", "1e-05", '  # Faces with zero map area
        cleanUpCommand += '"0", '  # ???
        cleanUpCommand += '"1", '  # exi geometry ( on { + } / off { - } ) geometry only = 2, normals and geometry = 1
        cleanUpCommand += '"0", '  # Lamina faces
        cleanUpCommand += '"0"};'  # Invalid Components

        print cleanUpCommand
        mel.eval(cleanUpCommand)


    #       iterate setup
    def setupObjects( self, processes=None ):
        if not processes:
            return
        sceneproc = ['uninstance']
        if 'uninstance' in processes:
            self.uninstance_proc()

        if 'normaifoldgeometry' in processes:
            self.normaifoldgeometry_proc()

        sceneproc += ['zeromovevertex']
        if 'zeromovevertex' in processes:
            for i in self.m_objects:
                tr = cmds.listRelatives(i, p=True, f=True)[0]
                self.zeromovevertex_proc( tr )

        sceneproc += ['freezetransform', 'zeropivot']
        if 'freezetransform' in processes or 'zeropivot' in processes:
            trnodes = []
            for i in self.m_objects:
                fullpath = cmds.listRelatives( i, p=True, f=True )[0]
                for x in fullpath.split('|')[1:]:
                    trnodes += cmds.ls(x, l=True)
            # unlock
            trs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz']
            for t in list(set(trnodes)):
                for x in trs:
                    cmds.setAttr( '%s.%s' % (t, x), lock=False )
            for i in list(set(trnodes)):
                if 'freezetransform' in processes:
                    self.freezetransform_proc( i )
                if 'zeropivot' in processes:
                    self.zeropivot_proc( i )

        for i in self.m_objects:
            tr = cmds.listRelatives(i, p=True, f=True)[0]
            #print 'mesh setup >> %s' % tr
            for p in ( set(processes)-set(sceneproc) ):
                eval( 'self.%s_proc(tr)' % p )
