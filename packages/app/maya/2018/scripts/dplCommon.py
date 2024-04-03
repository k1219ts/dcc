#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	Dexter Pipe-line Common procedural
#
#	2017.01.04	$1
#-------------------------------------------------------------------------------
#
#	JSON
#	- headerData, writeJsonLog
#
#	Look-Dev Attribute
#	- lkdv_setAttrs, lkdv_importAssetAttrs, lkdv_getAttrFile
#
#	Batch Common
#	- revisionCurrent, get_show_shot, get_show_shot_task
#
#-------------------------------------------------------------------------------

import os, sys
import re
import glob
import time
import json
import bson
import getpass

import maya.cmds as cmds
import maya.mel as mel

import sgComponent as sgc


#----------------------------------------------------------------------------
#
#	for Nautilus
#
#----------------------------------------------------------------------------
def openNautilus():
    proj = cmds.workspace(q=True, rd=True)
    command = 'nautilus %s' % proj
    os.system( command )



#-------------------------------------------------------------------------------
#
#	BSON
#
#-------------------------------------------------------------------------------
def readBSON( fileName ):
    f = open( fileName, 'r' )
    data = f.read()
    f.close()

    bstr = bson.BSON( data )
    return bstr.decode()



#-------------------------------------------------------------------------------
#
#	JSON
#
#-------------------------------------------------------------------------------
def headerData( Name=None, Context=None, User=None, Version=None ):
    _header = dict()
    _header['created'] = time.asctime()
    username = User
    if not username:
        username = getpass.getuser()
    _header['author'] = username
    if Name:
        _header['name'] = Name
    if Context:
        _header['context'] = Context
    if Version:
        _header['version'] = Version
    return _header

def writeJsonLog( File=None, Data=dict(), Context=None, User=None, Version=1.0 ):
    body = dict()
    body['_Header'] = headerData( File, Context, User, Version )
    body.update( Data )

    f = open( File, 'w' )
    json.dump( body, f, indent=4 )
    f.close()
    # debug
    mel.eval( 'print "# Result : \\"%s\\" log write <%s>\\n"' % (Data.keys()[0], File) )

#-------------------------------------------------------------------------------
#
#	look-dev attributes
#
#-------------------------------------------------------------------------------
_rmanAttr = {
    'rman__riattr___MatteObject': 'long',
    'rman__riattr___MotionFactor': 'float',
    'rman__riattr__sides_backfacetolerance': 'float',
    'rman__torattr___bakeContext': 'string',
    'rman__riattr__dice_binary': 'long',
    'rman__torattr___cacheShapeScript': 'string',
    'rman__riattr__visibility_camera': 'long',
    'rman__riattr__cull_backfacing': 'long',
    'rman__riattr__cull_hidden': 'long',
    'rman__torattr___curveBaseWidth': 'float',
    'rman__torattr___curveTipWidth': 'float',
    'rman__torattr___customShadingGroup': 'string',
    'rman__riattr__dice_referencecamera': 'string',
    'rman__riattr__dice_hair': 'long',
    'rman__riattr__dice_instancestrategy': 'string',
    'rman__riattr__dice_instanceworlddistancelength': 'float',
    'rman__riattr__dice_offscreenstrategy': 'string',
    'rman__riattr__dice_rasterorient': 'long',
    'rman__riattr__dice_strategy': 'string',
    'rman__riattr__dice_worlddistancelength': 'float',
    'rman__riattr__displacementbound_sphere': 'float',
    'rman__riattr__displacementbound_coordinatesystem': 'string',
    'rman__torattr___outputDisplacementShaders': 'long',
    'rman__torattr___outputImagerShaders': 'long',
    'rman__torattr___outputLightShaders': 'long',
    'rman__torattr___outputSurfaceShaders': 'long',
    'rman__torattr___outputVolumeShaders': 'long',
    'rman__torattr___evaluationFrequency': 'long',
    'rman__riattr___FocusFactor': 'float',
    'rman__riattr__grouping_membership': 'string',
    'rman__riattr__identifier_name': 'string',
    'rman__riattr__identifier_objectid': 'long',
    'rman__param___ignoreObjects': 'string',
    'rman__param___illuminateObjects': 'string',
    'rman__riattr__shade_indexofrefraction': 'float',
    'rman__riattr__visibility_indirect': 'long',
    'rman__riattr__trace_intersectpriority': 'long',
    'rman__torattr___invis': 'long',
    'rman__riattr__identifier_lpegroup': 'string',
    'rman__torattr___linearizeColors': 'long',
    'rman__riattr__trace_maxdiffusedepth': 'long',
    'rman__riattr__trace_maxspeculardepth': 'long',
    'rman__torattr___motionBlur': 'long',
    'rman__torattr___motionSamples': 'long',
    'rman__torattr___outputColorSets': 'long',
    'rman__torattr___outputTangents': 'long',
    'rman__torattr___passFilter': 'string',
    'rman__torattr___postShapeScript': 'string',
    'rman__torattr___preShapeScript': 'string',
    'rman__torattr___ptexFaceOffset': 'long',
    'rman__riattr__trace_samplemotion': 'long',
    'rman__torattr___readMethod': 'long',
    'rman__torattr___relativeMotionBlur': 'float',
    'rman__param___draResizeBBox': 'long',
    'rman__param___draFile': 'string',
    'rman__riattr__dice_roundcurve': 'long',
    'rman__param___draSequenceNumber': 'long',
    'rman__riattr__shade_frequency': 'string',
    'rman__riattr___ShadingRate': 'float',
    'rman__torattr___subdivFacevaryingInterp': 'long',
    'rman__torattr___subdivInterp': 'long',
    'rman__torattr___subdivScheme': 'long',
    'rman__riattr__trace_autobias': 'float',
    'rman__riattr__trace_bias': 'float',
    'rman__riattr__trace_displacements': 'long',
    'rman__riattr__visibility_transmission': 'long',
    'rman__param___draUseSequenceNumber': 'long',
    'rman__riattr__user_txFilePath': 'string',
    'rman__riattr__user_txAssetName': 'string',
    'rman__riattr__user_txLayerName': 'string',
    'rman__riattr__user_txVersion': 'long',
    'rman__riattr__user_txVarNum': 'long',
    'rman__riattr__user_txDoubleSide': 'long',
    'rman__riattr__user_txmultiUV': 'long',
    'rman__riattr__user_txUseSeq': 'long',
    'rman__riattr__user_txImgNum': 'long',
    'rman__riattr__user_object_id': 'long',
    'rman__riattr__user_group_id': 'long',
    'rman__riattr__dice_watertight': 'long',
    'rman__riattr__dice_micropolygonlength':'long'
}

_excludeAttrs = [ 'rman__riattr__user_txVarNum' ]

def lkdv_setAttrs( node, data ):	# set object, attributes data=dict()
    # delete attributes
    delAttrs = cmds.listAttr( node, st='rman*' )
    if delAttrs:
        delAttrs = list( set(delAttrs) - set(_excludeAttrs) )
        for at in delAttrs:
            if not data.has_key(at):
                cmds.deleteAttr( '%s.%s' % (node, at) )
                
    # add & set
    for at in data:
        if at in _excludeAttrs:
            continue

        setvalue = data[at]
        if _rmanAttr.has_key( at ):
            at_type = _rmanAttr[at]
        else:
            # at_type = type(setvalue).__name__
            print setvalue
            if re.compile(r"\d.\d").match(str(setvalue)):
                at_type = 'float'
            elif re.compile(r"\d").match(str(setvalue)):
                at_type = 'long'
            else:
                at_type = 'string'
                
        # add attr
        if not cmds.attributeQuery( at, n=node, ex=True ):
            print "addAttr :", at, at_type
            if at_type == 'string' or at_type == 'unicode':
                cmds.addAttr( node, ln=at, dt='string' )
            else:
                cmds.addAttr( node, ln=at, at=at_type )
        # set attr
        if cmds.getAttr('%s.%s' % (node, at), se = True) == False:
            continue
            
        if at_type == 'string' or at_type == 'unicode':
            setvalue = setvalue.replace( '"', '\\"' )
            cmds.setAttr( '%s.%s' % (node, at), setvalue, type='string' )
        else:
            if at_type == 'float':
                setvalue = float( setvalue )
            else:
                setvalue = int( setvalue )
            cmds.setAttr( '%s.%s' % (node, at), setvalue )

def lkdv_importAssetAttrs( rootNode, dataDict ):
    currentShape = cmds.ls( rootNode, dag=True, type='surfaceShape', ni=True )
    for shape in dataDict:
        at_data = dataDict[shape]
        for o in currentShape:
            if o.split(':')[-1] == shape:
                lkdv_setAttrs( o, at_data )


# look dev attribute file
def lkdv_getAttrFile( show, asset ):
    shaderRoot = '/show/%s/asset/shaders' % show
    assetName  = asset
    shaderPath = '%s/%s' % (shaderRoot, assetName)
    if not os.path.exists( shaderPath ):
        assetName = asset.split('_')[0]
        shaderPath = '%s/%s' % (shaderRoot, assetName)
    if not os.path.exists( shaderPath ):
        return

    versions = os.listdir( shaderPath )
    ld_ver = list()
    tx_ver = list()
    for i in versions:
        if i.find('tx') > -1:
            tx_ver.append( i )
        else:
            ld_ver.append( i )
    ld_ver.sort()
    tx_ver.sort()
    
    ld_ver.reverse()
    tx_ver.reverse()
    
    lastestVer = list()
    
    if len(tx_ver) > 0:
        lastestVer.append(tx_ver[0])
    
    if len(ld_ver) > 0:
        lastestVer.append(ld_ver[0])

    atfiles = list()
    for i in lastestVer:
        atfile = '%s/%s/at/%s_%s.json' % ( shaderPath, i, assetName, i )
        if os.path.exists( atfile ):
            atfiles.append(atfile)
    return atfiles

# reload attr
def reloadAttr():
    current = cmds.workspace(q=True, directory=True)
    print current
    src = current.split('/')
    showName = src[src.index('show') + 1]

    for node in cmds.ls(sl=True):
        nodeName = cmds.ls(node, dag=True, type='dxComponent', ni=True)[0]

        print nodeName

        cmds.setAttr('%s.mode' % nodeName, 0)
        sgc.componentReload("%s." % nodeName)

        assetName = ''
        if cmds.attributeQuery('assetName', n=nodeName, ex=True):
            assetName = cmds.getAttr('%s.assetName' % nodeName)
        if not assetName:
            assetName = nodeName.split(':')[-1].replace('_rig_GRP', '')
        print showName, assetName

        atfiles = lkdv_getAttrFile(showName, assetName)
        print '### atfile: ', atfiles
        print

        if atfiles:
            for atfile in atfiles:
                mel.eval('print "# Debug : import attributes : %s -> %s\\n"' % (assetName, atfile))
                body = json.loads(open(atfile, 'r').read())
                if body.has_key('Attributes'):
                    lkdv_importAssetAttrs(nodeName, body['Attributes'])

        cmds.setAttr('%s.mode' % nodeName, 1)
        sgc.componentReload("%s." % nodeName)


#-------------------------------------------------------------------------------
#
#	Cache Log
#
#-------------------------------------------------------------------------------
class CacheLogParser:
    def __init__( self, fileName ):
        self.m_body = json.loads( open(fileName, 'r').read() )

        cache_name = self.m_body.keys()
        cache_name.sort()
        self.m_cacheName = cache_name[0]

        self.m_data = self.m_body[self.m_cacheName]

    def getAssets( self ):
        assets = list()
        if self.m_data.has_key( 'zenn' ) and self.m_data['zenn']:
            for i in self.m_data['zenn']:
                basename = os.path.basename( i )
                name     = basename.split(':')[-1]
                assets.append( name )
        if self.m_data.has_key( 'mesh' ) and self.m_data['mesh']:
            for i in self.m_data['mesh']:
                basename = os.path.splitext( os.path.basename(i) )[0]
                name     = basename.split(':')[-1].split('_rig_GRP')[0]
                assets.append( name )
        return list(set(assets))

