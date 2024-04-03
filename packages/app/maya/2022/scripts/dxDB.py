#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	MongoDB common procedural
#
#	2017.01.04	$2
#-------------------------------------------------------------------------------
#
#	Get DB Data
#	- getCacheData, existsCacheCheckup
#
#	Insert DB Data
#	- mergeDict, getDictKeyValue, toDocument,
#	- insertDB
#
#-------------------------------------------------------------------------------

import os, sys
import copy
import json
import datetime
import dateutil.parser

from pymongo import MongoClient
import pymongo

try:
    _CONNECT = MongoClient( '10.0.0.12:27017, 10.0.0.13:27017',
                            serverSelectionTimeoutMS=5 )
    _CONNECT.server_info()
except pymongo.errors.ServerSelectionTimeoutError as err:
    _CONNECT = None



#-------------------------------------------------------------------------------
#
#	Get DB Data
#
#-------------------------------------------------------------------------------
def getCacheData( showName, shotName ):
    if not _CONNECT:
        return None
    db   = _CONNECT.SHOT[ showName ]
    docs = db.find( {'show':showName, 'shot':shotName, 'cache': {'$exists':True}} )
    if docs.count() == 1:
        return existsCacheCheckup( docs[0]['cache'] )


def existsCacheCheckup( data ):
    temp = copy.deepcopy( data )
    # mesh
    if data.has_key( 'mesh' ):
        for em in data['mesh']:
            for ver in data['mesh'][em]:
                fn = data['mesh'][em][ver]['file']
                if not os.path.exists( fn ):
                    temp['mesh'][em].pop( ver )
    # zenn
    if data.has_key( 'zenn' ):
        for em in data['zenn']:
            for ver in data['zenn'][em]:
                fn = data['zenn'][em][ver]['file']
                if not os.path.exists( fn ):
                    temp['zenn'][em].pop( ver )

    # version clean up
    cache = copy.deepcopy( temp )
    if temp.has_key( 'mesh' ):
        for em in temp['mesh']:
            if not temp['mesh'][em]:
                cache['mesh'].pop( em )
    if temp.has_key( 'zenn' ):
        for em in temp['zenn']:
            if not temp['zenn'][em]:
                cache['zenn'].pop( em )

    return cache


def getLastFile( data ):
    verMap = dict()
    for v in data:
        verMap[ data[v]['time'] ] = data[v]['file']
    keys = verMap.keys()
    keys.sort()
    return verMap[keys[-1]]

#-------------------------------------------------------------------------------
# ASSET
#	asset info
def getAssetInfo( showName, assetName ):
    if not _CONNECT:
        return None
    db   = _CONNECT.ASSET[ showName ]
    docs = db.find( {'show':showName, 'name':assetName} )
    if docs.count() == 1:
        return docs[0]

#	zenn template files
def getZennTemplates( showName, assetNameList ):
    zennFiles = list()
    for assetName in assetNameList:
        doc = getAssetInfo( showName, assetName )
        if doc:
            if doc.has_key('hair') and doc['hair'].has_key('pub'):
                zennFiles.append( getLastFile(doc['hair']['pub']) )
    return zennFiles


#-------------------------------------------------------------------------------
#
#	Insert DB Data
#
#-------------------------------------------------------------------------------
def mergeDict( dict1, dict2 ):
    for k in set( dict1.keys() ).union( dict2.keys() ):
        if k in dict1 and k in dict2:
            if isinstance( dict1[k], dict ) and isinstance( dict2[k], dict ):
                yield ( k, dict(mergeDict(dict1[k], dict2[k])) )
            elif isinstance( dict1[k], list ) and isinstance( dict2[k], list ):
                yield ( k, list(set(dict1[k]).union(dict2[k])) )
            else:
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


def getDictKeyValue( dictData, keyName ):
    for i in dictData:
        if type(dictData[i]).__name__ == 'dict':
            for k in dictData[i]:
                if k == keyName:
                    return dictData[i][k]


def toDocument( cacheLog ):
    data = dict()

    name = getDictKeyValue( cacheLog, 'name' )
    if name:
        name = name.split('.')[0]
    else:
        context = getDictKeyValue( cacheLog, 'context' )
        name    = os.path.splitext( os.path.basename(context) )[0]

    created = getDictKeyValue( cacheLog, 'created' )
    dt_time = dateutil.parser.parse( created )
    isotime = dt_time.isoformat()

    data['show']  = getDictKeyValue( cacheLog, 'SHOW' )
    data['shot']  = getDictKeyValue( cacheLog, 'SHOT' )
    data['start'] = getDictKeyValue( cacheLog, 'start' )
    data['end']   = getDictKeyValue( cacheLog, 'end' )
    fps = getDictKeyValue( cacheLog, 'fps' )
    if fps:
        data['fps'] = fps

    # Camera
    camera_file   = getDictKeyValue( cacheLog, 'abc_camera' )
    render_camera = getDictKeyValue( cacheLog, 'render_camera' )
    if camera_file:
        data['camera'] = dict()
        data['camera'][name] = {
                'file': camera_file,
                'render_camera': render_camera,
                'time': isotime
            }

    # Meshes
    meshTypes = ['mesh', 'mid', 'low', 'sim']
    for m in meshTypes:
        getValue = getDictKeyValue( cacheLog, m )
        if getValue:
            data[m] = dict()
            for f in getValue:
                char = os.path.splitext( os.path.basename(f) )[0]
                char = char.split('_rig_GRP')[0]
                data[m][char] = dict()
                data[m][char][name] = {
                        'file': f,
                        'time': isotime
                    }

    # ZENN
    zenn_data = getDictKeyValue( cacheLog, 'zenn' )
    if zenn_data:
        data['zenn'] = dict()
        for z in zenn_data:
            char = os.path.basename( z )
            data['zenn'][char] = dict()
            data['zenn'][char][name] = {
                    'file': z,
                    'nodes': zenn_data[z]['nodes'],
                    'ref-cache': zenn_data[z]['cache'],
                    'ref-file': zenn_data[z]['file'],
                    'time': isotime
                }
            if zenn_data[z].has_key( 'constant' ):
                data['zenn'][char][name]['constant'] = zenn_data[z]['constant']

    # Layout
    layout_file = getDictKeyValue( cacheLog, 'layout' )
    if layout_file:
        data['layout'] = dict()
        data['layout'][name] = {
                'file': layout_file,
                'time': isotime
            }

    return data


def insertDB( jsonFile ):
    if not _CONNECT:
        return
    data  = json.loads( open(jsonFile, 'r').read() )
    myDoc = toDocument( data )
    SHOW  = myDoc['show']
    SHOT  = myDoc['shot']

    db   = _CONNECT.SHOT[ SHOW ]
    docs = db.find( {'shot': SHOT, 'cache': {'$exists': True}} )
    num  = docs.count()
    if num > 0:
        currentDoc = docs[0]['cache']
        newDoc     = dict( mergeDict(currentDoc, myDoc) )
        db.update( {'shot': SHOT, 'show': SHOW}, {'$set': {'cache': newDoc}}, upsert=True )
    else:
        db.update( {'shot': SHOT, 'show': SHOW}, {'$set': {'cache': myDoc}}, upsert=True )
    # debug
    sys.stderr.write( '# Result : MongoDB push complete!\n' )

