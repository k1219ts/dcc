#encoding=utf-8
#--------------------------------------------------------------------------------
#
#    Dexter CG-Supervisor
#
#        Sanghun Kim, rman.td@gmail.com
#
#    rman.td 2016.08.07 $3
#
#            dexter mari common functions
#
#-------------------------------------------------------------------------------

import os, sys
import glob
import json
import subprocess
import platform

from PySide2 import QtWidgets, QtCore
# from pymodule.Qt import QtWidgets
# from pymodule.Qt import QtCore

import mari


def readTxLayer_byFile( filename ):
    file = open( filename, 'r' )
    body = json.load( file )
    file.close()

    if not 'TextureLayerInfo' in body.keys():
        return
    data = body['TextureLayerInfo']
    info = {}
    info['multiuv'] = {}
    for layer in data:
        key = data[layer]['txindex']
        if len(key) > 1:
            udim_guide = key[0]
            for i in range(key[0], key[-1]+1):
                info[i] = layer
                info['multiuv'][i] = 1001 + i - udim_guide
        else:
            info[key[0]] = layer
    return info


def getTextureChannels( txpath ):
    channels = []
    for layer in getMetadata_textureLayers():
        source_files = glob.glob( os.path.join(txpath, '%s_*' % layer) )
        if source_files:
            for i in source_files:
                base    = os.path.basename(i).split('.')[0]
                channel = base.split('%s_' % layer)
                channels.append( channel[-1] )
    channels = list(set(channels))
    channels.sort()
    return channels


def getMetadata_textureNameInfo():
    result = dict()
    geo = mari.geo.current()
    if 'TextureName' in geo.metadataNames():
        result = eval( geo.metadata('TextureName') )
    return result


def getMetadata_textureLayers():
    layerList = list()
    info = getMetadata_textureNameInfo()
    if info:
        for i in info:
            if type(i).__name__ == 'int':
                layerList.append( info[i] )
    return layerList


def geoMetadata_txname_update( textureNameInfo ):
    print("### DEBUG :", textureNameInfo)

    geo  = mari.geo.current()
    info = {}
    info['multiuv'] = {}
    for patch in geo.patchList():
        uv_index = patch.uvIndex()
        print(uv_index, "in", textureNameInfo.keys())
        if uv_index in textureNameInfo.keys():
            info[uv_index] = textureNameInfo[uv_index]
            if 'multiuv' in textureNameInfo.keys():
                if uv_index in textureNameInfo['multiuv'].keys():
                    info['multiuv'][uv_index] = textureNameInfo['multiuv'][uv_index]

    print(str(info))
    geo.setMetadata( 'TextureName', str(info) )


def updateTextureLayerInfo():
    startpath = mari.resources.path( 'MARI_DEFAULT_GEOMETRY_PATH' )
    fn = QtWidgets.QFileDialog.getOpenFileName( None, 'Select Texture Layer File', startpath, '*.json' )
    if fn[0]:
        texinfo = readTxLayer_byFile( fn[0] )
        geoMetadata_txname_update( texinfo )


def geoMetadata_outpath_update( outpath ):
    for obj in mari.geo.list():
        obj.setMetadata( 'OutPath', outpath )


#    clear LayerInfo metadata
def clearLayerInfo():
    for obj in mari.geo.list():
        if 'LayerInfo' in obj.metadataNames():
            obj.removeMetadata( 'LayerInfo' )
    print('# Result :Clear LayerInfo metadata.#')


def ClearHistory():
    proj = mari.current.project()
    proj.save()
    mari.history.clear(0)


def openNautilus():
    if mari.projects.current() is None:
        mari.utils.message( 'Open Project' )
        return
    startpath = os.path.join( mari.resources.path('MARI_DEFAULT_IMAGE_PATH'), 'dev' )
    geo = mari.geo.current()
    if 'OutPath' in geo.metadataNames():
        startpath = str( geo.metadata('OutPath') )
    command = 'nautilus %s' % startpath
    os.system( command )


def addVersion(fn = ""):
    startpath = mari.resources.path( 'MARI_DEFAULT_GEOMETRY_PATH' )
    if not fn:
        fn = QtWidgets.QFileDialog.getOpenFileName(
                None,
                'Select Object File',
                startpath
            )
    if fn:
        geo = mari.geo.current()
        jf = fn[0].replace( os.path.splitext(fn[0])[-1], '.json' )
        msg = None
        if os.path.exists( jf ):
            msg = mari.utils.messageResult(
                        'Do you want to import "Texture Info File".',
                        title = 'Question',
                        buttons = QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                        icon = QtWidgets.QMessageBox.Question )

        basename = os.path.basename( fn[0] )
        version_name = '%s_Merged' % os.path.splitext(basename)[0]
        geo.addVersion( fn[0], version_name )

        if msg == QtWidgets.QMessageBox.Ok:
            txinfo = readTxLayer_byFile( jf )
            geoMetadata_txname_update( txinfo )

def addObject(fn = ""):
    startpath = mari.resources.path( 'MARI_DEFAULT_GEOMETRY_PATH' )
    if not fn:
        fn = QtWidgets.QFileDialog.getOpenFileName(
                None,
                'Select Object File',
                startpath
            )
    if fn:
        jf = fn[0].replace( os.path.splitext(fn[0])[-1], '.json' )
        msg = None
        if os.path.exists( jf ):
            msg = mari.utils.messageResult(
                    'Do you want to import "Texture Info File".',
                    title = 'Question',
                    buttons = QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                    icon = QtWidgets.QMessageBox.Question )

        basename = os.path.basename( fn[0] )
        geo_name = '%s_Merged' % os.path.splitext(basename)[0]
        mari.geo.load( fn[0],
                       {'name': geo_name,
                        'CreateChannels': [mari.ChannelInfo('diffC', 8192, 8192, 8),]} )

        if msg == QtWidgets.QMessageBox.Ok:
            mari.geo.setCurrent( geo_name )
            txinfo = readTxLayer_byFile( jf )
            geoMetadata_txname_update( txinfo )

#-------------------------------------------------------------------------------
def dirReMap( filePath ):
    if filePath.find( 'show' ) == -1:
        return filePath.replace( os.sep, '/' ) # to linux path
    if platform.system() == 'Windows':
        if filePath.find( 'N:/dexter' ) > -1 or filePath.find( 'S:/' ) > -1:
            return filePath.replace( os.sep, '/' )
        else:
            if os.path.exists( 'S:/' ):
                new = os.path.join( 'S:/', filePath[1:] )
            else:
                new = os.path.join( 'N:/dexter', filePath[1:] )
            return new.replace( os.sep, '/' )
    else:
        return filePath
