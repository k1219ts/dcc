#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#       Sanghun Kim, rman.td@gmail.com
#
#	for texture pipe-line
#
#	2017.03.02 $4
#-------------------------------------------------------------------------------

import os, sys
import glob
import re
import shutil

import maya.cmds as cmds
import maya.mel as mel


try:
	from PIL import Image
except:
	print '# Result : Not support PILLOW module.'


#-------------------------------------------------------------------------------
#
#	Texture Attributes
#
#-------------------------------------------------------------------------------
def getTextureAttributesData():
    dataMap = {}
    selected = cmds.ls( sl=True, dag=True, type='surfaceShape', ni=True )
    if not selected:
        selected = cmds.ls( type='surfaceShape', ni=True )
    layerAttr = 'rman__riattr__user_txLayerName'
    usdlayerAttr='txLayerName'
    for i in selected:
        if cmds.attributeQuery( layerAttr, n=i, ex=True ):
            getValue = cmds.getAttr( '%s.%s' % (i,layerAttr) )
            if getValue:
                if not dataMap.has_key(getValue):
                    dataMap[getValue] = []
                dataMap[getValue].append( i )
        elif cmds.attributeQuery( usdlayerAttr , n=i, ex=True):
            getValue = cmds.getAttr( '%s.%s' % (i,usdlayerAttr) )
            if getValue:
                if not dataMap.has_key(getValue):
                    dataMap[getValue] = []
                dataMap[getValue].append( i )
        else:
            pass
    return dataMap


def create_TextureDisplayLayer():
    layerData = getTextureAttributesData()
    if not layerData:
        return
    for i in layerData:
        layerName = '%s_LYR' % i
        cmds.select( layerData[i] )
        layer = cmds.createDisplayLayer( name=layerName, nr=True )
    cmds.select( clear=True )



#-------------------------------------------------------------------------------
def versionCheckCopy( fileNames ):
    current_dir = os.path.dirname( fileNames[0] )
    current_version_files = list()
    for f in os.listdir( current_dir ):
        if os.path.isfile( os.path.join(current_dir,f) ):
            current_version_files.append( f )

    verCheck = re.compile(r'v(\d+)').findall( current_dir )
    if verCheck:
        ifever = 0
        print '# Result : Previous version file copy'

        rootPath = os.path.dirname( current_dir )
        version_dirs = list()
        for f in os.listdir( rootPath ):
            if os.path.isdir( os.path.join(rootPath, f) ) and re.compile(r'v(\d+)').findall(f):
                version_dirs.append( f )
        version_dirs.sort()

        previous_ver = version_dirs[ version_dirs.index('v%02d' % int(verCheck[0]))-1 ]
        previous_dir = os.path.join( rootPath, previous_ver )
        for f in os.listdir( previous_dir ):
            filename = os.path.join( previous_dir, f )
            if os.path.isfile( filename ):
                if not f in current_version_files:
                    ifever += 1
                    mel.eval( 'print "# Debug : %s -> %s\\n"' % (filename, current_dir) )
                    shutil.copy2( filename, current_dir )
        if ifever:
            mel.eval( 'print "# Result : version copy finished!\\n"' )
            return True

def proxyConvert( inputfile, outputfile, size ):
    resizefile = inputfile
    if os.path.splitext( inputfile )[-1] == '.tex':
        # tex -> jpg
        command = 'txmake -t:0 -ch rgb -byte %s %s' % ( inputfile, outputfile )
        os.system( command )
        resizefile = outputfile
    # resize
    readImage = Image.open( resizefile )
    readImage.thumbnail( (size,size), Image.ANTIALIAS )
    readImage.save( outputfile, 'JPEG' )
    # debug
    mel.eval( 'print "# Debug : proxy -> %s\\n"' % outputfile )


def textureProxy( Path=None, Type='diffC', Size=512 ):
    if not globals().has_key( 'Image' ):
        return
    if not Path:
        return
    apps = glob.glob( '/opt/pixar/RenderManProServer*' )
    if apps:
        apps.sort()
        os.environ['PATH'] += ':%s/bin' % apps[0]

    if Path.find('/dev') > -1:
        rootPath  = Path.split('/dev')[0]
        rootPath += '/dev'
    elif Path.find('/pub') > -1:
        rootPath  = Path.split('/pub')[0]
        rootPath += '/pub'
    else:
        rootPath  = os.path.dirname(Path)

    # proxy path
    proxyPath  = os.path.join( rootPath, 'proxy', os.path.basename(Path) )
    if not os.path.exists( proxyPath ):
        os.makedirs( proxyPath )


    files = os.listdir( Path )
    files.sort()
    for i in files:
        orig_file = os.path.join( Path, i )
        if os.path.isfile( orig_file ):
            if i.find(Type) > -1:
                source    = os.path.splitext( i )
                proxyfile = os.path.join( proxyPath, '%s.jpg' % source[0] )

                if os.path.exists( proxyfile ):
                    proxy_mtime = os.path.getmtime( proxyfile )
                    orig_mtime  = os.path.getmtime( orig_file )

                    if orig_mtime > proxy_mtime:
                        proxyConvert( orig_file, proxyfile, Size )

                else:
                    proxyConvert( orig_file, proxyfile, Size )

    #
    proxyfiles = list()
    for f in os.listdir( proxyPath ):
        filename = os.path.join( proxyPath, f )
        if os.path.isfile( filename ):
            proxyfiles.append( filename )

    mel.eval( 'print "# Result : convert proxy finished!\\n"' )
    if proxyfiles:
        versionCheckCopy( proxyfiles )


def textureProxyDialog():
    dirs = cmds.fileDialog2( fm =3,
                             cap='Create Proxy Texture (Select Directory)',
                             okc='convert' )
    if not dirs:
        return

    textureProxy( dirs[0] )

