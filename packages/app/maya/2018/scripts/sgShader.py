#encoding=utf-8
#!/usr/bin/env python
"""
for texture preview pipe-line

LAST RELEASE:
- 2017.09.19 $1 : PreviewSetup support ZGpuMeshShape
"""
#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#       Sanghun Kim, rman.td@gmail.com
#
#	for texture preview pipe-line
#
#	2017.02.21 $1
#-------------------------------------------------------------------------------

import os, sys
import glob
import re
import json
import string

import maya.cmds as cmds
import maya.mel as mel

import dplCommon

#-------------------------------------------------------------------------------
#
#	common
#
#-------------------------------------------------------------------------------
def getAssetPath(show, asset):
    for t in ['char', 'prop', 'vehicle', 'env']:
        dirpath = '/show/{SHOW}/asset/{TYPE}'.format(SHOW=show, TYPE=t)
        for a in os.listdir(dirpath):
            if a == asset:
                return 'asset/{TYPE}/{ASSET}'.format(TYPE=t, ASSET=asset)


def create_File( Name='file1', File=None, U=None, V=None ):
    fileNode  = cmds.shadingNode( 'file', n=Name, asTexture=True, isColorManaged=True )
    placeNode = cmds.shadingNode( 'place2dTexture', n='%s_place2d' % fileNode, asUtility=True )
    connectAttrs = ['coverage', 'translateFrame', 'rotateFrame', 'mirrorU', 'mirrorV',
            'stagger', 'wrapU', 'wrapV', 'repeatUV', 'offset', 'rotateUV', 'noiseUV',
            'vertexUvOne', 'vertexUvTwo', 'vertexUvThree', 'vertexCameraOne']
    for i in connectAttrs:
        cmds.connectAttr( '%s.%s' % (placeNode,i), '%s.%s' % (fileNode,i), f=True )
    # output connect
    cmds.connectAttr( '%s.outUV' % placeNode, '%s.uv' % fileNode )
    cmds.connectAttr( '%s.outUvFilterSize' % placeNode, '%s.uvFilterSize' % fileNode )
    # set attr
    if File:
        cmds.setAttr( '%s.fileTextureName' % fileNode, File, type='string' )
    if U != None or V != None:
        cmds.setAttr( '%s.wrapU' % placeNode, 0 )
        cmds.setAttr( '%s.wrapV' % placeNode, 0 )
        cmds.setAttr( '%s.translateFrameU' % placeNode, U )
        cmds.setAttr( '%s.translateFrameV' % placeNode, V )
        cmds.setAttr( '%s.defaultColor' % fileNode, 0, 0, 0, type='double3' )
    return fileNode

def create_Shader( Name=None, Type='lambert' ):
    if Name:
        shaderNode = cmds.shadingNode( Type, n=Name, asShader=True )
    else:
        shaderNode = cmds.shadingNode( Type, asShader=True )
    engineNode = cmds.sets( name='%sSG' % shaderNode, renderable=True, noSurfaceShader=True, empty=True )
    cmds.connectAttr( '%s.outColor' % shaderNode, '%s.surfaceShader' % engineNode )
    return shaderNode, engineNode

def getShadingEngine( node ):
    connections = cmds.listHistory( node, f=True )
    result = None
    for i in connections:
        if cmds.nodeType(i) == 'shadingEngine':
            return i

#-------------------------------------------------------------------------------
#
#	Texture Preview Shading by Texture Attributes
#
#-------------------------------------------------------------------------------
def previewShadingDialog():
    startpath = cmds.workspace( q=True, rd=True )
    fn = cmds.fileDialog2( fileMode = 3,
                           caption = 'Select Texture Directory',
                           okCaption = 'select',
                           startingDirectory = startpath )
    if not fn:
        return

    txpath = fn[0]
    src = txpath.split('/')
    if txpath.find( '/asset/' ) > -1:
        assetName = src[ src.index('asset')+2 ]
        assignInfo = textureAssignInfo( assetName )
        previewShading( assetName, assignInfo, txpath )



def textureAssignInfo( assetName ):
    result = dict()
    assetAttr = 'rman__riattr__user_txAssetName'
    layerAttr = 'rman__riattr__user_txLayerName'
    for s in cmds.ls( type='surfaceShape', ni=True ):
        if cmds.attributeQuery( assetAttr, n=s, ex=True ) and cmds.attributeQuery( layerAttr, n=s, ex=True ):
            getAssetName = cmds.getAttr( '%s.%s' % (s, assetAttr) )
            if getAssetName and getAssetName.split('/')[-1] == assetName:
                getLayerName = cmds.getAttr( '%s.%s' % (s, layerAttr) )
                if getLayerName:
                    if not result.has_key( getLayerName ):
                        result[getLayerName] = list()
                    result[getLayerName].append( s )
	return result

# result = { 0: filename, 1: filename, ... }
def getTextureIndex( sourceFiles ):
    result = dict()
    for f in sourceFiles:
        source = re.compile( r'\.(\d+)?\.' ).findall( f )
        if source:
            txIndex = int(source[-1]) - 1001
            result[txIndex] = f
    return result



# assetName = 'string'
# assignInfo = { 'layername':[shapename, ..], }
# texPath = 'string'
def previewShading( assetName, assignInfo, texPath ):
    for txlayer in assignInfo:
        filesource = glob.glob( os.path.join(texPath, '%s_diffC*' % txlayer) )
        filesource.sort()

        mtl = 'PV_%s_%s' % ( assetName, txlayer )
        # Create Shader
        if not cmds.objExists( mtl ):
            mtl, mtlSG = create_Shader( Name=mtl )
        else:
            mtlSG = '%sSG' % mtl

        # Create Texture
        txNode = None
        if filesource:
            if len(filesource) == 1:
                txNode = '%s_file' % mtl
                if not cmds.objExists( txNode ):
                    print '# debug : create txnode -> %s' % txNode
                    txNode = create_File( Name=txNode, File=filesource[0] )
                else:
                    cmds.setAttr( '%s.fileTextureName' % txNode, filesource[0], type='string' )
            elif len(filesource) > 1:
                txNode = '%s_txlayer' % mtl
                if not cmds.objExists( txNode ):
                    print '# debug : create txnode -> %s' % txNode
                    txNode = cmds.shadingNode( 'layeredTexture', n=txNode, asTexture=True )
                idInfo = getTextureIndex( filesource )
                if idInfo:
                    findexs = idInfo.keys()
                    findexs.sort()
                    for fi in findexs:
                        cmds.setAttr( '%s.inputs[%s].blendMode' % (txNode, fi), 7 )
                        fnNode = '%s_file%s' % ( mtl, fi )
                        if not cmds.objExists( fnNode ):
                            print '# debug : create txnode -> %s' % fnNode
                            fnNode = create_File( Name=fnNode, File=idInfo[fi], U=fi-fi/10*10, V=fi/10 )
                        else:
                            cmds.setAttr( '%s.fileTextureName' % fnNode, idInfo[fi], type='string' )
                        try:
                            cmds.connectAttr( '%s.outColor' % fnNode, '%s.inputs[%s].color' % (txNode, fi), force=True )
                        except:
                            pass
                    cmds.setAttr( '%s.inputs[%s].blendMode' % (txNode, fi), 0 )

        if txNode:
            try:
                cmds.connectAttr( '%s.outColor' % txNode, '%s.color' % mtl, force=True )
            except:
                pass

        cmds.sets( assignInfo[txlayer], e=True, forceElement=mtlSG )





#-------------------------------------------------------------------------------
#
#	Preview Shader Setup
#
#-------------------------------------------------------------------------------
class PreviewSetup:
    """
    Proxy Texture Maya Shading Setup
    """
    def __init__( self, show=None ):
        self.m_assets = dict()
        self.z_assets = dict()  # {asset_path:[node, node], asset_path: [...]}
        self.m_show = show

    # def XgetAssets( self ):
    #     # dxRig
    #     for i in cmds.ls( type='dxRig' ):
    #         asset_name = cmds.getAttr( '%s.assetName' % i )
    #         if not asset_name:
    #             asset_name = i.split(':')[-1].split('_rig_GRP')[0]
    #
    #         if self.m_assets.has_key( asset_name ):
    #             self.m_assets[asset_name].append( i )
    #         else:
    #             self.m_assets[asset_name] = [i]
    #
    #     # dxAbcArchive
    #     for i in cmds.ls( type=['dxComponent', 'dxAbcArchive'] ):
    #         abcFile = cmds.getAttr( '%s.abcFileName' % i )
    #         source  = abcFile.split('/')
    #         # ASSET
    #         if abcFile.find('/asset/') > -1:
    #             asset_name = source[ source.index('asset') + 2 ]
    #             if self.m_assets.has_key( asset_name ):
    #                 self.m_assets[asset_name].append( i )
    #             else:
    #                 self.m_assets[asset_name] = [i]
    #         # SHOT
    #         else:
    #             asset_name = source[-1].split(':')[-1].split('_rig_GRP')[0]
    #             if self.m_assets.has_key( asset_name ):
    #                 self.m_assets[asset_name].append( i )
    #             else:
    #                 self.m_assets[asset_name] = [i]

    def getAssets(self):
        """
        Get Assets in Scene
        mesh data - self.m_assets
        gpu data  - self.z_assets
        """
        # dxRig
        for i in cmds.ls(type='dxRig'):
            asset_name = cmds.getAttr('%s.assetName' % i)
            if not asset_name:
                asset_name = i.split(':')[-1].split('_rig_GRP')[0]

            if not self.m_assets.has_key(asset_name):
                self.m_assets[asset_name] = list()
            self.m_assets[asset_name].append(i)

        # dxComponent
        for i in cmds.ls(type=['dxComponent']):
            mode    = cmds.getAttr('%s.mode' % i)   # 0:mesh 1:gpu
            abcFile = cmds.getAttr('%s.abcFileName' % i)
            src = abcFile.split('/')
            # ASSET
            if abcFile.find('/asset/') > -1:
                asset_name = src[src.index('asset')+2]
            # SHOT
            else:
                asset_name = src[-1].split(':')[-1].split('_rig_GRP')[0]

            if mode == 0:
                if not self.m_assets.has_key(asset_name):
                    self.m_assets[asset_name] = list()
                self.m_assets[asset_name].append(i)
            else:
                asset_path = getAssetPath(self.m_show, asset_name)
                if asset_path:
                    if not self.z_assets.has_key(asset_path):
                        self.z_assets[asset_path] = list()
                    self.z_assets[asset_path].append(i)



    #---------------------------------------------------------------------------
    def import_attributes( self ):
        for a in self.m_assets:
            atFiles = None
            assetName = a
            atFiles    = dplCommon.lkdv_getAttrFile( self.m_show, assetName )
            if not atFiles:
                assetName = a.split('_')[0]
                atFiles    = dplCommon.lkdv_getAttrFile( self.m_show, assetName )
            if atFiles:
                mel.eval( 'print "# Debug : import attributes : %s -> %s\\n"' % (assetName, atFiles) )
                for atFile in atFiles:
                    body = json.loads( open(atFile, 'r').read() )
                    if body.has_key( 'Attributes' ):
                        for n in self.m_assets[a]:
                            dplCommon.lkdv_importAssetAttrs( n, body['Attributes'] )


    #---------------------------------------------------------------------------
    def getAssignInfo( self ):
        info = dict()
        assetAttr = 'rman__riattr__user_txAssetName'
        layerAttr = 'rman__riattr__user_txLayerName'
        for s in cmds.ls( type='surfaceShape', ni=True ):
            if cmds.attributeQuery( assetAttr, n=s, ex=True ) and cmds.attributeQuery( layerAttr, n=s, ex=True ):
                assetName = cmds.getAttr( '%s.%s' % (s, assetAttr) )
                if assetName:
                    if not info.has_key( assetName ):
                        info[assetName] = dict()

                    layerName = cmds.getAttr( '%s.%s' % (s, layerAttr) )
                    if layerName:
                        if not info[assetName].has_key( layerName ):
                            info[assetName][layerName] = list()
                        info[assetName][layerName].append( s )
        return info

    def getProxyTexturePath( self, assetPath ):
        rootPath = os.path.join( '/show', self.m_show, assetPath, 'texture', 'pub', 'proxy' )
        if not os.path.exists( rootPath ):
            return
        dirs = os.listdir( rootPath )
        dirs.sort()
        return os.path.join( rootPath, dirs[-1] )

    # def shaderSetup( self ):
    #     data = self.getAssignInfo()
    #     for asset in data:
    #         info = data[asset]
    #         txpath = self.getProxyTexturePath( asset )
    #         if txpath:
    #             previewShading( asset.split('/')[-1], data[asset], txpath )

    def meshShaderSetup(self):
        """
        Create Shader and Assign for Mesh
        data = {
            asset_path:{layername: [nodes, nodes], layername: [...]},
        }
        """
        data = self.getAssignInfo()
        for asset_path in data:
            txpath = self.getProxyTexturePath(asset_path)
            if txpath:
                previewShading(asset_path.split('/')[-1], data[asset_path], txpath)

    def gpuShaderSetup(self):
        """
        Create Shader and Assign for ZGpuMeshShape
        """
        for asset_path in self.z_assets:
            txpath = self.getProxyTexturePath(asset_path)
            if txpath:
                assignInfo = dict()
                assignInfo[asset_path.split('/')[-1]] = cmds.ls(self.z_assets[asset_path], dag=True, type='ZGpuMeshShape')
                previewShading(asset_path.split('/')[-1], assignInfo, txpath)




    #---------------------------------------------------------------------------
    def doIt( self ):
        if not self.m_show:
            cmds.error('Not found show')
            return

        self.getAssets()
        # look-dev attribute
        self.import_attributes()
        # previewShading
        self.meshShaderSetup()
        self.gpuShaderSetup()


def previewSetupProcess():
    prvClass = PreviewSetup()
    current = cmds.file( q=True, sn=True )
    if current:
        src = current.split('/')
        prvClass.m_show = src[ src.index('show')+1 ]
    else:
        proj = cmds.workspace( q=True, rd=True )
        if proj:
            src = proj.split('/')
            prvClass.m_show = src[ src.index('show')+1 ]
    prvClass.doIt()
