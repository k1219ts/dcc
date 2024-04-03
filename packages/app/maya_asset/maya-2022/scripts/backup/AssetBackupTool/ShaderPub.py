# if will you use mayapy? from pymel.all import *
# because load environment setting
import maya.standalone
maya.standalone.initialize(name='python')

import sys
import os

# sys.path.append('/netapp/backstage/pub/apps/maya2/versions/2017/global/linux/scripts')
# sys.path.append('/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts')

# print os.getenv('PYTHONPATH').split(':')
for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

import maya.cmds as cmds

if not cmds.pluginInfo("RenderMan_for_Maya", q=True, l=True):
    cmds.loadPlugin('RenderMan_for_Maya')

import rfm.rmanAssetsMaya as ram
import rfmShading
import dxRfmTemplate

# if __name__ == '__main__':
for argv in sys.argv:
    print argv

print "import shader file path :", sys.argv[1]
print "export FileName :", sys.argv[2]
filename = sys.argv[2]
assetName = sys.argv[3]
artistName = sys.argv[4]
print "author name :", sys.argv[4]

# 4 : import shader.ma
cmds.file(sys.argv[1], type = "mayaAscii", mnc = False, i = True)

# 5 : export shader.json
shaderObj = list()
for i in cmds.ls("%s*" % assetName, type = ['PxrSurface', 'PxrMarschnerHair', 'PxrLayerSurface']):
    connectNode = cmds.listConnections(i, type='shadingEngine')
    if connectNode:
        shaderObj.append(connectNode[0])

for i in cmds.ls("%s*" % assetName, type = 'transform'):
    shapes = cmds.ls( i, dag=True, s=True )
    for s in shapes:
        if cmds.nodeType(s).startswith( 'Pxr' ):
            shaderObj.append(s)
    # elif 'Pxr' in nodeType:
    #     sg = cmds.listConnections( i, type='shadingEngine' )
    #     if not sg == None:
    #         shaderObj.append(nodeType)
    #         print "sg :", sg[0]


label = os.path.splitext( os.path.basename(filename) )[0]
infoDict = {'label': label, 'author': artistName, 'version':0}

print shaderObj
print list(set(shaderObj))

dxRfmTemplate.exportAssetDexter( list(set(shaderObj)), 'nodeGraph',
                       infoDict, filename )

cmds.file( rename = filename.replace('.json', '.mb') )
cmds.file( force=True, save=True, options='v=0;', type='mayaBinary' )
