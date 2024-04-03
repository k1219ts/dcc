import maya.standalone
maya.standalone.initialize(name='python')

import sys
import os

for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

hairScene = sys.argv[1]
exportPath = sys.argv[2]

import maya.cmds as cmds
import maya.mel as mel

if not cmds.pluginInfo("AbcImport", q = True, l = True):
    cmds.loadPlugin('AbcImport')

if not cmds.pluginInfo("RenderMan_for_Maya", q=True, l=True):
    cmds.loadPlugin('RenderMan_for_Maya')

if not cmds.pluginInfo("ZENNForMaya", q=True, l=True):
    cmds.loadPlugin('ZENNForMaya')

# 1 : import base mesh
cmds.AbcImport("/dexter/Cache_DATA/ASSET/0.3Team/01.member/hyunjeong.baek/work/baseMan/baseMan.abc")
baseManAssetName = 'baseman_model_GRP_v03'

print cmds.file("/dexter/Cache_DATA/ASSET/0.3Team/01.member/hyunjeong.baek/work/baseMan/baseMan_hair_v01.mb",
# cmds.file(hairScene,
          i = True, type = "mayaBinary", ignoreVersion = True, #ra = True, renamingPrefix = "baseMan_hair_v01",
          mergeNamespacesOnClash = False, options = "v=0", pr = True,
          loadReferenceDepth = "none", importFrameRate = True, importTimeRange = "override")

bodyMeshList = cmds.ls("*_body_*", type="mesh")

removeBodyMeshList = []
for ZNimport in cmds.ls(type='ZN_Import'):
    removeBodyMeshList += cmds.listConnections('%s.inBodyMesh' % ZNimport, s=True, d=False, shapes=True)

list(set(removeBodyMeshList))

for mesh in removeBodyMeshList:
    bodyMeshList.remove(mesh)

newBodyMesh = bodyMeshList[0]
oldBodyMesh = removeBodyMeshList[0]

print oldBodyMesh + " => " + newBodyMesh

for ZNimport in cmds.ls(type='ZN_Import'):
    cmds.connectAttr('%s.w' % newBodyMesh, '%s.inBodyMesh' % ZNimport, f = True)

    cmds.setAttr('%s.updateMesh' % ZNimport)

cmds.delete(oldBodyMesh.split('|')[0])

cmds.select(cmds.ls(type = 'ZN_Global'))

cmds.xform('persp', rotation = [0, 0, 0])

cmds.viewFit(['persp'], f = 0.5)

cmds.setAttr("perspShape.backgroundColor", 1, 1, 1, type = 'double3')

cmds.file( exportPath, f=True, op='v=0', type='mayaBinary', pr=True, es=True )