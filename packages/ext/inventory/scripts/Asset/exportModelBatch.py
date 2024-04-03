# if will you use mayapy? from pymel.all import *
# because load environment setting
from pymel.all import *

import sys
import os
import string

for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

import dxExportMesh

import maya.cmds as cmds
import maya.mel as mel

# if __name__ == '__main__':
for argv in sys.argv:
    print argv

print "import alembic file path :", sys.argv[1]
print "add attr path :", sys.argv[2]
print "export path :", sys.argv[3]

# 1 : import Alembic
if not cmds.pluginInfo("AbcImport", q = True, l = True):
    cmds.loadPlugin('AbcImport')

if not cmds.pluginInfo("RenderMan_for_Maya", q=True, l=True):
    cmds.loadPlugin('RenderMan_for_Maya')

print mel.eval('AbcImport -mode import "{0}"'.format(sys.argv[1]))

print cmds.ls()

# 2 : add Attribute
for mesh in cmds.ls(type = 'mesh'):
    # mel.eval('rmanAddAttr "%s" "rman__riattr__user_txAssetLib" ""' % mesh)
    # cmds.setAttr(mesh + ".rman__riattr__user_txAssetLib", sys.argv[2], type = "string")
    # if cmds.attributeQuery('rman__riattr__user_txAssetName', n = mesh, ex = True):
    #     cmds.deleteAttr(mesh + ".rman__riattr__user_txAssetName")

    mel.eval('rmanAddAttr "%s" "rman__riattr__user_txAssetName" ""' % mesh)
    cmds.setAttr(mesh + ".rman__riattr__user_txAssetName", sys.argv[2], type="string")
    if cmds.attributeQuery('rman__riattr__user_txAssetLib', n=mesh, ex=True):
        cmds.deleteAttr(mesh + ".rman__riattr__user_txAssetLib")

# 3 : export Alembic
file_abc = sys.argv[3]
exportClass = dxExportMesh.ExportMesh(os.path.splitext(sys.argv[3])[0], None)
# exportClass.uvClass = dxExportMesh.UVLayOut()
# print "layers :", exportClass.uvClass.layers
# print "getTextureDisplayLayers :", dxExportMesh.getTextureDisplayLayers()
# txlayout = exportClass.uvClass.layoutInfo()
# print "txlayout :", txlayout
# txindexs = []
# for layer in txlayout:
#     txindexs.append( txlayout[layer]['txindex'][0] )
# txindexs = list(set(txindexs))
#
# # texture objects
# for i in txlayout:
#     for o in txlayout[i]['members']:
#         exportClass.textureObjects += cmds.listRelatives( o, f=True, p=True )

# exportClass.displaylayerinfo_export(os.path.splitext(sys.argv[3])[0] + '.json')

# if len(txindexs) > 1:
#     self.uvClass.uvposition(opt='init')

exportClass.alembic_export(file_abc)

# if len(txindexs) > 1:
#     self.uvClass.uvposition(opt='')

splitProjectPath = str(sys.argv[3]).split('/')
setProjectPath = string.join(splitProjectPath[:5], '/')

print setProjectPath

mel.eval('setProject "{0}"'.format(setProjectPath))

os._exit(0)