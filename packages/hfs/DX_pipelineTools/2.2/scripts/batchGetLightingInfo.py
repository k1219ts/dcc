# if will you use mayapy? from pymel.all import *
# because load environment setting
from pymel.all import *

import sys
import os
import json

# sys.path.append('/netapp/backstage/pub/apps/maya2/versions/2017/global/linux/scripts')
# sys.path.append('/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/scripts')

# print os.getenv('PYTHONPATH').split(':')


print os.getenv('PYTHONPATH')
print os.getenv('MAYA_PLUG_IN_PATH')

for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

import maya.cmds as cmds
import maya.mel as mel

if not cmds.pluginInfo("RenderMan_for_Maya", q=True, l=True):
    cmds.loadPlugin('RenderMan_for_Maya')

import rfm.rmanAssetsMaya as ram

# if __name__ == '__main__':
for argv in sys.argv:
    print argv

print "import maya scene file path :", sys.argv[1]
print "export Temp lighting Path", sys.argv[2]

lightTypeList = ['PxrAovLight', 'PxrDiskLight', 'PxrDistantLight', 'PxrDomeLight', 'PxrEnvDayLight', 'PxrMeshLight', 'PxrPortalLight', 'PxrRectLight', 'PxrSphereLight']

# 4 : import keyshot scene file
print "Open Maya File"
cmds.file(sys.argv[1], type = "mayaBinary", mnc = False, i = True)

# 5 : export light
print "Search Lighting", lightTypeList
lightObj = cmds.ls(type = lightTypeList)
lightDic = {}

print "Searching Light Shape :", lightObj
for light in lightObj:
    lightDic[light] = {}
    parentNode = cmds.listRelatives(light, parent = True)[0]
    lightDic[light]['type'] = cmds.nodeType(light)
    lightDic[light]['xForm'] = cmds.xform(parentNode, query = True, m = True)
    lightDic[light]['intensity'] = cmds.getAttr('%s.intensity' % light)
    lightDic[light]['exposure'] = cmds.getAttr('%s.exposure' % light)
    lightDic[light]['lightColor'] = cmds.getAttr('%s.lightColor' % light) # type double3
    if cmds.nodeType(light) == "PxrDomeLight":
        lightDic[light]['lightColorMap'] = cmds.getAttr('%s.lightColorMap' % light)

fileName = sys.argv[2]
print "fileName :", fileName

jsonFile = open( fileName, 'w' )
json.dump( lightDic, jsonFile, indent=4 )
jsonFile.close()
