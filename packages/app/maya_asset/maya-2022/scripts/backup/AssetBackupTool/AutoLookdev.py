# if will you use mayapy? from pymel.all import *
# because load environment setting
from pymel.all import *

import sys
import os
import getpass

for p in os.getenv('PYTHONPATH').split(':'):
    if p and not p in sys.path:
        sys.path.append( p )

from pymongo import MongoClient
import dxConfig

# plugin check
pluginInfo = cmds.pluginInfo(query=True, listPlugins=True)

if not "RenderMan_for_Maya" in pluginInfo:
    cmds.loadPlugin('RenderMan_for_Maya')

if not 'AbcImport' in pluginInfo:
    cmds.loadPlugin('AbcImport')

if not 'backstageLight' in pluginInfo:
    cmds.loadPlugin('backstageLight')

if not 'backstageMenu' in pluginInfo:
    cmds.loadPlugin('backstageMenu')

cmds.setAttr('defaultRenderGlobals.currentRenderer','renderManRIS',type='string')
print cmds.getAttr('defaultRenderGlobals.currentRenderer')

### RFM SETTING
mel.eval('rmanChangeRendererUpdate')

import sgUI
import dxRfmTemplate
import rfmShading
import lgtCommon
import lgtUI
import rfm.rmanAssetsLib as ral


def max(value1, value2):
    if value1 > value2:
        return value1
    else:
        return value2

DBIP = dxConfig.getConf("DB_IP")
DBNAME = "inventory"

client = MongoClient(DBIP)
database = client[DBNAME]
coll = database['assets']

# for i in range(1, 9731):
#     result = coll.insert({'name':'test' + str(i).zfill(4)})

from bson.objectid import ObjectId
objectID = ObjectId(sys.argv[1])
result = coll.find_one({'_id':objectID})

if result['files'].has_key('model'):
    modelPath = result['files']['model']

    ciClass = sgUI.ComponentImport( Files=[modelPath], World = 1 ) # 1 : Baked
    ciClass.m_display = 1 # 1 : "Render"
    ciClass.m_fitTime = True
    ciClass.m_mode = 1 # 1 : "gpumode"
    assetName = ciClass.doIt()[0]

if result['files'].has_key('shader_json'):
    dxRfmTemplate.importAssetShader(result['files']['shader_json'])

if result['files'].has_key('shader_xml'):
    rfmShading.importRlf(result['files']['shader_xml'], 'rlfAdd')

if result['files'].has_key('hair_cache'):
    trans, shape = lgtCommon.createArchiveNode('zennArchive')
    lgtUI.zennArchive_setCache('%s.cachePath' % shape, result['files']['hair_cache'])
    # cmds.setAttr('%s.cachePath' % shape, result['files']['hair_cache'], type = 'string')
    # zennGroup = sgZenn.zennStrandsViewer(result['files']['hair_cache'])

# default import success
cmds.setAttr("renderManRISGlobals.rman__riopt__Hider_maxsamples", 64)

hdriPath = '/dexter/Cache_DATA/ASSET/hdri/Stinson_Beach_1502_PM.rma/Env_StinsonBeach_1502PM_2k.18.tex'
envMapName = os.path.splitext(hdriPath)[0]

mel.eval('AbcImport -mode import "/netapp/backstage/pub/apps/maya2/versions/2017/team/asset/linux/etc/LookDev_Checker.abc"')

cmds.select(assetName)

wBBox = mel.eval("exactWorldBoundingBox;")
cmds.setAttr('CHECKER_GRP.translateX', wBBox[0] - 5)
cmds.setAttr('CHECKER_GRP.translateY', wBBox[4] - 5)

cmds.select(assetName, 'CHECKER_GRP')

cmds.setAttr('persp.translateX', 0)
cmds.setAttr('persp.translateY', 0)
cmds.setAttr('persp.translateZ', 0)
cmds.setAttr('persp.rotateX', 0)
cmds.setAttr('persp.rotateY', 0)
cmds.setAttr('persp.rotateZ', 0)

cmds.viewFit(['persp'], f = 0.8)
camZ = cmds.getAttr('persp.translateZ')

# Create hdr
shadingNode = cmds.shadingNode('PxrDomeLight', asLight=True, name=envMapName)
cmds.setAttr('{0}.lightColorMap'.format(shadingNode), hdriPath, type='string')

aovNode = cmds.createNode('stupidAOV')

cmds.setAttr("{0}.diffuse".format(aovNode), 1)
cmds.setAttr("{0}.indirectdiffuse".format(aovNode), 1)
cmds.setAttr("{0}.specular".format(aovNode), 1)

# mel.eval( '''setProject "%s";''' %renamePath.split('/renderBotTest')[0] )
cmds.file(rename=sys.argv[2])
cmds.file(force=True, save=True, options='v=0;', type='mayaBinary')
