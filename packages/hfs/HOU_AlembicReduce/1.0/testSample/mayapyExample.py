'''
 mayapy executing script
'''

'''
mayapy /dexter/Cache_DATA/RND/daeseok/AstPub.py /show/god/asset/char/tattooMan1/model/pub/scenes/tatooMan1_model_v03.abc /assetlib/3D/char/human/tattooTest /assetlib/3D/char/human/tattooTest/model/tattooTest_model
'''
'''
    workflow
    1. import .abc or .mb
    2. add Attribute (txAssetLib), write Path( path : /assetlib/3D/(TYPE)/(CATEGORY)/(ASSET_NAME)
    3. export abc
    4. shader export method
        4 - 1 : import maya scene and export shader file(ma, xml files)
        4 - 2 : import maya scene and export shader file(template file)
        4 - 3 : already pub shader file copy
    5. copy texture files (tiff & tex)

    *. afterwards add hair zenn cache
'''

# import custom Module
import sgUI
import dxExportMesh
import rfmShading
import rfmDataTemplate

# import Maya Module
import maya.cmds as cmds
import maya.mel as mel

# import base Module
import os
import subprocess

assetType = 'char'
assetCategory = "human"
assetName = "tattooMan1"

exportPath = '/assetlib/3D/{0}/{1}/{2}'.format(assetType, assetCategory, assetName)

alembicPath = "/show/god/asset/char/tattooMan1/model/pub/scenes/tatooMan1_model_v03.abc"
jsonPath = alembicPath.replace(".abc", ".json")

# chapter 1-1 : import Alembic
# def ImportAlembic(filePath):
importAlembciModule = sgUI.ComponentImport(Files = [alembicPath], World = 1)
importAlembciModule.m_display = 1
importAlembciModule.m_fitTime = True

abcObjName = importAlembciModule.doIt()[0]

# chapter 1-2 : import Scene
# scenePath = "/show/god/asset/char/tattooMan1/model/pub/scenes/tatooMan1_model_v03.mb"
# cmds.file(scenePath, force = True, open = True)

# chapter 2 : add Attr(txAssetLib)
attributePath = "%s/texture" % exportPath
for mesh in cmds.ls(type = 'mesh'):
    mel.eval('rmanAddAttr "%s" "rman__riattr__user_txAssetLib" ""' % mesh)
    cmds.setAttr(mesh + ".rman__riattr__user_txAssetLib", attributePath, type = 'string')

# chapter 3 : export abc
fileName = os.path.join(exportPath, "model", "{0}_model".format(assetName))
exportClass = dxExportMesh.ExportMesh(fileName, abcObjName)
exportClass.alembic_export(fileName)

# chapter 4-1 : export Shader
# init optionvar
cmds.optionVar(sv=[
    ('stpObjectStatus', 'allobjs'),
    ('stpNameStatus', 'removenamespace')
])

# rfm shader and binding
rfmshaderfile = os.path.join(exportPath, "shader", "{0}.ma".format(assetName))
if not os.path.exists(os.path.dirname(rfmshaderfile)):
    os.makedirs(os.path.dirname(rfmshaderfile))
rfmClass = rfmShading.RfMShaders()
rfmClass.exportProcess(File=rfmshaderfile, Mode="Assigned", Binding=True)

# chapter 4-2 : template Shader
fileName = os.path.join(exportPath, "shader", "{0}.json".format(assetName))
rfmDataTemplate.exportFile(fileName)

# chapter 4-3 : copy shader
copyShaderPath = "/show/god/asset/shaders/tattooMan1/txv04/rfm/tattooMan1_txv04.ma"
copyBindigPath = copyShaderPath.replace('.ma', '.xml')

command = ['cp', '-rf', copyShaderPath, os.path.join(exportPath, "shader", "{0}.ma".format(assetName))]
subprocess.Popen(command).wait()

command = ['cp', '-rf', copyBindigPath, os.path.join(exportPath, "shader", "{0}.xml".format(assetName))]
subprocess.Popen(command).wait()

# chapter 5 : copy texture
copyTiffPath = "/show/god/asset/char/tattooMan1/texture/pub/v02"
copyTexPath = "/show/god/asset/char/tattooMan1/texture/pub/tex/v02"

command = ['cp', '-rf', copyTiffPath, os.path.join(exportPath, "texture", "tiff")]
subprocess.Popen(command).wait()

command = ['cp', '-rf', copyTexPath, os.path.join(exportPath, "texture", "tex")]
subprocess.Popen(command).wait()

'''
 tractor jobspool
'''
#encoding=utf-8
#!/usr/bin/env python

import os
import getpass
import site

import dxConfig
site.addsitedir( dxConfig.getConf('TRACTOR_API') )

import tractor.api.author as author

job = author.Job()
job.title       = '(AssetLib) test'
job.comment     = ''
job.metadata    = ''
job.envkey      = [ 'cache2-2017' ]
job.service     = 'Cache'
job.maxactive   = 1
job.tier        = 'cache'
job.projects    = ['export']
# job.tags        = ['py']

# directory mapping
job.newDirMap( src='S:/', dst='/show/', zone='NFS' )
job.newDirMap( src='N:/', dst='/netapp/', zone='NFS' )
job.newDirMap( src='R:/', dst='/dexter/', zone='NFS' )

JobTask = author.Task( title='Job' )
command = "install -d -m 755 {0}".format('/assetlib/3D/char/human/tattooTest')
JobTask.addCommand(
                author.Command( argv=command, service='Cache')
                )
JobTask.serialsubtasks = 1

# abc
ScriptTask = author.Task( title='batchExportAbc' )

scriptRoot = '/dexter/Cache_DATA/RND/daeseok/Inventory'
exportPath = '/assetlib/3D/char/human/tattooTest'

command = [ 'mayapy', '%%D(%s/AstPub.py)' % scriptRoot,
                        '%%D(%s)' % '/show/god/asset/char/tattooMan3/model/pub/scenes/tattooMan3_model_v01.abc',
                        '%%D(%s)' % exportPath,
                        '%%D(%s)' % os.path.join(exportPath, 'model/tattooTest_model')]

ScriptTask.addCommand( author.Command( argv=command, service='Cache', tags=['py'] ) )
JobTask.addChild( ScriptTask )

ScriptTask = author.Task( title='batchExportShader' )
command = [ 'mayapy', '%%D(%s/ShaderPub.py)' % scriptRoot,
                        '%%D(%s)' % '/show/god/asset/shaders/tattooMan3/txv01/rfm/tattooMan3_txv01.ma',
                        '%%D(%s)' % os.path.join(exportPath, 'shader/tattooTest_model'),
                        '%s' % getpass.getuser()]

ScriptTask.addCommand( author.Command( argv=command, service='Cache', tags=['py'] ) )
JobTask.addChild( ScriptTask )

ScriptTask = author.Task( title='batchCopyTexture' )
copyImagePath = '/show/god/asset/char/tattooMan3/texture/pub/v01'
copyTexPath = '/show/god/asset/char/tattooMan3/texture/pub/tex/v01'

command = 'cp -rf %s %s' % (copyImagePath, os.path.join(exportPath, "texture", "images"))
ScriptTask.addCommand( author.Command( argv=command, service='Cache', tags=['py'] ) )

command = 'cp -rf %s %s' % (copyTexPath, os.path.join(exportPath, "texture", "tex"))
ScriptTask.addCommand( author.Command( argv=command, service='Cache', tags=['py'] ) )

JobTask.addChild( ScriptTask )

job.addChild( JobTask )

job.priority = 1000

author.setEngineClientParam( hostname=dxConfig.getConf('TRACTOR_IP'),
                             port=dxConfig.getConf('TRACTOR_PORT'),
                             user=getpass.getuser(), debug=True )
#        author.setEngineClientParam( hostname='10.0.0.30', port=80, user=getpass.getuser(), debug=True )
job.spool()
author.closeEngineClient()

job.asTcl()