from pymel.all import *
import maya.standalone
maya.standalone.initialize("Python")

import os
import sys
import maya.cmds as cmds
import dxsUsd
import optparse
from dxsUsd import DBQuery

pluginList = [u'svgFileTranslator', u'invertShape', u'mayaHIK', u'curveWarp', u'CloudImportExport', u'tiffFloatReader', u'MASH', u'poseInterpolator', u'hairPhysicalShader', u'pxrUsd', u'ikSpringSolver', u'ik2Bsolver', u'xgenToolkit', u'AbcExport', u'retargeterNodes', u'backstageMenu', u'pxrUsdTranslators', u'OpenEXRLoader', u'lookdevKit', u'Unfold3D', u'mayaCharacterization', u'Type', u'modelingToolkit', u'meshReorder', u'MayaMuscle', u'rotateHelper', u'matrixNodes', u'AbcImport', u'autoLoader', u'deformerEvaluator', u'sceneAssembly', u'gpuCache', u'Substance', u'OneClick', u'shaderFXPlugin', u'objExport', u'renderSetup', u'GPUBuiltInDeformer', u'ArubaTessellator', u'quatNodes', u'fbxmaya']
for i in pluginList:
    cmds.loadPlugin(i)

optparser = optparse.OptionParser()
optparser.add_option('--orgDir', dest='orgDir', type='string', default='',help='get original Directory')
optparser.add_option('--orgModelDir', dest='orgModelDir', type='string', default='',help='orgModelDir')
optparser.add_option('--newShow', dest='newShow', type='string', default='',help='New Show Directpry')
optparser.add_option('--orgAssetName', dest='orgAssetName', type='string', default='',help='original AssetName')
optparser.add_option('--assetName', dest='assetName', type='string', default='',help='New AssetName')
optparser.add_option('--Element', dest='Element', type='string', default='',help='Find Element')
optparser.add_option('--Purpose', dest='Purpose', type='string', default='',help='Find Purpose variant')
optparser.add_option('--Lod', dest='Lod', type='string', default='',help='Find Lod variant')
optparser.add_option('--overwrite', dest='overwrite', type='string', default='',help='Find overwrite')


opts, args = optparser.parse_args(sys.argv)

if opts.Element == 'True':
    Element = True
else:
    Element = False

if opts.Purpose == 'True':
    Purpose = True
else:
    Purpose = False

if opts.Lod == 'True':
    Lod = True
else:
    Lod = False

if opts.overwrite == 'True':
    overwrite = True
else:
    overwrite = False

print '\n'
print '-------------------------         Maya Start                --------------------'

#Define
orgDir = opts.orgDir #'/show/cdh_pub'
orgAssetName = opts.orgAssetName #'bear'
orgModelDir = opts.orgModelDir
orgAssetDir = orgModelDir.split('/model')[0] #/show/pipe/asset/bearTest


##########################################             New Info          ###############################################
newShow = opts.newShow #'/assetlib/3D'
assetName = opts.assetName  #'apple'
version = 'v001'

if Element == True:
    elementVersion = 'v001'
    elementAssetName = assetName.split('_')[0]
    elementName = assetName.replace('%s_' % elementAssetName, '', 1)
    txPath = 'asset/%s/element/%s/texture' % (elementAssetName, elementName)
    newAssetDir = "{SHOWDIR}/asset/{ASSETNAME}/element/{ELEMENTNAME}".format(SHOWDIR=newShow,
                                                                             ASSETNAME=elementAssetName,
                                                                             ELEMENTNAME= elementName) #'/assetlib/3D/asset/apple'
else:
    txPath = 'asset/%s/texture' % assetName
    newAssetDir = "{SHOWDIR}/asset/{ASSETNAME}".format(SHOWDIR=newShow,
                                                       ASSETNAME=assetName) #'/assetlib/3D/asset/apple'


print '-------------------------     Original Asset Infomation     --------------------'
print 'Get AssetInfo:   orgModelDir   :',orgModelDir
# print 'Get AssetInfo:   texturePath: ', txPath
print '-------------------------     New Asset Infomation     -------------------------'
print 'Get AssetInfo:   newAssetDir   :', newAssetDir


########################################## selected version model import###############################################
for fileName in os.listdir(orgModelDir):
    #print fileName
    if "high_geom.usd" in fileName or "mid_geom.usd" in fileName or "low_geom.usd" in fileName:
        # print "yes"
        fileDir = os.path.join(orgModelDir, fileName)
        print 'Get AssetInfo:   fileDir:',fileDir
        dxsUsd.dxsMayaUtils.UsdImport(fileDir)
    else:
        pass

getGRP = []
highM = cmds.ls('*_model_GRP')
midM = cmds.ls('*_model_mid_GRP')
lowM = cmds.ls('*_model_low_GRP')
list = [highM, midM, lowM]
for i in list:
    if i:
        getGRP.append(i[0])
print "Get AssetInfo:   exist GRP:", getGRP


#Set Attr--------------------------------------------------------------------------------------------------------------
newGRP = []
if orgAssetName == assetName:
    pass
else:
    # rename
    if 'element' in orgModelDir:
        orgAssetName = orgModelDir.split('/')[-5] + '_' + orgAssetName
        print 'Get AssetInfo:   Element Asset:  orgAssetName:   ', orgAssetName

    if Element == True:
        assetName = elementAssetName + '_' + elementName
        print 'Get AssetInfo:   New Element Asset:  assetName:   ', assetName

    for i in getGRP:
        new = i.replace(orgAssetName, assetName)
        new = cmds.rename(i, new)
        newGRP.append(new)
    getGRP = newGRP



node = cmds.ls(dag=True, type='surfaceShape', ni=True)
for i in node:
    if not cmds.attributeQuery("txBasePath", n=i, exists=True):
        cmds.addAttr(i, ln="txBasePath", nn="txBasePath", dt="string")
    if not cmds.attributeQuery("txVersion", n=i, exists=True):
        cmds.addAttr(i, ln="txVersion", nn="txVersion", dt="string")

    cmds.setAttr("%s.%s" % (i, "txBasePath"), txPath, type="string")
    cmds.setAttr("%s.%s" % (i, "txVersion"), version, type="string")

    # materialList = ["bronze", "chrome", "gold", "metal", "silver", "fabric", "glass", "leather", "plastic",
    #                 "rubber",
    #                 "feather",
    #                 "paint", "wood", "leaf", "ice", "ocean", "mineral", "rock", "snow", "skin", "fur", "eye",
    #                 "layer", "light", "layerB"]
    # #MaterialSet ReSet
    # if cmds.attributeQuery('MaterialSet', n=i, exists=True):
    #     getM = cmds.getAttr('%s.MaterialSet' % i)
    #     print getM
    #     if not getM in materialList:
    #         print 'MaterialSet is Wrong:   ', i
    #         #cmds.setAttr("%s.%s" % (i, 'MaterialSet'), 'plastic', type="string")
    # else:
    #     print 'MaterialSet is None:   ', i
    #     #cmds.addAttr(i, ln='MaterialSet', nn='MaterialSet', dt="string")
    #     #cmds.setAttr("%s.%s" % (i, 'MaterialSet'), 'plastic', type="string")


outDirs = []
if Element == True:
    for node in getGRP:
        mdExp = dxsUsd.ModelExport(node=node, isElement=Element, isPurpose=Purpose, isLod=Lod, showDir=newShow,
                                   asset=elementAssetName, version=elementVersion,overWrite= overwrite)
        mdExp.doIt()
        if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
            outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
        dxsUsd.DBQuery.assetInsertDB(newShow, elementAssetName, elementVersion, "element", outDirs,elementName=elementName, elementTask="model")
elif Element == False:
    for node in getGRP:
        mdExp = dxsUsd.ModelExport(
            node=node, isElement=Element, isPurpose=Purpose, isLod=Lod,
            showDir=newShow, asset=assetName, version=version,overWrite= overwrite)
        mdExp.doIt()
        if not os.path.join(mdExp.outDir, mdExp.version) in outDirs:
            outDirs.append(os.path.join(mdExp.outDir, mdExp.version))
        dxsUsd.DBQuery.assetInsertDB(newShow, assetName, version, "model", outDirs)
        print 'outDirs:   ',outDirs
else:
    pass

print 'Progress     :   Model Export '


