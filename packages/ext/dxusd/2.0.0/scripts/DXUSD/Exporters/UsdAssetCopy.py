# coding:utf-8
from __future__ import print_function
import pprint

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
from DXUSD.Exporters.Export import Export, AExport
import DXUSD.Structures as Arguments
import DXUSD.Compositor as cmp
import os
import glob
from pxr import Sdf, Usd, UsdGeom, Gf
import DXUSD.Tweakers as twk
import subprocess
import json

from pymongo import MongoClient
import dxConfig

gDBIP = dxConfig.getConf("DB_IP")
client = MongoClient(gDBIP)
gDB = client["ASSETLIB"]

try:
    import ice
except:
    pass


class AUsdExporter(AExport):
    def __init__(self, **kwargs):
        # input argument

        self.orgPath = ''

        self.show = ''
        self.overwrite = True
        self.ver = 'v001'

        self.newShow = ''
        self.newAssetName = ''
        # self.newBranchName = ''

        # treat compute
        self.dxusdVer = ''
        self.orgModeldir = ''
        self.orgShow = ''
        self.orgAsset = ''
        self.orgBranch = ''
        self.orgVer = ''

        # target compute
        self.dstdir = ''
        self.geomfiles = []  # export geom filename.
        self.master = ''
        self.maindir = ''
        self.versionExp = False

        # initialize
        AExport.__init__(self, **kwargs)

        # attributes
        self.task = 'model'
        self.taskProduct = 'TASKV'

    def Treat(self):

        if '_pub' in self.orgPath or '/assetlib/3D' in self.orgPath:
            self.dxusdVer = '1.0'
        else:
            self.dxusdVer = '2.0'

        asset, branch = self.getModelDir(self.orgPath)
        self.orgGeomfiles = glob.glob('%s/*geom.usd' % self.orgModeldir)

        if self.newAssetName:
            if '_' in self.newAssetName:
                if self.newAssetName[-1] == '_':
                    if not self.orgBranch:
                        branch = asset
                    else:
                        branch = branch
                else:
                    asset = self.newAssetName.split('_')[0]
                    branch = self.newAssetName.split('_')[-1]
            else:
                asset = self.newAssetName

        if self.has_attr('ovr_show'):
            self.newShow = self.ovr_show
            self.customdir = self.ovr_show
        else:
            self.newShow = self.D.PUB

        self.maindir = os.path.join(self.newShow, 'asset', asset) #asset directory
        if branch:
            self.maindir = os.path.join(self.newShow, 'asset', asset, 'branch', branch)

        self.D.SetDecode(self.maindir)
        self.dstdir = self.D[self.taskProduct]
        self.master = os.path.join(self.dstdir, self.F.MASTER)

        return var.SUCCESS


    def getModelDir(self, path):
        version =''
        stage = Usd.Stage.Open(path)
        dPrim = stage.GetDefaultPrim()
        splitPath = os.path.split(path)
        var = dPrim.GetVariantSets().GetAllVariantSelections()
        if var.get("modelVer"):
            version = var["modelVer"]
        elif var.get("modelVersion"):
            version = var["modelVersion"]
        else:
            orgasset = self.orgPath.split('.')[0].split('/')[-1]
            orgdir = self.orgPath.split('/' + orgasset)[0]
            modelpath = os.path.join(orgdir, orgasset, 'model', 'model.usd')
            mstage = Usd.Stage.Open(modelpath)
            mdPrim = mstage.GetDefaultPrim()
            mvar = mdPrim.GetVariantSets().GetAllVariantSelections()
            if mvar.get("modelVer"):
                version = mvar["modelVer"]
            elif mvar.get("modelVersion"):
                version = mvar["modelVersion"]

        self.orgVer = version
        self.orgModeldir = os.path.join(splitPath[0], 'model', version)

        asset = self.orgDecode(path)

        return asset

    def orgDecode(self, path):
        asset = path.split('asset/')[-1].split('/')[0]
        # asset = splitPath[4]
        branch = ''
        self.orgAsset = asset

        if not asset[0].islower():
            asset = asset.lower()
        if '_' in asset:
            asset = utl.renameAsset(asset)

        if 'element' in path or 'branch' in path:
            branch = path.split('element/')[-1].split('/')[0]
            if '_3d' in path:
                branch = path.split('branch/')[-1].split('/')[0]
            self.orgBranch = branch
            if not branch[0].islower():
                branch = branch.lower()
            if '_' in branch:
                branch = utl.renameAsset(branch)

        self.orgShow = path.split('/asset/')[0]

        return asset, branch


# ------------------------------------------------------------------------------


# UsdGeomCopy


# ------------------------------------------------------------------------------

class UsdGeomCopy(Export):
    ARGCLASS = AUsdExporter

    def Exporting(self):

        if self.arg.versionExp:
            self.versionCopy()
        else:
            self.Doit()

        TextureCopy(self.arg)
        return var.SUCCESS


    def versionCopy(self):
        msg.debug('[   VersionCopy   ]')

        lyr = utl.AsLayer(self.arg.orgPath)
        with utl.OpenStage(lyr) as stage:
            dPrim = stage.GetDefaultPrim()
            versions = dPrim.GetVariantSet("modelVer").GetVariantNames()

            for ver in versions:
                msg.debug('version:', ver)
                sourcedir = os.path.join(os.path.dirname(self.arg.orgPath), 'model', ver)  # model/v001
                dstdir = os.path.join(self.arg.D.TASK, ver)
                utl.MakeDir(dstdir)
                orgGeomfiles = glob.glob('%s/*geom.usd' % sourcedir)
                for source in orgGeomfiles:
                    self.arg.geomfiles.append( os.path.join(self.arg.dstdir, os.path.basename(source)) )
                    utl.CopyFile(source, dstdir)
                    self.attrCopy(source, sourcedir, dstdir)

    def Doit(self):
        # copy model
        utl.MakeDir(self.arg.dstdir)

        for i in range(len(self.arg.orgGeomfiles)):
            self.arg.geomfiles.append(os.path.join(self.arg.dstdir, os.path.basename(self.arg.orgGeomfiles[i])))
            utl.CopyFile(self.arg.orgGeomfiles[i], self.arg.dstdir)
            self.attrCopy(self.arg.orgGeomfiles[i], self.arg.orgModeldir, self.arg.dstdir)


    def attrCopy(self, source ,sourcedir,  dstdir):
        # geomname = os.path.split(source)[1]
        attrname = source.replace('_geom.usd', '_attr.usd')
        sourceATTR = os.path.join(sourcedir, attrname)
        dstATTR = utl.SJoin( dstdir, attrname)

        utl.CopyFile(sourceATTR, dstdir)

        if os.path.exists(dstATTR):
            self.editAttr(dstATTR)


    def editAttr(self, attrfile ):
        layer = utl.AsLayer(attrfile)

        if self.arg.dxusdVer == '1.0':
            self.eidtOldAttr(layer)

        else:
            for spec in layer.rootPrims:
                if spec.path.pathString.endswith('_txAttr'):
                    utl.SetModelVersion(spec)
        layer.Save()
        del layer

    def AssetName(self):
        assetName = self.arg.orgAsset
        if self.arg.orgBranch:
            assetName = self.arg.orgBranch

        return assetName

    def eidtOldAttr(self, layer):
        editor = Sdf.BatchNamespaceEdit()
        self.txLayerList = []

        for spec in layer.rootPrims:
            if spec.path.pathString.endswith('_Attr'):
                self.walk(spec, self.AssetName())

            if spec.path.pathString.endswith('_txAttr'):
                editor.Add(spec.path, Sdf.Path.emptyPath)
                layer.Apply(editor)

        if self.txLayerList:
            self.setReference(layer, self.AssetName())


    def setReference(self, layer, assetName ):
        # var.T.TEX # tex
        texusdfile = os.path.join(self.arg.maindir, 'texture', 'tex', 'tex.usd')
        for txLayer in self.txLayerList:
            txClassPath = '_'.join(['_' + assetName, txLayer, 'txAttr'])
            
            spec = utl.GetPrimSpec(layer, txClassPath, specifier='class')
            spec.payloadList.explicitItems.clear()
            relpath = utl.GetRelPath(layer.identifier, texusdfile)
            utl.ReferenceAppend(spec, relpath, '/' + txLayer)
            utl.SetModelVersion(spec)

    def walk(self, spec, asset):
        for s in spec.nameChildren:
            s.inheritPathList.prependedItems.clear()
            stage = Usd.Stage.Open(self.arg.orgPath)
            treeIter = iter(Usd.PrimRange.AllPrims(stage.GetPseudoRoot()))
            treeIter.next()
            for p in treeIter:
                if s.name == p.GetName():
                    txLayer = p.GetAttribute('primvars:txLayerName').Get()
                    if not txLayer == None:
                        txClassPath = '_'.join(['/', asset, txLayer, 'txAttr'])
                        utl.SetInherit(txClassPath, s)
                        if not txLayer in self.txLayerList:
                            self.txLayerList.append(txLayer)
            self.walk(s, asset)

    def Arguing(self):
        self.gArg = twk.AGeomAttrs()
        self.gArg.inputs = self.arg.geomfiles
        return var.SUCCESS

    def Tweaking(self):
        twks = twk.Tweak()
        twks << twk.PrmanMaterial(self.gArg)
        twks << twk.MasterModelPack(self.arg)
        twks << twk.Collection(self.arg)
        twks.DoIt()
        return var.SUCCESS

    def Compositing(self):
        cmp.Composite(self.arg.master).DoIt()

        if self.arg.customdir:
            Database(self.arg)

        return var.SUCCESS




# ------------------------------------------------------------------------------


# DCC Batch


# ------------------------------------------------------------------------------

class MayaExport(Export):
    ARGCLASS = AUsdExporter

    def Exporting(self):
        self.Doit()
        TextureCopy(self.arg)
        if self.arg.customdir:
            Database(self.arg)

    def Doit(self):
        newShow = self.arg.show
        if self.arg.customdir:
            newShow = self.arg.customdir

        dccpath = '/WORK_DATA/Develop/dcc'
        dccCmd = '%s/DCC dev' % dccpath
        if '/backstage' in __file__:
            dccpath = '/backstage/dcc'
            dccCmd = '%s/DCC' % dccpath

        mayaCmdPath = '%s/packages/ext/usdtoolkit/plugins/viewer/assetExportPlugin/mayaCmd.py' % dccpath

        hairPath = None
        if self.getHairScene(self.arg.orgPath):
            hairPath = self.getHairScene(self.arg.orgPath)


        cmd = '{DCCCMD} maya --zelos --terminal mayapy {CMDFILE} --orgModelDir {MODELDIR} --newShow {SHOW} --newAsset {ASSET} --newBranch {BRANCH} --hairPath {HAIR} --versionExp {VERSIONEXP}'.format(
            DCCCMD=dccCmd,
            CMDFILE=mayaCmdPath,
            MODELDIR=self.arg.orgModeldir,
            SHOW=newShow,
            ASSET=self.arg.asset,
            BRANCH=self.arg.branch,
            HAIR=hairPath,
            VERSIONEXP=self.arg.versionExp)
        subprocess.Popen(cmd, shell=True).wait()

    def getHairScene(self, path):
        rigVer = ''
        mayapath = ''

        orgasset = self.arg.orgPath.split('.')[0].split('/')[-1]
        orgdir = self.arg.orgPath.split('/' + orgasset)[0]
        with utl.OpenStage(path) as stage:
            dPrim = stage.GetDefaultPrim()
            var = dPrim.GetVariantSets().GetAllVariantSelections()
            if var.get('zennVersion'):
                zennVer = var['zennVersion']
                mayapath = os.path.join(orgdir, orgasset, 'zenn', 'scenes', '%s_hair_%s.mb' % (orgasset, zennVer))
                if not os.path.exists(mayapath):
                    files = os.listdir(os.path.join(orgdir, orgasset, 'zenn', 'scenes'))
                    files.sort()
                    mayapath = os.path.join(orgdir, orgasset, 'zenn', 'scenes', files[-1])

            elif var.get('task'):
                layer = utl.AsLayer(path)
                dprim = layer.defaultPrim
                prim = layer.GetPrimAtPath('/' + dprim)
                vspec = prim.variantSets.get('task')
                data = vspec.variants
                # print(data)
                if 'groom' in data.keys():
                    task = 'groom'
                    groompath = os.path.join(orgdir, orgasset, task, '%s.usd' % task)
                    stage = Usd.Stage.Open(groompath)
                    dPrim = stage.GetDefaultPrim()
                    var = dPrim.GetVariantSets().GetAllVariantSelections()
                    groomVer = var['groomVer']
                    mayapath = os.path.join(orgdir, orgasset, task, 'scenes', '%s.mb' % groomVer)

        return mayapath

    def Arguing(self):
        return

    def Tweaking(self):
        return

    def Compositing(self):
        return


# ------------------------------------------------------------------------------


# TextureCopy


# ------------------------------------------------------------------------------

class TextureCopy:
    def __init__(self, arg):
        self.arg = arg
        self.textureTasks = ['tex', 'proxy', 'images']

        utl.MakeDir(self.arg.maindir)
        utl.CopyFile(os.path.join(os.path.dirname(self.arg.orgPath), 'preview.jpg'), self.arg.maindir)
        
        if self.arg.versionExp:
            # version copy
            self.versionCopy()

        else:
            # single copy 
            dstver = 'v001'
            texdir = os.path.join(self.arg.maindir, 'texture', 'tex', dstver)
            proxydir = os.path.join(self.arg.maindir, 'texture', 'proxy', dstver)
            self.Doit(texdir, dstver)
            utl.texAttr(texdir)
            utl.proxyMtl(proxydir)


    def versionCopy(self):
        for type in self.textureTasks:
            typedir = os.path.join(os.path.dirname(self.arg.orgPath), 'texture', type) #texture/tex
            dsttypedir = os.path.join(self.arg.maindir,'texture', type)
            utl.MakeDir(dsttypedir)
            if type == 'tex' or type == 'proxy':
                typeusd = os.path.join(typedir,'%s.usd' %type)
                utl.CopyFile(typeusd, dsttypedir)

            for ver in os.listdir(typedir):
                sourcedir = os.path.join(typedir, ver) #texture/tex/v001
                dstdir = os.path.join(dsttypedir, ver)
                if os.path.isdir(sourcedir):
                    if len(os.listdir(sourcedir)) > 1:
                        utl.MakeDir(dstdir)

                    for filename in os.listdir(sourcedir):
                        source = os.path.join(sourcedir, filename)
                        utl.CopyFile(source , dstdir)
                        if 'tex.attr.usd' in filename:
                            self.editTexAttr(os.path.join(dstdir,filename))

    def editTexAttr(self, texattrfile):
        lyr = utl.AsLayer(texattrfile)
        for spec in lyr.rootPrims:
            # var.T.ATTR_TXBASEPATH # 'primvars:txBasePath'
            if spec.properties.get(var.T.ATTR_TXBASEPATH):
                del spec.properties[var.T.ATTR_TXBASEPATH]
            attrSpec = Sdf.AttributeSpec(spec, var.T.ATTR_TXBASEPATH, Sdf.ValueTypeNames.String)
            attrSpec.default = utl.GetBasePath(self.arg.asset, self.arg.branch)
            attrSpec.SetInfo('interpolation', 'constant')
            
        lyr.Save()
        del lyr

    def Doit(self,texdir, dstver):
        lyr = utl.AsLayer(self.arg.orgPath)
        with utl.OpenStage(lyr) as stage:
            treeIter = iter(Usd.PrimRange.AllPrims(stage.GetPseudoRoot()))
            treeIter.next()
            for p in treeIter:
                txPath = p.GetAttribute('primvars:txBasePath').Get()
                txLayer = p.GetAttribute('primvars:txLayerName').Get()
                if txPath:
                    if txLayer:
                        try:
                            texorgVer = self.getTexVer(self.arg.orgShow, txPath)
                        except:
                            return
                            # texorgVer = 'v001'

                        for t in self.textureTasks:
                            sourcedir = os.path.join(self.arg.orgShow, txPath, t, texorgVer)
                            dstdir = os.path.join(self.arg.maindir, 'texture', t, dstver)

                            if t == 'proxy':
                                self.proxy(sourcedir, dstdir, txLayer, texdir)
                            else:
                                self.copy(sourcedir, dstdir, txLayer)


    def proxy(self, sourcedir,dstdir,txLayer,texdir):
        if os.path.isdir(sourcedir):
            if len(os.listdir(sourcedir)) > 1:
                self.copy(sourcedir, dstdir, txLayer)
            else:
                self.proxyMake(texdir, dstdir, txLayer)
        else:
            self.proxyMake(texdir, dstdir, txLayer)


    def copy(self, sourcedir, dstdir, txLayer):

        if not os.path.exists(sourcedir):
            return

        if os.path.isdir(sourcedir):
            utl.MakeDir(dstdir)

        for fileName in os.listdir(sourcedir):
            if txLayer in fileName:
                source = os.path.join(sourcedir, fileName)
                try:
                    path = os.path.join(dstdir, fileName)
                    if not os.path.exists(path):
                        utl.CopyFile(source, dstdir)
                except:
                    pass


    def getTexVer(self, orgshow, txpath):
        texorgVer = ''
        if '_3d' in self.arg.orgPath:
            texusdfile = os.path.join(orgshow, txpath, 'tex', 'tex.usd')
            texdir = os.path.join(orgshow, txpath, 'tex')
        else:
            texusdfile = os.path.join(orgshow, txpath, 'tex', 'tex.attr.usd')
            texdir = os.path.join(orgshow, txpath, 'tex')

        if os.path.exists(texusdfile):
            layer = utl.AsLayer(texusdfile)
            texorgVer = layer.subLayerPaths[0].split('/')[1]

        if not texorgVer:
            verList = []
            for file in os.listdir(os.path.join(texdir)):
                if 'v' in file:
                    verList.append(file)
            verList.sort()
            texorgVer = verList[-1]
        return texorgVer


    def proxyMake(self, orgdir, proxydir, txLayer):
        
        utl.MakeDir(proxydir)
        proxyResolution = 512
        for fileName in os.listdir(orgdir):
            if not "diffC" in fileName:
                continue
            if not '_diffC_' in fileName:
                if txLayer in fileName:
                    orgfilepath = os.path.join(orgdir, fileName)
                    splitExtStr = list(os.path.splitext(fileName))
                    splitExtStr[-1] = ".jpg"
                    fileName = "".join(splitExtStr)
                    newFilepath = os.path.join(proxydir, fileName)
                    if not os.path.exists(newFilepath):
                        loadImg = ice.Load(orgfilepath)
                        # loadImg = self.srgb2lin(loadImg)
                        loadImg.Save(newFilepath, ice.constants.FMT_JPEG)

                        cmd = "convert {input} -resize {width} {output}".format(input=newFilepath,
                                                                                output=newFilepath,
                                                                                width=proxyResolution)
                        os.system(cmd)
                        self.deleteCache()

    def deleteCache(self):
        deleteCachePath = '/tmp'
        pattern = 'IceCachedImage'
        files = glob.glob('%s/%s*' % (deleteCachePath, pattern))
        for i in files:
            suCmd = 'rm -rf %s' % i
            os.system(suCmd)

        msg.debug('deleted caches')

    def srgb2lin(self, v):
        base = ice.Card(ice.constants.FLOAT, [0.0404482362771082])

        c1 = ice.Card(ice.constants.FLOAT, [12.92])
        c2 = ice.Card(ice.constants.FLOAT, [1.055])
        c3 = ice.Card(ice.constants.FLOAT, [2.4])
        c4 = ice.Card(ice.constants.FLOAT, [0.055])

        t1 = v.Add(c4).Divide(c2)
        t2 = t1.Pow(c3)

        t3 = v.Divide(c1)

        m = v.Gt(base)
        t4 = t2.Multiply(m)

        result = t4.Add(t3.Multiply(v.Le(base)))
        return result



# ------------------------------------------------------------------------------


# USD Database


# ------------------------------------------------------------------------------

class Database:
    def __init__(self, arg):
        self.arg = arg

        usdpath = os.path.join(self.arg.customdir, 'asset', self.arg.asset, self.arg.asset + '.usd')
        previewpath = os.path.join(self.arg.customdir, 'asset', self.arg.asset, 'preview.jpg')
        new = self.arg.asset
        org = self.arg.orgAsset

        if self.arg.branch:
            usdpath = os.path.join(self.arg.customdir, 'asset', self.arg.asset, 'branch', self.arg.branch,
                                   self.arg.branch + '.usd')
            previewpath = os.path.join(self.arg.customdir, 'asset', self.arg.asset, 'branch', self.arg.branch,
                                       'preview.jpg')

            new = self.arg.branch
            org = self.arg.orgBranch

        if os.path.exists(usdpath):
            item = gDB.item.find_one({"name": org})
            if item:
                item["files"]["preview"] = previewpath
                item["files"]["usdfile"] = usdpath
                gDB.item.update({'name': new}, {"$set": item})
                print('success: DBpath changed')

            else:
                # cmd = '/WORK_DATA/Develop/dcc/DCC dev rez-env assetbrowser pyside2 -- additem %s' % self.arg.maindir
                cmd = '/backstage/dcc/DCC rez-env assetbrowser pyside2 -- additem %s' % self.arg.maindir
                subprocess.Popen(cmd, shell=True).wait()
                # print('success: Added to DB ')

# ------------------------------------------------------------------------------


# Comp


# ------------------------------------------------------------------------------
#
# class AComp(AExport):
#     def __init__(self, **kwargs):
#         self.usdpath = ''
#         self.show = ''
#         self.overwrite = True
#         self.ver = 'v001'
#
#         # initialize
#         AExport.__init__(self, **kwargs)
#         # attributes
#         self.task = 'model'
#         self.taskProduct = 'TASKV'
#
#     def Treat(self):
#         assetdir = os.path.dirname(self.usdpath)
#         self.D.SetDecode(assetdir)
#         self.dstdir = self.D[self.taskProduct]
#         self.master = utl.SJoin(self.dstdir, self.F.MASTER)
#
#
# class Comp(Export):
#     ARGCLASS = AComp
#
#     def Compositing(self):
#         for file in os.listdir(os.path.dirname(self.arg.usdpath)):
#             if 'model' in file and os.path.exists(self.arg.master):
#                 cmp.Composite(self.arg.master).DoIt()
#                 return var.SUCCESS
#             else:
#                 print('branch asset')