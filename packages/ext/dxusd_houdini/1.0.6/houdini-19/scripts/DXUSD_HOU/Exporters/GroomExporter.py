#coding:utf-8
from __future__ import print_function

from pxr import Sdf

import DXRulebook as rb
import DXUSD_HOU.Message as msg
import DXUSD.Compositor as cmp
from DXUSD.Structures import Arguments

from DXUSD_HOU.Exporters.Export import Export, AExport

import DXUSD_HOU.Tweakers as twk
import DXUSD_HOU.Structures as srt
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var

import os
import shutil




class AGroomExporter(AExport):
    def __init__(self, **kwargs):
        # initialize
        AExport.__init__(self, **kwargs)

        # set default values
        self.task        = var.T.GROOM
        self.taskCode    = 'TASKNS'
        self.taskProduct = 'GEOM'
        self.lyrtype     = var.LYRGROOM

        # add attrs for other layers
        self.geoms = srt.Layers()
        self.geoms.AddLayer('high')
        self.geoms.AddLayer('low')
        self.geoms.AddLayer('guide')

        # dependency attrs
        self.dependRigVer = None
        self.dependOrgGroom  = None
        self.dependHighGroom = None
        self.dependLowGroom  = None


    def Treat(self):
        # ----------------------------------------------------------------------
        # check arguments and source layer
        if not self.CheckSourceLayer(self.taskProduct):
            return var.FAILED

        self.taskCode = self.taskCode[:-1]

        # ----------------------------------------------------------------------
        # set destination layers
        if not self.SetDestinationLayer('MASTER'):
            return var.FAILED

        # ----------------------------------------------------------------------
        # check sequenced
        if not self.SetSequenced():
            return var.FAILED

        # ----------------------------------------------------------------------
        # set other layers
        # high, low, guide geom
        for lod in [var.T.HIGH, var.T.LOW, var.T.GUIDE]:
            self.lod = lod
            path = self.D[self.taskCode]
            file = self.F.groom.LOD
            self.geoms[lod] = utl.SJoin(path, file)

        # ----------------------------------------------------------------------
        # check dependency to set customData.

        rigFile = self.FindRigFile()

        if not rigFile:
            return var.FAILED
        elif rigFile.task == var.T.MODEL:
            args = rigFile.CopyArgs()
            args.task = var.T.RIG
            args.ver = utl.Ver(0)
            self.dependRigVer = utl.FileName(args.F.MAYA.WORK)
        else:
            self.dependRigVer = rigFile.nslyr


        # ----------------------------------------------------------------------
        # set default prim
        if self.srclyr.defaultPrim:
            self.dprim = Sdf.Path('/%s'%self.srclyr.defaultPrim)
        elif self.srclyr.rootPrims:
            self.dprim = self.srclyr.rootPrims[0].path
        else:
            msg.errmsg('Cannot find defaultPrim')
            return var.FAILED

        customData = utl.AsLayer(self.srclyr).customLayerData
        print('customData:',customData)
        sceneFile = customData[var.T.CUS_SCENEFILE]


        # for shot
        if self.sequenced:
            groomFile     = self.FindGroomFile()
            inputCacheFile = self.FindInputCacheFile()
            if not groomFile:
                return var.FAILED

            # find high, low feather lod file for reference
            try:
                high = groomFile.arg.F.LOD
                groomFile.arg.lod = var.T.LOW
                low  = groomFile.arg.F.LOD
                groomFile.arg.pop('lod')
                high = utl.AsLayer(utl.SJoin(groomFile.dirname, high))
                low  = utl.AsLayer(utl.SJoin(groomFile.dirname, low))
                if not high or not low:
                    raise Exception('Groom layer does not exist (%s)'%high)

                # prim = high.GetPrimAtPath('/Groom')
                # prim = prim.nameChildren[0]
                # org  = prim.referenceList.prependedItems[0].assetPath
                # # org  = high.FindRelativeToLayer(high, org)
                # # print('org:', org)

            except Exception as e:
                msg.errmsg(e)
                msg.errmsg('Failed finding groom layers.')
                return var.FAILED

            # self.dependOrgGroom  = org
            self.dependHighGroom = high
            self.dependLowGroom  = low

            self.meta.customData[var.T.CUS_INPUTCACHE] = inputCacheFile.file
            self.meta.customData[var.T.CUS_GROOMFILE] = customData[var.T.CUS_GROOMFILE]

        #for asset
        else:
            self.SceneExport(sceneFile)


        return var.SUCCESS


    def SceneExport(self, sceneFile):
        # publish hip file
        arg = Arguments()
        arg.D.SetDecode(utl.DirName(self.FindRigFile().file))
        arg.task = 'groom'
        pubSceneFile = utl.SJoin(arg.D.TASK, 'scenes', self.nslyr + '.hip')


        if not os.path.exists(utl.DirName(pubSceneFile)):
            os.mkdir(utl.DirName(pubSceneFile))

        if os.path.exists(pubSceneFile):
            os.remove(pubSceneFile)

        shutil.copy(sceneFile, pubSceneFile)
        os.chmod(pubSceneFile, 0555)



class GroomExporter(Export):
    ARGCLASS = AGroomExporter
    def Arguing(self):
        self.texData   = dict()                   # CombineGroomLayers result for texture process
        self.meshFiles = list()
        self.cmArg = twk.ACombineLayers(**self.arg.AsDict())
        self.cmArg.inputs.Append(self.arg.srclyr)
        self.cmArgs = dict()
        geomfiles = []
        for lod in self.arg.LODs:
            self.cmArgs[lod] = twk.ACombineLayers(**self.cmArg.AsDict())
            s = "(/.*/(Geom|Render|Proxy))|(/Groom/%s/.*)" %lod
            # d = '/.' if self.arg.sequenced else '/Groom'
            d = '/Groom'
            self.cmArgs[lod].rules.append([s, d])
            self.cmArgs[lod].outputs.Append(self.arg.geoms[lod])
            geomfiles.append(self.arg.geoms[lod])

        self.cmArg_guide = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_guide.rules.append(['/.*/guides/.*', self.arg.dprim])
        self.cmArg_guide.outputs.Append(self.arg.geoms[var.T.GUIDE])

        if self.arg.sequenced:
            self.smArg_high = twk.AAddSublayers()
            self.smArg_high.inputs.Append(self.arg.dependHighGroom)
            self.smArg_high.outputs.Append(self.arg.geoms[var.T.HIGH])

            self.smArg_low = twk.AAddSublayers(**self.arg)
            self.smArg_low.inputs.Append(self.arg.dependLowGroom)
            self.smArg_low.outputs.Append(self.arg.geoms[var.T.LOW])

        self.mArg = twk.AMasterPack(**self.arg)
        self.mArg.master = self.arg.dstlyr

        self.tArg = twk.ATexture()              # Texture, ProxyMaterial arguments
        self.pArg = twk.APrmanMaterial()        # PrmanMaterial arguments
        self.pArg.inputs = geomfiles

        self.cArg = twk.ACollection()           # Collection arguments
        self.cArg.master = self.arg.dstlyr
        if self.arg.sequenced:
            groomusd = self.arg.dependHighGroom.realPath.replace('high.usd','usd')
            self.cArg.inputRigFile = groomusd


        return var.SUCCESS

    def Tweaking(self):
        #----------------------------------------------------------------------
        #Combining
        if self.arg.sequenced:
            # 1
            twks = twk.Tweak()
            for arg in self.cmArgs.values():
                twks << twk.CombineLayers(arg)
                twks << twk.HCombineGroomLayers(arg, self.texData)
            twks << twk.MasterGroomPack(self.mArg)
            twks.DoIt()

            # 2
            twks = twk.Tweak()
            twks << twk.Collection(self.cArg)  # create collection by master
            twks.DoIt()




        else:
            # 1
            twks = twk.Tweak()
            for arg in self.cmArgs.values():
                twks << twk.CombineLayers(arg)
                twks << twk.HCombineGroomLayers(arg, self.texData)
            twks << twk.CombineLayers(self.cmArg_guide)
            twks << twk.MasterGroomPack(self.mArg)
            twks.DoIt()
            # 2
            twks = twk.Tweak()
            for fn in self.texData:
                self.tArg.texAttrUsd = fn
                self.tArg.texData    = self.texData[fn]

                twks << twk.Texture(self.tArg)          # create or update tex.attr.usd
                twks << twk.ProxyMaterial(self.tArg)    # create or update proxy.mtl.usd

            twks << twk.PrmanMaterial(self.pArg)        # create prman material
            twks << twk.GroomLayerCompTex(self.cmArg)   # composite tex.usd by assetInfo
            twks << twk.Collection(self.cArg)           # create collection by master
            twks.DoIt()



        return var.SUCCESS

    def Compositing(self):
        cmp.Composite(self.arg.dstlyr).DoIt()
        return var.SUCCESS
