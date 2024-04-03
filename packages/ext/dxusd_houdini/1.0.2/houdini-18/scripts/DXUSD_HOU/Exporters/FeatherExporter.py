#coding:utf-8
from __future__ import print_function

import DXRulebook as rb
import DXUSD.Message as msg
import DXUSD.Compositor as cmp

from DXUSD_HOU.Exporters.Export import Export, AExport

import DXUSD_HOU.Tweakers as twk
import DXUSD_HOU.Structures as srt
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var


class AFeatherExporter(AExport):
    def __init__(self, **kwargs):
        # initialize
        AExport.__init__(self, **kwargs)

        # set default values
        self.task        = var.T.GROOM
        self.taskCode    = 'TASKNS'
        self.taskProduct = 'GEOM'
        self.lyrtype     = var.LYRFEATHER

        # add attrs for other layers
        self.geoms = srt.Layers()
        self.geoms.AddLayer('high')
        self.geoms.AddLayer('low')
        self.geoms.AddLayer('guide')

        # dependency attrs
        self.dependRigVer = None
        self.dependOrgFeather  = None
        self.dependHighFeather = None
        self.dependLowFeather  = None

    def Treat(self):
        # ----------------------------------------------------------------------
        # check arguments and source layer
        if not self.CheckSourceLayer(self.taskProduct):
            return var.FAILED

        # feather의 원본 layer는 항상 마스터 layer 경로 하위 폴더안에 들어 있다.
        # 예를 들어, asset일 경우 taskCode가 TASKNVS 이고, 원본 layer는 TASKVNS
        # 경로에, 마스터 layer는 TASKVN 이 된다. 따라서, srclyr를 확인한 이후에는
        # 마지막 S를 제거 한다.
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
        # dependency의 asset, branch 정보에서 feather 작업시 가져온 어셋정보에서 해당
        # rig version을 가져오고, 만약 없으면, (aset)_rig_v000 형식으로 셋팅한다.
        # 만약 model에서 작업을 시작했다면, USDPATH는 해당 model layer 경로를 알려준다.
        # USDPATH 경로가 없을시엔 (페더만 독립적으로 뽑을경우), USDPATH는 rig 버전을
        # 기반으로해서 rig layer 경로를 만들어 넣어준다.

        rigFile = ''
        featherFile = ''
        inputCache = ''

        # find rig file
        adata = {}
        if self.dependency.has_key(var.T.BRANCH):
            adata = self.dependency[var.T.BRANCH]
        elif self.dependency.has_key(var.T.ASSET):
            adata = self.dependency[var.T.ASSET]

        if adata.has_key(var.T.RIG):
            rigfile = adata[var.T.RIG][var.USDPATH]
            self.dependRigVer = adata[var.T.RIG][var.T.VAR_RIGVER]
        elif adata.has_key(var.T.MODEL):
            args = var.D.Decode(utl.DirName(adata[var.T.MODEL][var.USDPATH]))
            args.task = var.T.RIG
            args.ver = utl.Ver(0)
            F = rb.Coder('F')
            self.dependRigVer = F.Maya[var.T.RIG].WORK.Encode(**args)
            self.dependRigVer = utl.FileName(F)

        # find feather file
        for kind in [var.T.BRANCH, var.T.ASSET]:
            if not self.dependency.has_key(kind):
                continue
            if self.dependency[kind].has_key(var.T.GROOM):
                featherFile = self.dependency[kind][var.T.GROOM][var.USDPATH]
                break

        # find input cache (ani or sim ...)
        if self.dependency.has_key(var.T.SHOT):
            for task in [var.T.FX, var.T.SIM, var.T.CLIP, var.T.ANI]:
                if self.dependency[var.T.SHOT].has_key(task):
                    inputCache = self.dependency[var.T.SHOT][task][var.USDPATH]

        # ----------------------------------------------------------------------
        # set custom data
        if self.sequenced:
            self.customData[var.T.CUS_INPUTCACHE] = inputCache
            self.customData[var.T.CUS_FEATHERFILE] = featherFile
        else:
            self.customData[var.T.CUS_RIGFILE] = rigFile

        # ----------------------------------------------------------------------
        # if sequenced, find feather's original asset layer.
        if self.sequenced:
            try:
                path = utl.DirName(featherFile)
                args = var.D.Decode(path)

                high = var.F[args.task].LOD.Encode(**args)
                low  = var.F[args.task].LOD.Encode(lod=var.T.LOW, **args)

                high = utl.AsLayer(utl.SJoin(path, high))
                low  = utl.AsLayer(utl.SJoin(path, low))
                if not high or not low:
                    raise Exception('Feather layer does not exist (%s)'%high)

                prim = high.GetPrimAtPath(high.defaultPrim)
                prim = prim.nameChildren[0].nameChildren[0]
                org  = prim.referenceList.prependedItems[0].assetPath
                org  = high.FindRelativeToLayer(high, org)

            except Exception as e:
                msg.errmsg(e)
                msg.errmsg('Failed finding feahter layers.')
                return var.FAILED

            self.dependOrgFeather  = org
            self.dependHighFeather = high
            self.dependLowFeather  = low

        return var.SUCCESS


class FeatherExporter(Export):
    ARGCLASS = AFeatherExporter

    def Arguing(self):
        # cmArg gets prims from srclyr, fmArg is from asset feather layer
        self.cmArg = twk.ACombineLayers(**self.arg)
        self.cmArg.inputs.Append(self.arg.srclyr)
        if self.arg.sequenced:
            self.cmArg.inputs.Append(self.arg.dependOrgFeather)

        self.cmArg_high = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_high.rules.append(['Laminations', None])
        self.cmArg_high.rules.append(['/.*/Feather', None, self.arg.sequenced])
        self.cmArg_high.outputs.Append(self.arg.geoms[var.T.HIGH])

        self.cmArg_low = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_low.rules.append(['Proxy=Laminations', None])
        self.cmArg_low.rules.append(['/.*/Feather', None, self.arg.sequenced])
        self.cmArg_low.outputs.Append(self.arg.geoms[var.T.LOW])

        self.cmArg_guide = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_guide.rules.append(['Guides', None])
        self.cmArg_guide.outputs.Append(self.arg.geoms[var.T.GUIDE])

        self.mArg = twk.AMasterFeather(**self.arg)
        self.mArg.inputs.high  = self.arg.geoms.high
        self.mArg.inputs.low   = self.arg.geoms.low
        self.mArg.inputs.guide = self.arg.geoms.guide
        self.mArg.outputs.Append(self.arg.dstlyr)
        self.mArg.sequenced = self.arg.sequenced

        if self.arg.sequenced:
            self.smArg_high = twk.AAddSublayers()
            self.smArg_high.inputs.Append(self.arg.dependHighFeather)
            self.smArg_high.outputs.Append(self.arg.geoms[var.T.HIGH])

            self.smArg_low = twk.AAddSublayers(**self.arg)
            self.smArg_low.inputs.Append(self.arg.dependLowFeather)
            self.smArg_low.outputs.Append(self.arg.geoms[var.T.LOW])

        return var.SUCCESS


    def Tweaking(self):
        # Packing
        twks = twk.Tweak()

        twks << twk.CombineLayers(self.cmArg_high)
        twks << twk.CombineLayers(self.cmArg_low)

        if self.arg.sequenced:
            twks << twk.AddSublayers(self.smArg_high)
            twks << twk.AddSublayers(self.smArg_low)
        else:
            twks << twk.CombineLayers(self.cmArg_guide)

        # twks << twk.CorrectBasisCurves(self.cbArg)

        twks << twk.MasterFeather(self.mArg)

        twks.DoIt()

        # # Shadering
        # twks = twk.Tweak()
        # for fn in self.texData:
        #     self.tArg.texAttrUsd = fn
        #     self.tArg.texData    = self.texData[fn]
        #
        #     twks << twk.Texture(self.tArg)          # create or update tex.attr.usd
        #     twks << twk.ProxyMaterial(self.tArg)    # create or update proxy.mtl.usd
        #
        # twks << twk.PrmanMaterial(self.pArg)        # create prman material
        # # twks << twk.GroomLayerCompTex(self.cmArg)   # composite tex.usd by assetInfo
        # twks << twk.GeomAttrsCompTex(self.gArg)
        # twks << twk.Collection(self.cArg)           # create collection by master
        #
        # twks.DoIt()

        return var.SUCCESS


    def Compositing(self):
        cmp.Composite(self.arg.dstlyr).DoIt()
        return var.SUCCESS
