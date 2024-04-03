#coding:utf-8
from __future__ import print_function

import DXUSD.Message as msg
import DXUSD.Compositor as cmp
from DXUSD.Exporters.Export import Export, AExport

import DXUSD_HOU.Tweakers as twk
import DXUSD_HOU.Structures as srt
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var


class AFeatherExporter(AExport):
    def __init__(self, **kwargs):
        '''
        [Attributes]
        *srclyr (str)      : input usd layer
        *dependency (dict) : depend usd path and variants
            { 'asset': { __USDPATH__:'/show/pipe/...',
                         __ORDER__:['worldXform', 'preview'],
                         'worldXform':'on',
                         'preview':'off' },
              'shot': { ... },
              'branch': { ... } }

        dstlyr (str)     : master usd path
        customData (dict): custom layer data
        animation (bool) : animation cache or not

        dependRigVer : depend rig version

        '''
        # public attributes
        self.taskCode   = 'TASKN'
        self.dependency = {}

        # initialize
        AExport.__init__(self, **kwargs)

        # privite attributes
        self.task = var.T.FEATHER
        self.nameProduct = 'GEOM'

        self.srclyr = None
        self.dstlyr = None

        self.geoms = srt.Layers()
        self.geoms.AddLayer('high')
        self.geoms.AddLayer('low')
        self.geoms.AddLayer('guide')

        self.animation = False
        self.customData = dict()
        self.dependRigVer = None
        self.dependFeather = None

    def Treat(self):
        if not self.srclyr:
            msg.errmsg('No srclyr')
            return var.FAILED
        # ----------------------------------------------------------------------
        # check arguments
        res = self.CheckArguments(self.taskCode)
        if res != var.SUCCESS:
            # Decode srclyr's name
            try:
                self.F.SetDecode(utl.BaseName(self.srclyr),
                                 self.nameProduct)
                self.D.SetDecode(utl.DirName(self.srclyr),
                                 '%sS'%self.taskCode)
            except Exception as e:
                msg.errmsg(e)
                msg.errmsg('Cannot decode srclyr(%s)'%self.srclyr)
                return var.FAILED

        # ----------------------------------------------------------------------
        # set layers
        # feather task code is NS, VNVS(fx) or NVS(clip) but master file will be
        # in N, VNV(fx) or NV(clip).
        self.dstlyr  = utl.SJoin(self.D[self.taskCode], self.F.MASTER)

        # high, low, guide geom
        for lod in [var.T.HIGH, var.T.LOW, var.T.GUIDE]:
            self.lod = lod
            self.geoms[lod] = utl.SJoin(self.D[self.taskCode], self.F.LOD)

        # ----------------------------------------------------------------------
        # check dependency.

        # dependency의 asset, branch 정보에서 feather 작업시 가져온 어셋정보에서 해당
        # rig version을 가져오고, 만약 없으면, (aset)_rig_v000 형식으로 셋팅한다.
        # 만약 model에서 작업을 시작했다면, USDPATH는 해당 model layer 경로를 알려준다.
        # USDPATH 경로가 없을시엔 (페더만 독립적으로 뽑을경우), USDPATH는 rig 버전을
        # 기반으로해서 rig layer 경로를 만들어 넣어준다.
        assetdata = {}
        if self.dependency.has_key(var.T.BRANCH):
            assetdata = self.dependency[var.T.BRANCH]
        elif self.dependency.has_key(var.T.ASSET):
            assetdata = self.dependency[var.T.ASSET]

        if assetdata.has_key(var.T.VAR_RIGVER):
            self.dependRigVer = assetdata[var.T.VAR_RIGVER]
        else:
            _ = '%s_%s_%s'%(self.ABName(), var.T.RIG, utl.Ver(0))
            self.dependRigVer = _

        if assetdata.has_key(var.USDPATH) and assetdata[var.USDPATH]:
            rigfile = assetdata[var.USDPATH]
        else:
            rigfile = utl.SJoin(self.D.ASSET, var.T.RIG, self.dependRigVer)
            rigfile = utl.SJoin(rigfile, self.F.rig.MASTER)

        # shot 정보가 들어가 있으면, animation 체크를 하고, 사용하는 feather 어셋을
        # 찾아야 하고, 만약 없으면, Failed 처리 한다.
        if self.dependency.has_key(var.T.SHOT):
            self.animation = True
            shotdata = self.dependency[var.T.SHOT]

            if not (shotdata.has_key(var.USDPATH) and shotdata(var.USDPATH)):
                msg.errmsg('Need dependency usd layer path')
                return var.FAILED

            inputcache = shotdata[var.USDPATH]

            # TODO: find feather file
            featherfile = ''

        # customData 설정
        if self.animation:
            self.customData[var.T.CUS_INPUTCACHE] = inputcache
            self.customData[var.T.CUS_FEATHERFILE] = featherfile
        else:
            self.customData[var.T.CUS_RIGFILE] = rigfile

        return var.SUCCESS


class FeatherExporter(Export):
    ARGCLASS = AFeatherExporter

    def Arguing(self):
        self.cmArg = twk.ACombineLayers(**self.arg)
        self.cmArg.inputs.Append(self.arg.srclyr)

        self.cmArg_high = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_high.rules.append(['Laminations', None])
        self.cmArg_high.rules.append(['/.*/Feather', None])
        self.cmArg_high.outputs.Append(self.arg.geoms.high)

        self.cmArg_low = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_low.rules.append(['Proxy=Laminations', None])
        self.cmArg_low.rules.append(['/.*/Feather', None])
        self.cmArg_low.outputs.Append(self.arg.geoms.low)

        self.cmArg_guide = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_guide.rules.append(['Guides', None])
        self.cmArg_guide.outputs.Append(self.arg.geoms.guide)

        self.mArg = twk.AMasterFeather(**self.arg)
        self.mArg.inputs.high  = self.arg.geoms.high
        self.mArg.inputs.low   = self.arg.geoms.low
        self.mArg.inputs.guide = self.arg.geoms.guide
        self.mArg.outputs.Append(self.arg.dstlyr)
        self.mArg.isShot = self.arg.animation

        return var.SUCCESS


    def Tweaking(self):
        # Packing
        twks = twk.Tweak()

        twks << twk.CombineLayers(self.cmArg_high)
        twks << twk.CombineLayers(self.cmArg_low)
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
        # cmp.Composite(self.arg.master).DoIt()
        return var.SUCCESS
