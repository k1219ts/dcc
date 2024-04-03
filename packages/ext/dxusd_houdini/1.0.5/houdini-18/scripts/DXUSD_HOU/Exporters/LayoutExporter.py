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


class ALayoutExporter(AExport):
    def __init__(self, **kwargs):
        # # self.geomfiles = []
        # self.instfiles = []
        #
        # # initialize
        # AExport.__init__(self, **kwargs)
        #
        # # attributes
        # self.task = var.T.LAYOUT
        # initialize
        AExport.__init__(self, **kwargs)

        # set default values
        self.task        = var.T.MODEL
        self.taskCode    = 'TASKVS'
        self.taskProduct = 'GEOM'
        self.lyrtype     = var.LYRGEOM

        # add attrs for other layers
        self.geoms = srt.Layers()
        self.geoms.AddLayer('high')
        self.geoms.AddLayer('low')

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
        for lod in [var.T.HIGH, var.T.LOW]:
            self.lod = lod
            path = self.D[self.taskCode]
            file = self.F.model.LOD
            self.geoms[lod] = utl.SJoin(path, file)

        # ----------------------------------------------------------------------
        # set default prim

        if self.subdir:
            self.dprim = Sdf.Path('/%s'%self.subdir)
        else:
            msg.errmsg('Cannot find defaultPrim')
            return var.FAILED

        return var.SUCCESS


class LayoutExporter(Export):
    ARGCLASS = ALayoutExporter

    def Arguing(self):
        self.cmArg = twk.ACombineLayers(**self.arg.AsDict())
        self.cmArg.inputs.Append(self.arg.srclyr)

        self.cmArg_high = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_high.rules.append(['/Geom/high/%s'%self.arg.subdir, '/'])
        self.cmArg_high.outputs.Append(self.arg.geoms[var.T.HIGH])

        self.cmArg_low = twk.ACombineLayers(**self.cmArg.AsDict())
        self.cmArg_low.rules.append(['/Geom/low/%s' % self.arg.subdir, '/'])
        self.cmArg_low.outputs.Append(self.arg.geoms[var.T.LOW])

        self.mArg = twk.AMasterModelPack(**self.arg)
        self.mArg.master= self.arg.dstlyr

        return var.SUCCESS

    def Tweaking(self):
        # 1
        twks = twk.Tweak()
        for arg in [self.cmArg_high,self.cmArg_low]:
            twks << twk.CombineLayers(arg)
        twks << twk.MasterModelPack(self.mArg)  # master model geom package
        twks.DoIt()

        # 2
        twks = twk.Tweak()
        # twks << twk.PrmanMaterial(self.gArg)                # create prman material
        twks << twk.Collection(self.mArg)                    # create collection by master
        twks << twk.PrmanMaterialOverride(self.mArg)         # referenced asset material override
        twks.DoIt()

        return var.SUCCESS



    def Compositing(self):
        # print(self.arg.master)
        # cmp.Composite(self.arg.master).DoIt()
        return var.SUCCESS













#