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


class AInstanceExporter(AExport):
    def __init__(self, **kwargs):
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


    def Treat(self):
        print('self.taskProduct:',self.taskProduct)
        # ----------------------------------------------------------------------
        # check arguments and source layer
        if not self.CheckSourceLayer(self.taskProduct):
            msg.errmsg('Failed "CheckSourceLayer"')
            return var.FAILED
        self.taskCode = self.taskCode[:-1]
        print('self.taskCode:',self.taskCode)
        # ----------------------------------------------------------------------
        # set destination layers
        if not self.SetDestinationLayer('MASTER'):
            msg.errmsg('Failed "SetDestinationLayer"')
            return var.FAILED

        # ----------------------------------------------------------------------
        # check sequenced
        if not self.SetSequenced():
            msg.errmsg('Failed "SetSequenced"')
            return var.FAILED

        # ----------------------------------------------------------------------
        # set other layers
        # high, low, guide geom

        # find high, low, guide prim to set LOD
        if not self.LODs:
            self.LODs = [var.T.HIGH]

        for lod in self.LODs:
            self.lod = lod
            path = self.D[self.taskCode]
            if self.task == var.T.MODEL:
                file = self.F[self.task].LOD
            else:
                file = self.F[self.task].GEOM

            self.geoms[lod] = utl.SJoin(path, file)

        # ----------------------------------------------------------------------
        # set default prim

        if self.sequenced:
            if self.nslyr:
                self.dprim = Sdf.Path('/%s'%self.nslyr)
            else:
                msg.errmsg('Cannot find defaultPrim')
                return var.FAILED
        else:
            self.dprim = Sdf.Path('/Geom')

        return var.SUCCESS


class InstanceExporter(Export):
    ARGCLASS = AInstanceExporter
    def Arguing(self):
        # combines
        self.cmArg = twk.ACombineLayers(**self.arg.AsDict())
        self.cmArg.inputs.Append(self.arg.srclyr)
        geomfiles = []
        self.cmArgs = dict()
        for lod in self.arg.LODs:
            self.cmArgs[lod] = twk.ACombineLayers(**self.cmArg.AsDict())
            print('>>>>>self.arg.subdir:',self.arg.subdir)
            s = "(/Geom/%s/%s)"%(lod, str(self.arg.subdir))
            d = '/Geom'
            self.cmArgs[lod].rules.append([s, d])
            self.cmArgs[lod].outputs.Append(self.arg.geoms[lod])
            geomfiles.append(self.arg.geoms[lod])

        # dxusd
        self.mArg = twk.AMasterPack(**self.arg)
        self.mArg.master = self.arg.dstlyr

        self.pArg = twk.APrmanMaterial()        # PrmanMaterial arguments
        self.pArg.inputs = geomfiles

        # delete source layers
        self.rmArg = twk.ARemoveLayers()
        self.rmArg.inputs.Append(self.arg.srclyr)

        return var.SUCCESS


    def Tweaking(self):
        # ----------------------------------------------------------------------
        # Combining
        # # ----------------------------------------------------------------------
        twks = twk.Tweak()
        for arg in self.cmArgs.values():
            twks << twk.CombineLayers(arg)
        twks << twk.MasterModelPack(self.mArg)
        twks.DoIt()

        # ----------------------------------------------------------------------
        # material tweaks for model task
        twks = twk.Tweak()
        twks << twk.PrmanMaterial(self.pArg)
        twks << twk.Collection(self.mArg)
        twks << twk.PrmanMaterialOverride(self.mArg)
        twks.DoIt()

        return var.SUCCESS


    def Compositing(self):
        cmp.Composite(self.arg.dstlyr).DoIt()
        return var.SUCCESS
