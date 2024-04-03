#coding:utf-8
from __future__ import print_function

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg
import DXUSD.Compositor as cmp
from DXUSD.Exporters.Export import Export, AExport

import DXUSD_HOU.Tweakers as twk

class AModelExporter(AExport):
    def __init__(self, **kwargs):
        # export source data
        self.geomfiles = []     # export geom filename.
        self.instfiles = []
        self.master = ''
        self.overwrite = False

        # initialize
        AExport.__init__(self, **kwargs)

        # attributes
        self.task = var.T.MODEL
        self.taskProduct = 'TASKV'

    def Treat(self):
        try:
            self.dstdir = self.D[self.taskProduct]
            self.master = utl.SJoin(self.dstdir, self.F.MASTER)
        except:
            msg.errmsg('Failed to encode args.\n', self)
            return var.IGNORE

        return var.SUCCESS


class ModelExporter(Export):
    ARGCLASS = AModelExporter
    def Arguing(self):
        self.fparg = twk.AFixCompositePath(**self.arg)
        self.fparg.inputs = self.arg.geomfiles + self.arg.instfiles
        return var.SUCCESS

    def Tweaking(self):
        twks = twk.Tweak()
        # geom package
        twks << twk.FixCompositePath(self.fparg)
        twks << twk.MasterModelPack(self.arg)
        twks.DoIt()
        return var.SUCCESS

    def Compositing(self):
        cmp.Composite(self.arg.master).DoIt()
        return var.SUCCESS
