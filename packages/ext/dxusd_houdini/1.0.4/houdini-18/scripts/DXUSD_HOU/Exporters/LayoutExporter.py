#coding:utf-8
from __future__ import print_function

from DXUSD.Exporters.Export import Export, AExport
import DXUSD.Tweakers as twk
import DXUSD.Compositor as cmp

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg



class ALayoutExporter(AExport):
    def __init__(self, **kwargs):
        # self.geomfiles = []
        self.instfiles = []

        # initialize
        AExport.__init__(self, **kwargs)

        # attributes
        self.task = var.T.LAYOUT
        self.taskProduct = 'TASKNV'

    def Treat(self):

        return var.SUCCESS


class LayoutExporter(Export):
    ARGCLASS = ALayoutExporter

    def Arguing(self):
        
        return var.SUCCESS

    def Tweaking(self):
        # twks = twk.Tweak()
        # # geom package
        # twks << mtwk.ModifyMayaReference(self.arg)
        # # twks << twk.PackEnvGeom(self.arg)
        # twks.DoIt()
        return var.SUCCESS

    def Compositing(self):
        # print(self.arg.master)
        # cmp.Composite(self.arg.master).DoIt()
        return var.SUCCESS













#
