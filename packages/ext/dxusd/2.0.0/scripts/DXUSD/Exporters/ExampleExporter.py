#coding:utf-8
from __future__ import print_function

from .Export import Export, AExport
from DXUSD.Structures import Layers

import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

'''
import DXUSD.Tweakers as twk
import DXUSD.Exporters as exp
import DXUSD_MAYA.Example as test
import DXUSD.Arcs as arc

reload(twk)
reload(exp)
reload(test)
reload(arc)

import maya.cmds as cmds
cmds.select('example_model_GRP')
test.Export()
'''

class AExampleExporter(AExport):
    def __init__(self, **kwargs):
        # initialize
        AExport.__init__(self, **kwargs)

        # pre flags
        self.task = 'model'

        # attributes
        self.nameProduct = 'SRC_MODEL'
        self.taskProduct = 'TASKV'

        # set srclyr
        self.srclyr.AddLayer('place')
        self.srclyr.AddLayer('mesh')


    def Treat(self):
        # ----------------------------------------------------------------------
        # Decode srclyr's name
        try:
            self.N.SetDecode(self.srclyr.name, self.nameProduct)
        except:
            msg.errmsg('Cannot decode given name(%s)'%self.srclyr.name)
            return var.FAILED

        # ----------------------------------------------------------------------
        # check arguments
        res = self.CheckArguments(self.taskProduct)
        if res != var.SUCCESS:
            return res

        # ----------------------------------------------------------------------
        # Set layers
        dstdir       = self.D[self.taskProduct]
        self.srclyr.place = utl.SJoin(dstdir, '%s_place.usd'%self.asset)
        self.srclyr.mesh  = utl.SJoin(dstdir, '%s_mesh.usd'%self.asset)

        self.dstlyr  = utl.SJoin(dstdir, self.F.FINAL)

        return var.SUCCESS


class ExampleExporter(Export):
    ARGCLASS = AExampleExporter

    def Arguing(self):
        # ----------------------------------------------------------------------
        # tweaker argument
        self.emArg = twk.AExample_ExtractMesh(**self.arg)
        self.emArg.inputs[0]  = self.arg.srclyr.place
        self.emArg.outputs[0] = self.arg.srclyr.mesh

        self.rxArg = twk.AExample_RefereceToXform(**self.arg)
        self.rxArg.inputs.mesh  = self.arg.srclyr.mesh
        self.rxArg.inputs.place = self.arg.srclyr.place

        self.pgArg = twk.AExample_PackGeom(**self.arg)
        self.pgArg.inputs[0]  = self.arg.srclyr.place
        self.pgArg.outputs[0] = self.arg.dstlyr

        # ----------------------------------------------------------------------
        # compositor argument

        return var.SUCCESS


    def Tweaking(self):
        twks = twk.Tweak()

        twks << twk.Example_ExtractMesh(self.emArg)
        twks << twk.Example_RefereceToXform(self.rxArg)
        twks << twk.Example_PackGeom(self.pgArg)

        twks.DoIt()
        return var.SUCCESS


    def Compositing(self):
        return var.SUCCESS
