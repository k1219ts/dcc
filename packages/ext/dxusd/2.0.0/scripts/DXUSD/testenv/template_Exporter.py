#coding:utf-8
from __future__ import print_function

from .Export import Export, AExport
from DXUSD.Structures import Layers

import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


'''
class DCC_TempExport(exp.TempExport):
    def Exporting(self):
        # export something from DCC
        return var.SUCCESS

if __name__ == '__main__':
    # Set arguments
    arg = exp.AModelExport(show='pipe')
    arg.ver   = 'v001'
    arg.srclyr.name = 'lion_model_GRP'

    arg.srclyr[0] = True

    # Run Exporter
    DCC_ModelExport(arg)
'''


class ATempExport(AExport):
    def __init__(self, **kwargs):
        '''
        [Input Arguments]
        0-SourceLayer (bool)   : Input Source Layer
        '''
        # initialize
        AExport.__init__(self, **kwargs)

        # pre flags
        self.task = 'model'

        # attributes
        self.nameProduct = 'SRC_MODEL'
        self.taskProduct = 'TASKV'

        # set srclyr
        self.srclyr.AddLayer('SourceLayer')


    def Treat(self):
        res = AExport.Treat(self)
        if res != var.SUCCESS:
            return res

        # ----------------------------------------------------------------------
        # Set output layer
        dstdir       = self.D[self.taskProduct]
        self.dstlyr  = utl.SJoin(dstdir, self.F.FINAL)

        # ----------------------------------------------------------------------
        # Set source layers
        self.srclyr['SourceLayer'] = utl.SJoin(dstdir, self.F.GEOM)

        return var.SUCCESS


class TempExport(Export):
    ARGCLASS = AModelExport

    def Arguing(self):
        # ----------------------------------------------------------------------
        # Declare tweaker arguments
        self.snArg = twk.AGeomAttrs(**self.arg)

        # ----------------------------------------------------------------------
        # compositor
        return var.SUCCESS


    def Tweaking(self):
        twks = twk.Tweak()

        twks << twk.StripNamespace(self.snArg)
        twks << twk.NurbsToBasis(self.arg)

        twks.DoIt()
        return var.SUCCESS


    def Compositing(self):
        return var.SUCCESS
