#coding:utf-8
from __future__ import print_function
import os

from pxr import Sdf, Usd

from .Tweaker import Tweaker, ATweaker

import DXUSD.Arcs as arc
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

'''
arg = twk.ATemp()
arg.inputs[0]  = '/show/pipe/_3d/asset/lion/texture/tex/v002/tex.attr.usd'
arg.outputs[0] = '/show/pipe/_3d/asset/lion/texture/tex/tex.usd'
arg.Treat()
twk.Temp(arg).DoIt()
'''

class ATemp(ATweaker):
    '''
    [Input Layers]
    0-input (str) : input layer

    [Output Layers]
    0-output (str): output layer
    '''
    def __init__(self, **kwargs):
        # initialize
        ATweaker.__init__(self, **kwargs)

        # add input and output attributes
        self.inputs.AddLayer('input')
        self.outputs.AddLayer('output')

    def Treat(self):
        # inputs, outpus 확인
        if not self.inputs[0]:
            msg.warning('Treat@%s'%self.__name__, 'No inputs.')
            return var.IGNORE
        if not self.outputs[0]:
            msg.warning('Treat@%s'%self.__name__, 'No outputs.')
            return var.IGNORE

        return var.SUCCESS

class Temp(Tweaker):
    ARGCLASS = ATemp
    def DoIt(self):
        # do somthing for outputs
        return var.SUCCESS
