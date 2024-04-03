#coding:utf-8
from __future__ import print_function

import DXUSD.Utils as utl
from DXUSD.Exporters.Export import AExport as DXUSD_AExport
from DXUSD.Exporters.Export import Export as DXUSD_Export

import DXUSD_HOU.Vars as var

class AExport(DXUSD_AExport):
    def __init__(self, **kwargs):
        '''
        [Attributes from houdini]
        lyrtype (var.LYRTYPE): layer type (geom, inst, groom, feather, crwod...)
        prctype (var.PRCTYPE): post process type (none, clip, sim, fx)

        task (var.T.TASKS): task name
        taskCode (str)    : package task code (eg. 'TASKN') for rulebook

        srclyr (str)      : input usd layer
        cliprate (str)    : clip rate (eg. '0.8 1.0 1.2')
        dependency (dict) : depend usd path and variants
          {
            'asset/branch': {
                (task): { __USDPATH__:'/show/pipe/...',
                          __ORDER__:['vset1', 'vset2', ...],
                          'vset1':'on',
                          'vset2':'off' }, ...
            }
            'shot': { __USDPATH__:'/show/pipe/...',
                      __ORDER__:['vset1', 'vset2', ...],
                      'vset1':'on',
                      'vset2':'off' }, ...

          }

        [Attributes]
        sequenced (bool)  : if True, it's sequenced usd layer
        taskProduct (str) : task product name for rulebook

        dstlyr (str)     : master usd path
        customData (dict): custom layer data

        '''
        self.lyrtype = None
        self.prctype = var.PRCNONE
        self.sequenced = False

        self.task = None
        self.taskCode = None
        self.taskProduct = 'GEOM'

        self.cliprate = None
        self.dependency = None
        self.customData = dict()

        self.srclyr = None
        self.dstlyr = None

        DXUSD_AExport.__init__(self, **kwargs)

    def CheckSourceLayer(self, taskProduct, taskCode=None):
        if not taskCode:
            try:
                taskCode = self.taskCode
            except:
                msg.errmsg('Set taskCode argument.')
                return var.FAILED

        res = self.CheckArguments(taskCode)

        if res != var.SUCCESS:
            try:
                self.D.SetDecode(utl.DirName(self.srclyr),  taskCode)
                self.F.SetDecode(utl.BaseName(self.srclyr), taskProduct)
            except Exception as e:
                msg.errmsg(e)
                msg.errmsg('Cannot decode srclyr(%s)'%self.srclyr)
                return var.FAILED

        if not self.srclyr:
            try:
                self.srclyr = utl.SJoin(self.D[taskCode], self.F[taskProduct])
            except Exception as e:
                msg.errmsg(e)
                emsg = 'Cannot encode srclyr(%s, %s)'%(taskCode, taskProduct)
                msg.errmsg(emsg)
                msg.errmsg(self)
                return var.FAILED

        # set metadata
        self.srclyr = utl.AsLayer(self.srclyr)
        self.meta.Get(self.srclyr)
        return var.SUCCESS

    def SetDestinationLayer(self, taskProduct, taskCode=None):
        if not taskCode:
            try:
                taskCode = self.taskCode
            except:
                msg.errmsg('Set taskCode argument.')
                return var.FAILED

        try:
            self.dstlyr  = utl.SJoin(self.D[taskCode], self.F[taskProduct])
        except Exception as e:
            msg.errmsg(e)
            emsg = 'Cannot encode dstlyr(%s, %s)'%(taskCode, taskProduct)
            msg.errmsg(emsg)
            msg.errmsg(self)
            return var.FAILED
        return var.SUCCESS

    def SetSequenced(self):
        try:
            if self.IsShot() or self.prctype in [var.PRCCLIP, var.PRCSIM]:
                self.sequenced = True
        except Exception as e:
            msg.errmsg(e)
            emsg = 'Cannot figure out sequenced or not'
            msg.errmsg(emsg)
            msg.errmsg(self)
            return var.FAILED
        return var.SUCCESS

class Export(DXUSD_Export):
    ARGCLASS = AExport
