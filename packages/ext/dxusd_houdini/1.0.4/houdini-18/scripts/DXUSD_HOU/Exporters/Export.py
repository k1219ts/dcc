#coding:utf-8
from __future__ import print_function

import DXUSD.Utils as utl
import DXUSD.Message as msg
from DXUSD.Exporters.Export import AExport as DXUSD_AExport
from DXUSD.Exporters.Export import Export as DXUSD_Export

import DXUSD_HOU.Structures as srt
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
            'asset/branch/shot': {
                (task): { __USDPATH__:'/show/pipe/...',
                          __ORDER__:['vset1', 'vset2', ...],
                          'vset1':'on',
                          'vset2':'off' }, ...
            }
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

    def FindRigFile(self):
        res        = srt.DependentFile()
        customData = utl.AsLayer(self.srclyr).customLayerData

        if customData.has_key(var.T.CUS_RIGFILE):
            res.SetFile(customData[var.T.CUS_RIGFILE])

        if not res:
            adata = {}
            if self.dependency.has_key(var.T.BRANCH):
                adata = self.dependency[var.T.BRANCH]
            elif self.dependency.has_key(var.T.ASSET):
                adata = self.dependency[var.T.ASSET]

            if adata.has_key(var.T.RIG):
                res.SetFile(adata[var.T.RIG][var.USDPATH])
            elif adata.has_key(var.T.MODEL):
                res.SetFile(adata[var.T.MODEL][var.USDPATH])

        if not res:
            msg.errmsg('Cannot find rig file')

        return res

    def FindGroomFile(self):
        from DXUSD.Structures import Arguments

        res        = srt.DependentFile()
        customData = utl.AsLayer(self.srclyr).customLayerData

        cusname = var.NULL
        if self.lyrtype == var.LYRGROOM:
            cusname = var.T.CUS_GROOMFILE
            print('cusname:',cusname)

        elif self.lyrtype == var.LYRFEATHER:
            cusname = var.T.CUS_FEATHERFILE

        if customData.has_key(cusname):
            if self.lyrtype == var.LYRGROOM:
                arg = Arguments()
                gfile = customData[cusname]
                arg.D.SetDecode(gfile)
                arg.task = 'groom'
                arg.nslyr = utl.BaseName(utl.BaseName(gfile).split('.')[0])
                master = utl.SJoin(arg.D['TASKN'], arg.F.MASTER)
                res.SetFile(master)

            elif self.lyrtype == var.LYRFEATHER:
                res.SetFile(customData[cusname])

        if not res:
            for kind in [var.T.BRANCH, var.T.ASSET]:
                if not self.dependency.has_key(kind):
                    continue
                if self.dependency[kind].has_key(var.T.GROOM):
                    res.SetFile(self.dependency[kind][var.T.GROOM][var.USDPATH])
                    break

        if not res:
            msg.errmsg('Cannot find %s file'%self.lyrtype)

        return res


    def FindInputCacheFile(self):
        res        = srt.DependentFile()
        customData = utl.AsLayer(self.srclyr).customLayerData

        if customData.has_key(var.T.CUS_INPUTCACHE):
            res.SetFile(customData[var.T.CUS_INPUTCACHE])

        if not res and self.dependency.has_key(var.T.SHOT):
            for task in [var.T.FX, var.T.SIM, var.T.CLIP, var.T.ANI]:
                if self.dependency[var.T.SHOT].has_key(task):
                    res.SetFile(self.dependency[var.T.SHOT][task][var.USDPATH])
                    break

        if not res:
            msg.errmsg('Cannot find input cache file')

        return res


class Export(DXUSD_Export):
    ARGCLASS = AExport
