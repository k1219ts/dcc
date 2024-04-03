#coding:utf-8
from __future__ import print_function
import os

import DXUSD.Message as msg

import DXUSD_HOU.Vars as var
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Structures as srt


if msg.DEV:
    import DXUSD.moduleloader as md
    import DXUSD.Compositor as cmp
    import DXUSD_HOU.Tweakers as twk
    import DXUSD_HOU.Exporters as exps
    import HOU_Base.NodeUtils as ntl
    import DXUSD_HOU.HouExport as exp

    reload(md)
    reload(msg)
    reload(var)
    reload(utl)
    reload(srt)
    reload(twk)
    reload(exps)
    reload(exp)
    reload(cmp)
    reload(ntl)


class ROPLayers:
    def __init__(self):
        self._lyrs = list()
        self._idx  = 0

    def __len__(self):
        return len(self._lyrs)

    def __next__(self):
        if self._idx < len(self):
            res = self._lyrs[self._idx]
            self._idx += 1
            return res
        else:
            self._idx = 0
            raise StopIteration

    def __getitem__(self, k):
        return self._lyrs[k]

    def __repr__(self):
        res = ''
        for lyr in self._lyrs:
            res += '%s\n'%str(lyr)
        return res

    def Add(self, path):
        layer = ROPLayers.ROPLayer(path)
        self._lyrs.append(layer)
        return layer

    class ROPLayer:
        def __init__(self, path):
            self.path    = path
            self.type    = None
            self.frames  = None     # bool
            self.prctype = var.PRCNONE
            self.prcname = None
            self.cliprate = []     # eg. (0.8, 1.0, 1.5)
            self.nslyr   = ''
            self.lyrname = ''
            self.dprim   = ''
            self.dependency = {}

        def __repr__(self):
            res  = '%s : %s\n'%(self.path, self.type)
            res += '\t* post process type : %s\n'%self.prctype

            if self.prctype == var.PRCCLIP:
                res += '\t* clip rate : %s\n'%self.cliprate

            res += '\t* dependency : \n'
            for kind, data in self.dependency.items():
                res += '\t\t%s : %s\n'%(kind, data)

            res += '\t* nslyr : %s\n'%self.nslyr
            res += '\t* sublyr : %s\n'%self.sublyr
            res += '\t* lyrname  : %s\n'%self.lyrname
            res += '\t* dprim : %s\n'%self.dprim
            res += '\t* frame : %s\n'%self.frames
            return res

        def GetTaskCode(self, isShot):
            err = [None, '', '', {}]
            res = [[None, '', '', {}], [None, '', '', {}]]
            # res = [[Task, TaskCode, RuleBook USD Product, Extra args], ...]

            # ------------------------------------------------------------------
            if   self.type == var.LYRGEOM:
                pass

            # ------------------------------------------------------------------
            elif self.type == var.LYRINST:
                res = [[var.T.MODEL, 'V', 'GEOM',
                            {'layout':var.T.LAYOUT, 'nslyr':self.nslyr}],
                       [var.T.LAYOUT, 'NV', 'GEOM',
                            {}]]
                if   self.prctype == var.PRCSIM:
                    pass
                elif self.prctype == var.PRCCLIP:
                    pass
                elif self.prctype == var.PRCFX:
                    pass

            # ------------------------------------------------------------------
            elif self.type == var.LYRGROOM:
                res[0] = [var.T.GROOM, 'NS', 'GEOM',
                    {'subdir':self.lyrname}]
                res[1] = [var.T.GROOM, 'NVS', 'GEOM',
                    {'subdir':self.lyrname}]

                if   self.prctype == var.PRCSIM:
                    pass
                elif self.prctype == var.PRCCLIP:
                    pass
                elif self.prctype == var.PRCFX:
                    pass

            # ------------------------------------------------------------------
            elif self.type == var.LYRCROWD:
                if   self.prctype == var.PRCSIM:
                    pass
                elif self.prctype == var.PRCCLIP:
                    pass
                elif self.prctype == var.PRCFX:
                    pass

            # ------------------------------------------------------------------
            elif self.type == var.LYRFEATHER:
                res[0] = [var.T.GROOM, 'NS', 'GEOM',
                    {'desc':self.lyrname, 'subdir':self.lyrname}]
                res[1] = [var.T.GROOM, 'NVS', 'GEOM',
                    {'desc':self.lyrname, 'subdir':self.lyrname}]

                if   self.prctype == var.PRCSIM:
                    pass
                elif self.prctype == var.PRCCLIP:
                    res[0] = [var.T.CLIP, 'NVS', 'GEOM',
                        {'subdir':var.T.BASE, 'desc':self.lyrname}]
                    res[1] = res[0]
                elif self.prctype == var.PRCFX:
                    res[0] = [var.T.FX, 'VNVSVS', 'GEOM',
                        {'subdir1':self.lyrname, 'desc':self.lyrname}]
                    res[1] = res[0]

            res = res[isShot]
            return res[0], 'TASK%s'%res[1], res[2], res[3]


def ResolveTasks(arg, roplyrs):
    '''
    [Arguments]
    arg     (DXRulebook.Interface.Flags())
    roplyrs (ROPLayers)
    '''
    tasks = srt.Tasks(arg)
    if not arg:
        return tasks

    for roplyr in roplyrs:
        taskname, code, product, extra = roplyr.GetTaskCode(arg.IsShot())
        if not taskname:
            continue

        # ----------------------------------------------------------------------
        # Add task
        isNew = taskname not in tasks.keys()
        task  = tasks.Add(taskname, code)

        if isNew:
            # check task version
            if code.startswith('TASKV'):
                try:
                    task.vers = utl.GetVersions(task.arg.D.TASK, True, True)
                    task.arg.ver = task.vers[0]
                except Exception as e:
                    msg.warning(e, dev=True)
                    continue

        # ----------------------------------------------------------------------
        # Add nslyr
        if 'N' in code:
            nslyr = task.Add(roplyr.nslyr)
            task.arg.nslyr = roplyr.nslyr

            # find nslayer version
            if 'NV' in code:
                vercode = '%sN'%code.split('NV')[0]
                try:
                    nslyr.vers = utl.GetVersions(task.arg.D[vercode], True, True)
                    task.arg.nsver = nslyr.vers[0]
                except Exception as e:
                    msg.warning(e, dev=True)
                    continue
        else:
            nslyr = task.Add(var.NULL)

        # ----------------------------------------------------------------------
        # Add subdir (mostly it's fx process)
        if 'SV' in code:
            sublyr = nslyr.Add(roplyr.sublyr)
            task.arg.subdir = roplyr.sublyr

            # find nslayer version
            if 'SV' in code:
                vercode = '%sS'%code.split('SV')[0]
                try:
                    sublyr.vers = utl.GetVersions(task.arg.D[vercode], True, True)
                    task.arg.subver = sublyr.vers[0]
                except Exception as e:
                    msg.warning(e, dev=True)
                    continue
        else:
            sublyr = nslyr.Add(var.NULL)

        # ----------------------------------------------------------------------
        # Add layers
        task.arg.update(extra)
        try:
            lyr = sublyr.Add(var.F[taskname][product].Encode(**task.arg))
            lyr.outpath = task.arg.D[code]
        except Exception as e:
            msg.warning('Cannot add layer', dev=True)
            msg.warning(e, dev=True)
            lyr = None

        if lyr:
            lyr.cliprate    = roplyr.cliprate
            lyr.inputnode   = roplyr.path
            lyr.lyrtype     = roplyr.type
            lyr.prctype     = roplyr.prctype
            lyr.dependency  = roplyr.dependency

    return tasks



#
