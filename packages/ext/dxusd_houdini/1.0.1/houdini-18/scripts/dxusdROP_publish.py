#coding:utf-8
from __future__ import print_function
import os

import DXUSD.Utils as utl
import DXUSD_HOU.Vars as var
import DXUSD.Message as msg
import DXUSD_HOU.Structures as srt
import DXUSD_HOU.Vars as var


if msg.DEV:
    reload(srt)
    reload(utl)
    reload(var)


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
            self.path   = path
            self.type   = None
            self.frames = None     # bool
            self.fx     = False
            self.clip   = False
            self.cliprate = []     # eg. (0.8, 1.0, 1.5)
            self.nslyr  = ''
            self.desc   = ''
            self.dprim  = ''
            self.dependency = {}

        def __repr__(self):
            res  = '%s : %s\n'%(self.path, self.type)
            if self.fx: res += '\t* FX Task\n'
            if self.clip: res += '\t* Clip Task %s\n'%self.cliprate
            res += '\tdependency : \n'
            for kind, data in self.dependency.items():
                res += '\t\t%s : %s\n'%(kind, data)
            res += '\tnslyr : %s\n'%self.nslyr
            res += '\tdesc  : %s\n'%self.desc
            res += '\tdprim : %s\n'%self.dprim
            res += '\tframe : %s\n'%self.frames
            return res

        def GetTaskCode(self, kind):
            '''
            asset : kind = 0
            shot  : kind = 1
            '''
            res = (None, '', '', {})
            # res = (Task, TaskCode, RuleBook USD Product, Extra args)
            if self.type == 'geom':
                pass
            elif self.type == 'inst':
                res = [('model', 'V', 'GEOM',
                            {'layout':var.T.LAYOUT, 'nslyr':self.nslyr}),
                       ('layout', 'NV', 'GEOM',
                            {})]
            elif self.type == 'groom':
                pass
            elif self.type == 'crowd':
                pass
            elif self.type == 'feather':
                res = [('feather', 'NS', 'GEOM',
                            {'desc':self.desc, 'subdir':self.desc}),
                       ('feather', 'NS', 'GEOM',
                            {'desc':self.desc, 'subdir':self.desc})]

            res = res[kind]
            return res[0], 'TASK%s'%res[1], res[2], res[3]


def ResolveTasks(arg, roplyrs):
    '''
    [Arguments]
    arg     (DXRulebook.Interface.Flags())
    roplyrs (ROPLayers)
    '''
    tasks = srt.Tasks(arg)
    if arg.has_key('asset'):
        kind = 0
    elif arg.has_key('shot'):
        kind = 1
    else:
        return tasks

    for roplyr in roplyrs:
        taskname = None
        lyrtask, code, product, extra = roplyr.GetTaskCode(kind)
        if not lyrtask:
            continue

        # ----------------------------------------------------------------------
        # Check clip and fx task
        if roplyr.clip: # when clip or fxlyr, change code
            taskname = 'clip'
            code = 'TASKNVC'
            extra['clip'] = 'base'
        elif roplyr.fx:
            taskname = 'fx'
            code = 'TASKVNV'
        else:
            taskname = lyrtask

        # ----------------------------------------------------------------------
        # Add or find task
        if taskname in tasks.items:
                task = tasks[taskname]
        else:
            task = tasks.Add(taskname, code)

            # check task version
            if code.startswith('TASKV'):
                task.vers = utl.GetVersions(task.arg.D.TASK, True, True)
                task.arg.ver = task.vers[0]

        # ----------------------------------------------------------------------
        # Add or find layer group (nslyr)
        if 'N' in code:
            nslyr = task.Add(roplyr.nslyr)
            task.arg.nslyr = roplyr.nslyr

            # find nslayer version
            if 'NV' in code:
                vercode = '%sN'%code.split('NV')[0]
                nslyr.vers = utl.GetVersions(task.arg.D[vercode], True, True)
                task.arg.nsver = nslyr.vers[0]
        else:
            nslyr = task.Add(var.NULL)

        # ----------------------------------------------------------------------
        # Add layers
        task.arg.update(extra)
        try:
            lyr = nslyr.Add(var.F[lyrtask][product].Encode(**task.arg))
            lyr.outpath = task.arg.D[code]
        except Exception as e:
            msg.warning('Cannot add layer', dev=True)
            msg.warning(e, dev=True)

        lyr.lyrtask = lyrtask
        lyr.cliprate = roplyr.cliprate
        lyr.inputnode = roplyr.path
        lyr.inputtype = roplyr.type
        lyr.dependency = roplyr.dependency

    return tasks







#
