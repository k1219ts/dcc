#coding:utf-8
from __future__ import print_function

import DXUSD.Message as msg
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var
import DXUSD_HOU.Exporters as exp

if msg.DEV:
    import DXUSD.moduleloader as md
    reload(md)
    import DXUSD.Vars as dxvar
    reload(dxvar)
    import DXUSD.Structures as srt
    reload(srt)
    import DXUSD.Utils as dxutl
    reload(dxutl)
    reload(utl)
    print('reload utl')
    dxvar.rb.Reload()

    reload(var)
    reload(exp)

    import DXUSD_HOU.Tweakers as twk
    reload(twk)

def ModelExport(arg, layers):
    arg = exp.AModelExporter(**arg)
    for layer in layers:
        if layer.inputtype == var.TYPEINST:
            arg.instfiles.append(utl.SJoin(layer.outpath, layer.name))
        elif layer.inputtype == var.TYPEGEOM:
            arg.geomfiles.append(utl.SJoin(layer.outpath, layer.name))
    exp.ModelExporter(arg)


def LayoutExport(nslyrs):
    for nslyr in nslyrs:
        arg = exp.ALayoutExporter(**nslyr.arg)
        for lyr in nslyr.items:
            if lyr.inputtype == var.TYPEINST:
                arg.instfiles.append(utl.SJoin(lyr.outpath, lyr.name))
        exp.LayoutExporter(arg)


def FeatherExport(task):
    for nslyr in task.items:
        for lyr in nslyr.items:
            arg = exp.AFeatherExporter(**lyr.arg)
            arg.srclyr = lyr.outpath
            arg.dependency = lyr.dependency
            exp.FeatherExporter(arg)


def HouExport(tasks, meta):
    msg.debug('#'*80)
    msg.debug('# Start Exporting')
    msg.debug('#'*80)
    msg.debug(tasks)
    msg.debug('#'*80)

    for task in tasks.items:
        if not task.items:
            continue

        if task.name == var.T.MODEL:
            ModelExport(task.arg, task.items[0].items)
        elif task.name == var.T.LAYOUT:
            LayoutExport(task.items)
        elif task.name == var.T.FEATHER:
            FeatherExport(task)
