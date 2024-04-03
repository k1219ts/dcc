#coding:utf-8
from __future__ import print_function

import DXUSD.Message as msg
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Vars as var
import DXUSD_HOU.Exporters as exp


def GeomExport(lyrs):
    if lyrs:
        arg = exp.AModelExporter(**lyrs[0].parent().arg)
        for lyr in lyrs:
            arg.geomfiles.append(utl.SJoin(lyr.outpath, lyr.name))
        exp.ModelExporter(arg)


def InstanceExport(lyrs):
    if lyrs:
        arg = exp.ALayoutExporter(**lyrs[0].parent().arg)
        for lyr in lyrs:
            arg.instfiles.append(utl.SJoin(lyr.outpath, lyr.name))
        exp.LayoutExporter(arg)


def GroomExport(lyrs):
    pass


def FeatherExport(lyrs):
    for lyr in lyrs:
        arg = exp.AFeatherExporter(**lyr.arg)
        arg.task       = lyr.task.name
        arg.taskCode   = lyr.task.code
        arg.prctype    = lyr.prctype
        arg.srclyr     = lyr.outpath
        arg.dependency = lyr.dependency
        arg.cliprate   = lyr.cliprate
        exp.FeatherExporter(arg)


def CrowdExport(lyrs):
    pass


def HouExport(tasks, meta):
    msg.debug('#'*80)
    msg.debug('# Start Exporting')
    msg.debug('#'*80)
    msg.debug(tasks)
    msg.debug('#'*80)

    for task in tasks.items():
        for nslyr in task.items():
            for sublyr in nslyr.items():
                geomlyrs = []
                instlyrs = []
                groomlyrs = []
                featherlyrs = []
                crowdlyrs = []

                for lyr in sublyr.items():
                    if lyr.lyrtype == var.LYRGEOM:
                        geomlyrs.append(lyr)
                    elif lyr.lyrtype == var.LYRINST:
                        instlyrs.append(lyr)
                    elif lyr.lyrtype == var.LYRGROOM:
                        groomlyrs.append(lyr)
                    elif lyr.lyrtype == var.LYRFEATHER:
                        featherlyrs.append(lyr)
                    elif lyr.lyrtype == var.LYRCROWD:
                        crowdlyrs.append(lyr)

                GeomExport(geomlyrs)
                InstanceExport(instlyrs)
                GroomExport(groomlyrs)
                FeatherExport(featherlyrs)
                CrowdExport(crowdlyrs)
