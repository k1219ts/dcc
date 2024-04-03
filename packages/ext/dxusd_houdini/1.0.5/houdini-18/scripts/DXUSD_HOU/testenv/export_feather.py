#coding:utf-8
from __future__ import print_function

import DXUSD_HOU.Vars as var
import DXUSD_HOU.Exporters as exp
import DXUSD_HOU.Utils as utl
import DXUSD_HOU.Structures as srt

def Test(srclyr):
    arg = exp.AFeatherExporter()

    dirname = utl.DirName(srclyr)
    arg.D.SetDecode(dirname)
    arg.srclyr = utl.AsLayer(srclyr)
    arg.taskCode = 'TASKNVS'
    arg.sequenced = True
    arg.dependency = srt.BaseStructure()
    arg.dependency.asset = {}
    arg.dependency.asset[var.T.TASK] = var.T.GROOM

    customdata = dict(arg.srclyr.customLayerData)
    if 'penguinB' in customdata.get('rigFile', ''):
        arg.dependency.asset[var.USDPATH] = '/show/prat2/_3d/asset/penguinB/groom/penguinB_feather_v001/penguinB_groom.usd'
    else:
        arg.dependency.asset[var.USDPATH] = '/show/prat2/_3d/asset/penguin/groom/penguin_feather_v001/penguin_groom.usd'

    # print('>>> Treat :', arg.Treat())
    print(arg)
    exp.FeatherExporter(arg)


if __name__ == '__main__':
    srclyrs = [
        '/show/prat2/_3d/shot/PS84/PS84_0080/groom/penguin4/v001/dxSOP_FeatherAttach1/dxSOP_FeatherAttach1.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0080/groom/penguin5/v001/dxSOP_FeatherAttach2/dxSOP_FeatherAttach2.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0080/groom/penguin6/v001/dxSOP_FeatherAttach3/dxSOP_FeatherAttach3.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0080/groom/penguinB/v001/dxSOP_FeatherAttach8/dxSOP_FeatherAttach8.high_geom.usd'
    ]
    srclyrs = [
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin/v001/dxSOP_FeatherAttach1/dxSOP_FeatherAttach1.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin5/v001/dxSOP_FeatherAttach2/dxSOP_FeatherAttach2.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin6/v001/dxSOP_FeatherAttach3/dxSOP_FeatherAttach3.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin7/v001/dxSOP_FeatherAttach4/dxSOP_FeatherAttach4.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin10/v002/dxSOP_FeatherAttach5/dxSOP_FeatherAttach5.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin11/v001/dxSOP_FeatherAttach6/dxSOP_FeatherAttach6.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguin12/v001/dxSOP_FeatherAttach7/dxSOP_FeatherAttach7.high_geom.usd',
        '/show/prat2/_3d/shot/PS84/PS84_0100/groom/penguinB/v001/dxSOP_FeatherAttach8/dxSOP_FeatherAttach8.high_geom.usd'
    ]

    for srclyr in srclyrs:
        Test(srclyr)
