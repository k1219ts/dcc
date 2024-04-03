#coding:utf-8
from __future__ import print_function
import os

import DXUSD.Utils as utl
import DXUSD.Exporters as exp

def setUsdCopy(show,input):
    for i, path in enumerate(utl.GetUsdPath(input)):
        if '_set' in path:
            for n, p in enumerate(utl.GetUsdPath(path).get()):
                assetCopy(show,p)
            createSet(show, path)
        assetCopy(show, path)
    createSet(show, input)

def createSet(show, path, ovr_asset=None):
    arg = exp.AUsdReferenceExporter()
    arg.show = show
    arg.input = path
    arg.ovr_asset = ovr_asset
    exp.UsdReferenceExporter(arg)


def assetCopy(show,path):
    arg = exp.AUsdExporter()
    arg.show = show
    # arg.oldGeomfiles = path
    arg.orgPath =path
    exp.UsdExporter(arg)


reload(exp)
newShow = 'bmt'
# usdpath='/show/srh_pub/asset/hachiSatllite/element/lightA/lightA.usd'
# usdpath='/assetlib/3D/asset/lijangEnv_lanternC/lijangEnv_lanternC.usd'
usdpath ='/show/srh_pub/asset/victoryShipB/victoryShipB.usd'
# assetCopy(newShow,usdpath)


# setpath = '/show/srh_pub/asset/hachiSatllite_set/hachiSatllite_set.usd'
# setpath = '/show/srh_pub/asset/hatchBridge_set/hatchBridge_set.usd'
setpath='/show/srh_pub/asset/hachiSatllite_light_set/hachiSatllite_light_set.usd'
ovr_asset = 'hachiSatllite_light'
createSet(newShow,setpath,ovr_asset)
