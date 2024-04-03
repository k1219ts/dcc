import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

#  camera ----------------------------------------------------------------------
arg = twk.AConvertAlembic()
arg.geomfiles = ['/show/pipe/template/Camera/convertABC/PKL_0240_main1_cam.geom.usd',
                 '/show/pipe/template/Camera/convertABC/LIM_0578_main1_cam.geom.usd']

if arg.Treat():
    cvt = twk.ConvertAlembic(arg)
    cvt.DoIt()
