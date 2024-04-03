import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

#-------------------------------------------------------------------------------
# model asset
# arg = twk.APackGeom()
# arg.master = '/show/pipe/_3d/asset/houseA/model/v003/houseA_model.usd'
# if arg.Treat():
#     print(arg)
#     PKG = twk.PackGeom(arg)
#     PKG.DoIt()

#-------------------------------------------------------------------------------
# rig asset
# arg = twk.APackGeom()
# arg.master = '/show/pipe/_3d/asset/bear/rig/bear_rig_v005/bear_rig.usd'
# if arg.Treat():
#     print(arg)
#     PKG = twk.PackGeom(arg)
#     PKG.DoIt()

#-------------------------------------------------------------------------------
# rig shot
arg = twk.APackGeom()
arg.master = '/show/pipe/_3d/shot/S26/S26_0450/ani/bear/v002/bear.usd'
if arg.Treat():
    # print(arg)
    PKG = twk.PackGeom(arg)
    PKG.DoIt()
