import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

#-------------------------------------------------------------------------------
# rig asset
# arg = twk.APackRigGeom()
# arg.master = '/show/pipe/_3d/asset/bear/rig/bear_rig_v004/bear_rig.usd'
# if arg.Treat():
#     print(arg)
#     PKG = twk.PackRigGeom(arg)
#     PKG.DoIt()

#-------------------------------------------------------------------------------
# rig shot
arg = twk.AMasterRigPack()
arg.master = '/show/pipe/_3d/shot/S26/S26_0450/ani/bear/v003/bear.usd'
if arg.Treat():
    # print(arg)
    PKG = twk.MasterRigPack(arg)
    PKG.DoIt()
