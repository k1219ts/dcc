import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

#-------------------------------------------------------------------------------
arg = twk.ACollection()

# model
# arg.master = '/show/pipe/_3d/asset/asdalCityTown/branch/houseC/model/v001/houseC_model.usd'
arg.master = '/show/pipe/_3d/asset/asdalCityTown/model/v001/asdalCityTown_model.usd'
# arg.master = '/show/wdl/_3d/asset/iss/rig/iss_rig_v001/iss_rig.usd'

# golaem agent
# arg.master = '/show/pipe/_3d/asset/soldier/agent/soldier/v010/soldier.usd'

if arg.Treat():
    print(arg)
    COL = twk.Collection(arg)
    COL.DoIt()
