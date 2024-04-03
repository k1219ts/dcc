
import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

import DXUSD_MAYA.Tweakers as mtwk


#-------------------------------------------------------------------------------
# Crowd Agent
arg = mtwk.AGeomAgentAttrs()
arg.inputs = [
    '/show/pipe/_3d/asset/crdMainStreet/agent/crdMainStreet_man/v002/OriginalAgent_crdMainStreet_man.geom.usd'
]
if arg.Treat():
    print(arg)
    GAA = mtwk.GeomAgentAttrs(arg)
    GAA.DoIt()
