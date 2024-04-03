import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


arg = twk.AMasterLayoutPack()
arg.master = '/show/pipe/_3d/shot/PKL/PKL_0350/layout/buildA/v002/buildA_layout.usd'
if arg.Treat():
    print(arg)
    TWK = twk.MasterLayoutPack(arg)
    TWK.DoIt()
