import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

import pprint

arg = twk.AGeomAttrs()
arg.inputs = [
    # '/show/pipe/_3d/shot/CLF/asset/asdalCityTown/model/v001/asdalCityTown_model_GRP.high_geom.usd',
    '/show/cdh1/_3d/shot/ELV/ELV_0010/asset/guardA/model/v001/guardA_model_GRP.high_geom.usd'
]

if arg.Treat():
    print(arg)

    texDir = utl.SearchInDirs(arg.searchPath, 'asset/guardA')
    print('>>>', utl.SJoin(texDir, 'texture', var.T.TEX))
