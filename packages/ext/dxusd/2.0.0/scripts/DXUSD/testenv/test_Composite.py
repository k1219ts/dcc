#coding:utf-8
from __future__ import print_function

import DXUSD.Compositor as cmp

import DXUSD.Utils as utl

#-------------------------------------------------------------------------------
# ASSET
# filename = '/show/pipe/_3d/asset/asdalCityTown/model/v002/asdalCityTown_model.usd'
# filename = '/show/pipe/_3d/asset/asdalCityTown/branch/houseA/model/v001/houseA_model.usd'
# filename = '/show/pipe/_3d/asset/fox/rig/fox_rig_v004/fox_rig.usd'
# filename = '/show/pipe/_3d/asset/fox/groom/fox_hair_v004/fox_groom.usd'
# filename = '/show/pipe/_3d/asset/fox/clip/walk/v007/walk_clip.usd'
# filename = '/show/pipe/_3d/asset/crdGhost/agent/crdGhost/v001/crdGhost.usd'

#   Golaem Asset
# filename = '/show/pipe/_3d/asset/soldier/agent/soldier/v008/soldier.usd'
# filename = '/show/pipe/_3d/asset/soldier/agent/soldier_spear/v001/soldier_spear.usd'

#   Golaem Shot
filename = '/show/pipe/_3d/shot/CTC/CTC_0020/crowd/v009/crowd.usd'

#-------------------------------------------------------------------------------
# MTK
# filename = '/show/slc/works/AST/e45dog/scenes/tmp11/asset/e45dog/groom/e45dog_new_hair_v01_w08_jin/e45dog_groom.usd'

#-------------------------------------------------------------------------------
# SHOT
# filename = '/show/pipe/_3d/shot/CLF/CLF_0050/cam/v003/camera.usd'
# filename = '/show/pipe/_3d/shot/CLF/CLF_0050/ani/fox/v001/fox_ani.usd'
# filename = '/show/pipe/_3d/shot/CLF/CLF_0050/groom/fox/v001/fox_groom.usd'
# filename = '/show/pipe/_3d/shot/CLF/CLF_0050/sim/fox/v001/fox_sim.usd'
# filename = '/show/pipe/_3d/shot/S26/S26_0010/layout/ModelExport/v009/ModelExport_layout.usd'

#-------------------------------------------------------------------------------
comp = cmp.Composite(filename)
# print(comp.arg)
comp.DoIt()
