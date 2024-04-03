import DXUSD.Tweakers as twk


#-------------------------------------------------------------------------------
#
#   PrmanMaterial
#
#-------------------------------------------------------------------------------
arg = twk.APrmanMaterial()
# arg.inputs = [
#     # '/show/pipe/_3d/asset/asdalCityTown/branch/houseA/model/v001/asdalCityTown_houseA_model_GRP.mid_geom.usd',
#     # '/show/pipe/_3d/asset/asdalCityTown/branch/houseB/model/v001/asdalCityTown_houseB_model_GRP.mid_geom.usd',
#     # '/show/pipe/_3d/asset/asdalCityTown/branch/houseC/model/v001/asdalCityTown_houseC_model_GRP.high_geom.usd',
#     # '/show/pipe/_3d/asset/asdalCityTown/model/v002/asdalCityTown_model_GRP.high_geom.usd',
#     # '/show/pipe/_3d/asset/asdalCityTown/model/v011/asdalCityTown_model_GRP_ABlock.high_geom.usd',
#     # '/show/pipe/_3d/asset/fox/rig/fox_rig_v004/fox_rig_fox_rig_GRP.high_geom.usd',
#     '/show/pipe/_3d/asset/fox/groom/fox_hair_v004/fox_groom.high.usd',
#     '/show/pipe/_3d/asset/fox/groom/fox_hair_v004/fox_groom.low.usd',
#     # '/show/pipe/_3d/asset/e9dog/rig/e9dog_rig_v004/e9dog_rig_GRP.high_geom.usd',
#     # '/show/pipe/_3d/asset/e9dog/groom/e9dog_groom_v005/e9dog_groom.usd'
#     # '/show/pipe/_3d/asset/e9dog/model/v013/e9dog_model_GRP.high_geom.usd'
# ]
arg.dstdir = '/show/pipe/_3d/asset/asdalCityTown/branch/houseA/model/v003'
if arg.Treat():
    print(arg)
    TPM = twk.PrmanMaterial(arg)
    TPM.DoIt()


#-------------------------------------------------------------------------------
#
#   PrmanMaterialOverride
#
#-------------------------------------------------------------------------------
# arg = twk.APrmanMaterialOverride()
# arg.dstdir = '/show/pipe/_3d/asset/asdalCityTown/branch/houseA/model/v003'
# # arg.dstdir = '/show/pipe/_3d/asset/asdalCityTown/model/v001'
# # arg.dstdir = '/show/pipe/_3d/asset/parkingLot/branch/carSet/model/v002'
# # arg.dstdir = '/show/pipe/_3d/shot/PKL/PKL_0350/layout/carSet/v001'
# # arg.dstdir = '/show/pipe/_3d/shot/PKL/PKL_0350/layout/carSet/v002'
# # arg.dstdir = '/show/pipe/_3d/shot/PKL/PKL_0350/layout/buildA/v003'
# # arg.dstdir = '/show/pipe/_3d/shot/CLF/CLF_0050/ani/fox1/v002'
# # arg.dstdir = '/show/pipe/_3d/shot/CLF/CLF_0050/groom/fox/v001'
#
# if arg.Treat():
#     print(arg)
#     TPMO = twk.PrmanMaterialOverride(arg)
#     TPMO.DoIt()
