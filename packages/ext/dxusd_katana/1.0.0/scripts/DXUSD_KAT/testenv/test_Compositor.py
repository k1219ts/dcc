import DXUSD_KAT.Compositor as cmp

#   assetlib _global material
# input = '/assetlib/_3d/usd/material/prman/shaders/wood/v002/Wood_SHD2.usd'

#   show _global material
input = '/show/pipe/_3d/asset/_global/material/prman/shaders/wood/v001/Wood_SHD11.usd'

#   show asset
# input = '/show/pipe/_3d/asset/asdalCityTown/branch/houseC/material/prman/shaders/wood/v003/Wood_SHD2.usd'
input = '/show/pipe/_3d/asset/asdalCityTown/branch/houseC/material/prman/shaders/rock/v001/Rock_SHD2.usd'

# cmp.Composite(input).DoIt()
cmp.MaterialComposite(input).DoIt()
