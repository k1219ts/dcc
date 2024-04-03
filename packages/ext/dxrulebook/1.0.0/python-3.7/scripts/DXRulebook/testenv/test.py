import DXRulebook.Rulebook as rulebook
reload(rulebook)

import DXRulebook.Interface as rb
flags = rb.Flags()
# flags.D.SetDecode('/show/pipe/works/CSP/sanghun.kim/MTK_TEST/asset/asdalCityTown/model/v000/ddd', 'ROOTS')
# flags.D.SetDecode('/show/pipe/_3d/asset/fox/rig/fox_rig_v003', 'ROOTS')
# print flags

coder = rb.Coder()
# print coder.D.ROOTS.Decode('/show/pipe/works/CSP/sanghun.kim/MTK_TEST/asset/asdalCityTown/model/v001')



# flags.D.ASSETLIB.SetDecode('/assetlib/3D/usd/material/prman/shaders/bronze/v001')
# print flags.D.ASSETLIB.MATRMNV

path = '/show/slc/works/AST/e45dog/groom/tmp/asset/e45dog/groom/e45dog_new_hair_v01_w08_jin'
path = '/show/pipe/_3d/shot/S26/S26_0450/ani/bear/v001/'

# flags.D.SetDecode(path)
# print flags
# print flags.D.SHOT
# print flags.D.ASSET
# print flags.D.ROOTS
path = '/knotaa/show/pipe/_3d/asset/house'
# path = '/knotdd/show'
# print flags.D.OTHER.Decode(path, "EXCLUSIVEROOT")
print flags.D.Decode(path)
