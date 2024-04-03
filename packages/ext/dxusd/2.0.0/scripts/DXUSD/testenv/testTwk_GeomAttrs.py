import DXUSD.Tweakers as twk

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

import pprint

#-------------------------------------------------------------------------------
# model asset
# arg = twk.AGeomAttrs()
# arg.extracts = [var.T.ATTR_MATERIALSET]
# arg.inputs = [
#     '/show/pipe/_3d/asset/houseA/model/v003/houseA_model_GRP.high_geom.usd',
#     '/show/pipe/_3d/asset/houseA/model/v003/houseA_model_GRP.mid_geom.usd',
#     '/show/pipe/_3d/asset/houseA/model/v003/houseA_model_GRP.low_geom.usd'
# ]
# if arg.Treat():
#     texData = dict()
#     TGA = twk.GeomAttrs(arg, texData)
#     TGA.DoIt()
#     pprint.pprint(texData)


#-------------------------------------------------------------------------------
# model branch
arg = twk.AGeomAttrs()
arg.extracts = [var.T.ATTR_MATERIALSET]
arg.inputs = [
    '/show/pipe/_3d/asset/bear/branch/fatbear/model/v001/bear_fatbear_model_GRP.high_geom.usd',
    '/show/pipe/_3d/asset/bear/branch/fatbear/model/v001/bear_fatbear_model_GRP.low_geom.usd'
]
if arg.Treat():
    print(arg)
    texData = dict()
    TGA = twk.GeomAttrs(arg, texData)
    TGA.DoIt()
    pprint.pprint(texData)


#-------------------------------------------------------------------------------
# Crowd Agent
# arg = twk.AGeomAttrs()
# arg.extracts = [var.T.ATTR_MATERIALSET]
# arg.inputs = [
#     # '/show/pipe/_3d/asset/crdMainStreet/agent/crdMainStreet_man/v002/OriginalAgent_crdMainStreet_man.geom.usd',
#     '/show/pipe/_3d/asset/crdMainStreet/agent/crdMainStreet_woman/v004/OriginalAgent_crdMainStreet_woman.geom.usd'
# ]
# if arg.Treat():
#     print(arg)
#     txarg = twk.ATexture()
#     txarg.Treat()
#     texData = dict()
#     TGA = twk.GeomAttrs(arg, texData)
#     TGA.DoIt()
#     pprint.pprint(texData)
#
#     twks = twk.Tweak()
#     for f in texData:
#         txarg.texAttrUsd = f
#         txarg.texData    = texData[f]
#         twks << twk.Texture(txarg)
#     #     twks << twk.ProxyMaterial(txarg)
#     # twks << twk.PrmanMaterial(arg)
#     twks.DoIt()


#-------------------------------------------------------------------------------
# texData
# texData = {'/show/pipe/_3d/asset/crdMainStreetA/texture/tex/v001/tex.attr.usd': {'crdMainStreetA': {'attrs': {'primvars:modelVersion': 'v001',
#                                                                                                     'primvars:txmultiUV': 0},
#                                                                                           'classPath': '/_crdMainStreetA_crdMainStreetA_txAttr'}},
#  '/show/pipe/_3d/asset/crdMainStreetB/texture/tex/v001/tex.attr.usd': {'crdMainStreetB': {'attrs': {'primvars:modelVersion': 'v001',
#                                                                                                     'primvars:txmultiUV': 0},
#                                                                                           'classPath': '/_crdMainStreetB_crdMainStreetB_txAttr'}},
#  '/show/pipe/_3d/asset/crdMainStreetC/texture/tex/v001/tex.attr.usd': {'crdMainStreetC': {'attrs': {'primvars:modelVersion': 'v001',
#                                                                                                     'primvars:txmultiUV': 0},
#                                                                                           'classPath': '/_crdMainStreetC_crdMainStreetC_txAttr'}},
#  '/show/pipe/_3d/asset/crdMainStreetD/texture/tex/v001/tex.attr.usd': {'crdMainStreetD': {'attrs': {'primvars:modelVersion': 'v001',
#                                                                                                     'primvars:txmultiUV': 0},
#                                                                                           'classPath': '/_crdMainStreetD_crdMainStreetD_txAttr'}}}
# pprint.pprint(texData)
# txarg = twk.ATexture()
# twks = twk.Tweak()
# for f in texData:
#     txarg.texAttrUsd = f
#     txarg.texData = texData[f]
#     twks << twk.Texture(txarg)
#     twks << twk.ProxyMaterial(txarg)
# twks.DoIt()


#-------------------------------------------------------------------------------
# rig asset
# arg = twk.AGeomAttrs()
# # arg.extracts= [var.T.ATTR_MATERIALSET]
# arg.extracts.append('purpose')
# arg.inputs  = [
#     '/show/pipe/_3d/asset/bear/rig/bear_rig_v005/bear_rig_GRP.high_geom.usd',
#     # '/show/pipe/_3d/asset/bear/rig/bear_rig_v005/bear_rig_GRP.sim_geom.usd'
# ]
# if arg.Treat():
#     print(arg)
#     texData = dict()
#     TGA = twk.GeomAttrs(arg, texData)
#     TGA.DoIt()
#     pprint.pprint(texData)

# # rig shot
# arg = twk.AGeomRigAttrs()
# arg.inputs = [
#     '/show/pipe/_3d/shot/S26/S26_0450/ani/bear/v003/bear_rig_GRP.high_geom.usd'
# ]
# if arg.Treat():
#     print(arg)
#     TGA = twk.GeomRigAttrs(arg)
#     TGA.DoIt()
