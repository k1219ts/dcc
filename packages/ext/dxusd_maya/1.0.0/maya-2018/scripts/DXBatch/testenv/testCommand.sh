# ani cache
DCC.local dev maya -v 2018 --zelos --terminal DXUSD_MAYA_Batch --host local -p both -f /show/pipe/template/fox/CLF_0050_ani_v001_sample.mb --mesh v003=fox:fox_rig_GRP


# groom cache-merged
DCC.local dev maya -v 2018 --zelos --terminal DXUSD_MAYA_Batch --host local -p both -f /show/pipe/template/fox/CLF_0050_ani_v001_sample.mb --mesh v004=fox:fox_rig_GRP --onlyGroom
# or
DCC.local dev maya -v 2018 --zelos --terminal DXBatchGroom -p both -i /show/pipe/_3d/shot/CLF/CLF_0050/ani/fox/v004/fox_ani.usd


# sim cache
DCC.local dev maya -v 2018 --terminal DXBatchMain --host local -p both -f /show/pipe/template/fox/CLF_0050_cloth_v002_sample.mb --simMesh v002=fox:fox_rig_GRP
