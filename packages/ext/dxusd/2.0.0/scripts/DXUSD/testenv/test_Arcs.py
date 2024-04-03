import sys
from pxr import Sdf

path = '/works/dev/libs/pylibs/1.0.0/python2.7/site-packages'
if path not in sys.path:
    sys.path.append(path)

import DXUSD.Message as msg
import DXUSD.Vars as var
import DXUSD.moduleloader as mld
import DXUSD.Utils as utl
import DXUSD.Arcs as arc


geomfile = ['/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion_model_GRP.high_geom.usd',
            '/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion_model_GRP.low_geom.usd']


hgeom = utl.AsLayer(geomfile[0])
lgeom = utl.AsLayer(geomfile[1])
col = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/v007/collection.usd')

# test payload and purpos
reload(arc)
pkgfile = '/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion.usd'
stgmeta = utl.StageMetadata(hgeom)
pkglyr = utl.AsLayer(pkgfile, create=True, clear=True)
arcs = arc.Arcs(pkglyr, stgmeta)
arcs.Payload(geomfile[0], '/lion', purpose=var.RENDER)
arcs.Payload(geomfile[1], '/lion', purpose=var.PROXY)
arcs.Reference(col, '/$D')
arcs.DefaultPrim(assetName='lion')
arcs.DoIt()


# variant set test
reload(arc)
reload(utl)
srclayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion.usd')
pkglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion.payload.usd', create=True, clear=True)
stgmeta = utl.StageMetadata(srclayer)
arcs = arc.Arcs(pkglayer, stgmeta)
arcs.Payload(srclayer, '/lion{modelVersion=v007}')
arcs.DoIt()


# sublayer
reload(arc)
srclayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/v007/lion.payload.usd')
pkglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/model.usd', create=True, clear=True)
stgmeta = utl.StageMetadata(srclayer)
arcs = arc.Arcs(pkglayer, stgmeta)
arcs.Sublayer(srclayer)
arcs.DoIt()


# multi rererences, taskVariants
reload(arc)
mdllayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/model/model.payload.usd')
riglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/rig/rig.payload.usd')
pkglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/lion/lion.usd', create=True, clear=True)
stgmeta  = utl.StageMetadata(riglayer)
arcs = arc.Arcs(pkglayer, stgmeta)
arcs.Reference(mdllayer, '/lion')
arcs.Reference(riglayer, '/lion')
arcs.DefinePrim('/lion{taskVariant=model}')
arcs.DefaultPrim(kind='')
arcs.DoIt()


# payload then reference
reload(utl)
hgmlayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/bridge2_clip/crows_rig_GRP.high_geom.usd')
collayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/rig/usd/crows_rig_v01_latice/collection.usd')
pkglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/bridge2_clip/bridge2_clip.usd', create=True, clear=True)
stgmeta  = utl.StageMetadata(hgmlayer)
arcs = arc.Arcs(pkglayer, stgmeta)
arcs.DefinePrim('/crows')
arcs.Payload(hgmlayer, '/$D')
arcs.Reference(collayer, '/$D')
arcs.DoIt()


# sublayer overprim
reload(arc)
sublayers = [utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/bridge2_loop0_8/bridge2_loop0_8.payload.usd'),
             utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/bridge2_loop1_0/bridge2_loop1_0.payload.usd'),
             utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/bridge2_loop1_5/bridge2_loop1_5.payload.usd')]
pkglayer = utl.AsLayer('/works/tasks/DXUSD_test/show/asset/crow/clip/v008/loopClip.usd', create=True, clear=True)
stgmeta = utl.StageMetadata(sublayers[0])
for sub in sublayers:
    arcs = arc.Arcs(pkglayer, stgmeta)
    arcs.Sublayer(sub)
    arcs.OverPrim('/crows{loopVariant=bridge2_loop1_0}')
    arcs.DoIt()


# camera
reload(var)
path = '/works/tasks/DXUSD_test/show/shot/TST/TST_0010/cam/v002'
imglayer = utl.AsLayer('%s/FST_0080_main1_matchmove_imagePlane1.imp.usd'%path)
camlayer = utl.AsLayer('%s/FST_0080_main1_matchmove.geom.usd'%path)
pkglayer = utl.AsLayer('%s/camera.usd'%path, create=True, clear=True)
stgmeta = utl.StageMetadata(camlayer)
cmt = 'Generated with /show/imt/shot/FST/FST_0080/ani/dev/scenes/FST_0080_ani_v01_w05.mb'

arcs = arc.Arcs(pkglayer, stgmeta, comment=cmt)
arcs.DefinePrim('/cameras', custom={'scene':'FST_0080_ani_v01_w05.mb'})
arcs.DefaultPrim(kind=var.KIND.COM)
arcs.Payload(camlayer, '/$D/main_cam')
arcs.Payload(imglayer, '/$D/extra/$S')
arcs.AddAttribute('/$D/extra', 'primvars:ri:attributes:visibility:camera', 0)
arcs.AddAttribute('/$D/extra', 'primvars:ri:attributes:visibility:indirect', 0)
arcs.AddAttribute('/$D/extra', 'primvars:ri:attributes:visibility:transmission', 0)
arcs.DoIt()


# test copySpec
reload(arc)
fdir = '/show/pipe/template/DXUSD-2.0/show/tst/asset/lion/v001'
srclyr = utl.AsLayer('%s/lion_model_GRP.high_geom_org.usd'%fdir)
dstlyr = utl.AsLayer('%s/lion_model_GRP.high_attr.usd'%fdir, clear=True, create=True)

stgmeta = utl.StageMetadata()
arcs = arc.Arcs(dstlyr, stgmeta)

dspec = utl.GetDefaultPrim(srclyr)
arcs.ClassPrim('_%s'%dspec.name)

sspec = srclyr.GetPrimAtPath('/lion_model_GRP/lion_claw_GRP/lion_claw_FR_GRP/lion_claw_FR_001_Mplastic_PLY')
arcs.CopySpec(sspec,
              '/_lion_model_GRP/lion_claw_GRP/lion_claw_FR_GRP/lion_claw_FR_001_Mplastic_PLY',
              ['doubleSided', 'primvars:txLayerName'])
arcs.DoIt()

# test variants
reload(arc)
fdir = '/show/pipe/template/DXUSD-2.0/show/tst/asset/lion/v001'
lyr  = utl.AsLayer('%s/test_variants.usd'%fdir, clear=True, create=True)

arcs = arc.Arcs(lyr)
arcs.DefinePrim('/a/b/c{k=v}d/e{f=a}g')
arcs.DoIt()

# test inherit
reload(arc)
fdir = '/show/pipe/template/DXUSD-2.0/test_arcs'
lyr  = utl.AsLayer('%s/test_inherit_org.usd'%fdir)
lyr.Export('%s/test_inherit.usda'%fdir)
lyr  = utl.AsLayer('%s/test_inherit.usda'%fdir)
lyr.Reload()

arcs = arc.Arcs(lyr)
arcs.Inherit(Sdf.Path('/_class_Tree'), lyr.GetPrimAtPath('/TreeA'))
arcs.Inherit('/_class_Tree2', '/TreeC')
arcs.DoIt()
