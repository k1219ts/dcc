#coding:utf-8
from __future__ import print_function
import os

from DXUSD.Structures import Arguments
import DXUSD.Vars as var
import DXUSD.Utils as utl


#-------------------------------------------------------------------------------
#
#   Model
#
#-------------------------------------------------------------------------------
def asset_model(inputname):
    print('----------------------')
    print('    Model -- asset')
    print('----------------------')
    # inputname = 'bear_model_GRP'

    arg = Arguments()
    arg.show = 'pipe'
    arg.taskProduct = 'TASKV'

    arg.N.model.SetDecode(inputname)

    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    arg.ver = utl.GetLastVersion(arg.D.TASK)
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKV, arg.F.MASTER))
    for lod in [var.T.HIGH, var.T.MID, var.T.LOW]:
        arg.lod = lod
        print('> gome\t\t:', utl.SJoin(arg.D.TASKV, arg.F.GEOM))


def asset_model_set(inputname):
    print('--------------------------')
    print('    Model Set -- asset')
    print('--------------------------')
    # inputname = 'asdalTown_set_asb'

    arg = Arguments()
    arg.show = 'pipe'
    arg.taskProduct = 'TASKV'

    arg.N.model.SetDecode(inputname)

    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    arg.ver = utl.GetLastVersion(arg.D.TASK)
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKV, arg.F.MASTER))
    print('> gome\t\t:', utl.SJoin(arg.D.TASKV, arg.F.GEOM))


#-------------------------------------------------------------------------------
#
#   Rig
#
#-------------------------------------------------------------------------------
def rig_asset():
    print('----------------------')
    print('    Rig -- asset')
    print('----------------------')
    inputname = 'bear_rig_GRP'
    workfile  = '/show/pipe/works/rig/bear_rig_v004.mb'
    basename  = os.path.basename(workfile).split('.')[0]

    arg = Arguments()
    arg.N.rig.SetDecode(inputname)
    arg.D.SetDecode(utl.DirName(workfile), 'ROOTS')
    arg.F.MAYA.SetDecode(utl.BaseName(workfile), 'WORK')
    arg.nslyr= basename

    print('> inputname\t:', inputname)
    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKN, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKN, arg.F.MASTER))
    arg.lod = var.T.HIGH
    print('> geom\t\t:', utl.SJoin(arg.D.TASKN, arg.F.GEOM))
    arg.lod = var.T.LOW
    print('> geom\t\t:', utl.SJoin(arg.D.TASKN, arg.F.GEOM))


def rig_shot():
    print('----------------------')
    print('    Rig -- shot')
    print('----------------------')
    inputname = 'bear:bear_rig_GRP'
    workfile  = '/show/srh/works/ani/S26_0450_ani_v003_baked.mb'

    arg = Arguments()
    arg.N.ani.SetDecode(inputname)
    arg.D.SetDecode(utl.DirName(workfile), 'ROOTS')
    arg.F.MAYA.SetDecode(utl.BaseName(workfile), 'WORK')

    print('> inputname\t:', inputname)
    print(arg)
    print('> shot dir\t:', arg.D.SHOT)
    print('> shot payload\t:', utl.SJoin(arg.D.SHOT, arg.F.SHOT_PAY))
    print('> shot file\t:', utl.SJoin(arg.D.SHOT, arg.F.SHOT))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> nslyr payload\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR_PAY))
    print('> nslyr file\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR))
    arg.nsver = utl.GetLastVersion(arg.D.TASKN)
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKNV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKNV, arg.F.MASTER))
    arg.lod = var.T.HIGH
    print('> geom\t\t:', utl.SJoin(arg.D.TASKNV, arg.F.GEOM))
    arg.lod = var.T.LOW
    print('> geom\t\t:', utl.SJoin(arg.D.TASKNV, arg.F.GEOM))
    print('> xform\t\t:', utl.SJoin(arg.D.TASKNV, arg.F.XFORM))


#-------------------------------------------------------------------------------
#
#   Groom
#
#-------------------------------------------------------------------------------
def groom_asset():
    print('----------------------')
    print('    Groom -- asset')
    print('----------------------')
    inputname = 'bear_ZN_GRP'

    arg = Arguments()
    arg.show = 'pipe'
    arg.taskProduct = 'TASKN'
    arg.N.groom.SetDecode(inputname, 'ASSET')
    arg.nslyr = 'bear_hair_v005'

    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKN, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKN, arg.F.MASTER))
    zn_nodes = ['bear_body_ZN_Deform', 'bear_ear_ZN_Deform', 'bear_head_ZN_Defrom']
    for s in zn_nodes:
        arg.subdir = s
        print('> output\t:', utl.SJoin(arg.D.TASKNS, arg.subdir))


def groom_shot():
    print('----------------------')
    print('    Groom -- shot')
    print('----------------------')
    inputname = 'bear_ZN_GRP'
    inputcache= '/show/pipe/_3d/shot/S26/S26_0450/ani/bear/v001/bear.usd'

    arg = Arguments()
    arg.taskProduct = 'TASKNVS'
    arg.D.SetDecode(utl.DirName(inputcache), 'ROOTS')
    arg.N.groom.SetDecode(inputname, 'SHOT')

    print(arg)
    print('> shot dir\t:', arg.D.SHOT)
    print('> shot file\t:', utl.SJoin(arg.D.SHOT, arg.F.SHOT))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> nslyr payload\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR_PAY))
    print('> nslyr file\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR))
    arg.nsver = utl.GetLastVersion(arg.D.TASKN)
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKNV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKNV, arg.F.MASTER))
    zn_nodes = ['bear_body_ZN_Deform', 'bear_ear_ZN_Deform', 'bear_head_ZN_Defrom']
    for s in zn_nodes:
        arg.subdir = s
        print('> output\t:', utl.SJoin(arg.D.TASKNVS, arg.subdir))
        # for lod in var.T.LODS:
        #     arg.lod = lod
        #     for frame in [0, 1000]:
        #         arg.frame = '%04d'%frame
        #         print('> output %s\t:'%lod, utl.SJoin(arg.D.TASKNVS, arg.F.GEOM))





#-------------------------------------------------------------------------------
#
#   Clip
#
#-------------------------------------------------------------------------------
def clip_asset():
    print('------------------------------')
    print('    Clip -- asset -- rig')
    print('------------------------------')
    inputname= 'pudacuoHorse_rig_GRP'
    arg = Arguments()
    arg.show = 'pipe'
    arg.taskProduct = 'TASKNVC'
    arg.nslyr = 'runA'
    arg.clip  = 'base'
    # arg.task = 'clip'
    # arg.asset= 'pudacuoHorse'

    arg.N.clip.SetDecode(inputname, 'ASSET')
    arg.desc = inputname

    print('> inputname\t:', inputname)
    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> nslyr payload\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR_PAY))
    print('> nslyr file\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR))
    arg.nsver = utl.GetLastVersion(arg.D.TASKN)
    #print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKNV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKNV, arg.F.MASTER))
    # clip base
    arg.clip = 'base'
    print('> base payload\t:', utl.SJoin(arg.D.TASKNVC, arg.F.PAYLOAD))
    print('> base file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.CLIP))
    arg.lod = var.T.HIGH
    print('> geom file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.GEOM))
    arg.lod = var.T.LOW
    print('> geom file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.GEOM))

    time_scales = ['loop_0_8', 'loop_1_0', 'loop_1_5']
    for s in time_scales:
        arg.clip = s
        print('> loop payload\t:', utl.SJoin(arg.D.TASKNVC, arg.F.PAYLOAD))
        print('> loop file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.CLIP))
        arg.lod = var.T.HIGH
        print('> geom file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.GEOM))
        arg.lod = var.T.LOW
        print('> geom file\t:', utl.SJoin(arg.D.TASKNVC, arg.F.GEOM))

    print('------------------------------')
    print('    Clip -- asset -- groom')
    print('------------------------------')
    time_scales.insert(0, 'base')
    zn_nodes   = ['pudacuoHorse_eye_ZN_Deform', 'pudacuoHorse_tail_A_ZN_Deform']

    for s in time_scales:
        arg.clip = s
        print('> groom final\t:', utl.SJoin(arg.D.TASKNVC, arg.F.GROOM))
        for n in zn_nodes:
            arg.subdir = n
            print('> output\t:', utl.SJoin(arg.D.TASKNVCS, arg.subdir))
            # print('> output base\t:',  utl.SJoin(arg.D.TASKNVCS, arg.F.GROOMGEOM))



#-------------------------------------------------------------------------------
#
#   Crowd
#
#-------------------------------------------------------------------------------
def crowd_asset():
    print('-----------------------------')
    print('    Crowd -- asset(agent)')
    print('-----------------------------')
    inputname = 'OriginalAgent_crdMainStreet_man'   #   *_$ASSETNAME_$AGTYPE
    # inputname = 'Geometry_crdMainStreet_man'
    arg = Arguments()
    arg.show = 'pipe'
    arg.taskProduct = 'TASKNV'
    arg.N.agent.SetDecode(inputname, 'ASSET')
    arg.N.agent.SetDecode(arg.nslyr, 'SETASSET')

    print('> inputname\t:', inputname)
    print(arg)
    print('> asset dir\t:', arg.D.ASSET)
    print('> asset file\t:', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('> task payload\t:', utl.SJoin(arg.D.TASK, arg.F.TASK_PAY))
    print('> task file\t:', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('> agt payload\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR_PAY))
    print('> agt file\t:', utl.SJoin(arg.D.TASKN, arg.F.NSLYR))
    arg.nsver = utl.GetLastVersion(arg.D.TASKN)
    print('> GEOM PAYLOAD\t:', utl.SJoin(arg.D.TASKNV, arg.F.PAYLOAD))
    print('> GEOM MASTER\t:', utl.SJoin(arg.D.TASKNV, arg.F.MASTER))
    print('> collection\t:', utl.SJoin(arg.D.TASKNV, arg.F.COLLECTION))
    print('> attr file\t:', utl.SJoin(arg.D.TASKNV, arg.F.ATTR))
    print('> geom file\t:', utl.SJoin(arg.D.TASKNV, arg.F.GEOM))
    print('> skel file\t:', utl.SJoin(arg.D.TASKNV, arg.F.SKEL))


#-------------------------------------------------------------------------------
#
# MTK
#
#-------------------------------------------------------------------------------
def mtk_model():
    print('-----------------------------')
    print('    MTK -- model')
    print('-----------------------------')
    filename = '/show/slc/works/AST/e45dog/tmp/asset/e45dog/model/v000/e45dog_model.usd'
    arg = Arguments()
    arg.D.SetDecode(utl.DirName(filename))
    print(arg)
    print('>', filename)
    print('>', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('>', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('>', utl.SJoin(arg.D.ROOTS, 'asset', arg.F.ASSETS))

def mtk_groom():
    print('-----------------------------')
    print('    MTK -- groom')
    print('-----------------------------')
    # filename = '/show/slc/works/AST/e45dog/tmp/asset/e45dog/groom/e45dog_new_hair_v01_w08_jin/e45dog_groom.usd'
    filename = '/show/slc/works/AST/e45dog/groom/tmp/asset/e45dog/groom/e45dog_new_hair_v01_w08_jin/e45dog_groom.usd'
    arg = Arguments()
    arg.D.SetDecode(utl.DirName(filename))
    print(arg)
    print('>', filename)
    print('>', utl.SJoin(arg.D.TASK, arg.F.TASK))
    print('>', utl.SJoin(arg.D.ASSET, arg.F.ASSET))
    print('>', utl.SJoin(arg.D.ROOTS, 'asset', arg.F.ASSETS))


#-------------------------------------------------------------------------------
#
# DOIT
#
#-------------------------------------------------------------------------------
mtk_model()
mtk_groom()
