#coding:utf-8
from __future__ import print_function

import DXUSD_HOU.Vars as var
import DXUSD_HOU.PostJobs as post
from DXUSD_HOU.Exporters import AFeatherExporter


if __name__ == '__main__':

    arg = AFeatherExporter()

    arg.show = 'pipe'
    arg.asset = 'babyPenguin'
    arg.task = 'groom'
    arg.nslyr = 'babyPenguin_feather_v003'
    arg.dependRigVer = 'babyPenguin_rig_v003'
    arg.dstlyr = '/show/pipe/_3d/asset/babyPenguin/groom/babyPenguin_feather_v003/babyPenguin_groom.usd'



    # <Dictionary>
	# 	task : groom
	# 	lod : guide
	# 	show : pipe
	# 	pub : _3d
	# 	nslyr : babyPenguin_feather_v003
	# 	asset : babyPenguin
	# 	root : /show
	# 	subdir : groomer_allFeathers
	#     <Attributes>
	# 	sequenced : False
	# 	meta : [Stage Metadata]
	# 	      frames : 2 - 2
	# 	      fps : 24
	# 	      tps : 24
	# 	      up axis : Y
	# 	      customData : {'dxusd': '2.0.0', 'sceneFile': '/show/pipe/works/CSP/wonchul.kang/babyPenguin/babyPenguin_jin_v001.hip', 'rigFile': '/show/pipe/_3d/asset/babyPenguin/rig/babyPenguin_rig_v003/babyPenguin_rig.usd'}
    #
	# 	customData : {}
	# 	prctype : none
	# 	dependRigVer : babyPenguin_rig_v003
	# 	cliprate : None
	# 	dependLowFeather : None
	# 	dependHighFeather : None
	# 	dstlyr : /show/pipe/_3d/asset/babyPenguin/groom/babyPenguin_feather_v003/babyPenguin_groom.usd
	# 	srclyr : Sdf.Find('/show/pipe/_3d/asset/babyPenguin/groom/babyPenguin_feather_v003/groomer_allFeathers/groomer_allFeathers.high_geom.usd')
	# 	isRigSrc : False
	# 	taskProduct : GEOM
	# 	dependOrgFeather : None
	# 	lyrtype : feather
	# 	taskCode : TASKN

    job = post.FeatherPostJobs(arg)
    if job.Treat() == var.SUCCESS:
        job.DoIt()
