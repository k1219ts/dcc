#encoding=utf-8
#!/usr/bin/env python

import os, sys
import getpass
import site
import dxConfig

platform = 'linux'
pyversion = 'Tractor-2.2/lib/python2.7'
packpath = 'site-packages'

if len(os.getenv('BACKSTAGE_PATH')):
    root = os.getenv('BACKSTAGE_PATH')
else:
    if sys.platform == 'darwin':  # mac
        root = '/Volumes/10.0.0.248/dexter/netapp/backstage/pub'
        platform = 'mac'
    elif sys.platform == 'linux2':  # linux
        root = '/netapp/backstage/pub'
        platform = 'linux'
    else:  # window
        root = 'N:/backstage/pub'
        platform = 'win64'
        packpath = 'Lib/site-packages'

TractorSite = '%s/apps/tractor/%s/%s/%s' % (root, platform, pyversion, packpath)
Tractor_IP = '10.0.0.25'
try:
    Tractor_IP = dxConfig.getConf('TRACTOR_REDSHIFT_IP')
except:
    print ' <<WARNNING>> Check dxConfig...!!'

NukeVers = '9.0v5'
if Tractor_IP != '10.0.0.25': # DEXTER CHINA
    NukeVers = '10.0v4'
NukePath = '/usr/local/Nuke%s/Nuke%s' % ( NukeVers, NukeVers.split('v')[0] )

site.addsitedir( TractorSite )
import tractor.api.author as author

class JobScript():
    def __init__( self, options ):
        self.m_opt          = options
        self.speedck = self.m_opt['speedck']
        self.start_frame    = self.m_opt['start']  # 6
        self.frames_origin  = self.m_opt['frame_range'] # [(6, 17), (18, 30)]
        self.frames         = self.m_opt['frame_range_retime'] # [(6, 30), (31, 55)]
        self.step           = self.m_opt['step']  # 2.0 - 리타임 값
        self.fps            = self.m_opt['fps']   # 24.0

        self.m_script_path  = str(self.m_opt['script_path'])#rnd
        self.m_scene_file_origin = str(self.m_opt['scene_file_origin'])   # /netapp/dexter/asset/redshift/test/scenes/redshift_test_proxy_v010.mb
        self.m_scene_file   = str(self.m_opt['scene_file'])               # /netapp/dexter/asset/redshift/test/render/redshift_test_proxy_v010_seonku.kim_rs8864.mb
        self.m_scene_path   = os.path.dirname(self.m_scene_file)          # /netapp/dexter/asset/redshift/test/render

        if self.m_scene_file.find('/scenes/') != -1:  # scene 폴더가 있는 경우
            self.m_scene_proj = '/'.join(self.m_scene_path.split('/')[:-1]) # /netapp/dexter/asset/redshift/test
        else:
            self.m_scene_proj = self.m_scene_path

        self.m_scene_name   = os.path.splitext(os.path.basename(self.m_scene_file))[0]  # redshift_test_proxy_v010_seonku.kim_rs8864
        self.m_scene_base   = '_'.join(self.m_scene_name.split('_')[:-2])               # redshift_test_proxy_v010
        self.m_out_format   = self.m_opt['output_format']  # exr

        # 레이어 이름이 rs_로 시작되는 경우 rs_ 제거
        self.m_layer_name   = str(self.m_opt['layer_name'])
        if self.m_layer_name.find('rs_') == 0:
            self.m_layer_name = self.m_layer_name[3:]
        self.m_camera_name  = str(self.m_opt['camera_name'])
        self.m_shot         = str(self.m_opt['shot'])
        self.m_camera_seqc  = self.m_opt['camera_seqc'] # 0 or 1

        # 카메라 시퀀스를 사용한 경우 shot 네임 변경 처리.
        self.m_shot_name = self.m_scene_base
        if self.m_camera_seqc == 1:
            self.m_shot_name = self.m_shot

        self.m_camera_lens  = self.m_opt['camera_lens'] # 35.0
        self.m_camera_type  = self.m_opt['camera_type'] # standard, fisheye ...

        # 레이어 이름과 카메라 이름이 같은 경우 레이어 이름만 표시
        self.layer_name     = self.rmNamespace(self.m_layer_name)
        self.camera_name    = self.rmNamespace(self.m_camera_name.replace('Shape', ''))

        # 렌더링 이미지 이름 설정
        self.image_name = '%s' % self.camera_name
        self.title_name     = '[%s/%s]' % (self.layer_name, self.camera_name)
        if self.layer_name == self.camera_name:
            self.title_name = '[%s]' % self.layer_name

        self.output_root_path = '%s/images' % self.m_scene_proj
        self.m_output_path    = '%s/%s/%s' % (self.output_root_path, self.m_scene_base, self.layer_name)
        self.m_render_width = self.m_opt['render_width']
        self.m_render_height = self.m_opt['render_height']
        self.m_mov_format = self.m_opt['mov_format']

        # image plane 렌더링 여부 체크
        self.m_imageplane   = self.m_opt['renderImagePlane']  # 'FG 레이어갯수 BG 레이어갯수' --> '0:1'
        self.numFG = self.m_imageplane.split(':')[0]
        self.numBG = self.m_imageplane.split(':')[1]
        self.render_layer_list = self.m_opt['render_layer_list']

        if int(self.numFG) > 0 or int(self.numBG) > 0 or self.render_layer_list:
            self.preLayerOption = ['-rl', 'True', '-preLayer', 'redshift_preLayer']
        else:
            self.preLayerOption = ['-im', '%s' % self.image_name]

        self.m_preview_path = self.m_opt['preview_path']
        self.m_tags         = self.m_opt['tags'] # ['gpu_cache', 'gpu_render'] 리스트 타입만 사용가능
        self.m_envkey       = self.m_opt['envkey']  # redshift-2.5.27-2017
        self.m_service      = self.m_opt['service'] # Redshift||Cache

        self.m_user         = '%s' % self.m_opt['user']
        self.m_team         = '%s' % self.m_opt['team']
        self.m_show         = '%s' % self.m_opt['show']
        self.m_metadataStr  = self.getMovMetadata(self.m_scene_file_origin, self.m_user)
        self.m_log_level    = self.m_opt['log_level']

    def getMovMetadata(self, MayaFileFullPath, artistName):
        movMetadata = '\'{"mayaFilePath":"%s","artist":"%s"}\'' % (MayaFileFullPath, artistName)
        return movMetadata

    def getDuration(self, frames):
        st = int(frames[0])
        et = int(frames[1])
        dur = et - st + 1
        return st, et, dur

    def rmNamespace(self, src):
        result = src
        if src.find(':') != -1:
            result = src.split(':')[-1]
        return result

    def makePreviewTask(self, frames, frames_origin, step):
        export_frames = (frames[0][0], frames[-1][1])  # (1, 100)
        export_frames_origin = (frames_origin[0][0], frames_origin[-1][1])  # (1, 50)

        st, et, dur = self.getDuration(export_frames_origin)
        st2, et2, dur2 = self.getDuration(export_frames)

        job_title = str('Make Preview (%04d-%04d/%df)' % (st, et, dur))
        if step != 1.00:
            job_title = str('Make Preview (%04d-%04d/%df)' % (st2, et2, dur2))

        # 최종 태스크 정의
        makePreviewTaskGrp = author.Task( title=job_title )
        # makePreviewTaskGrp.serialsubtasks = 1
        command_preview =  [NukePath, '-t']
        command_preview += ['%D({0}/nukeMov.py)'.format(self.m_script_path)]
        command_preview += ['%%D(%s/%s)' % (self.m_output_path, self.image_name), '%%D(%s)' % self.m_preview_path,
                            str(st2), str(et2),
                            self.m_scene_base, self.m_show, self.m_user, step,
                            self.fps, self.m_camera_name, self.m_camera_lens,
                            self.m_metadataStr, self.m_out_format,
                            self.m_imageplane, self.m_camera_seqc,
                            self.m_shot_name,
                            self.m_mov_format,
                            self.m_camera_type, # sys.argv[18]
                            self.speedck # speed append
                            ]
        makePreviewTaskGrp.addCommand(
            author.Command(service='Nuke', tags=['%s' % self.m_envkey], envkey=[self.m_envkey], argv=command_preview))

        if (self.m_camera_type == 'Fisheye' and self.speedck == 'ok') or self.speedck == 'ok':
        # Direct Render 태스크 생성
            insertHeaderTask = self.insertToExrHeaderTask(st, et, pad=5)
            self.directRenderTask(st, et, step, frames, frames_origin, insertHeaderTask)
            makePreviewTaskGrp.addChild(insertHeaderTask)
        else:
            self.directRenderTask(st, et, step, frames, frames_origin, makePreviewTaskGrp)
        return makePreviewTaskGrp  # makePreviewTaskGrp 노드를 태스크 루트 노드로 반환

    def directRenderTask(self, st, et, step, frames, frames_origin, parentTaskGrp):
    # Direct Render 태스크 생성 : frame별 랜더테스트 생성
        increment = round(1.0/float(step), 3) # 1.0 / 1.6 = 0.625
        increment_st = round(float(st), 3)
        et_cal = 0.0
        command_retime = list()

        for f in range(len(frames)):  # [(1, 25), (26, 50), (51, 75), (76, 100)]
            st, et, dur = self.getDuration(frames_origin[f])  # (1, 50)
            st = round(float(st), 3)
            et = round(float(et), 3)

            job_title = str( 'Render (%04d-%04d/%df)' % (st, et, dur) )

            # 리타임인 경우 처리
            st2, et2, dur2 = self.getDuration(frames[f])  # (1, 25)
            st2 = round(float(st2), 3)
            et2 = round(float(et2), 3)
            if step != 1.00:
                st = increment_st
                et_cal = st + ( (dur2 - 1)  * increment )
                et = round( et_cal, 2 ) + ( increment/2 )

                job_title = str( 'Render (%04d-%04d/%df)' % (st2, et2, dur2) )
                command_retime = ['-rfs', st2, '-rfb', 1]

                command_cleanup = ['python', '%D({0}/cleanup.py)'.format(self.m_script_path),
                                   '%%D(%s/%s)' % (self.m_output_path, self.image_name),
                                   st2, et2, 1, '.lock'
                                   ]
            else:
                command_cleanup = ['python', '%D({0}/cleanup.py)'.format(self.m_script_path),
                                   '%%D(%s/%s)' % (self.m_output_path, self.image_name),
                                   st, et, increment, '.lock'
                                   ]

            directRender = author.Task( title=job_title )

            command_render = [
                              'Render', '-r', 'redshift',
                              '-s', st, '-e', et , '-b', increment, '-pad', '5'
                             ]
            command_output = [
                              '-cam', self.m_camera_name,
                              '-proj', self.m_scene_proj,
                              '-rd', '%%D(%s)' % self.m_output_path,
                              '-x', self.m_render_width,
                              '-y', self.m_render_height
                             ]
            command_post   = ['-preRender', 'redshift_preRender', '-logLevel', self.m_log_level, #'-progressive', 1024,
                               '%%D(%s)' % self.m_scene_file]

            command_render.extend(command_retime)# 리타임 옵션 추가
            command_render.extend(command_output)# 카메라 및 아웃풋 옵션 추가
            command_render.extend(self.preLayerOption)# 렌더레이어 옵션 추가
            command_render.extend(command_post)# 최종 렌더링 커맨드 조합
            directRender.addCommand(# 클린업 프레임 구간 별 실행 명령어
                author.Command(service='', tags=['%s' % self.m_envkey], envkey=[self.m_envkey], argv=command_cleanup))

            directRender.addCommand(# 최종 렌더링 실행 명령어
                author.Command(service='', tags=['%s' % self.m_envkey], envkey=[self.m_envkey], argv=command_render))

            # 다음 프레임 구간을 시작할 때 이전 프레임 구간동안의 프레임 증가량 계산 적용
            increment_st = et_cal + increment

            # Make Preview 태스크 그룹에 서브로 지정
            parentTaskGrp.addChild(directRender)

    def insertToExrHeaderTask(self, st, et, pad=5):
    # speed > exr
        insertEXRTask = author.Task(title='Insert EXR Header')
        command = [
            'mayapy', '%D({0}/insert_exr_header.py)'.format(self.m_script_path),
            '-s', str(st).zfill(pad), '-e', str(et).zfill(pad), '-pad', pad,
            '-cam', self.m_camera_name,
            '-rd', '%%D(%s)' % self.m_output_path,
            '-im', '%s' % self.image_name,
            '-mb', '%%D(%s)' % self.m_scene_file ]

        insertEXRTask.addCommand(
            author.Command(service='', tags=['%s' % self.m_envkey], envkey=[self.m_envkey], argv=command)
        )
        return insertEXRTask

    def doIt( self ):
        job = author.Job()
        # Check frame_range limit이 2이상일 경우 시작 프레임과 끝 프레임 계산 처리
        if len(self.frames) > 1: # [(6, 30), (31, 55)]
            chk_frames = (self.frames[0][0], self.frames[-1][-1]) # (6, 55)
        else:
            chk_frames = self.frames[0] # (6, 55)

        st, et, dur = self.getDuration(chk_frames) # 리타임 적용된 프레임 정보 구하기 : st = 6, et = 55, dur = 50
        frame_range = '%d-%d(%df)' % (st, et, dur)

        if self.step != 1.0: # 리타임샷의 경우 뒤에 정보 표시 추가
            frame_range = '%s Retime: %s' % (frame_range, self.step)

        # job 기본 정보 설정
        job.title     = '(RED-%s) %s %s %s' % (self.m_opt['tier'], self.m_scene_base, self.title_name, frame_range)
        job.envkey    = [ '%s' % self.m_envkey ] #['redshift-2.0.87-2017']
        job.service   = self.m_service #'Redshift||Cache' #'Lofn-041'
        job.maxactive = self.m_opt['maxactive']
        job.tier      = self.m_opt['tier']  #'GPU'
        job.projects  = ['%s' % self.m_opt['projects']]  #['ani, render']
        job.tags      = self.m_opt['tags']
        job.metadata  = str(self.m_output_path)
        job.comment   = str(self.m_preview_path)

        # directory mapping
        job.newDirMap(src='X:/', dst='/netapp/dexter/show/', zone='NFS')
        job.newDirMap(src='X:/', dst='/dexter/show/', zone='NFS')
        job.newDirMap(src='X:/', dst='/show/', zone='NFS')
        job.newDirMap(src='N:/', dst='/netapp/', zone='NFS')
        job.newDirMap(src='R:/', dst='/dexter/', zone='NFS')
        job.newDirMap(src='R:/', dst='/data/', zone='NFS')
        job.newDirMap(src='T:/', dst='/tactic/', zone='NFS')

        # direct render task( makepreview >> insert exr >> direct )
        job.serialsubtasks = 1 # 서브 태스크들이 순서대로 진행되도록 설정
        makePreviewTaskGrp = self.makePreviewTask(self.frames, self.frames_origin, self.step)

        # job의 서브태스크들 지정
        job.addChild(makePreviewTaskGrp)

        # spool
        job.priority = self.m_opt['priority']
        author.setEngineClientParam(
                hostname=Tractor_IP, port=80,
                user=getpass.getuser(), debug=True )
        job.spool()
        author.closeEngineClient()

