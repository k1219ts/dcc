# encoding=utf-8
# !/usr/bin/env python
# -------------------------------------------------------------------------------
#
#   DEXTER STUDiOS
#
#   CG Supervisor	: seonku.kim
#
# -------------------------------------------------------------------------------
import os, sys, random, string, site, shutil, getpass
from datetime import timedelta, datetime, date, time
from collections import namedtuple
import main
try:
    import maya.cmds as cmds
    import maya.mel as mel
    import pymel.core as pm
except ImportError:
    pass

##############################################
# Global Variables
##############################################
# Set Script Path
if len(os.getenv('BACKSTAGE_PATH')):
    root = os.getenv('BACKSTAGE_PATH')
else:
    if sys.platform == 'darwin':  # mac
        root = '/Volumes/10.0.0.248/dexter/netapp/backstage/pub'
    elif sys.platform == 'linux2':  # linux
        root = '/netapp/backstage/pub'
    else:  # window
        root = 'N:/backstage/pub'

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
script_path = CURRENT_DIR  # '/dexter/Cache_DATA/RND/youkyoung/redshiftmaya/fource'
# sys.path.insert(0, CURRENT_DIR)
site.addsitedir(script_path)
##############################################
# Set Script Path and Import Modules
##############################################
import redshift_Spool
reload(redshift_Spool)

class LayoutRedshift():
    def __init__(self, args):
        self.chunkSize = args['chunkSize']
        self.maxActive = args['maxActive']
        self.speed = args['speed']
        self.limitTag = args['limitTag']
        self.frames = args['frames']
        self.teamName = args['team']

        self.settingInit()
        self.getSceneInfo(self.currentScene)
        self.farmSetting(self.limitTag)
        self.renderList(self.frames) # self.renderlist
        self.getRenderLayers() # self.renderlayer
        self.jobRender()

    def settingInit(self):
        # 0:standard, 2:fisheye, 3:spherical, 4:cylindrical, 5:stereo spherical
        self.stampFormatDict = ['Standard', 'Standard', 'Fisheye', 'Spherical', 'Cylindrical', 'Stereo Spherical']
        self.username = getpass.getuser()
        self.projects = 'redshift'
        self.currentScene = cmds.file(q=True, sn=True)
        maya_version = str(cmds.about(v=True))  # '2017'
        try:
            redshift_version = str(cmds.pluginInfo('redshift4maya', q=True, v=True))#'redshift-2.5.21-2017'
        except Exception, e:  # 플러그인 로딩이 안되는 경우 기본 버전 처리
            redshift_version = '2.5.52'#2.5.70
        self.envkey = 'redshift-%s-%s' % (redshift_version, maya_version)

    def getSceneInfo(self, currentScene = None):
    # preview 폴더와 씬이름 확장자 분리 저장 : self.mayaProj, self.scene_base, self.scene_ext
        filename_origin = os.path.basename(currentScene)  # redshift_test_proxy_v010.mb
        if currentScene.find('/scenes/') != -1:  # scene 폴더가 있는 경우
            self.mayaProj = '/'.join(os.path.dirname(currentScene).split('/')[:-1])
        else:
            self.mayaProj = os.path.dirname(currentScene)
        self.scene_base = os.path.splitext(filename_origin)[0]  # redshift_test_proxy_v010
        self.scene_ext = os.path.splitext(filename_origin)[-1]  # .mb

    def farmSetting(self, limitTag = None):
    # Tractor Render Farm service key setting
        self.service = 'GPUFARM'
        if limitTag:
            if 'gpu_cache' in limitTag:
                self.service += '||Cache'
            if 'gpu_user' in limitTag:
                self.service += '||USERGPU'

    def renderList(self, frames = None):
    # 랜더 정보를 쿼리
        self.renderlist = {}
        self.getCameraSequencer() # 카메라 시퀀스를 사용할 경우 그에 대한 shot 정보들을 가져옴.
        if not self.renderlist: # 카메라 시퀀스를 사용하지 않는 경우
            st, et, by = self.getFrameInfo(frames)
            renders = self.getRenderCameras()
            for cam in renders:
                focalLengh = self.getFocalLength(cam)  # 35.0
                rsCameraType = self.getCameraType(cam)  # S or F or P
                output_format = self.getOutputFormat()  # iff

                camname = cam.replace('Shape', '')
                self.renderlist[camname] = {'start': st, 'end': et, 'step': by, 'focalLength': focalLengh,
                                   'camera_type': rsCameraType, 'camera': cam,
                                   'layer': camname, 'shot': '', 'useCameraSequencer': 0,
                                   'output_format': output_format}

    def getCameraSequencer(self):
        shots = cmds.sequenceManager(listShots=True)
        if shots:  # 카메라 시퀀스를 사용하는 경우
            shots.sort()
            for i in shots:  # 카메라 시퀀스의 shot 노드이름 별로 필요한 정보를 가져옴.
                muteshot = cmds.shot(i, q=True, sm=True)  # mute 처리된 shot 체크
                if not muteshot:  # mute 되지 않은 샷의 경우 렌더링 진행할 수 있게 처리
                    sc = i
                    sn = cmds.shot(i, q=True, shotName=True)
                    st = cmds.shot(i, q=True, st=True)  # start frame is st
                    et = cmds.shot(i, q=True, et=True)  # end frame is et
                    by = cmds.shot(i, q=True, scale=True)  # scale is by

                    # 렌더링 카메라가 없는 경우 예외 처리
                    try:
                        cam = cmds.shot(i, q=True, currentCamera=True)
                        rendercam = self.getCameraShape(cam) #camera shapename get
                        focalLengh = self.getFocalLength(rendercam)  # 35.0
                        rsCameraType = self.getCameraType(rendercam) # standard, fisheye.....
                        output_format = self.getOutputFormat()  # jpg, exr, iff
                        self.renderlist[i] = {'start': st, 'end': et, 'step': by, 'focalLength': focalLengh,
                                              'camera_type': rsCameraType, 'camera': rendercam, 'layer': sc,
                                              'shot': sn, 'useCameraSequencer': 1, 'output_format': output_format}
                    except:
                        main.messageBox('Not Found Camera to Rendering.', 'error', ['OK'])
                        main.closeUI()

    def getCameraShape(self, cam):
    # camera shape name get
        camList = cmds.listRelatives(cam)
        for item in camList:
            if pm.objectType(item) == 'camera':
                return item

    def getFocalLength(self, cameraShape):
    # camera length
        result = cmds.getAttr('%s.focalLength' % cameraShape)
        return result

    def getCameraType(self, cameraShape):
    # 카메라 타입 설정 : 랜더러가 redshift 일 경우 설정 가능
    # redshift plugin not loading > standard camera setting
        try:
            result = cmds.getAttr('%s.rsCameraType' % cameraShape)  # 2 - Fisheye
        except:
            result = 0
        result = self.stampFormatDict[result]
        return result

    def getOutputFormat(self):
        ext = {0: 'iff', 1: 'exr', 2: 'png', 3: 'tga', 4: 'jpg', 5: 'tif'}
        # picnum = 0
        imgformatIndex = cmds.getAttr("redshiftOptions.imageFormat")
        # if cameratype == 'Fisheye' and self.speed == 'ok':# fisheye >> exr output
        #     picnum = 1
        # else:
        #     picnum = imgformatIndex
        result = ext[imgformatIndex]
        # cmds.setAttr("redshiftOptions.imageFormat",imgformatIndex)
        return result

    def getFrameInfo(self, frames):
        if frames:  # 프레임 정보가 사전에 주어진 경우
            by = 1.0
            frames = frames.split('/')  # "1001.0/1200.0/1.0"
            st = frames[0]  # '1001.0' or '1001'
            et = frames[1]
            by = frames[2]
        else:  # 프레임 정보가 사전에 주어지지 않은 경우 씬 파일에서 가져와 사용
            st = cmds.playbackOptions(q=True, min=True)
            et = cmds.playbackOptions(q=True, max=True)
            by = cmds.playbackOptions(q=True, by=True)  # 1.0
        return st, et, by

    def getRenderCameras(self):
        result = []
        for cam in cmds.ls(type='camera'):
            if cmds.getAttr('%s.renderable' % cam):
                result.append(cam)
        return result

    def getRenderLayers(self):
        self.renderlayer = []
        for i in cmds.ls(type='renderLayer'):
            if len(i.split(':')) == 1 and cmds.getAttr('%s.renderable' % i):
                if i.find('defaultRenderLayer') != -1 or i.find('globalRender') != -1:
                    # 만약 defaultRenderLayer만 있는 경우엔 레이어 구분을 하지 않게 처리
                    pass
                else:
                    self.renderlayer.append(i)

    def jobRender(self):
        jobList = []
        for i in self.renderlist:
            options = {}
            filename, renderScene, previewRoot = self.renderFileName()
            imagePlanes = cmds.ls(type='imagePlane')
            chkFG, chkBG = self.imagePlanLayer(imagePlanes)
            render_width, render_height = self.renderResolution()
            fps = mel.eval('currentTimeUnitToFPS')

            # Preview mov 경로 정의
            cameraname = self.renderlist[i]['camera']
            layername = self.renderlist[i]['layer']
            previewName = self.previewMov(cameraname, layername)

            # Preview mov 사이즈 정의
            if render_width >= 1920:
                options['mov_format'] = 'B'  # 1920x1080
            else:
                options['mov_format'] = 'A'  # 1280x720

            # Render Image task split
            # [(1001, 1125), (1126, 1250), (1251, 1375), (1376, 1500), (1501, 1625), (1626, 1750)]
            st = int(float(self.renderlist[i]['start']))
            et = int(float(self.renderlist[i]['end']))
            by = int(float(self.renderlist[i]['step']))
            framerange = self.retimeScale(st, et, by)
            # show name get
            showname = self.checkShow(self.mayaProj)

            options['renderImagePlane'] = '{0}:{1}'.format(chkFG, chkBG)
            options['render_layer_list'] = self.renderlayer
            options['layer_name'] = self.renderlist[i]['layer']
            options['camera_name'] = self.renderlist[i]['camera']
            options['camera_seqc'] = self.renderlist[i]['useCameraSequencer']
            options['camera_lens'] = self.renderlist[i]['focalLength']
            options['camera_type'] = self.renderlist[i]['camera_type']
            options['shot'] = self.renderlist[i]['shot']
            options['render_width'] = render_width
            options['render_height'] = render_height
            options['start'] = self.renderlist[i]['start']
            options['end'] = self.renderlist[i]['end']
            options['step'] = self.renderlist[i]['step']
            options['fps'] = fps  # 24.0
            options['frame_range'] = framerange
            options['frame_range_retime'] = framerange
            options['script_path'] = script_path
            options['scene_file_origin'] = self.currentScene
            options['scene_file'] = renderScene
            options['output_format'] = self.renderlist[i]['output_format']
            options['preview_path'] = '%s/%s_rs.mov' % (previewRoot, previewName)
            options['maxactive'] = self.maxActive
            options['priority'] = 100
            options['projects'] = self.projects  # 'redshift'
            options['tier'] = 'GPU'
            options['tags'] = self.limitTag  # 'gpu_render, gpu_cache'['GPU']
            options['envkey'] = self.envkey  # redshift-2.5.27-2017
            options['service'] = self.service  # 'GPUFARM||Cache||USERGPU'
            options['user'] = self.username
            options['show'] = showname
            options['team'] = self.teamName
            options['log_level'] = 0  # verb = 0 is None, 2 is Detail
            options['speedck'] = self.speed
            jobList.append(options)

        if jobList:
            num = 0
            self.checkDiskSize()
            makepath = os.path.dirname(renderScene)
            if not os.path.exists(makepath):
                print 'makedir = ', makepath
                os.makedirs(makepath)
            self.mbReName(renderScene)
            for job in jobList: # job 갯수만큼 tractor spool
                num += 1
                # render/ 씬 파일 복사 *맨 처음 씬을 저장했던 씬 파일은 중복되므로 패스 처리
                if renderScene != job['scene_file']:
                    self.copyRenderScene(renderScene, job['scene_file'])
                # 스크립트 에디터 및 메세지 박스에 출력할 내용 표시
                line1 = '-' * 60
                msg = '\n >> %s Job Spooled : %s\n' %(str(num).zfill(4), job['layer_name'])
                sendmsg = line1 + msg + line1
                print sendmsg
                keylist = job.keys()
                for keys in keylist:
                    jobps = '>> %s : %s'%(keys, job[keys])
                    print jobps
                # Send Job to Tractor
                jobClass = redshift_Spool.JobScript(job)
                jobClass.doIt()

            msg2 = ' >>> %d Job(s) submitted successfully..!!\n' % num
            main.messageBox(msg2, 'information', ['OK'])
        else:
            main.messageBox('Render List Not Checked !! >> Check Please.', 'warning', ['OK'])

    def imagePlanLayer(self, imagePlanes):
        chkFG = 0
        chkBG = 0
        if imagePlanes:
            for imp in imagePlanes:
                hidden = 0
                getImageMode = cmds.getAttr('%s.displayMode' % imp)
                if getImageMode == 0:
                    hidden = 1

                getImageName = cmds.getAttr('%s.imageName' % imp)
                chkWord = ['_bar', 'bar_', '_letterbox', 'letterbox_', '_right', 'frame_']
                for chk in chkWord:
                    if getImageName.lower().find(chk) != -1:
                        hidden = 1
                        break
                if (hidden == 0) and (os.path.splitext(getImageName)[-1][1:].lower() == 'png'):
                    chkFG += 1
                elif (hidden != 1) and (getImageMode != 0):
                    chkBG += 1
        return chkFG, chkBG

    def renderResolution(self):
        render_width = cmds.getAttr("defaultResolution.w")
        render_height = cmds.getAttr("defaultResolution.h")
        # ani team 1280 초과 랜더 제한
        if self.teamName == 'Ani':
            render_width = 1280
            aspRatio = cmds.getAttr('defaultResolution.deviceAspectRatio')
            y = int(round(render_width / aspRatio))
            if y % 2:# 짝수는 0, 홀수는 1
                y = y + 1
            render_height = y # int
        return render_width, render_height

    def previewMov(self, cameraname, layername):
        previewName = self.scene_base  # redshift_test_proxy_v010_rs.mov
        cameraname = cameraname.replace('Shape', '')
        layername = layername.replace('rs_', '')

        if len(self.renderlist) > 1:  # 레이어가 2개 이상인 경우 뒤에 레이어명 추가
            previewName += '_%s' % layername
        if len(self.getRenderCameras()) > 1:  # 카메라가 2개 이상인 경우 뒤에 카메라명 추가
            if not cameraname == layername:
                previewName += '_%s' % cameraname
        return previewName

    def checkShow(self, showpath):
        if showpath.find('show') > 0:
            imsi = showpath.split('/show/')[-1]
            showname = imsi.split('/')[0]
        return showname

    def retimeScale(self, st, et, by):
        if by != 1:
            et = ((et - st + 1) * by) + (st - 1)  # int( (init_et - st + 1) * by )
        else:
            et = et
        dur = et - st + 1
        if dur <= self.chunkSize:
            limit = 1
        else:
            limitFrame = int(round(dur / self.chunkSize))  # 283 / 6
            limit = int(round(dur / limitFrame))

        frameRange = '%s/%s' % (st, et)
        result = self.iterateFrame(frameRange, limit)
        return result

    def iterateFrame(self, frameRange, limit):
        result = []
        for i in frameRange.split(','):
            if len(i.split('/')) > 1:
                source = i.split('/')
                start_frame = int(source[0])
                end_frame = int(source[-1])

                hostbyframe = (end_frame - start_frame + 1) / limit
                chk_point = hostbyframe * limit + start_frame - 1

                if hostbyframe > 1:
                    for x in range(limit):
                        sf = start_frame + (x * hostbyframe)
                        ef = sf + hostbyframe - 1
                        if x == limit - 1:
                            result.append((sf, end_frame))
                        else:
                            result.append((sf, ef))
                else:
                    for x in range(start_frame, end_frame + 1):
                        result.append((x, x))
            else:
                result.append((i, i))
        return result

    def checkDiskSize(self):
        saveMsgErr = 'Check Storage and Retry please.'
        path = os.path.dirname(self.currentScene)
        freesize = self.getDiskSize(path)[2]
        currentsize = os.path.getsize(self.currentScene)
        if freesize <= currentsize:
            main.messageBox(saveMsgErr, 'critical', ['OK'])
            main.closeUI()

    def getDiskSize(self, path):
        _diskusage = namedtuple('usage', 'total used free')
        disk = os.statvfs(path)
        total = disk.f_bsize * disk.f_blocks
        used = disk.f_bsize * (disk.f_blocks - disk.f_bavail)
        free = disk.f_bsize * disk.f_bavail
        return _diskusage(total, used, free)
        # usage(total=549755813888, used=316236365824, free=233519448064)

    def renderFileName(self):
        random.seed()
        random_name = str(random.randrange(1, 10000, 1))
        filename = '{0}_rs{1}{2}'.format(self.scene_base, random_name, self.scene_ext)
        renderScene = '{0}/{1}_{2}_rs{3}{4}'.format(os.path.dirname(self.currentScene),
                                                    self.scene_base, self.username,
                                                    random_name, self.scene_ext)
        previewRoot = self.mayaProj + '/preview'
        return filename, renderScene, previewRoot

    def copyRenderScene(self, file1, file2):
        # 만약 기존에 파일이 존재할 경우 파일크기와 변경시간을 체크하고 같지 않을 경우 삭제 처리
        if os.path.isfile(file2):
            if os.path.getsize(file1) != os.path.getsize(file2):
                if os.path.getmtime(file1) != os.path.getmtime(file2):
                    os.remove(file2)
        # copy current scene to render
        try:
            shutil.copy2(file1, file2)
        except:
            main.messageBox('Check Scene Render File Copy please !!', 'critical', ['OK'])
            main.closeUI()

    def mbReName(self, renderScene):
        try:
            # 렌더링 할 위치에 새 이름으로 저장할 수 있도록 이름 변경
            cmds.file(rename = renderScene)  # render/redshift_test_proxy_v010_seonku.kim_rs8864.mb
            cmds.file(save = True, f = True)
        except:
            main.messageBox('Check Rename Scene please.', 'critical', ['OK'])
            main.closeUI()
        finally:
            # 에러가 나도 원래 이름으로 다시 돌아가게 처리. 현재 열려 있는 씬 파일은 저장 하지 않음.
            cmds.file(rename = self.currentScene)  # ../redshift_test_proxy_v010.mb


