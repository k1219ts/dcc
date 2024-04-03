'ffmpeg -i {INPUT} -map 0 -map -0:d -c copy -timecode 00:00:00:01 {OUTPUT}'
from tactic_client_lib import TacticServerStub
from Define import TACTIC
import requests, os, sys, ffmpy, subprocess
from core import calculator
import Msg
import json
import opentimelineio as otio

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = TACTIC.API_KEY
    requestParam['name'] = projectName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC.IP), params=requestParam)

    projectInfo = responseData.json()[0]
    return projectInfo['code'], projectInfo['sync']

def getMOVInfo(movFile):
    fpsMapper = {'24000/1001':23.98,
                 '23.976': 23.98,
                 '23.98': 23.98,
                 '24': 24,
                 '24/1': 24}
    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=['-v', 'error',
                                           '-select_streams', 'v:0',
                                           '-show_entries', 'stream=avg_frame_rate',
                                           '-of', 'default=noprint_wrappers=1:nokey=1',
                                           '-print_format', 'json']
                          ).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
    # fps = fpsMapper[meta['streams'][0]['avg_frame_rate']]
    fps = round(eval(meta['streams'][0]['avg_frame_rate'] + '.0'), 2)

    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=[
                               '-show_format', '-pretty',
                               '-loglevel', 'quiet',
                               '-print_format', 'json'
                           ]).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
    timecode = otio.opentime.RationalTime.from_timecode(meta['format']['start_time'], fps).to_timecode()

    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=[
                               '-v', 'error',
                               '-show_entries', 'format=duration',
                               '-of', 'default=noprint_wrappers=1:nokey=1',
                               '-print_format', 'json']).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
    duration = int(float(meta['format']['duration']) * fps)
    return timecode, duration, fps

codecOptions = {
  "h264": "-c:v libx264 -profile:v baseline -b 30000k -tune zerolatency -preset slow -pix_fmt yuv420p",
  "h265": "-c:v libx265 -crf 10 -tune fastdecode -pix_fmt yuv420p -tag:v hvc1",
  "mjpeg": "-c:v mjpeg -q:v 2 -pix_fmt yuv420p",
  "prores4444": "-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le",
  "proresProxy": "-c:v prores_ks -profile:v 0 -q:v 4 -pix_fmt yuv422p10le",
  "proresLT": "-c:v prores -profile 1"
}

# 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 /stuff/prat2/stuff/ftp/edit/from_dexter/20210304/PS83/PS83_0030_ani_animation_animation_v009.mov' # DURATION
# 'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 /stuff/prat2/stuff/ftp/edit/from_dexter/20210304/PS83/PS83_0030_ani_animation_animation_v009.mov' # FPS
# 'ffprobe -show_format -pretty -loglevel quiet -print_format json -i /stuff/prat2/stuff/ftp/edit/from_dexter/20210304/PS83/PS83_0030_ani_animation_animation_v009.mov'

def main(rootDir):
    showName = calculator.parseShowName(rootDir)
    print showName
    project_code, sync = getShowCode(showName.lower())

    # Find ConfigFile
    configFile = '/show/{SHOWNAME}/_config/Project.config'.format(SHOWNAME=showName)

    if not os.path.exists(configFile):
        Msg.error("Not Find Config File : %s" % configFile)
        return

    f = open(configFile, 'r')
    configData = json.load(f)

    deliveryRule = configData['deliveryMOV']
    fps = deliveryRule['fps']
    codec = deliveryRule['codec']
    resolution = deliveryRule['resolution']
    # FPS, RESOLUTION, CODEC

    convertDir = os.path.join(rootDir, 'convert')
    if not os.path.exists(convertDir):
        os.makedirs(convertDir)

    tactic = TacticServerStub(login=TACTIC.LOGIN, password=TACTIC.PASSWORD, server=TACTIC.IP, project=project_code)
    tactic.start()

    shot_search_type = '%s/shot' % project_code

    excludeShotNameList = []
    for filename in os.listdir(rootDir):
        if not filename.startswith('.') and os.path.isfile(os.path.join(rootDir, filename)):
            print filename
            splitFileName = filename.split('.')[0].split('_')
            shotName = '%s_%s' % (splitFileName[0], splitFileName[1])
            queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
            if not queryData:
                excludeShotNameList.append(shotName)
                continue

            # VELOZ TC CHECK
            movFile = os.path.join(rootDir, filename)
            movTC, movDUR, movFPS = getMOVInfo(movFile)
            print movTC, movDUR, movFPS

            velozFrameIn = queryData[0]['frame_in']
            velozFrameOut = queryData[0]['frame_out']
            velozDuration = velozFrameOut - velozFrameIn + 1
            velozStartTC = otio.opentime.RationalTime(velozFrameIn, fps)
            print velozFrameIn, velozFrameOut, velozDuration, velozStartTC.to_timecode()

            command = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
            command += ['ffmpeg', '-r', str(fps)]
            command += ['-i', movFile]
            command += [codecOptions[codec]]
            command += ['-timecode', velozStartTC.to_timecode()]
            command += ['-s', '%sx%s' % (resolution[0], resolution[1])]
            command += ['-y', os.path.join(convertDir, filename)]

            strCmd = ' '.join(command)
            os.system(strCmd)

if __name__ == '__main__':
    # main(sys.argv[-1])
    main('/stuff/prat2/stuff/ftp/edit/from_dexter/20210304/PS83')