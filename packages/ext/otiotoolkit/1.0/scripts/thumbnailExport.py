'ffmpeg -i {INPUT} -map 0 -map -0:d -c copy -timecode 00:00:00:01 {OUTPUT}'
from tactic_client_lib import TacticServerStub
from Define import TACTIC
import requests, os, sys, ffmpy, subprocess
from core import calculator
import Msg
import json
import opentimelineio as otio

def main(rootDir):
    convertDir = os.path.join(rootDir, 'thumbnail')
    if not os.path.exists(convertDir):
        os.makedirs(convertDir)

    excludeShotNameList = []
    for filename in os.listdir(rootDir):
        if not filename.startswith('.') and os.path.isfile(os.path.join(rootDir, filename)) and filename.endswith('.mov'):
            print filename

            # os.path.join()
            # # VELOZ TC CHECK
            # movFile = os.path.join(rootDir, filename)
            # movTC, movDUR, movFPS = getMOVInfo(movFile)
            # print movTC, movDUR, movFPS
            #
            # velozFrameIn = queryData[0]['frame_in']
            # velozFrameOut = queryData[0]['frame_out']
            # velozDuration = velozFrameOut - velozFrameIn + 1
            # velozStartTC = otio.opentime.RationalTime(velozFrameIn, fps)
            # print velozFrameIn, velozFrameOut, velozDuration, velozStartTC.to_timecode()
            #
            # command = ['/backstage/dcc/DCC', 'rez-env', 'ffmpeg_toolkit', '--']
            # command += ['ffmpeg', '-r', str(fps)]
            # command += ['-i', movFile]
            # command += [codecOptions[codec]]
            # command += ['-timecode', velozStartTC.to_timecode()]
            # command += ['-s', '%sx%s' % (resolution[0], resolution[1])]
            # command += ['-y', os.path.join(convertDir, filename)]
            #
            # strCmd = ' '.join(command)
            # os.system(strCmd)

if __name__ == '__main__':
    # main(sys.argv[-1])
    main('/stuff/prat2/stuff/ftp/edit/from_dexter/20210304/PS83')