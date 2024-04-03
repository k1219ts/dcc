#coding:utf-8
import os
import argparse
import nuke
import projectSetting
import subprocess
from tactic_client_lib import TacticServerStub

deliveryToProjectionSize = {'2048x858': (2048, 858, 'height'),
                            '2288x858': (2288, 858, 'height'),
                            '2059x858': (2059, 858, 'height'),
                            '2324x858': (2324, 858, 'height'),
                            '2050x858': (2050, 858, 'height'),
                            '2048x1080': (2048, 858, 'width'),
                            '2048x1152': (2048, 858, 'width'),
                            '2048x1318': (2048, 858, 'width'),
                            '2048x1024': (2048, 858, 'width'),
                            '2048x1090': (2048, 858, 'width')
                            }

print nuke.rawArgs
argparser = argparse.ArgumentParser()

# mov file name
# argparser.add_argument('--colorSpaceIn', dest='colorSpaceIn', type=str, required=True, help='input color space')
# argparser.add_argument('--colorSpaceOut', dest='colorSpaceOut', type=str, required=True, help='output color space')

colorSpaceIn = "ACES - ACES2065-1"
colorSpaceOut = "Output - Rec.709"
argparser.add_argument('--exrDir', dest='exrDir', type=str, required=True, help='input exr directory')
argparser.add_argument('--stamp', dest='stamp', type=str, default='Stamp_DEFAULT', help='input exr directory')

argStartIndex = nuke.rawArgs.index(__file__) + 1

print 'argument setup'
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])
print args

# exr parse
exrDirName = os.path.basename(args.exrDir)
# splitExrDirName = exrDirName.split('_')
# seq = splitExrDirName[0]
# shot = splitExrDirName[1]
# version = splitExrDirName[3]

# OCIO Config Setup
try:
    projectSetting.settingBDS()
except:
    pass

nuke.tcl('drop', args.exrDir)
exrRead = nuke.selectedNode()
exrRead['colorspace'].setValue(colorSpaceIn)
exrWidth = exrRead.metadata()['input/width']
exrHeight = exrRead.metadata()['input/height']

deliverySize = '{WIDTH}x{HEIGHT}'.format(WIDTH=exrWidth, HEIGHT=exrHeight)
# if not deliveryToProjectionSize.has_key(deliverySize):
#     assert False, "delivery Size Error"

start = int(nuke.knob('first_frame'))
end = int(nuke.knob('last_frame'))

# stamp setup
if args.stamp == "slate_vendor_cds":
    stampNode = nuke.createNode('slate_vendor_cds')

    stampNode['seq'].setValue(seq)
    stampNode['shot'].setValue(shot)
    stampNode['version'].setValue(version)
    stampNode['artist'].setValue('wonhee')

    stampNode['input.first_1'].setValue(start)
    stampNode['input.last_1'].setValue(end)

    cropBoxSize = deliveryToProjectionSize[deliverySize]
    stampNode['crop_box_width'].setValue(2048)
    stampNode['crop_box_height'].setValue(cropBoxSize[1])
    stampNode['resize'].setValue(cropBoxSize[2])

    stampNode.setInput(1, exrRead)
else:
    stampNode = nuke.createNode('stamp_default')
    stampNode.setInput(0, exrRead)
    splitExrDir = args.exrDir.split('/')
    prjName = 'unknown'
    if 'show' in splitExrDir:
        prjName = splitExrDir[splitExrDir.index('show') + 1]
    prjName = prjName.upper()

    stampNode['Project_name'].setValue(prjName)
    stampNode['Artist_name'].setValue('DEXTER')
    stampNode['LetterBox'].setValue(0)

    shotName = ''
    if 'shot' in splitExrDir:
        seq = splitExrDir[splitExrDir.index('shot') + 1]
        shotName = splitExrDir[splitExrDir.index('shot') + 2]
    stampNode['Shotname'].setValue(shotName)


print args.exrDir, 'jpg', exrDirName + '.%04d.jpg'
outputPath = os.path.join(args.exrDir, 'jpg', exrDirName + '.%04d.jpg')

if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

write = nuke.nodes.Write(file=outputPath, file_type='jpg', _jpeg_quality=1.0)
write['colorspace'].setValue(colorSpaceOut)
write.setInput(0, stampNode)

nuke.execute(write, start=start, end=end, incr=1)

# Make MOV
movCmd = '/backstage/dcc/DCC rez-env ffmpeg_toolkit -- ffmpeg_converter -i %s -o %s -c proresLT' % (os.path.dirname(outputPath), os.path.dirname(args.exrDir))
print movCmd
p = subprocess.Popen(movCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
while p.poll() is None:
    output = p.stdout.readline()
    if output:
        print output.strip()

rmDirCmd = 'rm -rf %s' % os.path.dirname(outputPath)
print rmDirCmd
os.system(rmDirCmd)

# command
# DCC.local rez-env ffmpeg nuke-10 -- nukeX -i -t /WORK_DATA/develop/dcc/packages/ext/otiotoolkit/examples/tmp/exrToMovUsingNuke.py --exrDir /show/yys/stuff/ftp/_257/to_dexter/20200914/comp/exr/OK/XYB_0160_comp_v011