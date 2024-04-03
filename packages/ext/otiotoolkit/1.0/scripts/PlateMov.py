#coding:utf-8
import os
import argparse
import nuke
# import projectSetting
import subprocess

print nuke.rawArgs
argparser = argparse.ArgumentParser()

colorSpaceIn = "ACES - ACES2065-1"
colorSpaceOut = "Output - Rec.709"
argparser.add_argument('--exrDir', dest='exrDir', type=str, required=True, help='input exr directory')
argparser.add_argument('--reelname', dest='reelname', type=str, default='', help='input plate original reel name')
argparser.add_argument('--resolution', dest='resolution', type=str, default='', help='input plate image resolution')
argparser.add_argument('--stamp', dest='stamp', type=str, default='stamp_plate', help='input exr directory')

argStartIndex = nuke.rawArgs.index(__file__) + 1

print 'argument setup'
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])
print args

# exr parse
exrDirName = os.path.basename(args.exrDir)

nuke.tcl('drop', args.exrDir)
for i in nuke.allNodes('Read'):
    if i.error():
        nuke.delete(i)
exrRead = nuke.allNodes('Read')[0]
fileExt = exrRead['file'].getValue().split('.')[-1]

if 'exr' in fileExt:
    nuke.root().knob('colorManagement').setValue('OCIO')
elif 'dpx' in fileExt:
    nuke.root().knob('colorManagement').setValue('Nuke')
    colorSpaceIn = 'Cineon'
    colorSpaceOut = 'rec709'

exrRead['colorspace'].setValue(colorSpaceIn)
if args.resolution:
    exrWidth = args.resolution.split('x')[0]
    exrHeight = args.resolution.split('x')[-1]
else:
    exrWidth = exrRead.metadata()['input/width']
    exrHeight = exrRead.metadata()['input/height']

start = int(nuke.knob('first_frame'))
end = int(nuke.knob('last_frame'))

# stamp setup
stampNode = nuke.createNode(args.stamp)
stampNode.setInput(0, exrRead)
splitExrDir = args.exrDir.split('/')

prjName = 'unknown'
if 'show' in splitExrDir:
    prjName = splitExrDir[splitExrDir.index('show') + 1]
prjName = prjName.upper()

shotName = ''
if 'shot' in splitExrDir:
    seq = splitExrDir[splitExrDir.index('shot') + 1]
    shotName = splitExrDir[splitExrDir.index('shot') + 2]

type = splitExrDir[-2]
version = splitExrDir[-1]
# knob setup
stampNode['Project_name'].setValue(prjName)
stampNode['Shotname'].setValue(shotName)
stampNode['Plate_Version'].setValue(version)
stampNode['Plate_type'].setValue(type)
stampNode['Clip_name'].setValue(args.reelname)

print args.exrDir, 'jpg', exrDirName + '.%04d.jpg'
outputPath = os.path.join(args.exrDir, 'jpg', exrDirName + '.%04d.jpg')

if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

write = nuke.nodes.Write(file=outputPath, file_type='jpg', _jpeg_quality=1.0)
write['colorspace'].setValue(colorSpaceOut)
write.setInput(0, stampNode)

nuke.execute(write, start=start, end=end, incr=1)

# Make MOV
movCmd = '/backstage/dcc/DCC rez-env ffmpeg_toolkit -- ffmpeg_converter -i %s -o %s -c h264' % (os.path.dirname(outputPath),
                                                                                                '{DIR}/{SHOTNAME}_{TYPE}_{VERSION}.mov'.format(DIR=os.path.dirname(args.exrDir),
                                                                                                                                               SHOTNAME=shotName,
                                                                                                                                               TYPE=type,
                                                                                                                                               VERSION=version))
print movCmd
p = subprocess.Popen(movCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
while p.poll() is None:
    output = p.stdout.readline()
    if output:
        print output.strip()

rmDirCmd = 'rm -rf %s' % os.path.dirname(outputPath)
print rmDirCmd
os.system(rmDirCmd)
