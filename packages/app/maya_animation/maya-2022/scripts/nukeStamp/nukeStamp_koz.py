import os, shutil
import argparse
import datetime
import getpass
import subprocess
import DXRulebook.Interface as rb
from dxstats import inc_tool_by_user as log

argparser = argparse.ArgumentParser()
argparser.add_argument('--jpgPath', dest='jpgPath', type=str, required=True, help='input seq directory')
argparser.add_argument('--impPath', dest='impPath', type=str, default=False, help='input seq directory')
argparser.add_argument('--fileName', dest='fileName', type=str, required=True, help='input filename.')
argparser.add_argument('--comments', dest='comments', type=str, nargs='+', required=True, help='input comments.')

argStartIndex = nuke.rawArgs.index(__file__) + 1
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])

print 'jpgPath:', args.jpgPath
print 'fileName:', args.fileName
print 'comments:', ' '.join(args.comments)

nuke.tcl('drop', args.jpgPath)

platePath = ''
if args.impPath:
    print 'impPath:', args.impPath
    coder = rb.Coder()
    argv = coder.D.IMAGEPLANE.IMAGES.Decode(args.impPath)
    platePath = coder.D.PLATES.IMAGES.Encode(**argv)
    print 'platePath:', platePath
    nuke.tcl('drop', platePath)

for i in nuke.allNodes('Read'):
    if i.error():
        nuke.delete(i)
        continue

    if '.jpg' in i['file'].getValue():
        readJpg = i
    elif '.exr' in i['file'].getValue():
        readPlate = i

if platePath:
    timeCode = nuke.createNode('CopyMetaData')
    timeCode.setInput(0, readJpg)
    timeCode.setInput(1, readPlate)
else:
    timeCode = nuke.createNode('AddTimeCode')
    timeCode['startcode'].setValue('00:00:00:00')
    timeCode.setInput(0, readJpg)

first = readJpg['first'].getValue()
last = readJpg['last'].getValue()

stamp = nuke.createNode('stamp_koz')
stamp['first_frame'].setValue(first)
stamp['last_frame'].setValue(last)
stamp['shotcb'].setValue(1)
stamp['Shotname'].setValue(args.fileName)
stamp['Note'].setValue(' '.join(args.comments))
stamp.setInput(0, timeCode)

ref = nuke.createNode('Reformat')
ref['type'].setValue('to box')
ref['box_width'].setValue(1920)
ref['box_height'].setValue(1080)
ref.setInput(0, stamp)

ref2 = nuke.createNode('Reformat')
ref2['type'].setValue('to box')
ref2['box_width'].setValue(2048)
ref2['box_height'].setValue(1152)
ref2.setInput(0, stamp)

copyFiles = []
filename = args.fileName + '.mov'
dir = os.path.dirname(os.path.dirname(readJpg['file'].getValue()))
output = os.path.join(dir, filename)

copyFiles.append(output)
print 'mov:', output

write = nuke.createNode('Write')
write['file'].setValue(output)
write['file_type'].setValue('mov')
write['colorspace'].setValue('rec709')
write['mov_prores_codec_profile'].setValue('ProRes 4:2:2 LT 10-bit')
write.setInput(0, ref2)

write2 = nuke.createNode('Write')
write2['file'].setValue(output.replace('.mov', '.mxf'))
write2['file_type'].setValue('mxf')
write2['mxf_op_pattern_knob'].setValue('OP-Atom')
write2['colorspace'].setValue('rec709')
write2.setInput(0, ref)

copyFiles.append(output.replace('.mov', '.mxf'))
print 'mxf:', output.replace('.mov', '.mxf')

start = int(nuke.knob('first_frame'))-1
end = int(nuke.knob('last_frame'))

print 'start, end:', start, end
nuke.execute(write, start=start, end=end, incr=1)
nuke.execute(write2, start=start, end=end, incr=1)

# set metadata for Avid (material_package_name)
cmd = 'ffmpeg -i "{INPUT}" -metadata material_package_name="{FILENAME}" -c:v copy -f mxf_opatom "{OUTPUT}" -y'.format(INPUT=output.replace('.mov', '_v1.mxf'), FILENAME=args.fileName, OUTPUT=output.replace('.mov', '.mxf'))
print 'cmd:', cmd
p = subprocess.Popen(cmd, shell=True)
p.wait()

# temporary mxf file remove (ex *_v1.mxf
os.remove(output.replace('.mov', '_v1.mxf'))

today = datetime.datetime.today().strftime("%Y%m%d")
stuff = '/show/koz/stuff/_feedback/_confirm/%s' % today

if not os.path.isdir(stuff):
    os.makedirs(stuff)

for file in copyFiles:
    print 'copy: %s -> %s' % (file, os.path.join(stuff, os.path.basename(file)))
    shutil.copy(file, os.path.join(stuff, os.path.basename(file)))

log.run('action.nukeStamp.koz', getpass.getuser())
