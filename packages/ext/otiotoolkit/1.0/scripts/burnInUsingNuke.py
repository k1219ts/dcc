#coding:utf-8
import sys
import os
import argparse
import nuke
import subprocess

print(nuke.rawArgs)
argparser = argparse.ArgumentParser()

# mov file name
argparser.add_argument('-sb', '--shotNameBurn', dest='shotNameBurn', type=str, default="True", help='this argument default true, but set argument is false')
argparser.add_argument('--shotname', dest='shotname', type=str, default='', help='burn in from shotname')
argparser.add_argument('--jpgdir', dest='jpgdir', type=str, required=True, help='exported jpg files.')
argparser.add_argument('--color', dest='color', type=str, nargs=4, default=('1', '0', '0', '1'), help='text color ex) 1,0,0,1')
argparser.add_argument('-eb', '--effectBurn', dest='effectBurn', type=str, default="True", help='this argument default true, but set argument is false')
argparser.add_argument('--effect', dest='effect', type=str, default='', help='burn in effect issue')
# argparser.add_argument('--textpos', dest='textpos', type=str, default='RT', choices=('LT', 'RT', 'LB', 'RB'), help='choice text position [LT, RT, LB, RB]')

argStartIndex = nuke.rawArgs.index(__file__) + 1

print('argument setup')
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])
print('jpgDir :', args.jpgdir)
print('color :', args.color)
print('shotName', args.shotname)
print('shotNameBurn', args.shotNameBurn)
print('effectBurn', args.effectBurn)

shotName = args.shotname
if not shotName:
    shotName = os.path.basename(args.jpgdir)

outputPath = os.path.join(args.jpgdir, 'burnin', '{SHOTNAME}.%06d.jpg'.format(SHOTNAME=shotName))

if os.path.exists(os.path.dirname(outputPath)):
    proc = subprocess.Popen('rm -rf %s' % os.path.dirname(outputPath), shell=True)
    proc.wait()
    print("Remove")

nuke.tcl('drop', args.jpgdir)
movReadNode = nuke.selectedNode()
movReadNode['colorspace'].setValue('sRGB')

# print(args.textpos)
color = ' '.join(args.color)#[0], args.color[1], args.color[2], args.color[3])
print(color)
# 'RT'
# shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
#                                font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='-40 -40',
#                                size='{floor(width*0.04)}', xjustify='right', yjustify='top', box='0 0 width height')
# 'RB'
# shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
#                                font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='-40 40',
#                                size='{floor(width*0.04)}', xjustify='right', yjustify='bottom', box='0 0 width height')
# 'LB'
# shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
#                                font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='40 40',
#                                size='{floor(width*0.04)}', xjustify='left', yjustify='bottom', box='0 0 width height')

if args.shotNameBurn == "False":
    shotNameBurn = False
else:
    shotNameBurn = True
print(shotNameBurn)
if not shotNameBurn:
    shotName = ''

shotNameText = nuke.nodes.Text(
    cliptype='no clip',
    message=shotName,
    color=color,
    font='/usr/share/fonts/gnu-free/FreeSansBold.ttf',
    translate='40 -40',
    size='{floor(width*0.04)}',
    xjustify='left',
    yjustify='top',
    box='0 0 width height'
)

shotNameText.setInput(0, movReadNode)

effect = ''
if args.effectBurn == "False":
    effectBurn = False
else:
    effectBurn = True
print(effectBurn)
if args.effect and effectBurn:
    effect = args.effect
    effect = effect.replace('--', '\n').replace(',', ' ')

effectText = nuke.nodes.Text(
    cliptype='no clip',
    message=effect,
    font='/usr/share/fonts/gnu-free/FreeSansBold.ttf',
    translate='-40 -50',
    size='{floor(width*0.013)}',
    xjustify='right',
    yjustify='top',
    box='0 0 width height'
)

effectText.setInput(0, shotNameText)

# outputPath = os.path.join(args.jpgdir, 'burnin', '{SHOTNAME}.%04d.jpg'.format(SHOTNAME=shotName))

if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

write = nuke.nodes.Write(file=outputPath, file_type='jpg', _jpeg_quality=1.0)
write['colorspace'].setValue('rec709')
write.setInput(0, effectText)
