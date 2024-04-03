#coding:utf-8
import sys
import os
import argparse
import nuke

print nuke.rawArgs
argparser = argparse.ArgumentParser()

# mov file name
argparser.add_argument('--shotname', dest='shotname', type=str, default='', help='burn in from shotname')
argparser.add_argument('--jpgdir', dest='jpgdir', type=str, required=True, help='exported jpg files.')
argparser.add_argument('--color', dest='color', type=str, nargs=4, default=('1', '0', '0', '1'), help='text color ex) 1,0,0,1')
argparser.add_argument('--textpos', dest='textpos', type=str, default='RT', choices=('LT', 'RT', 'LB', 'RB'), help='choice text position [LT, RT, LB, RB]')

argStartIndex = nuke.rawArgs.index(__file__) + 1

print 'argument setup'
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])
print 'jpgDir :', args.jpgdir
print 'textPos :', args.textpos
print 'color :', args.color

for filename in os.listdir(args.jpgdir):
    if os.path.isdir(os.path.join(args.jpgdir, filename)):
        os.system('rm -rf %s' % os.path.join(args.jpgdir, filename))

nuke.tcl('drop', args.jpgdir)
movReadNode = nuke.selectedNode()

shotName = args.shotname
if not shotName:
    shotName = os.path.basename(args.jpgdir)
print args.color
print args.textpos

color = ' '.join(args.color)#[0], args.color[1], args.color[2], args.color[3])
print color
if args.textpos == "RT":
    shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
                                   font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='-40 -40',
                                   size='{floor(width*0.04)}', xjustify='right', yjustify='top', box='0 0 width height')
elif args.textpos == 'LT':
    shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
                                   font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='40 -40',
                                   size='{floor(width*0.04)}', xjustify='left', yjustify='top', box='0 0 width height')
elif args.textpos == 'RB':
    shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
                                   font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='-40 40',
                                   size='{floor(width*0.04)}', xjustify='right', yjustify='bottom', box='0 0 width height')
else:
    shotNameText = nuke.nodes.Text(cliptype='no clip', message=shotName, color=color,
                                   font='/usr/share/fonts/gnu-free/FreeSansBold.ttf', translate   ='40 40',
                                   size='{floor(width*0.04)}', xjustify='left', yjustify='bottom', box='0 0 width height')
shotNameText.setInput(0, movReadNode)


outputPath = os.path.join(args.jpgdir, 'burnin', '%s.#.jpg' % shotName)

if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

write = nuke.nodes.Write(file=outputPath, file_type='jpg', _jpeg_quality=1.0)
write.setInput(0, shotNameText)