import argparse
import nuke
import os
import subprocess


def getEndFrame(out):
    import glob
    dir = os.path.dirname(out)
    print('dir:',dir)
    filename = out.split('/')[-1]
    filename = filename.split('.')[0]
    files = glob.glob('%s/%s*' % (dir, filename))
    filelist = []
    for file in files:
        if len(file.split('.')) < 4:
            filelist.append(file)
    filelist.sort()
    print('filelist:',filelist)
    endf = filelist[-1]
    print('lastfile',filelist[-1])
    endf = endf.split('/')[-1].split('.')[1]
    endf = int(endf)
    return endf


argparser = argparse.ArgumentParser()
argparser.add_argument('--exrFiles', dest='exrFiles', type=str, required=True, help='input exr directory')
argparser.add_argument('--outJpgPath', dest='outJpgPath', type=str, required=True, help='input exr directory')

argStartIndex = nuke.rawArgs.index(__file__) + 1

print('argument setup')
args = argparser.parse_args(nuke.rawArgs[argStartIndex:])
print('args:', args)

nuke.root().knob('colorManagement').setValue('OCIO')

nuke.tcl('drop', args.exrFiles)
exrRead = nuke.allNodes('Read')[0]
# exrRead['colorspace'].setValue('ACES - ACES2065-1')
exrRead['colorspace'].setValue('ACES - ACEScg')



outdir = os.path.dirname(args.outJpgPath)
if not os.path.exists(outdir):
    os.makedirs(outdir)

write = nuke.nodes.Write(file=args.outJpgPath, file_type='jpg', _jpeg_quality=1.0, colorspace='Output - Rec.709')
write.setInput(0, exrRead)

start = int(nuke.knob('first_frame'))
end = getEndFrame(args.exrFiles)
exrRead['last'].setValue(end)

nuke.execute(write, start=start, end=end, incr=1)
