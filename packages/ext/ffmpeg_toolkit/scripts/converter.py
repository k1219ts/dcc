import os, sys
import json
import glob
import getpass
import argparse
import pprint
from PIL import Image

import opentimelineio as otio

from pymongo import MongoClient
import dxConfig

_EXTS = ['jpg', 'jpeg', 'tif', 'png', 'exr', 'mov']
DB_IP = dxConfig.getConf("DB_IP")
DB_NAME = 'Editorial'


class EncodingMov:
    def __init__(self, args, ext):
        self.args = args
        self.ext = ext
        self.show = ''
        self.shot = ''
        self.num, self.inputfile = self.GetImageFile(self.args.input)
        self.codec = self.loadCodecConfig()
        self.shotInfo = None # self.getDbRecord()

        print '### num:', self.num
        print '### inputfile:',self.inputfile

    def loadCodecConfig(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        confPath = pwd + '/defaultCodec.json'

        path = self.inputfile.split('/')
        if 'show' in path:
            self.show = path[path.index('show')+1]
            tmp = os.path.basename(self.inputfile)
            self.shot = '_'.join(tmp.split('_')[:2])

            print '### show:', self.show
            print '### shot:', self.shot

            path = '/show/{show}/_config/ffmpegCodec.json'.format(show=self.show)
            if os.path.isfile(path):
                confPath = path

        f = open(confPath, "r")
        codec = json.load(f)

        print '### Codec config path:', confPath
        return codec

    def GetImageFile(self, input):
        start_num = None
        filename  = None
        data = dict()

        # mov to mov
        if self.ext =='mov':
            return '', args.input

        # seq dir
        elif os.path.isdir(input):
            for ext in _EXTS:
                searchstr = '%s/*.%s' % (input, ext)

                files = glob.glob(searchstr)
                files.sort()
                data[self.ext] = files
                break

        # seq image files
        elif '%0' in input:
            name = os.path.basename(input).split('.')[0]
            input = os.path.dirname(input)

            searchstr = '%s/%s*.%s' % (input, name, self.ext)
            files = glob.glob(searchstr)
            files.sort()
            data[self.ext] = files
        else:
            name = os.path.basename(input)
            path = os.path.dirname(input)

            src = name.split('.')
            start_num = src[-2]
            filename  = os.path.join(path,
                                     name.replace(start_num, '%0' + str(len(start_num)) + 'd'))
            return start_num, filename

        for ext, files in data.items():
            if len(files) > 1:
                name = os.path.basename(files[0])
                path = os.path.dirname(files[0])

                src = name.split('.')
                start_num = src[-2]
                filename  = os.path.join(path,
                                         name.replace(start_num, '%0' + str(len(start_num)) + 'd'))
                break
            else:
                filename = files[0]
                name = os.path.basename(filename)
                src = name.split('.')
                start_num = src[-2]

        return start_num, filename

    def getDbRecord(self):
        if self.show and self.shot:
            client = MongoClient(DB_IP)
            db = client[DB_NAME]
            coll = db[self.show]
            recentDoc = coll.find_one({'$and':[{'shot_name':self.shot},
                                      {'$or':[{'type':'main1'}, {'type':'main1_org'}]}]})
            if recentDoc:
                pprint.pprint(recentDoc)
                return recentDoc
        return ''

    def to_mov(self):
        if self.args.output.split('.')[-1] == 'mov':
            outfile = self.args.output
        else:
            basename = os.path.basename(self.inputfile).split('.')[0]
            outfile  = os.path.join(self.args.output, basename + '.mov')
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        user = self.args.user
        if not user:
            user = getpass.getuser()

        command  = 'ffmpeg'
        command += ' -r %s' % self.args.rate     # source rate
        if self.ext != 'mov':
            command += ' -start_number %s' % self.num
        command += ' -i %s' % self.inputfile
        if self.args.inputaudio:
            command += ' -r %s -i %s' % (self.args.rate, self.args.inputaudio)
            if self.args.aframes:
                command += ' -aframes %s' % self.args.aframes
            self.args.audio = True
        command += ' ' + self.codec[self.args.codec]
        command += ' -r %s' % self.args.rate     # output rate
        if not self.args.audio:
            command += ' -an'

        # timecode
        if self.ext != 'mov':
            if self.shotInfo:
                try:
                    if int(self.num) == int(self.shotInfo['frame_in']) or int(self.num) == 1:
                        tc = self.shotInfo['tc_in']
                    else:
                        diff = int(self.num) - int(self.shotInfo['frame_in'])
                        tc_frame = otio.opentime.from_timecode(self.shotInfo['tc_in'],
                                                               float(self.args.rate)).to_frames()
                        tc_frame += diff
                        tc = otio.opentime.from_frames(tc_frame,
                                                       float(self.args.rate)).to_timecode()
                except:
                    tc = self.shotInfo['tc_in']
            else:
                try:
                    tc = otio.opentime.from_frames(int(self.num),
                                                   float(self.args.rate)).to_timecode()
                    tc = tc.replace(';',':')
                except:
                    tc = '00:00:00:00'
            if tc:
                command += ' -timecode %s' % tc
                print '### timecode:', tc

        # metadata
        command += ' -movflags use_metadata_tags'
        command += ' -metadata artist="%s"' % str(user)
        command += ' -metadata: srcSequence="%s"' % str(self.inputfile)
        command += ' -metadata: vendor="DEXTER STUDIOS"'
        if self.args.metadata:
            command += ' -metadata: %s' % self.args.metadata

        # mov resolution
        if self.args.s:
            command += ' -s %s' % self.args.s
        else:
            if self.ext != 'mov':
                try:
                    im = Image.open(self.inputfile.replace('%0' + str(len(self.num)) + 'd', self.num))
                    width, height = im.size

                    if width%2: width += 1
                    if height%2: height += 1

                    command += ' -s %sx%s' % (width, height)
                    print '### resolution: %sx%s' % (width, height)
                except:
                    pass

        command += ' -y ' + outfile
        print 'command:', command

        if self.args.command:
            sys.exit(command)
        else:
            os.system(command)
            print '# result :', outfile
            sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--codec', type=str, default='h265',
        help='Codec: h264, h265, ,h265HDR, proresProxy, proresLT, mjpeg')
    parser.add_argument('-i', '--input', type=str, help='Input directory')
    parser.add_argument('-a', '--inputaudio', type=str, help='Input audio')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('--command', action='store_true', default=False, help='get command')
    parser.add_argument('-au', '--audio', action='store_true', default=False, help='enable audio')
    parser.add_argument('-af', '--aframes', type=str, default=False, help='audio start frame')
    parser.add_argument('-u', '--user', type=str, help='Encoding user')
    parser.add_argument('-r', '--rate', type=float, default=23.976, help='set frame rate')
    parser.add_argument('-s', type=str, default=False, help='mov resolution')
    parser.add_argument('-metadata', type=str, default=False, help='add metadata')

    args, unknown = parser.parse_known_args()

    if not args.input:
        print '# Error : not found input.'
        sys.exit(1)

    args.input = os.path.abspath(args.input)

    if args.output:
        args.output = os.path.abspath(args.output)
    else:
        args.output = os.path.dirname(args.input)

    # images to mov
    ext = args.input.split('.')[-1]
    EncodingMov(args, ext).to_mov()
    # print args
