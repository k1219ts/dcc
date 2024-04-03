# -*- coding: utf-8 -*-
import os, json
import platform
import ffmpy
import subprocess
from PySide2 import QtWidgets

def chkPlatform():
    # print 'platform:', platform.system()
    return str(platform.system())

def loadCodecConfig(input):
    confPath = ''
    ffmpegRoot = os.path.join(os.environ['REZ_FFMPEG_TOOLKIT_ROOT'], 'scripts')

    if os.path.isdir(ffmpegRoot):
        confPath = ffmpegRoot + '/defaultCodec.json'

    path = input.split('/')
    if 'Linux' == chkPlatform() and 'show' in path:
       show = path.index('show')+1
       path = '/show/{show}/_config/ffmpegCodec.json'.format(show=show)
       if os.path.isfile(path):
           confPath = path

    f = open(confPath, "r")
    codec = json.load(f)

    print '### Codec config path:', confPath
    return codec

def getOutputPath(input):
    output = ''
    if os.path.isdir(input) and '.mov' not in input:
        output = os.path.join(input, 'convert')
    else:
        tmp = input.split('/')
        output = os.path.join('/'.join(tmp[:-1]), 'convert', tmp[-1])

    return output

def resolvePath(path):
    resolve = {'data2': 'dexter',
               'mach': 'show',
               'knot': 'show',
               'data': 'prod_nas'
               }

    if 'Darwin' == chkPlatform():
        if '/Volumes' in path or '/opt' in path:
            path = path.replace('/Volumes', '')
            path = path.replace('/opt', '')

        tmp = path.split('/')
        for key, value in resolve.items():
            for idx, i in enumerate(tmp):
                if key == i:
                    print key, tmp[idx]
                    tmp[idx] = resolve[key]
                    break

        path = '/'.join(tmp)
    return path

def getMOVInfo(movFile):
    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=['-v', 'error',
                                           '-select_streams', 'v:0',
                                           '-show_entries', 'stream=avg_frame_rate',
                                           '-of', 'default=noprint_wrappers=1:nokey=1',
                                           '-print_format', 'json']
                          ).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
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

def getMovMetadata(movFile):
    result = ffmpy.FFprobe(inputs={movFile: None},
                           global_options=['-v', 'quiet',
                                           '-print_format', 'json',
                                           '-show_format', '-show_streams']
                          ).run(stdout=subprocess.PIPE)
    meta = json.loads(result[0].decode('utf-8'))
    return meta

def getSrcSeq(movFile):
    infos = getMovMetadata(movFile)

    metadata = {}
    if infos.has_key('streams'):
        for i in infos['streams']:
            if i.has_key('width'):
                metadata['width'] = i['width']
            if i.has_key('height'):
                metadata['height'] = i['height']
            if i.has_key('avg_frame_rate'):
                if i['avg_frame_rate'] != '0/0':
                    if not metadata.has_key('fps'):
                        metadata['fps'] = round(eval(i['avg_frame_rate'] + '.0'), 2)

    if infos['format']['tags'].has_key('srcSequence'):
        srcPath = os.path.dirname(infos['format']['tags']['srcSequence'])
        srcSeq = os.path.basename(infos['format']['tags']['srcSequence'])
        fileName = srcSeq.split('.')[0]

        find = False
        if os.path.exists(srcPath):
            for file in os.listdir(srcPath):
                if fileName in file:
                    find = True
                    break

        if find:
            metadata['srcSeq'] = infos['format']['tags']['srcSequence']
            metadata['artist'] = infos['format']['tags']['artist']

    return metadata


class inputLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent):
        super(inputLineEdit, self).__init__(parent)

        self.parent = parent
        self.setDragEnabled(True)

    def dragEnterEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            event.acceptProposedAction()

    def dropEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            filePath = '/' + str(urls[0].path())[1:]
            self.setText(filePath)
        self.parent.window().setOutputPath(filePath)
